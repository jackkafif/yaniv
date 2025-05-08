import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from game.state import GameState
from game.globals import *
from game.model import M, MDQN
import os


class YanivAgent:
    def __init__(self, model_dir, state_size=STATE_SIZE, M1 = MDQN, M2 = MDQN, M3 = MDQN):

        self.state_size = state_size

        self.model_dir = model_dir

        self.model_phase1 = M1(state_size, PHASE1_ACTION_SIZE)
        self.model_phase2 = M2(state_size, PHASE2_ACTION_SIZE)
        self.model_phase3 = M3(state_size, PHASE3_ACTION_SIZE)

        self.epsilon = 1.0
        self.epsilon_decay = 0.995

    def load_models(self, i: str):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        try:
            self.model_phase1.model.load_state_dict(
                torch.load(f"{MODEL_DIR}/{self.model_dir}/agent{i}_phase1.pt"))
            self.model_phase2.model.load_state_dict(
                torch.load(f"{MODEL_DIR}/{self.model_dir}/agent{i}_phase2.pt"))
            self.model_phase3.model.load_state_dict(
                torch.load(f"{MODEL_DIR}/{self.model_dir}/agent{i}_phase3.pt"))
        except:
            self.save_models(i)

    def save_models(self, i: str):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        torch.save(self.model_phase1.model.state_dict(),
                   f"{MODEL_DIR}/{self.model_dir}/agent{i}_phase1.pt")
        torch.save(self.model_phase2.model.state_dict(),
                   f"{MODEL_DIR}/{self.model_dir}/agent{i}_phase2.pt")
        torch.save(self.model_phase3.model.state_dict(),
                   f"{MODEL_DIR}/{self.model_dir}/agent{i}_phase3.pt")

    def choose_action_phase1(self, state: GameState, hand: np.ndarray, other: np.ndarray, top_cards : list):
        if not state.can_yaniv(hand):
            return 0
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            q_vals = self.model_phase1.model(
                state.to_tensor(hand, other, top_cards).unsqueeze(0))
        return int(torch.argmax(q_vals))

    def choose_action_phase2(self, state: GameState, hand: np.ndarray, other: np.ndarray, top_cards : list):
        valid_moves = state.valid_move_indices(hand)
        if random.random() < self.epsilon:
            return random.choice(np.where(valid_moves > 0)[0])
        with torch.no_grad():
            q_vals = self.model_phase2.model(
                state.to_tensor(hand, other, top_cards).unsqueeze(0))
            q_vals = q_vals.squeeze(0)
            q_vals[valid_moves.T <= 0] = -np.inf
        return int(torch.argmax(q_vals))

    def choose_action_phase3(self, state: GameState, hand: np.ndarray, other: np.ndarray, top_cards : list):
        valid_draws_mask = state.valid_draws()
        if random.random() < self.epsilon:
            return random.choice(valid_draws_mask.nonzero()[0])
        with torch.no_grad():
            tensor = state.to_tensor(hand, other, top_cards).unsqueeze(0)
            print(f"Tensor is {tensor}")
            q_vals = self.model_phase3.model(
                tensor)
            q_vals = q_vals.squeeze(0)
            q_vals[valid_draws_mask == 0] = -np.inf
        move = int(torch.argmax(q_vals))
        return move

    def schoose_turn_moves(self, state: GameState, hand: np.ndarray, other: np.ndarray):
        # Convert the state to a tensor
        state_tensor = state.to_tensor(hand, other)

        # Choose actions for each phase
        phase1 = self.choose_action_phase1(state, hand, other)
        phase2 = self.choose_action_phase2(state, hand, other)
        phase3 = self.choose_action_phase3(state, hand, other)

        # Return the chosen actions
        return phase1, phase2, phase3

    def play_agent(self, game: GameState, hand: np.ndarray, other: np.ndarray, debug=True):
        state_tensor = game.to_tensor(hand, other, game.top_cards)

        a1 = self.choose_action_phase1(game, hand, other, game.top_cards)

        # Initialize intermediate losses/rewards
        phase_1_intermediate_loss = 0
        phase_2_intermediate_loss = 0
        phase_3_intermediate_loss = 0

        # Check if player calls Yaniv prematurely or correctly
        if a1 == 1:
            won = game.yaniv(hand, [other])
            if won:
                reward = 1
            else:
                reward = -50 if a1 == 1 else -1  # Strong penalty for premature Yaniv
            self.model_phase1.add_episode(state_tensor, a1, reward)
            return True, won

        a2 = self.choose_action_phase2(game, hand, other, game.top_cards)
        discard_indices = POSSIBLE_MOVES[int(a2)]
        move_played = list(discard_indices)

        valid_moves = list(game.valid_moves(hand))
        move_values = {POSSIBLE_MOVES[valid_moves[i]]: game.move_value(
            hand, POSSIBLE_MOVES[valid_moves[i]]) for i in range(len(valid_moves))}
        # Phase 2: Discard cards and draw new card
        hc = hand.copy()
        game.play(hand, move_played)

        # Penalize inefficient discards
        if 2 * move_values[tuple(move_played)] <= move_values[max(move_values)]:
            phase_2_intermediate_loss -= 20

        # Improved phase 3 intermediate loss logic
        state_tensor = game.to_tensor(hand, other, game.tc_holder)
        a3 = self.choose_action_phase3(game, hand, other, game.tc_holder)
        top_card_values = [game.card_value(card) for card in game.tc_holder]
        
        if a3 == 52:  # Draw unknown card
            if min(top_card_values) <= 3:
                phase_3_intermediate_loss -= 15  # Penalize skipping low-value visible card
        else:
            if len(game.tc_holder) > 1:

                chosen_card = a3
                other_card = game.tc_holder[0] if game.tc_holder[0] != a3 else game.tc_holder[1]

                completes, _ = game.completes_move(hand, chosen_card)
                if completes:
                    phase_3_intermediate_loss += 9  # Reward drawing card completing a combination
                elif game.card_value(chosen_card) > game.card_value(other_card):
                    phase_3_intermediate_loss -= 14  # Penalize taking higher-value card
                else:
                    # Slight reward for drawing low-value card
                    phase_3_intermediate_loss += 5 if game.card_value(
                        chosen_card) <= 3 else 0
            else:
                completes, _ = game.completes_move(hand, a3)
                if completes:
                    phase_3_intermediate_loss += 10
                if game.card_value(a3) <= 3:
                    phase_3_intermediate_loss += 4

        if len(move_played) == 0:
            raise ValueError("No cards played in phase 2")
        # Print debug information
        if debug:
            played = []
            nzs = np.nonzero(hc)[0]
            counter = 0
            for nz in nzs:
                if counter in move_played:
                    played.append(game.card_to_name(nz))
                counter += 1

            print(f"""
                Top card values: {top_card_values}
                Top cards: {[game.card_to_name(game.tc_holder[i]) for i in range(len(game.tc_holder))]}
                Hand: {game.hand_to_cards(hc)}
                Discarded cards: {played}
                Drawn card: {game.card_to_name(a3) if a3 != 52 else "deck"}
                Phase 3 intermediate loss: {phase_3_intermediate_loss}
                """)
        # phase_3_intermediate_loss = 0
        # Reward/penalize changes in hand value after move
        hand_value_before = game.get_hand_value(hand)

        drawn_card_idx = game.draw(hc, move_played, a3)
        hand[drawn_card_idx] += 1

        hand_value_after = game.get_hand_value(hand)
        hand_value_change = hand_value_before - hand_value_after

        if hand_value_change >= 5:
            phase_2_intermediate_loss += 10  # Reward significantly lowering hand value
        elif hand_value_change < 0:
            phase_2_intermediate_loss -= 10  # Penalize increasing hand value

        # Apply episodes with calculated intermediate losses
        self.model_phase1.add_episode(
            state_tensor, a1, phase_1_intermediate_loss)
        self.model_phase2.add_episode(
            state_tensor, a2, phase_2_intermediate_loss)
        self.model_phase3.add_episode(
            state_tensor, a3, phase_3_intermediate_loss)

        return False, False

    def train(self, reward):
        # Train the models
        self.model_phase1.replay_game(reward)
        self.model_phase2.replay_game(reward)
        self.model_phase3.replay_game(reward)

        # Decay epsilon
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay


def play_agent(state: GameState, m: YanivAgent, hand: np.ndarray, other: np.ndarray):
    # Convert the state to a tensor
    state_tensor = state.to_tensor(hand, other)

    # Choose actions for each phase
    phase1 = m.choose_action_phase1(state, hand, other)
    phase2 = m.choose_action_phase2(state, hand, other)
    phase3 = m.choose_action_phase3(state, hand, other)

    return phase1, phase2, phase3


def run_game(p1: YanivAgent, p2: YanivAgent):
    # Returns True if player 1 wins, False if player 2 wins
    game = GameState()

    p1_hand = game.player_1_hand
    p2_hand = game.player_2_hand

    while True:
        done, won = p1.play_agent(game, p1_hand, p2_hand)
        if done:
            return won

        # Player 2's turn
        done, won = p2.play_agent(game, p2_hand, p1_hand)
        if done:
            return not won
