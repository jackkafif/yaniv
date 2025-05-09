import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from game.state import GameState
from game.globals import *
from game.model import M, MDQN, ML1
import os

debug = False
d = False
debug = True
d = True

p2_state_size = 137
class YanivAgent:
    def __init__(self, model_dir, state_size=STATE_SIZE, M1=MDQN, M2=MDQN, M3=MDQN):

        self.state_size = state_size

        self.model_dir = model_dir

        self.model_phase1 = M1(state_size, PHASE1_ACTION_SIZE)
        self.model_phase2 = M2(137, PHASE2_ACTION_SIZE)
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

    def choose_action_phase1(self, state: GameState, hand: np.ndarray, other: np.ndarray, top_cards: list , debug=debug):
        if not state.can_yaniv(hand):
            return 0
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            q_vals = self.model_phase1.model(
                state.to_tensor(hand, other, top_cards).unsqueeze(0))
        # if debug:
        #     print(f"Phase 1 Q values: {q_vals}")
        return int(torch.argmax(q_vals))

    def choose_action_phase2(self, state: GameState, hand: np.ndarray, other: np.ndarray, top_cards: list, debug=debug):
        valid_moves = state.valid_move_indices(hand)
        if random.random() < self.epsilon:
            return random.choice(np.where(valid_moves > 0)[0])
        with torch.no_grad():
            feats = state.get_features_2(hand, other, top_cards).unsqueeze(0)
            # print(feats.shape)
            q_vals = self.model_phase2.model(feats)
            q_vals = q_vals.squeeze(0)
            q_vals[valid_moves.T <= 0] = -np.inf
        if debug:
            print(f"Phase 2 Q values: {q_vals}")
        return int(torch.argmax(q_vals))

    def choose_action_phase3(self, state: GameState, hand: np.ndarray, other: np.ndarray, top_cards: list, debug=debug):
        valid_draws_mask = state.valid_draws(top_cards)
        if random.random() < self.epsilon:
            return random.choice(valid_draws_mask.nonzero()[0])
        with torch.no_grad():
            tensor = state.to_tensor(hand, other, top_cards).unsqueeze(0)
            # print(f"Tensor is {tensor}")
            q_vals = self.model_phase3.model(
                tensor)
            q_vals = q_vals.squeeze(0)
            q_vals[valid_draws_mask == 0] = -np.inf
        # if debug:
        #     print(f"Phase 3 Q values: {q_vals}")
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

    def play_agent(self, game: GameState, hand: np.ndarray, other: np.ndarray, debug=debug):
        if debug:
            print("Starting turn...")
        state_tensor = game.to_tensor(hand, other, game.top_cards)

        a1 = self.choose_action_phase1(game, hand, other, game.top_cards)

        # Initialize intermediate losses/rewards
        phase_1_intermediate_loss = 0
        phase_2_intermediate_loss = 0
        phase_3_intermediate_loss = 0

        # Check if player calls Yaniv prematurely or correctly
        # Noam was here
        if a1 == 1:
            won = game.yaniv(hand, [other])
            if won:
                reward = 1
            else:
                reward = -30 # Strong penalty for premature Yaniv
            self.model_phase1.add_episode(state_tensor, a1, reward)
            if debug:
                print("End of game", reward)
            return True, won
        state_2_tensor = game.get_features_2(hand, other, game.top_cards)

        # Phase 2: Discard cards and draw new card
        a2 = self.choose_action_phase2(game, hand, other, game.top_cards)
        discard_indices = POSSIBLE_MOVES[int(a2)]
        move_played = list(discard_indices)
        hc = hand.copy()
        game.play(hand, move_played) # Hand now discarded cards played, hc has old hand

        # Phase 2 intermediate loss logic
        valid_moves = list(game.valid_moves(hc))  # Valid moves before playing
        move_values = {
            POSSIBLE_MOVES[i]: game.move_value(hc, POSSIBLE_MOVES[i])
            for i in valid_moves
        }
        # Penalize inefficient discards

        best_value = max(move_values.values())
        played_value = game.move_value(hc, discard_indices)

        # Encourage playing the highest value possible
        diff = best_value - played_value
        if diff > 0:
            phase_2_intermediate_loss -= diff * 5  # higher penalty for poor discard choice
        else:
            phase_2_intermediate_loss += played_value  # small reward for optimal discard

        # Strongly discourage playing Aces unless strategically necessary
        num_aces = sum(1 for idx in discard_indices if idx % 13 == 0)
        if num_aces > 0 and played_value < best_value:
            phase_2_intermediate_loss -= 20 * num_aces

        # Improved phase 3 intermediate loss logic
        tops = game.tc_holder
        state_tensor = game.to_tensor(hand, other, game.tc_holder)
        a3 = self.choose_action_phase3(game, hand, other, game.tc_holder)
        top_card_values = [game.card_value(card) for card in game.tc_holder]
        drawn_card_idx = game.draw(hc, move_played, a3)
        hand[drawn_card_idx] += 1

        if a3 == 52:  # drew from deck
            for card in game.tc_holder:
                name = game.card_to_name(card)
                c, v = game.completes_move(hc, card)
                if debug:
                    print(name, c, v)
                if c:  # this card completes a move
                    phase_3_intermediate_loss -= max(10, v)
                else:
                    if game.card_value(card) <= 3:
                        phase_3_intermediate_loss -= 3 * (10 - game.card_value(card))
        else:
            our = game.card_value(a3)
            if len(game.tc_holder) == 1:
                if d:
                    print(f"{our, game.card_to_name(a3)}")
                c, v = game.completes_move(hc, our)
                value = game.card_value(our)
                if not (c or value <= 3):
                    phase_3_intermediate_loss -= our
                else:
                    phase_3_intermediate_loss += max(v, value)
            else:
                us_completes, us_value = game.completes_move(hc, a3)
                a3_val = game.card_value(a3)
                if not us_completes and a3_val > 3:
                    phase_3_intermediate_loss -= a3_val
                for card in game.tc_holder:
                    if card == a3:
                        if not us_completes and game.get_hand_value(hand) > 7:
                            phase_3_intermediate_loss -= a3_val
                        else:
                            phase_3_intermediate_loss += 2
                    else:
                        completes, value = game.completes_move(hc, card)
                        v = game.card_value(card)
                        if completes:
                            phase_3_intermediate_loss -= value
                        elif v < a3_val:
                            phase_3_intermediate_loss -= 10 - v
        
        # Apply episodes with calculated intermediate losses
        self.model_phase1.add_episode(
            state_tensor, a1, phase_1_intermediate_loss)
        self.model_phase2.add_episode(
            state_2_tensor, a2, phase_2_intermediate_loss)
        self.model_phase3.add_episode(
            state_tensor, a3, phase_3_intermediate_loss)
        
        # Print debug information
        if d:
            played = []
            nzs = np.nonzero(hc)[0]
            counter = 0
            for nz in nzs:
                if counter in move_played:
                    played.append(game.card_to_name(nz))
                counter += 1

            print(f"""
                  Top cards: {[game.card_to_name(tops[i]) for i in range(len(tops))]}
                  Hand: {game.hand_to_cards(hc)}
                  Move played: {move_played}, Move Value: {move_values[tuple(move_played)]}
                  Move values: {move_values}
                  Discarded cards: {played}
                  Drawn card: {game.card_to_name(a3) if a3 != 52 else "deck"}
                  Phase 1 intermediate loss: {phase_1_intermediate_loss}
                  Phase 2 intermediate loss: {phase_2_intermediate_loss}
                  Phase 3 intermediate loss: {phase_3_intermediate_loss}
                  """)


        if debug:
            print("End of turn")
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
    num_turns = 0

    while True:
        done, won = p1.play_agent(game, p1_hand, p2_hand)
        num_turns += 1
        if done:
            return (won, num_turns)

        # Player 2's turn
        done, won = p2.play_agent(game, p2_hand, p1_hand)
        num_turns += 1
        if done:
            return (not won, num_turns)
