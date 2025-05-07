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
    def __init__(self, state_size = STATE_SIZE):
        
        self.state_size = state_size
        
        self.model_phase1 = MDQN(state_size, 2)
        self.model_phase2 = MDQN(state_size, len(POSSIBLE_MOVES))
        self.model_phase3 = MDQN(state_size, 3)

        self.epsilon = 1.0
        self.epsilon_decay = 0.999

    def load_models(self, i: str):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        try:
            self.model_phase1.model.load_state_dict(
                torch.load(f"{MODEL_DIR}/{self.model_phase1.model.path}/agent{i}_phase1.pt"))
            self.model_phase2.model.load_state_dict(
                torch.load(f"{MODEL_DIR}/{self.model_phase2.model.path}/agent{i}_phase2.pt"))
            self.model_phase3.model.load_state_dict(
                torch.load(f"{MODEL_DIR}/{self.model_phase3.model.path}/agent{i}_phase3.pt"))
        except:
            self.save_models(i)

    def save_models(self, i: str):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        torch.save(self.model_phase1.model.state_dict(),
                   f"{MODEL_DIR}/{self.model_phase1.model.path}/agent{i}_phase1.pt")
        torch.save(self.model_phase2.model.state_dict(),
                   f"{MODEL_DIR}/{self.model_phase2.model.path}/agent{i}_phase2.pt")
        torch.save(self.model_phase3.model.state_dict(),
                   f"{MODEL_DIR}/{self.model_phase3.model.path}/agent{i}_phase3.pt")

    def choose_action_phase1(self, state : GameState, hand : np.ndarray, other : np.ndarray):
        if not state.can_yaniv(hand):
            return 0
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            q_vals = self.model_phase1.model(state.to_tensor(hand, other).unsqueeze(0))
        return int(torch.argmax(q_vals))

    def choose_action_phase2(self, state : GameState, hand : np.ndarray, other : np.ndarray):
        valid_moves = state.valid_move_indices(hand)
        if random.random() < self.epsilon:
            return random.choice(np.where(valid_moves > 0)[0])
        with torch.no_grad():
            q_vals = self.model_phase2.model(state.to_tensor(hand, other).unsqueeze(0))
            q_vals = q_vals.squeeze(0)
            q_vals[valid_moves.T <= 0] = -np.inf
        return int(torch.argmax(q_vals))

    def choose_action_phase3(self, state : GameState, hand : np.ndarray, other : np.ndarray):
        valid_draws = state.valid_draws()
        valid_draws_mask = np.zeros(3)
        for i in valid_draws:
            valid_draws_mask[i] = 1
        if random.random() < self.epsilon:
            return random.choice(valid_draws)
        with torch.no_grad():
            q_vals = self.model_phase3.model(state.to_tensor(hand, other).unsqueeze(0))
            q_vals = q_vals.squeeze(0)
            q_vals[valid_draws_mask < 1] = -np.inf
        return int(torch.argmax(q_vals))
    
    def choose_turn_moves(self, state : GameState, hand : np.ndarray, other : np.ndarray):
        # Convert the state to a tensor
        state_tensor = state.to_tensor(hand, other)

        # Choose actions for each phase
        phase1 = self.choose_action_phase1(state, hand, other) 
        phase2 = self.choose_action_phase2(state, hand, other)
        phase3 = self.choose_action_phase3(state, hand, other)

        # Return the chosen actions
        return phase1, phase2, phase3
    
    def play_agent(self, game: GameState, hand: np.ndarray, other: np.ndarray):
        state_tensor = game.to_tensor(hand, other)

        a1, a2, a3 = self.choose_turn_moves(game, hand, other)
        discard_indices = POSSIBLE_MOVES[int(a2)]
        self.model_phase1.add_episode(state_tensor, a1, 0)
        if a1 == 1 or len(discard_indices) == 0:
            won = game.yaniv(hand, [other])
            if won:
                reward = 1
            else:
                reward = -1
            return True, won
        
        move_played = list(discard_indices)
        
        # Check if the player played a low card when they could have played a higher card
        # Ex: if the player has a straight and they play a low card, they should have played a higher card
        # Ex: if the player plays a single when they have a pair or a triple (and the other card doesn't complete a straight)

        valid_moves = list(game.valid_moves(hand))
        # print(game.hand_to_cards(hand))
        # print(valid_moves)
        move_values = {POSSIBLE_MOVES[valid_moves[i]] : game.move_value(hand, POSSIBLE_MOVES[valid_moves[i]]) for i in range(len(valid_moves))}
        # print(move_values)
        if 2 * move_values[tuple(move_played)] <= move_values[max(move_values)]:
            phase_2_intermediate_loss = -20
        else:
            phase_2_intermediate_loss = 0

        # Check if the cards in the discard pile complete a set/straight
        # Ex: If the player draws from the deck when they could have drawn from the discard pile to complete a set/straight
        # Ex: If the player draws from the deck when a low card is on the discard pile
        phase_3_intermediate_loss = 0
        if a3 == 0:
            if len(game.top_cards) > 0:
                for top_card in game.top_cards:
                    if game.card_value(top_card) <= 3 or game.completes_move(hand, top_card):
                        phase_3_intermediate_loss = -20

        # Phase 2: Discard cards
        hc = hand.copy()
        game.play(hand, move_played)
        hand[game.draw(hc, move_played, a3)] += 1

        self.model_phase2.add_episode(state_tensor, a2, phase_2_intermediate_loss)
        self.model_phase3.add_episode(state_tensor, a3, phase_3_intermediate_loss)

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

    
def run_game(p1 : YanivAgent, p2 : YanivAgent):
    # Returns True if player 1 wins, False if player 2 wins
    game = GameState()

    p1_hand = game.player_1_hand
    p2_hand = game.player_2_hand

    while True:
        # Player 1's turn
        done, won = p1.play_agent(game, p1_hand, p2_hand)
        if done:
            return won

        # Player 2's turn
        done, won = p2.play_agent(game, p2_hand, p1_hand)
        if done:
            return not won
        
