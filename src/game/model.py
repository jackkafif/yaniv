import numpy as np
from state import GameState
import helpers
import copy

class Model:
    def __init__(self):
        self.yaniv_weights = {
            'Yaniv': 0.0,
            'Continue': 0.0,
        }

        self.move_value = {
            i : 0.0 for i in range(1, 48)
        }

        self.draw_or_from_discard = {
            'Deck' : 0.0,
            'Card 0' : 0.0,
            'Card 1' : 0.0
        }

        self.learning_rate = 0.01
        self.discount_factor = 0.90

    def features(self, state : GameState):
        our_hand_value, _, other_player_num_cards, turn, last_five, completes_set, completes_straight = state.get_features()
        last = np.array([i for i in last_five])
        completes_s = np.array([c for c in completes_set])
        completes_str = np.array([c for c in completes_straight])
        return np.array([our_hand_value, turn, other_player_num_cards]).extend(completes_s).extend(completes_str).extend(last)

    def predict_q_first(self, state : GameState, action : str):
        feature = self.feature(state)
        return (self.weights[action] * feature).sum()
    
    def predict_q_second(self, move : tuple[int, list[str]], feature):
        value, action = move
        feat = feature.extend(value)
        return (self.weights[action] * feat).sum()

    def can_yaniv(self, state : GameState, hand):
        return state.can_yaniv(hand)
    
    def valid_move_values(self, state : GameState, hand : np.ndarray[int]) -> list[tuple[int, list[int]]]:
        valid_moves = []
        nonzeros = np.nonzero(hand)
        for i in range(nonzeros):
            valid_moves += state.move_value(hand, i)
        return valid_moves

    def update_first_weights(self, state : GameState, action : str, reward : int, next_state : GameState):
        next_action = 'Continue' if self.predict_q(next_state, 'Continue') > self.predict_q(next_state, 'Yaniv') else 'Yaniv'
        best_future_q = 0 if next_action == "Yaniv" else self.predict_q_first(next_state, next_action)

        current_q = self.predict_q(state, action)
        
        self.weights[action] += self.learning_rate * (reward + self.discount_factor * best_future_q - current_q) * self.feature(state)

    def choose_first_action(self, state : GameState):
        first_moves = ["Continue"] + ["Yaniv"] if self.can_yaniv(state.player_1_hand) else []

        if len(first_moves) == 1:
            return "Continue"
        elif np.random.rand() < 0.4:  # 40% chance to explore
            if np.random.choice(first_moves) == "Yaniv":
                return "Yaniv"
        else:
            return "Yaniv" if self.predict_first_q(state, "Yaniv") > self.predict_first_q(state, "Continue") else "Continue"
        
    def update_second_weights(self, state : GameState, action : list[int], reward : int, next_state : GameState) -> None:

        state_features = self.features(state)

        next_valid_moves = self.valid_move_values(state, state.player_1_hand)
        next_state_features = self.features(next_state)
        next_valid_moves.sort(key=lambda x : self.predict_q_second(x[1], next_state_features))

        next_action = next_valid_moves[0]
        best_future_q = self.predict_q_second(next_action, next_state_features)

        current_q = self.predict_q_second(state, action, state_features)
        
        self.weights[action] += self.learning_rate * (reward + self.discount_factor * best_future_q - current_q) * self.feature(state)
        
    def choose_second_action(self, state : GameState):
        
        valid_moves = self.valid_move_values(state, state.player_1_hand)

        if np.random.rand() < 0.4:
            return np.random.choice(valid_moves)[1]
        
        state_features = self.features(state)
        valid_moves.sort(key=lambda x : self.predict_q_second(state, x, state_features))
        return valid_moves[0][1]
    
    def update_third_weights(self, state : GameState, action, reward, next_state):
        # TODO
        pass

    def update_weights(self, state : GameState, action1 : str, action2 : list, action3 : str, reward : float, next_state : GameState):
        # TODO 
        pass

    def choose_third_action(self, state : GameState):
        # TODO
        pass
    
    def play_opponent_turn(self, state : GameState):
        # TODO play the highest value move
        pass
    
    def play_step(self, state : GameState, action1 : str, action2=[], action3="Deck"):
        curr = copy.deepcopy(state)
        if action1 == "Yaniv":
            win = state.yaniv(state.player_1_hand, [state.player_2_hand])
            # Update 
        draw = -1 if action3 == "Deck" else 0 if action3 == "Draw 0" else 1
        state.play(state.player_1_hand, action2, draw)
        self.play_opponent_turn(state)
        
        return state, reward, win
    
def main():
    for episode in range(10000):
        sim = Model()
        state = GameState()
        done = False
        while not done:
            action = sim.choose_action(state)
            next_state, reward, done = sim.play_step(state, action)
            sim.update_weights(state, action, reward, next_state, done)
            state = next_state
