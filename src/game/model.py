import numpy as np
from state import GameState
import helpers

class Model:
    def __init__(self):
        self.yaniv_weights = {
            'Yaniv': 0.0,
            'Continue': 0.0,
        }

        self.draw_or_from_discard = {
            'Deck' : 0.0,
            'Card 0' : 0.0,
            'Card 1' : 0.0
        }

        self.move_value = {
            i : 0.0 for i in range(1, 48)
        }

        self.learning_rate = 0.01
        self.discount_factor = 0.90

    def features(self, state : GameState):
        our_hand_value, _, other_player_num_cards, turn, last_five, completes_set, completes_straight = state.get_features()
        last = np.array([i for i in last_five])
        completes_s = np.array([c for c in completes_set])
        completes_str = np.array([c for c in completes_straight])
        return np.array([our_hand_value, turn, other_player_num_cards]).extend(completes_s).extend(completes_str).extend(last)

    def predict_q_first(self, state, action):
        feature = self.feature(state)
        return (self.weights[action] * feature).sum()
    
    def can_yaniv(self, state : GameState, hand):
        return state.can_yaniv(hand)
    
    def valid_move_values(self, state : GameState, hand):
        valid_moves = []
        nonzeros = np.nonzero(hand)
        for i in range(nonzeros):
            valid_moves += state.move_value(hand, i)
        return valid_moves

    def update_first_weights(self, state, action, reward, next_state):
        # TODO (this is incorrect)
        yaniv = self.can_yaniv(state[0])

        next_action = 'hit' if self.predict_q(next_state, 'hit') > self.predict_q(next_state, 'stand') else 'stand'
        best_future_q = self.predict_q(next_state, next_action)

        current_q = self.predict_q(state, action)
        
        self.weights[action] += self.learning_rate * (reward + self.discount_factor * best_future_q - current_q) * self.feature(state)

    def choose_first_action(self, state : GameState):
        # TODO (this is incorrect)
        first_moves = ["Continue"] + ["Yaniv"] if self.can_yaniv(state.player_1_hand) else []

        if len(first_moves) == 1:
            return "Continue"
        elif np.random.rand() < 0.4:  # 40% chance to explore
            if np.random.choice(first_moves) == "Yaniv":
                return "Yaniv"
        else:
            return "Yaniv" if self.predict_first_q(state, "Yaniv") > self.predict_first_q(state, "Continue") else "Continue"
        
    def update_second_weights(self, state, action, reward, next_state):
        # TODO (this is incorrect)

        next_action = 'hit' if self.predict_q(next_state, 'hit') > self.predict_q(next_state, 'stand') else 'stand'
        best_future_q = self.predict_q(next_state, next_action)

        current_q = self.predict_q(state, action)
        
        self.weights[action] += self.learning_rate * (reward + self.discount_factor * best_future_q - current_q) * self.feature(state)
        
    def choose_second_action(self, state : GameState):
        
        valid_moves = self.valid_move_values(state, state.player_1_hand).sort(key=lambda x : self.predict_q(state, x))

        if np.random.rand() < 0.4:
            return np.random.choice(valid_moves)[1]

        return 'hit' if self.predict_q(state, 'hit') > self.predict_q(state, 'stand') else 'stand'
