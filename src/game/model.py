import numpy as np
from state import GameState

import itertools

def generate_combinations(n=5, max_length=6):
    all_combinations = []

    for length in range(1, max_length):
        for combination in itertools.combinations(range(n), length):
            all_combinations.append(combination)

    return all_combinations

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
            i : 0.0 for i in range(1, 41)
        }

        self.learning_rate = 0.01
        self.discount_factor = 0.90

    def features(self, state : GameState):
        our_hand_value, _, other_player_num_cards, turn, last_five, completes_set, completes_straight = state.get_features()
        last = np.array([i for i in last_five])
        completes_s = np.array([c for c in completes_set])
        completes_str = np.array([c for c in completes_straight])
        return np.array([our_hand_value, other_player_num_cards]).extend(completes_s).extend(completes_str)

    def predict_q(self, state, action):
        feature = self.feature(state)
        return (self.weights[action] * feature).sum()
    
    def can_yaniv(self, state : GameState, hand):
        return state.can_yaniv(hand)

    def update_weights(self, state, action, reward, next_state):
        # TODO (this is incorrect)
        yaniv = self.can_yaniv(state[0])

        next_action = 'hit' if self.predict_q(next_state, 'hit') > self.predict_q(next_state, 'stand') else 'stand'
        best_future_q = self.predict_q(next_state, next_action)

        current_q = self.predict_q(state, action)
        
        self.weights[action] += self.learning_rate * (reward + self.discount_factor * best_future_q - current_q) * self.feature(state)

    def choose_action(self, state : GameState):
        # TODO (this is incorrect)
        yaniv = self.can_yaniv(state.player_1_hand)

        if np.random.rand() < 0.4:  # 10% chance to explore
            return np.random.choice(['hit', 'stand'])

        return 'hit' if self.predict_q(state, 'hit') > self.predict_q(state, 'stand') else 'stand'
