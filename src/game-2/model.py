import numpy as np

class Model:
    def __init__(self, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.Q_table = np.zeros(())
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_state(self):
        # our_hand = np.zeros((4, 13))
        # top_card = np.zeros((4, 13))
        # pick_deck = np.one((1))
        state = np.zeros(106)
        state[105] += 1

    def translate_state(self, state):
        our_hand = state[:52]
        top_card = state[52:104]
        pick_deck = state[105]
        return our_hand, top_card, pick_deck

    def play_step(self, state, turn):
        pass

    def update_q_table(self, state, action, reward, next_state):
        pass

    def compute_reward(self, state):
        pass

    def choose_action(self, state):
        pass