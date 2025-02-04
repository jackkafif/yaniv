import numpy as np

class Model:
    def __init__(self, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.Q_table = np.zeros(())
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_state(self):
        state = np.zeros(106)
        state[105] += 1

    def translate_state(self, state):
        our_hand = state[:52]
        top_card = state[52:104]
        pick_deck = state[105]
        return our_hand, top_card, pick_deck

    def choose_action(self, state):
        valid_actions = self.valid_actions(state)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            action = max(self.Q_table.get(state, {}), default=valid_actions[0])
        return action
    
    def valid_actions(self, state):
        actions = ["Play and Draw From Deck", "Play and Draw From Pile", "Yaniv"]
        also = ["What to play, if yaniv nothing"]


    def play_step(self, state, action):
        """
        Advance state based on action
        """
        pass

    def update_q_table(self, state, action, reward):
        """
        Update Q Learning Table[state] based on reward
        """
        pass

    def compute_reward(self, state):
        """
        Computes reward of current game state
        """
        pass
