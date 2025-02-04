import numpy as np
import math

index_to_combination = {
    0 : [0],
    1 : [1],
    2 : [2],
    3 : [3],
    4 : [4],
    5 : [0, 1],
    6 : [0, 2],
    7 : [0, 3],
    8 : [0, 4],
    9 : [1, 2],
    10 : [1, 3],
    11 : [1, 4],
    12 : [2, 3],
    13 : [2, 4],
    14 : [3, 4],
    15 : [0, 1, 2],
    16 : [0, 1, 3],
    17 : [0, 1, 4],
    18 : [0, 2, 3],
    19 : [0, 2, 4],
    20 : [0, 3, 4],
    21 : [1, 2, 3],
    22 : [1, 2, 4],
    23 : [1, 3, 4],
    24 : [2, 3, 4],
    25 : [0, 1, 2, 3],
    26 : [0, 1, 2, 4],
    27 : [0, 1, 3, 4],
    28 : [0, 2, 3, 4],
    29 : [1, 2, 3, 4],
    30 : [0, 1, 2, 3, 4]
}
combination_to_index = {v: k for k, v in index_to_combination.items()}

class Model:
    NUM_SUITS = 4
    NUM_RANKS = 13
    CARD_REPR = NUM_SUITS * NUM_RANKS
    MAX_HAND_TOTAL = 50
    COMBINATIONS = 31

    def __init__(self, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.Q_table = 
        self.state_size = 3 * self.CARD_REPR # First card_repr is our hand, next is player 2 hand, next is top card(s)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_state(self):
        state = np.zeros(self.state_size)

    def state_to_q_table_index(self, state):
        hand = state[:52]
        sum_of_hand = (hand.sum(axis=0) * np.arange(1, 14)).sum()
        top_card = state[2*self.CARD_REPR:]
    
    def combinations_check(self, hand):
        card_indices = np.nonzero(hand)
        hand_indices = {card_indices[i] : i for i in enumerate(card_indices)}

        same_ranks = {i : [] for i in range(self.NUM_RANKS)}
        for card in card_indices:
            rank = card % 13
            same_ranks[rank].append(hand_indices[card])

        def sublists(list):
            holder = [list]
            if len(list) <= 2:
                return holder
            for i in range(len(list)):
                sublist = list[:i] + list[i+1:]
                holder += sublists(sublist)
            return holder        

        combinations = np.zeros(31)
        for i in range(self.NUM_RANKS):
            if same_ranks[i] >= 2:
                combinations[combination_to_index(list(set(sublists(same_ranks[i]))))] = 1

        return combinations

    def choose_action(self, state):
        valid_actions = self.valid_actions(state)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            action = max(self.Q_table.get(state, {}), default=valid_actions[0])
        return action
    
    def valid_actions(self, state):
        pass


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
