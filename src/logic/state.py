import numpy as np

class Deck():
    def __init__(self):
        self.cards = [(i, j) for i in range(0, 3) for j in range(0, 13)]
        np.random.shuffle(self.cards)
        self.discard = []
        self.top = []

    def deal(self):
        if len(self.cards) > 0:
            return self.cards.pop()
        else:
            self.shuffle()
            return self.cards.pop() if len(self.cards) > 0 else None

    def shuffle(self):
        self.cards = [(i, j) for i in range(0, 3) for j in range(0, 13)]
        np.random.shuffle(self.cards)

class GameState():

    def __init__(self, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.initialize_state()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_state(self):
        self.deck = Deck()
        our_hand = np.zeros((4, 13))
        player_1 = np.zeros((4, 13))
        for hand in [our_hand, player_1]:
            hand[self.deck.deal()] += 1
        top_card = np.zeros((5, 52))
        possible_moves = np.zeros(22) # Contains possible sets, and single cards
        top_card[self.deck.deal()] += 1
        state = (sum(our_hand), our_hand, player_1, top_card, possible_moves)
        self.Q_table = np.zeros((50, our_hand.shape, player_1.shape, top_card.shape, possible_moves.shape))
        return state
    
    def play_step(self, state, action):
        if action == 0:
            pass


    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q_table[next_state])  # Choose the best next action
        td_target = reward + self.gamma * self.Q_table[next_state][best_next_action]
        self.Q_table[state][action] += self.alpha * (td_target - self.Q_table[state][action])


    def compute_reward(our_hand, player_hand):
        our_total = np.sum(our_hand.flatten())
        other_total = np.sum(player_hand.flatten())
        if our_total < other_total:
            return 1
        else:
            return -1
    
    def choose_action(self, state, Q_table):
        has_sets, rank = self.get_sets(state[1])
        has_yaniv = state[0] <= 7
        valid_actions = []
        if has_sets:
            valid_actions.append(rank)
        if has_yaniv:
            valid_actions.append("Yaniv")
        if np.random.rand() < self.epsilon:
            action = np.random.choice(valid_actions)
        # Choose the best-known action with probability 1 - epsilon (exploit)
        else:
            # Filter Q-values for valid actions only
            valid_q_values = self.Q_table[state][valid_actions]
            action = valid_actions[np.argmax(valid_q_values)]
        return action


    def choose_pickup(self, state):
        pass

    def get_sets(self, hand):
        sets = []
        for rank in range(hand.shape[1]):
            total_cards = np.sum(hand[:, rank])
            if total_cards >= 2:
                sets.append(rank)
        return sets


def main():
    for episode in range(10000):
        sim = GameState()
        state = sim.initialize_state()
        done = False
        while not done:
            action = sim.choose_action(state)
            next_state, reward, done = sim.play_step(state, action)
            sim.update_q_table(state, action, reward, next_state, done)
            state = next_state

if __name__ == "__main__":
    print("Playing")
    main()