import numpy as np

class Deck:
    def __init__(self):
        # Cards are tuples (suit, rank)
        self.cards = [(i, j) for i in range(4) for j in range(13)]  # Fixed suit range from 3 to 4
        np.random.shuffle(self.cards)
    
    def deal(self):
        return self.cards.pop() if self.cards else None

class GameState:
    def __init__(self, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.deck = Deck()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q_table = {}
        self.state = self.initialize_state()

    def initialize_state(self):
        our_hand = [0] * 52  # Fixed size to 52 for a standard deck
        player_1 = [0] * 52
        top_card = [0] * 52
        
        for i in range(5):  # Dealing 5 cards to each player
            for hand in [our_hand, player_1]:
                card = self.deck.deal()
                if card:
                    hand[card[0] * 13 + card[1]] += 1
        card = self.deck.deal()
        if card:
            top_card[card[0] * 13 + card[1]] += 1
        
        state = (tuple(our_hand), tuple(player_1), tuple(top_card))
        return state
    
    def play_step(self, state, action, draw_from_deck=True):
        if action == "Yaniv":
            return state, True
        our_hand, opponent_hand, top_card = state
        our_hand = list(our_hand)
        new_top_card = list(top_card)
        if isinstance(action, int):  # Playing a single card
            suit, rank = divmod(action, 13)
            if our_hand[action] > 0:
                our_hand[action] -= 1
                new_top_card = [0] * 52  # Reset top card state
                new_top_card[action] = 1  # Set this card as the new top card
                print(f"Played card: Suit {suit}, Rank {rank}")
            else:
                return state  # Return the original state if the action is invalid
        elif isinstance(action, list):  # Playing a set of cards
            for card in action:
                suit, rank = divmod(card, 13)
                if our_hand[card] > 0:
                    our_hand[card] -= 1
                    print(f"Played card from set: Suit {suit}, Rank {rank}")
                else:
                    print("Invalid action: Card not in hand")
                    return state  # Return the original state if the action is invalid
            new_top_card = [0] * 52
            new_top_card[action[-1]] = 1  # The last card played is the new top card
        # Drawing a card after playing
        if draw_from_deck:
            drawn_card = self.deck.deal()
            if drawn_card:
                our_hand[drawn_card[0] * 13 + drawn_card[1]] += 1
                print(f"Drew card from deck: Suit {drawn_card[0]}, Rank {drawn_card[1]}")
        else:
            # Draw the top card from the pile if not empty and if not drawing from deck
            if sum(top_card) > 0:
                top_card_index = top_card.index(1)  # Get the index of the top card
                our_hand[top_card_index] += 1
                new_top_card = [0] * 52  # Reset top card after drawing
                print(f"Drew top card from pile: Suit {divmod(top_card_index, 13)[0]}, Rank {divmod(top_card_index, 13)[1]}")

        # Assume the opponent plays randomly for now (this can be replaced with more strategic AI)
        opponent_action = np.random.choice([i for i in range(len(opponent_hand)) if opponent_hand[i] > 0])
        print(type(opponent_hand[opponent_action]))
        opponent_hand[opponent_action] -= 1
        new_top_card = [0] * 52
        new_top_card[opponent_action] = 1  # Opponent's played card becomes the new top card

        # Convert lists back to tuples before returning the new state
        new_state = (tuple(our_hand), tuple(opponent_hand), tuple(new_top_card))
        return new_state


    def update_q_table(self, state, action, reward, next_state):
        best_next_action = max(self.Q_table.get(next_state, {}), key=self.Q_table.get(next_state, {}).get, default=0)
        td_target = reward + self.gamma * self.Q_table.get(next_state, {}).get(best_next_action, 0)
        old_value = self.Q_table.get(state, {}).get(action, 0)
        self.Q_table[state] = self.Q_table.get(state, {})
        self.Q_table[state][action] = old_value + self.alpha * (td_target - old_value)

    def compute_reward(self, our_hand, player_hand):
        our_total = np.sum(np.array(our_hand))
        other_total = np.sum(np.array(player_hand))
        return 1 if our_total < other_total else -1

    def choose_action(self, state):
        valid_actions = self.valid_actions(state, 0)
        if np.random.rand() < self.epsilon:
            # Explore
            action = np.random.choice(valid_actions)
        else:
            # Exploit
            action = max(self.Q_table.get(state, {}), key=self.Q_table.get(state, {}).get, default=valid_actions[0])
        return action
    
    def valid_actions(self, state, player=0):
        hand = np.array(state[player]).reshape((4, 13))
        valid_actions = []
        for idx, count in enumerate(hand.flatten()):
            if count > 0:
                valid_actions.append(idx)
        for rank in range(13):  # Checking for possible sets
            if hand[:, rank].sum() >= 2:
                valid_actions.append(13 + rank)  # Adding index offset for clarity
        return valid_actions


if True:
    # Initialize the game state
    game = GameState(alpha=0.1, gamma=0.9, epsilon=0.1)
    episodes = 10000  # Total number of games to simulate

    for episode in range(episodes):
        game.deck = Deck()  # Reinitialize the deck each episode
        game.state = game.initialize_state()  # Reset the initial state
        done = False
        
        while not done:
            current_state = game.state
            action = game.choose_action(current_state)  # Decide action based on current policy
            new_state, game_over = game.play_step(current_state, action)  # Take action, get new state
            
            if game_over:
                reward = game.compute_reward(new_state[0], new_state[1])
                game.update_q_table(current_state, action, reward, None)
                done = True
            else:
                reward = game.compute_reward(new_state[0], new_state[1])  # Intermediate rewards (optional)
                game.update_q_table(current_state, action, reward, new_state)  # Update Q-table
                game.state = new_state  # Move to the new state
            
            # Optionally decrease epsilon over time to reduce exploration
            game.epsilon = max(0.01, game.epsilon * 0.999)

        if episode % 100 == 0:
            print(f"Episode {episode}: Done training with epsilon {game.epsilon}")

    print("Training completed.")

if __name__ == "__main__":
    main()
