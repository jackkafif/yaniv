import numpy as np

class GameState:
    def __init__(self):
        self.deck = np.random.shuffle(np.arange(53))
        self.player_1_hand = np.zeros(53)
        self.player_2_hand = np.zeros(53)
        self.turn = 0
        self.discard = []
        self.top_cards = []
        self.cards_played = []
        self.over = False

    def completes_set(self, hand, card):
        hand_copy = self.hand.copy().clip(0, 1)
        hand_copy[card] += 1
        return np.nonzero(hand_copy) >= 1
    
    def completes_straight(self, hand, card):
        suit = card // 13
        rank = card % 13
        hand_reshaped = hand.copy().reshape((4, 13))
        hand_suit = hand_reshaped[suit]
        between = (hand_suit[(rank - 1) % 13] == 1 and hand_suit[(rank + 1) % 13] == 1)
        before = (hand_suit[(rank - 1) % 13] == 1 and hand_suit[(rank - 2) % 13] == 1)
        after = (hand_suit[(rank + 1) % 13] == 1 and hand_suit[(rank + 2) % 13] == 1)
        return between or before or after

    def get_features(self):
        our_hand_value = self.get_hand_value(self.player_1_hand)
        our_cards = len(self.player_1_hand)
        other_player_num_cards = len(self.player_2_hand)
        turn = self.turn
        last_five = self.cards_played[-5:]
        top_cards = self.top_cards
        completes_set = [self.completes_set(our_cards, card) for card in top_cards]
        completes_straight = [self.completes_straight(our_cards, card) for card in top_cards]
        return our_hand_value, our_cards, other_player_num_cards, turn, last_five, completes_set, completes_straight 

    def get_hand_value(self, hand):
        sum = (hand.reshape((4, 13)) * np.arange(1, 13)).clip(0, 10).sum()
        return sum

    def deal(self):
        if len(self.deck) == 0:
            self.deck = np.random.shuffle(np.arange(53))
        self.discard = []
        self.top_cards = []
        return self.deck.pop()
    
    def can_yaniv(self, hand):
        return self.get_hand_value(hand) <= 7
    
    def yaniv(self, hand, other_hands):
        our_value = self.get_hand_value(hand)
        for others in other_hands:
            if our_value <= self.get_hand_value(others):
                return False
        return True
    
    def draw(self, idx):
        if idx <= len(self.top_cards) - 1:
            card = self.top_cards[idx]
        else:
            card = self.top_cards[0]
        self.top_cards = []
        return card
    
    def play(self, cards, draw_idx):
        self.discard += self.top_cards
        if draw_idx >= 0:
            card_drawn = self.draw(draw_idx)
        else:
            card_drawn = self.deal()
        if len(cards) <= 1:
            self.top_cards = cards
        self.top_cards = [cards[0], cards[-1]]
        return card_drawn