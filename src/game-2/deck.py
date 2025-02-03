import numpy as np

class Deck:
    SUITS = 4
    RANKS = 13
    def __init__(self):
        # Cards are tuples (suit, rank)
        self.cards = [(i, j) for i in range(self.SUITS) for j in range(self.RANKS)]
        self.top_card = [0] * 52
        self.discard = []
        np.random.shuffle(self.cards)
    
    def deal(self):
        if len(self.cards) == 0:
            self.cards = [(i, j) for i in range(self.SUITS) for j in range(self.RANKS)]
            np.random.shuffle(self.cards)

        return self.cards.pop() if self.cards else None
    
    def place(self, cards):
        self.top_card = [0] * 52
        for suit, rank in cards:
            self.discard.append((suit, rank))
            self.top_card[suit * self.RANKS + rank] += 1
