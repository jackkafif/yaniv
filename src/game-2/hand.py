import numpy as np
from deck import Deck

class Hand:
    def __init__(self, deck: Deck):
        self.hand = np.zeros((52))
        for i in range(5):
            rank, suit = deck.deal()
            self.hand[rank * 13 + suit]

    def valid_actions(self):
        """
        Returns an array of length 19 where the first 5 indices indicate whether the player
        can play card i alone and the next 13 whether the player has sets of that suit and
        the last index whether the player can play yaniv
        """
        moves = np.zeros(19)
        moves[0] = 1 if self.value() <= 7 else 0
        pass

    def draw(self, card, deck, topCard=False):
        pass

    def value(self):
        return np.sum(self.hand.flatten())