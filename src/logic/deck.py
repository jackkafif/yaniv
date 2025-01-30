import random
from card import Card

class Deck:
    def __init__(self):
        self.begin()
        self.discard = []
        self.top = []
        random.shuffle(self.cards)

    def begin(self):
        self.cards = [Card(suit, number) for suit in ["Spade", "Club", "Heart", "Diamond"] for number in range(1, 14)]

    def shuffle(self):
        self.cards = self.discard
        self.discard = []
        random.shuffle(self.cards)

    def deal(self):
        if not self.isEmpty():
            return self.cards.pop()
        else:
            self.shuffle

    def play(self, cards):
        self.discard += self.top
        self.top = cards

    def isEmpty(self):
        return self.size() == 0

    def size(self):
        return len(self.cards)