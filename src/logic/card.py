class Card:
    def __init__(self, suit, number):
        assert suit in ["Spade", "Club", "Heart", "Diamond"]
        assert number >= 1 and number <= 13
        self.suit = suit
        self.value = number

    def numerical_value(self):
        return self.number