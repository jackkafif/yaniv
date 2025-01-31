class Card:
    def __init__(self, suit, number):
        assert suit in ["Spade", "Club", "Heart", "Diamond"], "Suit must be Spade, Club, Heart, or Diamond"
        assert 1 <= number <= 13, "Number must be between 1 and 13"
        self.suit = suit
        self.value = number

    def numerical_value(self) -> int:
        return self.value

    def __le__(self, c2) -> bool:
        return self.value <= c2.value

    def __eq__(self, c2) -> bool:
        return self.same_number(c2) and self.same_suit(c2)

    def __add__(self, i):
        new_value = (self.value + i - 1) % 13 + 1
        return Card(self.suit, new_value)

    def same_number(self, c2) -> bool:
        return self.value == c2.value

    def same_suit(self, c2) -> bool:
        return self.suit == c2.suit

    def __str__(self):
        return f"{self.value} of {self.suit}s"
