from card import Card

class Player:
    def __init__(self):
        self.hand = {i : None for i in range(5)}

    def add_to_hand(self, card):
        self.hand.append(card)

    def play(self, move):
        if self.validate_move(move):
            cards = [self.hand[i] for i in move]
            for i in range(i):
                self.hand[i] = None
            return True, cards
        else:
            return False, []


    def validate_move(self, move):
        return True