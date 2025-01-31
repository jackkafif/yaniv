from card import Card

class InvalidMoveException(Exception):
    pass

class Player:
    def __init__(self):
        self.hand = {i : None for i in range(5)}
        self.open = [0, 1, 2, 3, 4]

    def play_turn(self, move, card_to_pick_up, yaniv=False):
        if yaniv:
            try:
                return "Yaniv", self.yaniv()
            except InvalidMoveException:
                return "", False
            return 
        try:
            self.play_cards(move)
        except InvalidMoveException:
            return "", False
        self.add_to_hand(card_to_pick_up)
        return "Continue", True

    def add_to_hand(self, card):
        assert len(self.open) > 0
        self.hand[self.open.pop()] = card

    def yaniv(self):
        if self.hand_value() <= 7:
            return self.hand_value()
        else:
            raise InvalidMoveException("Cannot call Yaniv if hand has > 7 value")

    def hand_value(self):
        val = 0
        for idx in self.hand:
            val += self.hand[idx].value if self.hand[idx] else 0
        return val

    def play_cards(self, move):
        if self.validate_move(move):
            cards = [self.hand[i] for i in move]
            for i in move:
                self.hand[i] = None
                self.open.append(i)
            return True, cards
        else:
            raise InvalidMoveException("This move is not valid")
        
    def print_hand(self):
        sb = ""
        for idx in self.hand:
            sb += str(self.hand[idx]) + ", " if self.hand[idx] else "empty, "
        print(sb)


    def validate_move(self, move):
        if len(move) == 0:
            return False
        if len(move) == 1:
            return True
        cards = []
        for i in move:
            if self.hand[i] is None:
                return False
            cards.append(self.hand[i])

        cards.sort(key=lambda card : card.value)
        same_number = True
        straight = len(cards) >= 3
        c1 = cards[0]
        for card in cards[1:]:
            same_number &= c1.same_number(card)
            straight &= c1 + 1 == card
            c1 = card
        return same_number or straight