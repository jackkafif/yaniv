import numpy as np
import helpers

class GameState:
    def __init__(self) -> None:
        self.deck = np.random.shuffle(np.arange(52))
        self.player_1_hand = np.zeros(52)
        self.player_2_hand = np.zeros(52)
        self.turn = 0
        self.discard = []
        self.top_cards = []
        self.cards_played = []
        self.over = False

    def move_value(self, hand : np.ndarray, card : int) -> tuple[int, int]:
        suit = card // 13
        rank = card % 13

        hand_copy = hand.reshape((4, 13))

        set_combinations = helpers.COMBINATIONS[4]

        hand_suit = hand_copy[suit]
        value_tups = []
        after = (hand_suit[(rank + 1) % 13] == 1 and hand_suit[(rank + 2) % 13] == 1)
        if after:
            print(rank, rank + 1, rank + 2)
            after_value, indices = min((rank + 1) % 13, 10) + min((rank + 2) % 13, 10) + min((rank + 3) % 13, 10), [rank, rank+1, rank+2]
            value_tups.append((after_value, indices))

        for set_comb in set_combinations:
            length = len(set_comb)
            if length >= 2:
                works = True
                for i in set_comb:
                    works = True and hand_copy[i, rank] == 1
                if works:
                    value_tups.append((min(rank, 10) * length, set_comb))
        return value_tups

    def card_indices(self, hand : np.ndarray) -> list[int]:
        return np.nonzero(hand)

    def completes_set(self, hand : np.ndarray, card : int) -> bool:
        hand_copy = self.hand.copy().reshape((4, 13)).clip(0, 1)
        hand_copy[card] += 1
        return np.nonzero(hand_copy) > 1
    
    def completes_straight(self, hand : np.ndarray, card : int) -> bool:
        suit = card // 13
        rank = card % 13
        hand_reshaped = hand.copy().reshape((4, 13))
        hand_suit = hand_reshaped[suit]
        between = (hand_suit[(rank - 1) % 13] == 1 and hand_suit[(rank + 1) % 13] == 1)
        before = (hand_suit[(rank - 1) % 13] == 1 and hand_suit[(rank - 2) % 13] == 1)
        after = (hand_suit[(rank + 1) % 13] == 1 and hand_suit[(rank + 2) % 13] == 1)
        return between or before or after

    def get_features(self) -> tuple:
        our_hand_value = self.get_hand_value(self.player_1_hand)
        our_cards = len(self.player_1_hand)
        other_player_num_cards = len(self.player_2_hand)
        turn = self.turn
        last_five = self.cards_played[-5:]
        top_cards = self.top_cards
        completes_set = [self.completes_set(our_cards, card) for card in top_cards]
        completes_straight = [self.completes_straight(our_cards, card) for card in top_cards]
        return our_hand_value, our_cards, other_player_num_cards, turn, last_five, completes_set, completes_straight

    def get_hand_value(self, hand : np.ndarray) -> int:
        sum = (hand.reshape((4, 13)) * np.arange(1, 13)).clip(0, 10).sum()
        return sum

    def deal(self) -> int:
        if len(self.deck) == 0:
            self.deck = np.random.shuffle(np.arange(53))
        self.discard = []
        self.top_cards = []
        return self.deck.pop()
    
    def can_yaniv(self, hand : np.ndarray) -> bool:
        return self.get_hand_value(hand) <= 7
    
    def yaniv(self, hand : np.ndarray, other_hands : list[np.ndarray]) -> bool:
        our_value = self.get_hand_value(hand)
        for others in other_hands:
            if our_value <= self.get_hand_value(others):
                return False
        return True
    
    def draw(self, idx : int) -> int:
        if idx <= len(self.top_cards) - 1:
            card = self.top_cards[idx]
        else:
            card = self.top_cards[0]
        self.top_cards = []
        return card
    
    def play(self, hand : np.ndarray, cards : list[int], draw_idx : int) -> int:
        self.discard += self.top_cards
        if draw_idx >= 0:
            card_drawn = self.draw(draw_idx)
        else:
            card_drawn = self.deal()
        if len(cards) <= 1:
            self.top_cards = cards
        self.top_cards = [cards[0], cards[-1]]
        return card_drawn