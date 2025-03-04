from __future__ import annotations
import numpy as np
import game.helpers as helpers

SUITS = {
    0 : "Clubs",
    1 : "Spades",
    2 : "Hearts",
    3 : "Diamonds"
}

RANKS = {
    0 : "Ace", 1 : "Two", 2 : "Three", 3 : "Four",
    4 : "Five", 5 : "Six", 6 : "Seven", 7 : "Eight", 8 : "Nine",
    9 : "Ten", 10 : "Jack", 11 : "Queen", 12 : "King"
}

SUITS_REVERSE = {value : key for key, value in SUITS.items()}
RANKS_REVERSE = {value : key for key, value in RANKS.items()}

class GameState:
    def __init__(self) -> None:
        self.deck = np.arange(52)
        np.random.shuffle(self.deck)
        self.player_1_hand = np.zeros(52)
        self.player_2_hand = np.zeros(52)
        self.curr_idx = -1
        for i in range(5):
            self.player_1_hand[self.deal()] += 1
            self.player_2_hand[self.deal()] += 1
        self.turn = 0
        self.discard = []
        self.top_cards = [self.deal()]
        self.cards_played = []
        self.over = False

    def reset(self) -> GameState:
        return GameState()

    def card_to_name(self, card):
        suit = SUITS[card // 13]
        rank = RANKS[card % 13]
        return f"{rank} of {suit}"
    
    def name_to_card(self, name : str) -> int:
        word = name.split(" ")
        rank, suit = RANKS_REVERSE[word[0]], SUITS_REVERSE[word[2]]
        return suit * 13 + rank

    def hand_to_cards(self, hand: np.ndarray[int]) -> list[str]:
        cards = []
        for card in range(len(hand)):
            # suit = SUITS[card // 13]
            # rank = RANKS[card % 13]
            if hand[card] == 1:
                cards.append(self.card_to_name(card))
        return cards
    
    def get_top_cards(self) -> np.ndarray[int]:
        top = np.zeros(52)
        for card in self.top_cards:
            top[card] += 1
        return top

    def valid_move_values(self, hand: np.ndarray[int]) -> list[tuple[int, list[int]]]:
        valid_moves = []
        nonzeros = np.nonzero(hand)
        for i in range(len(nonzeros[0])):
            #print(i)
            valid_moves += [(self.card_value(nonzeros[0][i] % 13), [i])]
        # print(f"VMS {valid_moves}")
        for idx, i in enumerate(nonzeros[0]):
            valid_moves += self.move_value(hand, i)
        # print(f"VMS ARE NOW {valid_moves}")
        return valid_moves

    def valid_third_moves(self) -> list[str]:
        valid_moves = ["Deck", "Card 0"]
        if len(self.top_cards) == 2:
            valid_moves.append("Card 1")
        return valid_moves
    
    def card_value(self, rank : int):
        if rank == 0:
            return 1
        elif rank > 9:
            return 10
        else:
            return rank + 1

    def move_value(self, hand: np.ndarray, card: int) -> list[tuple[int, list[int]]]:
        suit = card // 13
        rank = card % 13

        hand_copy = hand.reshape((4, 13))  # Reshape hand to suits and ranks

        # Assuming COMBINATIONS is defined elsewhere and suitable for the context
        #print("HEE "  + str(len(np.nonzero(hand)[0])))
        set_combinations = helpers.COMBINATIONS[4]

        hand_suit = hand_copy[suit]
        value_tups = []
        
        # Calculate potential sequence after the given card
        nonzeros = list(np.nonzero(hand)[0])
        work = True
        for i in range(3):
            work = work and hand[(card + i) % 52] == 1
        if work:
            value_tups.append((self.card_value(rank) + self.card_value(rank + 1) + self.card_value(rank + 2), [nonzeros.index(card) + i for i in range(3)]))

        for set_comb in set_combinations:
            if len(set_comb) >= 2:
                works = True
                for s in set_comb:
                    if hand_copy[s, rank] == 0:
                        works = False
            if len(set_comb) >= 2 and all(hand_copy[s, rank] == 1 for s in set_comb):
                set_value = sum(self.card_value(rank) for s in set_comb)  # Calculate total value for set
                value_tups.append((set_value, [nonzeros.index((s * 13) + rank) for s in set_comb])) # Need it to be index in hand of nzs

        return value_tups

    def card_indices(self, hand: np.ndarray[int]) -> list[int]:
        return np.nonzero(hand)

    def completes_set(self, hand: np.ndarray[int], card: int) -> bool:
        hand_copy = hand.copy()
        hand_copy[card] += 1
        hand_copy.reshape((4, 13)).clip(0, 1)
        return len(np.nonzero(hand_copy.reshape((52)))) > 1

    def completes_straight(self, hand: np.ndarray[int], card: int) -> bool:
        suit = card // 13
        rank = card % 13
        hand_reshaped = hand.copy().reshape((4, 13))
        hand_suit = hand_reshaped[suit]
        between = (hand_suit[(rank - 1) % 13] ==
                   1 and hand_suit[(rank + 1) % 13] == 1)
        before = (hand_suit[(rank - 1) % 13] ==
                  1 and hand_suit[(rank - 2) % 13] == 1)
        after = (hand_suit[(rank + 1) % 13] ==
                 1 and hand_suit[(rank + 2) % 13] == 1)
        return between or before or after

    def get_features(self, hand : np.ndarray, other_hand : np.ndarray) -> tuple[np.ndarray[int], int, int, np.ndarray]:
        our_hand_value = self.get_hand_value(hand)
        other_player_num_cards = len(other_hand)
        turn = self.turn
        top_cards = self.get_top_cards()
        return other_player_num_cards, turn, top_cards

    def get_hand_value(self, hand: np.ndarray[int]) -> int:
        sum = (hand.reshape((4, 13)) * np.arange(1, 14)).clip(0, 10).sum()
        return sum

    def deal(self) -> int:
        if self.curr_idx == 51:
            self.deck = np.arange(52)
            np.random.shuffle(self.deck)
            self.curr_idx = -1
        self.discard = []
        self.top_cards = []
        self.curr_idx += 1
        return self.deck[self.curr_idx]

    def can_yaniv(self, hand: np.ndarray) -> bool:
        return self.get_hand_value(hand) <= 7

    def yaniv(self, hand: np.ndarray[int], other_hands: list[np.ndarray]) -> bool:
        our_value = self.get_hand_value(hand)
        # print(our_value)
        for others in other_hands:
            other_value = self.get_hand_value(others)
            # print(other_value)
            if our_value >= other_value:
                return False
        return True

    def draw(self, idx: int) -> int:
        if idx <= len(self.top_cards) - 1:
            card = self.top_cards[idx]
        else:
            card = self.top_cards[0]
        self.top_cards = []
        return card
    
    def play(self, hand : np.ndarray[int], cards : list[int], draw_idx : int) -> int:
        #print("-----\n\n\n")
        self.discard += self.top_cards
        nzs = np.nonzero(hand)[0]
        h = hand.copy()

        counter = 0
        for nz in nzs:
            if counter in cards:
                hand[nz] -= 1
            counter += 1

        if draw_idx >= 0:
            card_drawn = self.draw(draw_idx)
        else:
            card_drawn = self.deal()
        if len(cards) <= 1:
            self.top_cards = [nzs[cards[0]]]
        else:
            try:
                self.top_cards = [nzs[cards[0]], nzs[cards[-1]]]
            except Exception:
                print(cards)
                print(cards[0])
                print(cards[-1])
                print(nzs)
                print(hand)
        hand[card_drawn] += 1
        return card_drawn

    def playOpponentTurn(self) -> tuple[bool, bool]:
        done = False
        won = False
        if self.can_yaniv(self.player_2_hand):
            won = self.yaniv(self.player_2_hand, [self.player_1_hand])
            done = True
        else:
            moves = self.valid_move_values(self.player_2_hand)
            moves.sort(key=lambda move: move[0])
            move_i = moves[0]
            curr = "Deck"
            drew = False
            for move in self.valid_third_moves():
                if move == "Deck":
                    pass
                elif move == "Card 0":
                    if self.completes_set(self.player_2_hand, self.top_cards[0]):
                        idx = 0
                        drew = True
                else:
                    if self.completes_set(self.player_2_hand, self.top_cards[1]):
                        idx = 1
                        drew = True
            if not drew:
                idx = -1
            self.play(self.player_2_hand, move_i[1], idx)
        return done, won
