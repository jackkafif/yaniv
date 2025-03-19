# game_state.py
from __future__ import annotations
import numpy as np
import helpers

SUITS = {
    0: "Clubs",
    1: "Spades",
    2: "Hearts",
    3: "Diamonds"
}

RANKS = {
    0: "Ace", 1: "Two", 2: "Three", 3: "Four",
    4: "Five", 5: "Six", 6: "Seven", 7: "Eight", 8: "Nine",
    9: "Ten", 10: "Jack", 11: "Queen", 12: "King"
}

SUITS_REVERSE = {value: key for key, value in SUITS.items()}
RANKS_REVERSE = {value: key for key, value in RANKS.items()}


class GameState:
    def __init__(self) -> None:
        self.deck = np.arange(52)
        np.random.shuffle(self.deck)
        self.player_1_hand = np.zeros(52, dtype=int)
        self.player_2_hand = np.zeros(52, dtype=int)
        self.curr_idx = -1
        for i in range(7):
            self.player_1_hand[self.deal()] += 1
            self.player_2_hand[self.deal()] += 1
        self.turn = 0
        self.discard = []
        self.top_cards = [self.deal()]
        self.cards_played = []
        self.over = False

    def reset(self) -> GameState:
        return GameState()

    def card_to_name(self, card: int) -> str:
        suit = SUITS[card // 13]
        rank = RANKS[card % 13]
        return f"{rank} of {suit}"
    
    def name_to_card(self, name: str) -> int:
        words = name.split(" ")
        rank, suit = RANKS_REVERSE[words[0]], SUITS_REVERSE[words[2]]
        return suit * 13 + rank

    def hand_to_cards(self, hand: np.ndarray) -> list[str]:
        cards = []
        for card in range(len(hand)):
            if hand[card] == 1:
                cards.append(self.card_to_name(card))
        return cards

    def get_top_cards(self) -> np.ndarray:
        top = np.zeros(52, dtype=int)
        for card in self.top_cards:
            top[card] += 1
        return top

    def valid_move(self, cards: list[int]) -> bool:
        if len(cards) == 1:
            return True
        if len(cards) < 3:
            return cards[0] % 13 == cards[1] % 13
        rank = cards[0] % 13
        curr = cards[0]
        rank_same = True
        straight = True
        for card in cards:
            rank_same = rank_same and (rank == card % 13)
            straight = straight and (card == curr + 1)
            curr = card
        return straight or rank_same

    def valid_moves(self, hand: np.ndarray) -> np.ndarray:
        nonzeros = np.nonzero(hand)[0]
        valids = np.zeros(31)
        for idx, comb in helpers.COMBINATIONS[len(nonzeros)].items():
            if all(i < len(nonzeros) for i in comb):
                cards = [nonzeros[i] for i in comb]
                if self.valid_move(cards):
                    valids[idx] = 1
        return valids

    def valid_move_values(self, hand: np.ndarray) -> list[tuple[int, list[int]]]:
        valid_moves = []
        nonzeros = np.nonzero(hand)[0]
        # Single card moves.
        for i in range(len(nonzeros)):
            valid_moves.append((self.card_value(nonzeros[i] % 13), [i]))
        # Combination moves.
        for idx, card in enumerate(nonzeros):
            valid_moves += self.move_value(hand, card)
        return valid_moves

    def valid_third_moves(self) -> list[str]:
        valid_moves = ["Deck", "Card 0"]
        if len(self.top_cards) == 2:
            valid_moves.append("Card 1")
        return valid_moves

    def card_value(self, rank: int) -> int:
        if rank == 0:
            return 1
        elif rank > 9:
            return 10
        else:
            return rank + 1

    def move_value(self, hand: np.ndarray, card: int) -> list[tuple[int, list[int]]]:
        suit = card // 13
        rank = card % 13
        hand_copy = hand.reshape((4, 13))
        set_combinations = helpers.COMBINATIONS[4]
        value_tups = []
        nonzeros = list(np.nonzero(hand)[0])
        work = True
        for i in range(3):
            work = work and hand[(card + i) % 52] == 1
        if work:
            value_tups.append((
                self.card_value(rank) + self.card_value(rank + 1) + self.card_value(rank + 2),
                [nonzeros.index(card) + i for i in range(3)]
            ))
        for set_comb in set_combinations:
            if len(set_comb) >= 2 and all(hand_copy[s, rank] == 1 for s in set_comb):
                set_value = sum(self.card_value(rank) for s in set_comb)
                value_tups.append((
                    set_value,
                    [nonzeros.index((s * 13) + rank) for s in set_comb]
                ))
        return value_tups

    def card_indices(self, hand: np.ndarray) -> list[int]:
        return list(np.nonzero(hand)[0])

    def completes_set(self, hand: np.ndarray, card: int) -> bool:
        hand_copy = hand.copy()
        hand_copy[card] += 1
        hand_copy = hand_copy.reshape((4, 13))
        return np.count_nonzero(hand_copy) > 1

    def completes_straight(self, hand: np.ndarray, card: int) -> bool:
        suit = card // 13
        rank = card % 13
        hand_reshaped = hand.copy().reshape((4, 13))
        hand_suit = hand_reshaped[suit]
        between = (hand_suit[(rank - 1) % 13] == 1 and hand_suit[(rank + 1) % 13] == 1)
        before = (hand_suit[(rank - 1) % 13] == 1 and hand_suit[(rank - 2) % 13] == 1)
        after = (hand_suit[(rank + 1) % 13] == 1 and hand_suit[(rank + 2) % 13] == 1)
        return between or before or after

    def get_features(self, hand: np.ndarray, other_hand: np.ndarray) -> tuple[np.ndarray, int, int, np.ndarray]:
        our_hand_value = self.get_hand_value(hand)
        other_player_num_cards = int(np.count_nonzero(other_hand))
        turn = self.turn
        top_cards = self.get_top_cards()
        return (hand, our_hand_value, turn, top_cards)

    def get_hand_value(self, hand: np.ndarray) -> int:
        reshaped = hand.reshape((4, 13))
        card_vals = np.array([self.card_value(i) for i in range(13)])
        total = (reshaped * card_vals).sum()
        return total

    def deal(self) -> int:
        if self.curr_idx >= 51:
            self.deck = np.arange(52)
            np.random.shuffle(self.deck)
            self.curr_idx = -1
        self.discard = []
        self.top_cards = []
        self.curr_idx += 1
        return int(self.deck[self.curr_idx])

    def can_yaniv(self, hand: np.ndarray) -> bool:
        return self.get_hand_value(hand) <= 7

    def yaniv(self, hand: np.ndarray, other_hands: list[np.ndarray]) -> bool:
        our_value = self.get_hand_value(hand)
        for other in other_hands:
            if our_value >= self.get_hand_value(other):
                return False
        return True

    def draw(self, idx: int) -> int:
        if idx < len(self.top_cards):
            card = self.top_cards[idx]
        else:
            card = self.top_cards[0]
        self.top_cards = []
        return card

    def play(self, hand: np.ndarray, cards: list[int], draw_idx: int) -> int:
        self.discard += self.top_cards
        nzs = np.nonzero(hand)[0]
        counter = 0
        for nz in nzs:
            if counter in cards:
                hand[nz] -= 1
            counter += 1

        if draw_idx >= 0:
            card_drawn = self.draw(draw_idx)
        else:
            card_drawn = self.deal()
        if cards:
            if len(cards) <= 1:
                self.top_cards = [nzs[cards[0]]]
            else:
                try:
                    self.top_cards = [nzs[cards[0]], nzs[cards[-1]]]
                except Exception:
                    print("Error in play:", cards, nzs)
        hand[card_drawn] += 1
        return card_drawn

    def get_state_vector(self) -> np.ndarray:
        """
        Build a state vector including:
          - The player's hand (52 dimensions)
          - The top cards (52 dimensions)
          - The opponent's hand value (1 dimension)
        Total length: 105
        """
        hand_vector = self.player_2_hand.copy()
        top_vector = self.get_top_cards()
        opp_hand_value = np.array([self.get_hand_value(self.player_1_hand)])
        state = np.concatenate([hand_vector, top_vector, opp_hand_value])
        return state

    def playOpponentTurnHeuristic(self) -> tuple[bool, bool]:
        """
        A heuristic opponent turn for testing or fallback.
        """
        done = False
        won = False

        my_hand_value = self.get_hand_value(self.player_2_hand)
        opponent_hand_value = self.get_hand_value(self.player_1_hand)
        
        if self.can_yaniv(self.player_2_hand) and my_hand_value < min(7, opponent_hand_value):
            won = self.yaniv(self.player_2_hand, [self.player_1_hand])
            done = True
            return done, won

        moves = self.valid_move_values(self.player_2_hand)
        if not moves:
            self.play(self.player_2_hand, [], -1)
            return done, won

        best_score = float('inf')
        best_move = None
        best_draw_idx = -1
        current_hand_value = my_hand_value
        nonzeros = np.nonzero(self.player_2_hand)[0]
        
        for move_value, indices in moves:
            discard_value = sum(self.card_value(nonzeros[i] % 13) for i in indices)
            new_hand_value_estimate = current_hand_value - discard_value

            draw_idx = -1 
            for idx, top_card in enumerate(self.top_cards):
                if self.completes_set(self.player_2_hand, top_card) or self.completes_straight(self.player_2_hand, top_card):
                    bonus = 3 
                    new_hand_value_estimate -= bonus
                    draw_idx = idx
                    break 

            if new_hand_value_estimate < best_score:
                best_score = new_hand_value_estimate
                best_move = indices
                best_draw_idx = draw_idx

        if best_move is None:
            best_move = moves[0][1]
            best_draw_idx = -1

        self.play(self.player_2_hand, best_move, best_draw_idx)
        return done, won
