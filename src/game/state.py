from __future__ import annotations
import numpy as np
import game.helpers as helpers

POSSIBLE_MOVES = helpers.generate_combinations(5)
N_MOVES = len(POSSIBLE_MOVES)

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
        """
        Initializes Yaniv game state

        Shuffles the deck, deals 5 cards to each of the 2 players, and opens a card at the top of the discard
        """
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
        """
        Reshuffles the cards and redeals hands
        """
        return GameState()

    def card_to_name(self, card):
        """
        The name of the card in English
        
        Args:
            card (int) : The index of the card in the card array

        Returns:
            string: a string of the form {rank} of {suit}
        """
        suit = SUITS[card // 13]
        rank = RANKS[card % 13]
        return f"{rank} of {suit}"
    
    def name_to_card(self, name : str) -> int:
        """
        The index of the card in the card array
        
        Args:
            name (int) : The name of the card in English

        Returns:
            int: The index of the card in the card array
        """
        word = name.split(" ")
        rank, suit = RANKS_REVERSE[word[0]], SUITS_REVERSE[word[2]]
        return suit * 13 + rank

    def hand_to_cards(self, hand: np.ndarray[int]) -> list[str]:
        """
        A list of the names of the cards in hand
        
        Args:
            hand (np.ndarray[int]) : The one hot array of cards representing the hand

        Returns:
            list[string]: A list of the English names of the cards in hand
        """
        cards = []
        for card in range(len(hand)):
            if hand[card] == 1:
                cards.append(self.card_to_name(card))
        return cards
    
    def move_value(self, hand : np.ndarray[int], move: list[int]) -> int:
        """
        Value of the cards being played in move {move} from hand {hand}

        Params: 
            hand (np.ndarray[int]) : The one hot array of cards representing the hand
            move (list[int]) : List of indices of nonzero cards in hand (0 - 4) representing move
        
        Returns:
            int : Value of the cards played
        """
        card_idxs = self.card_indices(hand)
        value = 0
        for m in move:
            val = self.card_value(card_idxs[m]) 
            value += val
        return value
    
    def move_to_cards(self, hand: np.ndarray[int], move: list[int]) -> str:
        """
        String of the cards being played in move {move} from hand {hand}

        Params: 
            hand (np.ndarray[int]) : The one hot array of cards representing the hand
            move (list[int]) : List of indices of nonzero cards in hand (0 - 4) representing move
        
        Returns:
            str : English language list of cards played in move
        """
        card_idxs = self.card_indices(hand)
        sb = "[ "
        for idx, i in enumerate(move):
            sb += self.card_to_name(card_idxs[i]) 
            if idx != len(move) - 1:
                sb += ", "
        sb += " ]"
        return sb
    
    def get_top_cards(self) -> np.ndarray[int]:
        """
        Returns:
            cards (np.ndarray[int]) : The one hot array of cards representing the top of the discard pile
        """
        top = np.zeros(52)
        for card in self.top_cards:
            top[card] += 1
        return top
    
    def valid_move(self, cards : list[int]) -> bool:
        """
        Whether playing the cards in cards is a valid move

        Args:
            cards (np.ndarray[int]) : The one hot array of cards representing the hand

        Returns:
            bool : Whether or not the move playing the cards in cards is valid
        """
        if len(cards) == 1:
            return True
        if len(cards) < 3:
            return cards[0] % 13 == cards[1] % 13
        rank = cards[0] % 13
        curr = cards[0]
        rank_same = True
        straight = True
        for card in cards[1:]:
            rank_same = rank_same and (rank == card % 13)
            straight = straight and (card == curr + 1)
            curr = card
        return straight or rank_same
    
    def valid_move_indices(self, hand: np.ndarray[int]) -> np.ndarray[int]:
        """
        The valid playable moves from hand in vector form

        Args:
            cards (np.ndarray[int]) : The one hot array of cards representing the hand

        Returns:
            np.ndarry[int] : A list of tuples combinations of playable cards in the hand
        """
        try:
            nonzeros = np.nonzero(hand)[0]
            valids = np.zeros(31)
            if len(nonzeros) == 0:
                return valids
            for idx, comb in helpers.COMBINATIONS[len(nonzeros)].items():
                if all(i < len(nonzeros) for i in comb):
                    cards = [nonzeros[i] for i in comb]
                    if self.valid_move(cards):
                        valids[idx] = 1
            return valids
        except:
            print(hand)
            raise Exception
    
    def valid_moves(self, hand: np.ndarray[int]) -> list[tuple[int]]:
        """
        The valid playable moves from hand

        Args:
            cards (np.ndarray[int]) : The one hot array of cards representing the hand

        Returns:
            list[tuple[int]] : A list of tuples combinations of playable cards in the hand
        """
        valids = self.valid_move_indices(hand)
        return np.where(valids == 1)[0]
    
    def card_value(self, idx : int):
        """
        The yaniv value for a card of rank rank

        Params:
            idx (int) : The index of the card in the hand

        Returns:
            int : The yaniv value for that card
        """
        if idx % 13 == 0:
            return 1
        elif idx % 13 > 9:
            return 10
        else:
            return idx % 13 + 1

    def card_indices(self, hand: np.ndarray[int]) -> list[int]:
        """
        The indices in hand that have the cards in hand

        Params:
            hand (np.ndarray[int]) : The one hot array of cards representing the hand

        Returns:
            list[int] : The list of indices of the cards in the hand
        """
        return np.nonzero(hand)[0]

    def get_features(self, hand : np.ndarray, other_hand : np.ndarray) -> np.ndarray[int]:
        """
        The features of the game known to player with hand {hand} playing against opponent with hand {other_hand}

        Params:
            hand (np.ndarray[int]) : The one hot array of cards representing the player's hand
            other_hand (np.ndarray[int]) : The one hot array of cards representing the opponents hand

        Returns:
            np.ndarray[int] : The features of the game known to player
        """
        other_player_num_cards = len(other_hand)
        turn = self.turn
        top_cards = self.get_top_cards()
        return np.concatenate([hand.flatten(), top_cards, [other_player_num_cards, turn]])

    def get_hand_value(self, hand: np.ndarray[int]) -> int:
        """
        The Yaniv value of the hand

        Params:
            hand (np.ndarray[int]) : The one hot array of cards representing the player's hand

        Returns:
            int : The summed total score of the player's current hand
        """
        sum = (hand.reshape((4, 13)) * np.arange(1, 14)).clip(0, 10).sum()
        return sum

    def deal(self) -> int:
        """
        Deals a card

        Returns:
            int : The index in the 52 length array where the dealt card should be placed
        """
        self.curr_idx += 1
        if self.curr_idx >= len(self.deck):
            self.deck = self.discard
            self.discard = []
            np.random.shuffle(self.deck)
            self.curr_idx = 0
        return self.deck[self.curr_idx]
            

    def can_yaniv(self, hand: np.ndarray) -> bool:
        """
        Whether player with hand {hand} can call Yaniv

        Params:
            hand (np.ndarray[int]) : The one hot array of cards representing the player's hand

        Returns:
            bool : Whether this player has hand value <= 7, can call Yaniv
        """
        return self.get_hand_value(hand) <= 7

    def yaniv(self, hand: np.ndarray[int], other_hands: list[np.ndarray]) -> bool:
        """
        Calls Yaniv for player with hand {hand} and checks whether player won game

        Params:
            hand (np.ndarray[int]) : The one hot array of cards representing the player's hand
            other_hands (list[np.ndarray[int]]) : List of one hot array of cards representing the opponent players' hand

        Returns:
            bool : Whether player won against all the rest of the players
        """
        our_value = self.get_hand_value(hand)
        for others in other_hands:
            other_value = self.get_hand_value(others)
            if our_value >= other_value:
                return False
        return True

    def draw_card(self, idx: int) -> int:
        """
        Draws card with index {idx} from the discard

        Params:
            idx (int) : The index of the card in the dicard pile

        Returns:
            int : The index of the picked up card in the 52 length array representing a players hand
        """
        # print(self.top_cards, idx)
        try:
            if idx <= len(self.top_cards) - 1:
                card = self.top_cards[idx]
            else:
                card = self.top_cards[0]
            self.top_cards = []
            return card
        except:
            print(self.top_cards, idx)
            raise Exception
    
    def play(self, hand : np.ndarray[int], cards : list[int]) -> int:
        """
        Plays the cards in {cards} from {hand} and draws {draw_idx} card from discard or deck

        Params:
            hand (np.ndarray[int]) : The one hot array of cards representing the player's hand
            cards (list[int]) : A list of indices in hand of the cards to play
            draw_idx (int) : -1 if drawing from deck, 0 if drawing first card from discard, 1 if drawing second card from dicard

        Returns:
            int : The card draw after playing {cards} from player's {hand}
        """
        self.discard += self.top_cards
        nzs = np.nonzero(hand)[0]

        counter = 0
        for nz in nzs:
            if counter in cards:
                hand[nz] -= 1
            counter += 1

        return
    
    def draw(self, hand : np.ndarray[int], cards : list[int], draw_idx : int) -> int:
        """
        Draws {draw_idx} card from discard or deck and places {cards} on top of discard pile

        Params:
            hand (np.ndarray[int]) : The one hot array of cards representing the player's hand before playing
            cards (list[int]) : A list of indices in hand of the cards to play
            draw_idx (int) : -1 if drawing from deck, 0 if drawing first card from discard, 1 if drawing second card from dicard

        Returns:
            int : The card draw after playing {cards} from player's {hand}
        """
        nzs = np.nonzero(hand)[0]
        if draw_idx == 0:
            card_drawn = self.deal()
        else:
            card_drawn = self.draw_card(draw_idx - 1)

        # print(nzs, cards)
        try:
            if len(cards) <= 1:
                self.top_cards = [nzs[cards[0]]]
            else:
                self.top_cards = [nzs[cards[0]], nzs[cards[-1]]]
            return card_drawn
        except:
            print(self.hand_to_cards(hand), nzs, cards)
            raise Exception
    
    def completes_move(self, hand : np.ndarray[int], card : int) -> bool: 
        """
        Whether card of index {card} completes a set or a straight

        Params:
            hand (np.ndarray[int]) : The one hot array of cards representing the player's hand
            card (int] : index in hand of the cards

        Returns:
            bool : Whether card of index {card} completes a set or a straight
        """
        completes = False
        completes = completes or (hand[card - 1] == 1 and hand[card + 1] == 1)
        for i in [card - 13, card - 26, card - 39]:
            completes = completes or hand[i] == 1
        return completes
    
    def valid_draws(self) -> list[int]:
        """
        Returns: 
            list[int] : Valid draw indices for (-1 for deck, 0 for first index of discard, 1 for second index of discard)
        """
        draws = [0]
        for i in range(1, len(self.top_cards)):
            draws.append(i + 1)
        return draws
    
    def playOpponentTurn(self, hand: np.ndarray, other : np.ndarray) -> tuple[bool, bool]:
        """
        Plays opponents' turns (using self.player_2_hand as opponent hand) based on heuristic

        Returns:
            tuple[bool, bool] : tuple of whether the game is over and if the game is over whether opponent won
        """
        done = False
        won = False
        if self.can_yaniv(hand):
            won = self.yaniv(hand, [other])
            done = True
        else:
            moves = list(self.valid_moves(hand))
            moves.sort( 
                key = lambda move : -1 * self.move_value(
                    self.player_2_hand, list(POSSIBLE_MOVES[move])
                )
            )
            move_i = moves[0]
            drew = False
            for move in self.valid_draws():
                if move == 0:
                    pass
                else:
                    if self.completes_move(hand, self.top_cards[move - 1]):
                        idx = move
                        drew = True
            if not drew:
                idx = 0
            play_move = list(POSSIBLE_MOVES[move_i])
            hand_copy = hand.copy()
            self.play(hand, play_move)
            hand[self.draw(hand_copy, play_move, idx)] += 1
        return done, won
