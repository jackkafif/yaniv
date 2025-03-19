import numpy as np
from game.state import GameState
from game.helpers import *

POSSIBLE_MOVES = generate_combinations(5)
N_MOVES = len(POSSIBLE_MOVES)

def test_state():
    state = GameState()

    # Testing converting card name to index and index to card name
    suits = ["Clubs", "Spades", "Hearts", "Diamonds"]
    ranks = ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Jack", "Queen", "King"]
    for i in range(len(suits)):
        for j in range(len(ranks)):
            suit = suits[i]
            rank = ranks[j]
            assert state.name_to_card(f"{rank} of {suit}") == i * 13 + j
            assert state.card_to_name(i * 13 + j) == f"{rank} of {suit}"

    hand1 = state.player_1_hand
    hand2 = state.player_2_hand
    hand1[True] = 0
    hand2[True] = 0

    cards1 = ["Ace of Spades", "Two of Spades", "Three of Spades", "Ten of Diamonds", "Jack of Clubs"]
    for card in cards1:
        idx = state.name_to_card(card)
        hand1[idx] += 1

    assert hand1.sum() == 5
    reshaped = hand1.reshape((4, 13))
    per_suits = [1, 3, 0, 1]
    for i in range(len(per_suits)):
        assert len(np.nonzero(reshaped[i])[0]) == per_suits[i]

    hand1[True] = 0

    for card in ['Ace of Clubs', 'Ace of Spades', 'Three of Diamonds']:
        hand1[state.name_to_card(card)] += 1

    hand1[state.name_to_card("Ace of Diamonds")] += 1
    print(state.hand_to_cards(state.player_1_hand), state.get_hand_value(state.player_1_hand))

    valid_moves = state.valid_moves(state.player_1_hand)
    for idx in valid_moves:
        move = list(POSSIBLE_MOVES[idx])
        print(state.move_to_cards(state.player_1_hand, move), state.move_value(state.player_1_hand, move))

    print("----")
    for card in ['Six of Hearts', 'Seven of Hearts', 'Jack of Hearts', 'Eight of Diamonds', "Eight of Hearts"]:
        hand2[state.name_to_card(card)] += 1
        
    print(state.card_indices(hand2))
    print(state.hand_to_cards(hand2), state.get_hand_value(hand2))  
    valid_moves2 = state.valid_moves(hand2)
    for idx in valid_moves2:
        move = list(POSSIBLE_MOVES[idx])
        print(state.move_to_cards(hand2, move), state.move_value(hand2, move))

    print("Removing Eight of Hearts")
    hand2[state.name_to_card("Eight of Hearts")] = 0
    print(state.hand_to_cards(hand2))
    valid_moves2 = state.valid_moves(hand2)
    print("Valid moves")
    for idx in valid_moves2 == 1:
        move = list(POSSIBLE_MOVES[idx])
        print(state.move_to_cards(hand2, move), state.move_value(hand2, move))

    print(state.completes_move(hand2, state.name_to_card("Eight of Hearts")))
    print(state.hand_to_cards(state.player_2_hand))
    done, won = state.playOpponentTurn()
    print(state.hand_to_cards(state.player_2_hand))

    print("Adding Eight of Hearts, seeing if heuristic plays it")
    hand2[state.name_to_card("Eight of Hearts")] = 1
    print(state.hand_to_cards(state.player_2_hand))
    done, won = state.playOpponentTurn()
    print(state.hand_to_cards(state.player_2_hand))

    hand2[True] = 0
    for card in ['Ace of Hearts', 'Two of Hearts', 'Three of Hearts', 'Eight of Diamonds']:
        hand2[state.name_to_card(card)] += 1

    state.player_2_hand = hand2
    print(state.hand_to_cards(state.player_2_hand))
    done, won = state.playOpponentTurn()
    print(state.hand_to_cards(state.player_2_hand))

if __name__ == "__main__":
    test_state()