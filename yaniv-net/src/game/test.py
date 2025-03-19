import numpy as np
from game.state import GameState

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

    # print(state.valid_move_values(hand1))

    hand1[True] = 0
    for card in ['Six of Hearts', 'Seven of Hearts', 'Jack of Hearts', 'Eight of Diamonds']:
        hand2[state.name_to_card(card)] += 1
    
    for card in ['Ace of Clubs', 'Ace of Spades', 'Three of Diamonds']:
        hand1[state.name_to_card(card)] += 1

    print(state.get_hand_value(hand1))
    print(state.get_hand_value(hand2))
    print(state.yaniv(hand1, [hand2]))

if __name__ == "__main__":
    test_state()