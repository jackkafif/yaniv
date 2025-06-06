from game.state import GameState
import numpy as np
import time
import os

to_suit = {
    0: "C",
    1: "S",
    2: "H",
    3: "D"
}

to_value = {
    0: "A",
    1: "2",
    2: "3",
    3: "4",
    4: "5",
    5: "6",
    6: "7",
    7: "8",
    8: "9",
    9: "T",
    10: "J",
    11: "Q",
    12: "K"
}


def display_hand(hand, show=True, add_deck=False):
    total_cards = int(np.sum(hand))
    print_string = ""
    if add_deck:
        add = 1
    else:
        add = 0
    if not show:
        print_string += (" _______    "*(total_cards+add) + "\n" +
                         ("|       |   " * (total_cards+add) + "\n")*4 +
                         "|_______|   "*(total_cards+add))
        print(print_string)
    else:
        cards_list = []
        for i in range(52):
            if hand[i] == 1:
                suit = to_suit[i//13]
                value = to_value[i % 13]
                cards_list.append((suit, value))
        suit_string = ""
        value_string = ""
        for suit, value in cards_list:
            value_string += ("|   " + value + "   |   ")
            suit_string += ("|   " + suit + "   |   ")
        if add_deck:
            value_string = "|       |   " + value_string
            suit_string = "|       |   " + suit_string
        print_string += (" _______    "*(total_cards+add) + "\n" +
                         ("|       |   " * (total_cards+add) + "\n") +
                         value_string + "\n" +
                         ("|       |   " * (total_cards+add) + "\n") +
                         suit_string + "\n" +
                         "|_______|   "*(total_cards+add))
        print(print_string)


if __name__ == "__main__":
    hand1 = np.zeros(52)
    hand1[0] = 1
    hand1[19] = 1
    hand1[38] = 1
    hand1[42] = 1

    display_hand(hand1)
