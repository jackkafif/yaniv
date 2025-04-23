from game.train import train_models, play_agent
from game.state import GameState
from gameplay.userinterface import display_hand
import time
import os
from game.globals import *
from game.train import train_models, play_agent
import numpy as np


def play(visual=False):
    w1, w2, m1, m2 = train_models(1000)
    os.system('clear')
    if w1 > w2:
        m = m1
        print(w1)
    else:
        m = m2
        print(m2)
    m.epsilon = 0.1
    play = True
    while play:
        # input("Would you like to play Yaniv? Y/N: ") == "Y":
        os.system('clear')
        state = GameState()
        # state.player_1_hand = np.zeros(52)
        # state.player_1_hand[state.name_to_card("Ace of Clubs")] = 1
        # state.player_1_hand[state.name_to_card("Two of Clubs")] = 1
        # state.player_1_hand[state.name_to_card("Three of Clubs")] = 1
        # state.player_1_hand[state.name_to_card("Queen of Diamonds")] = 1
        # state.player_1_hand[state.name_to_card("King of Diamonds")] = 1
        win = False
        done = False
        print("Dealing Cards...")
        while done != True:
            if not visual:
                print(
                    f"Your hand is: {state.hand_to_cards(state.player_1_hand)}")
                print(
                    f"The card(s) on the top is/are {[state.card_to_name(card) for card in state.top_cards]} ")
            else:
                print("Opponent's hand:")
                display_hand(state.player_2_hand, True, False)
                # print("Value: " + str(state.get_hand_value(state.player_2_hand)))
                print("Top cards: ")
                display_hand(state.get_top_cards(), True, True)
                print("Your hand:")
                display_hand(state.player_1_hand, True, False)
                # print("Value: " + str(state.get_hand_value(state.player_1_hand)))

            valid_moves = list(state.valid_moves(state.player_1_hand))
            valid_moves = [POSSIBLE_MOVES[move] for move in valid_moves]
            # print(valid_moves)
            while True:
                move = input(
                    "Enter Y if you'd like to call Yaniv. " +
                    "Otherwise, input the cards you want to play in their indices 0 1 2 ...: ")
                if (move == "Y"):
                    if state.can_yaniv(state.player_1_hand):
                        done = True
                        if state.yaniv(state.player_1_hand, [state.player_2_hand]):
                            print("You won!")
                        else:
                            print("You lost!")
                        break
                    else:
                        print("Invalid move, please try again")
                        continue

                try:
                    indices = tuple([int(i) for i in move.split(" ")])
                    if indices in valid_moves:
                        break
                    else:
                        print("Invalid move, please try again")
                except:
                    print("Invalid move, please try again")

            if done:
                break

            while True:
                draw = input(
                    "Input 0 to draw from deck, 1 to draw first index of discard, 2 for second: ")
                try:
                    if draw in ["0", "1", "2"]:
                        break
                    print("Invalid draw, please try again ")
                except:
                    print("Invalid draw, please try again ")

            # print(indices)
            cp = state.player_1_hand.copy()
            state.play(state.player_1_hand, indices)
            state.player_1_hand[state.draw(cp, indices, int(draw))] += 1

            # state.play(state.player_2_hand, move)
            # state.draw(cp2, move, int(draw))
            done, won = play_agent(
                state, m, state.player_2_hand, state.player_1_hand)
            if done:
                print("Your opponent called Yaniv! " +
                      "They have " + str(int(state.get_hand_value(state.player_2_hand))) +
                      " to your " + str(int(state.get_hand_value(state.player_1_hand))) + ".")
                # print(
                #     f"Opponent has {state.hand_to_cards(state.player_2_hand)}")
                message = "won!" if won == False else "lost!"
                print(f"You have {message}")
                break

            # print(f"Your hand is now {state.player_2_hand}")
        again = input("Enter Y to play again, anything else to quit: ")
        play = (again == "Y")


play(True)
