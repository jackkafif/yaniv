from game.combined_model import CombinedModel, play_step
from game.state import GameState
from gameplay.userinterface import display_hand
import time
import os


def play(model: CombinedModel, visual=False):
    os.system('clear')
    while True:  # input("Would you like to play Yaniv? Y/N: ") == "Y":
        state = GameState()
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
                display_hand(state.player_2_hand, False)
                print("Top cards: ")
                display_hand(state.get_top_cards(), True)
                print("Your hand:")
                display_hand(state.player_1_hand, True)
            yaniv = input(
                "Enter Y if you'd like to call Yaniv, anything else otherwise: ")
            if "Y" == yaniv and state.can_yaniv(state.player_1_hand):
                if state.can_yaniv(state.player_1_hand):
                    done = True
                    if state.yaniv(state.player_1_hand, [state.player_2_hand]):
                        print("You won!")
                    else:
                        print("You lost!")
                    break
            valid_moves = list(state.valid_moves(state.player_1_hand))
            print(valid_moves)
            while True:
                move = input(
                    "Input the cards you want to play in their indices 0 1 2 ...: ")
                try:
                    indices = tuple([int(i) for i in move.split(" ")])
                    if indices in valid_moves:
                        break
                except:
                    print("Invalid move, please try again")

            draw = input(
                "Input -1 to draw from deck, 0 to draw first index of discard, 1 for second: ")
            while True:
                try:
                    if int(draw) in [-1, 0, 1]:
                        break
                    else:
                        print("Invalid draw, please try again ")
                except:
                    pass
            state.play(state.player_1_hand, move, draw)

            if done:
                print(f"Your opponent called Yaniv! \
                        They have {state.get_hand_value(state.player_1_hand)} \
                        to your {state.get_hand_value(state.player_2_hand)}")
                print(
                    f"Opponent has {state.hand_to_cards(state.player_1_hand)}")
                message = "won!" if win == False else "lost!"
                print(f"You have {message}")
                break

            state.play(state.player_2_hand, move, int(draw))
            # print(f"Your hand is now {state.player_2_hand}")


play("", True)
