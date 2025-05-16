from game.state import GameState
from gameplay.userinterface import display_hand
import time
import os
from game.globals import *
from game.train import train_models
from game.agent import YanivAgent, play_agent
from game.model import MDQN, M, linear_vs_linear, linear_vs_multi, multi_vs_multi
import numpy as np
import random
import sys
import torch


def set_seed():
    np.random.seed(44)
    random.seed(44)
    torch.manual_seed(44)
    torch.cuda.manual_seed_all(44)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def play(visual=False, i=0, model_comb=linear_vs_linear):
    set_seed()
    m1, m2 = train_models(model_comb, i)
    input("Press Enter to start playing...")
    os.system('clear')
    if random.choice([True, False]):
        m = m1
    else:
        m = m2
    m.epsilon = 0.0
    play = True
    while play:
        os.system('clear')
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
                display_hand(state.player_2_hand, True, False)
                # print("Value: " + str(state.get_hand_value(state.player_2_hand)))
                print("Top cards: ")
                display_hand(state.get_top_cards(state.top_cards), True, True)
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
                        cp = state.player_1_hand.copy()
                        state.play(state.player_1_hand, indices)
                        draw = int(draw)
                        if draw == 0:
                            draw = state.draw(cp, indices, 52)
                        else:
                            tcs = state.tc_holder
                            draw = state.draw(cp, indices, tcs[draw-1])
                        state.player_1_hand[draw] += 1
                        break
                    print("Invalid draw, please try again ")
                except:
                    print("Invalid draw, please try again ")

            # print(indices)

            # state.play(state.player_2_hand, move)
            # state.draw(cp2, move, int(draw))
            done, won = m.play_agent(
                state, state.player_2_hand, state.player_1_hand, True)
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        NUM_EPISODES = int(sys.argv[1])
        model_comb = linear_vs_linear if sys.argv[
            2] == "linear" else multi_vs_multi if sys.argv[2] == "multi" else linear_vs_multi
    else:
        NUM_EPISODES = 10000
        model_comb = linear_vs_linear
    if len(sys.argv) > 2:
        model_comb = linear_vs_linear if sys.argv[
            2] == "linear" else multi_vs_multi if sys.argv[2] == "multi" else linear_vs_multi
    else:
        model_comb = linear_vs_linear
    if len(sys.argv) > 3:
        vis = sys.argv[3] == "visual"
    else:
        vis = False
    play(vis, NUM_EPISODES, model_comb)
