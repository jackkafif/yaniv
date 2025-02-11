from __future__ import annotations
from combined_model import CombinedModel, sim_step
from model_a import ModelA
from model_b import ModelB
from model_c import ModelC
from helpers import *
from state import GameState
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt


def visualize_wins_over_time(args):
    x_array = []
    y_array = []

    won_games = 0
    num_episodes = int(args[1])
    sim = CombinedModel(A_explore_prob=0.8,
                        B_explore_prob=0.8, C_explore_prob=0.8)
    sim2 = CombinedModel(A_explore_prob=0.8,
                        B_explore_prob=0.8, C_explore_prob=0.8)
    for episode in range(num_episodes):
        # print(f"Running episode {episode}")
        state = GameState()
        done = False
        player1_hand = state.player_1_hand
        player2_hand = state.player_2_hand
        while not done:
            state, done, win = sim_step(sim, state, player1_hand, player2_hand)
            if done and win:
                break
            state, done, win = sim_step(sim2, state, player2_hand, player1_hand)
            if done and win:
                win = False
                break

    sim3 = CombinedModel(A_explore_prob=0.8,
                        B_explore_prob=0.8, C_explore_prob=0.8)
    for episode in range(num_episodes):
        # print(f"Running episode {episode}")
        state = GameState()
        done = False
        player1_hand = state.player_1_hand
        player2_hand = state.player_2_hand
        while not done:
            state, done, win = sim_step(sim, state, player1_hand, player2_hand)
            if done and win:
                break
            state, done, win = sim_step(sim3, state, player2_hand, player1_hand)
            if done and win:
                win = False
                break
        won_games += 1 if win else 0
        x_array.append(episode)
        y_array.append(won_games)


    plt.plot(x_array, y_array)
    plt.show()


def visualize_probs_vs_win_percentage():
    x_array = []
    y_array = []
    for prob in (.1, .2, .3, .4, .5, .6, .7, .8, .9):
        won_games = 0
        num_episodes = 500
        for episode in range(num_episodes):
            # print(f"Running episode {episode}")
            sim = CombinedModel(A_explore_prob=prob,
                                B_explore_prob=prob, C_explore_prob=prob)
            state = GameState()
            done = False
            while not done:
                features = sim.features(state)
                m1, m2, m3, p1, p2, p3 = sim.choose_actions(features, state)
                action = (m1, m2, m3)
                possible = (p1, p2, p3)
                next_state, reward, done, win = sim.play_step(state, action)
                sim.update_weights(features, action, possible, reward)
                state = next_state
            won_games += 1 if win else 0
        x_array.append(won_games/1000)
        y_array.append(prob)
        print("Loop Done")

    plt.plot(y_array, x_array)
    plt.show()


def main():
    visualize_wins_over_time(sys.argv)


if __name__ == "__main__":
    main()
