import matplotlib.pyplot as plt
import sys
from game.nets import *
import random
import numpy as np
from game.globals import *
from game.model import MDQN, M
from game.state import GameState
from game.agent import YanivAgent, run_game
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def compare(agent1, agent2, num_trials=10, show=False):
    """
    Compare two agents by running a specified number of trials and counting wins for each agent.
    """
    wins_agent1 = 0
    wins_agent2 = 0

    if show:
        agent1_total_wins = []
        total_games = []
        total_game_length = []

    for i in range(num_trials):
        print(i)
        win, turns = run_game(agent1, agent2)
        if win:
            wins_agent1 += 1
        else:
            wins_agent2 += 1
        if show:
            agent1_total_wins.append(wins_agent1)
            total_games.append(wins_agent1 + wins_agent2)
            total_game_length.append(turns)
    if show:
        # print(total_games)
        # print(agent1_total_wins)
        plt.plot(total_games, agent1_total_wins)
        plt.xlabel("Total Games")
        plt.ylabel("Agent 1 Wins")
        plt.title("Agent 1 Wins Over Time")
        plt.savefig(
            "src/game/compare_plots/good_multi-vs-linear_multi_wins_over_time.png")
        plt.show()

        plt.plot(total_games, total_game_length)
        plt.xlabel("Total Games")
        plt.ylabel("Game Length")
        plt.title("Game Length Over Time")
        plt.savefig(
            "src/game/compare_plots/good_multi-vs-linear_game_length_over_time.png")
        plt.show()

    return wins_agent1, wins_agent2


agent1 = YanivAgent("multi-vs-multi", STATE_SIZE, MDQN, MDQN, MDQN)
agent1.load_models(1)
agent1.epsilon = 0.01

agent2 = YanivAgent("linear-vs-linear", STATE_SIZE, MDQN, MDQN, MDQN)
agent2.load_models(2)
agent2.epsilon = 0.01

compare(agent1, agent2, num_trials=20, show=True)
