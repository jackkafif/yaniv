# train_yaniv_agent.py

import torch
from game.agent import YanivAgent, run_game
from game.model import MDQN, M, linear_vs_linear, linear_vs_multi, multi_vs_multi
from game.state import GameState
from game.globals import *
import numpy as np
import os
import random
from game.nets import *
import sys

def set_seed():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_models(model_comb, NUM_EPISODES=1000):

    set_seed()

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    dir = model_comb["dir"]
    M1 = model_comb["model1"]
    M2 = model_comb["model2"]

    agent1 = YanivAgent(dir, STATE_SIZE, M1, M1, M1)
    agent2 = YanivAgent(dir, STATE_SIZE, M2, M2, M2)
    agent1.epsilon = 1.0
    agent2.epsilon = 1.0

    for idx, agent in enumerate([agent1, agent2], start=1):
        agent.load_models(idx)

    if NUM_EPISODES == 0:
        print("No training episodes specified. Exiting.")
        return agent1, agent2

    # Updated training loop for self-play
    e = 1
    w1 = 0
    w2 = 0
    win_rates_1 = []
    win_rates_2 = []
    for episode in range(1, NUM_EPISODES + 1):
        print(f"Episode {episode}", '\r', end='')
        # Randomly select True or False
        if random.choice([True, False]):
            if run_game(agent1, agent2):
                agent1.train(1)
                agent2.train(-1)
                w1 += 1
            else:
                agent1.train(-1)
                agent2.train(1)
                w2 += 1
        else:
            if run_game(agent2, agent1):
                agent1.train(-1)
                agent2.train(1)
                w2 += 1
            else:
                agent1.train(1)
                agent2.train(-1)
                w1 += 1
        win_rates_1.append(w1 / e)
        win_rates_2.append(w2 / e)
        e += 1

        if episode % SAVE_EVERY == 0:
            print("Saving models...")
            agent1.save_models(1)
            agent2.save_models(2)
            w1 = 0
            w2 = 0
            e = 1

        if episode % 100 == 0:
            print(
                f"Episode {episode}: Agent 1 Win Rate: {win_rates_1[-1]:.2f}, Agent 2 Win Rate: {win_rates_2[-1]:.2f}")

    print(f"Final Win Rates: Agent 1: {w1}, Agent 2: {w2}")
    print(
        f"Agent 1 Win Rate: {w1 / NUM_EPISODES:.2f}, Agent 2 Win Rate: {w2 / NUM_EPISODES:.2f}")

    return (w1, win_rates_1), (w2, win_rates_2), agent1, agent2


if __name__ == "__main__":
    (w1, _), (w2, _), _, _ = train_models(multi_vs_multi,
                                          int(sys.argv[1]) if len(sys.argv) > 1 else 10000)
    print(f"Agent 1 won {w1} to 2's {w2}")
