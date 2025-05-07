# train_yaniv_agent.py

import torch
from game.agent import YanivAgent, run_game
from game.state import GameState
from game.globals import *
import numpy as np
import os
import random
from game.nets import *
import sys

def train_models(NUM_EPISODES=1000):

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    agent1 = YanivAgent(state_size=STATE_SIZE)
    agent2 = YanivAgent(state_size=STATE_SIZE)

    for idx, agent in enumerate([agent1, agent2], start=1):
        agent.load_models(idx)

    for i in range(12):

        if NUM_EPISODES == 0:
            print("No training episodes specified. Exiting.")
            return agent1, agent2

        # Updated training loop for self-play
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
            win_rates_1.append(w1 / episode)
            win_rates_2.append(w2 / episode)

            if episode % SAVE_EVERY == 0:
                print("Saving models...")
                agent1.save_models(1)
                agent2.save_models(2)

            if episode % 100 == 0:
                print(f"Episode {episode}: Agent 1 Win Rate: {win_rates_1[-1]:.2f}, Agent 2 Win Rate: {win_rates_2[-1]:.2f}")

        print(f"Final Win Rates: Agent 1: {w1}, Agent 2: {w2}")
        print(f"Agent 1 Win Rate: {w1 / NUM_EPISODES:.2f}, Agent 2 Win Rate: {w2 / NUM_EPISODES:.2f}")
        if w1 < w2:
            agent1 = agent2
        else:
            agent1 = agent1
        agent2 = YanivAgent(state_size=STATE_SIZE)
    
    agent1.save_models(1)
    # agent2.save_models(2)
    return (w1, win_rates_1), (w2, win_rates_2), agent1, agent2

if __name__ == "__main__":
    (w1, _), (w2, _), a1, a2 = train_models(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
    print(f"Agent 1 won {w1} to 2's {w2}")
