import torch
from game.train import train_models
import numpy as np
import os
from game.globals import MODEL_DIR
import matplotlib.pyplot as plt
import sys

def train_and_visualize(NUM_EPISODES=1000):

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    os.makedirs("vizs", exist_ok=True)

    episodes = np.arange(1, NUM_EPISODES + 1)

    (w1, win_rates_1), (w2, win_rates_2), _, _ = train_models(NUM_EPISODES)

    plt.figure()
    plt.plot(episodes, win_rates_1, label='Agent 1 Win %')
    plt.plot(episodes, win_rates_2, label='Agent 2 Win %')
    plt.xlabel('Episode')
    plt.ylabel('Win Percentage')
    plt.title('Win Percentage Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig("vizs/win_percentage.png")
    plt.show()

if __name__ == "__main__":
    train_and_visualize(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
