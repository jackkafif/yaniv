import torch
from game.agent import YanivAgent
from game.state import GameState
from game.globals import *
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from game.nets import *
import sys
import game.train as trainer

def train_and_visualize(NUM_EPISODES=1000):
    agent1 = YanivAgent(state_size=STATE_SIZE, Net1=MultipleLayers, Net2=MultipleLayers, Net3=MultipleLayers)
    agent2 = YanivAgent(state_size=STATE_SIZE, Net1=SimpleNN, Net2=SimpleNN, Net3=SimpleNN)

    trainer.load_model(agent1, 1)
    trainer.load_model(agent2, 2)

    w1 = 0
    w2 = 0

    win_rates_1 = []
    win_rates_2 = []
    avg_rewards_1 = []
    avg_rewards_2 = []

    for episode in range(1, NUM_EPISODES + 1):
        game = GameState()
        done = False

        p1, p2 = agent1, agent2
        ep_r1_total = 0
        ep_r2_total = 0

        print(f"Episode {episode}", '\r', end='')

        while not done:
            r1a, r2a, done, won = trainer.play_turn(game, p1, p2, game.player_1_hand, game.player_2_hand)
            ep_r1_total += r1a
            ep_r2_total += r2a

            if done:
                w1 += 1
                break

            r1b, r2b, done, won = trainer.play_turn(game, p2, p1, game.player_2_hand, game.player_1_hand)
            ep_r1_total += r2b  # rewards flip since agents flip
            ep_r2_total += r1b

            if done:
                w2 += 1
                break

        agent1.train()
        agent2.train()

        # Track win rate and reward over time
        win_rates_1.append(w1 / episode)
        win_rates_2.append(w2 / episode)
        avg_rewards_1.append(ep_r1_total)
        avg_rewards_2.append(ep_r2_total)

        if episode % SAVE_EVERY == 0:
            trainer.save_model(agent1, 1)
            trainer.save_model(agent2, 2)

    # Plot results

    os.makedirs("vizs", exist_ok=True)

    episodes = np.arange(1, NUM_EPISODES + 1)

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

    plt.figure()
    plt.plot(episodes, avg_rewards_1, label='Agent 1 Avg Reward')
    plt.plot(episodes, avg_rewards_2, label='Agent 2 Avg Reward')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Episode Reward Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig("vizs/episode_rewards.png")
    plt.show()

if __name__ == "__main__":
    train_and_visualize(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
