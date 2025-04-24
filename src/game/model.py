import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from game.state import GameState
from game.globals import *
from game.nets import *

class M:
    def __init__(self, state_size = STATE_SIZE, output_size = 2):
        self.state_size = state_size
        self.model = SimpleNN(state_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.episode_memory = []
        self.gamma = 0.95
        self.policy_net = self.model

    def forward(self, x):
        return self.model(x)

    def add_episode(self, state_tensor, action_idx):
        self.episode_memory.append((state_tensor, action_idx))

    def replay_game(self, final_reward):
        # Monte Carlo update with discounting:
        for t, (state_tensor, action_idx) in enumerate(self.episode_memory):
            discounted_reward = final_reward * (self.gamma ** (len(self.episode_memory) - t - 1))

            predicted_q = self.policy_net(state_tensor)[action_idx]
            target_q = torch.tensor(discounted_reward, dtype=torch.float32)

            loss = self.criterion(predicted_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.episode_memory = []
