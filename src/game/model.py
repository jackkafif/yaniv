import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from game.state import GameState
from game.globals import *
from game.nets import *

class MDQN:
    def __init__(self, state_size = STATE_SIZE, output_size = 2):
        self.state_size = state_size
        self.model = MultipleLayers(state_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.episode_memory = []
        self.gamma = 0.95
        self.policy_net = self.model
        self.intermediate_loss_weight = 1.5

    def forward(self, x):
        return self.model(x)

    def add_episode(self, state_tensor, action_idx, intermediate_loss=0):
        self.episode_memory.append((state_tensor, action_idx, intermediate_loss))

    def replay_game(self, final_reward):
        # Monte Carlo update with discounting:
        for t, (state_tensor, action_idx, intermediate_loss) in enumerate(self.episode_memory):
            discounted_reward = final_reward * (self.gamma ** (len(self.episode_memory) - t - 1))
            discounted_reward += intermediate_loss * self.intermediate_loss_weight

            predicted_q = self.policy_net(state_tensor)[action_idx]
            target_q = torch.tensor(discounted_reward, dtype=torch.float32)

            loss = self.criterion(predicted_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.episode_memory = []

class M:
    def __init__(self, state_size = STATE_SIZE, output_size = 2):
        self.state_size = state_size
        self.model = SimpleNN(state_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.episode_memory = []
        self.gamma = 0.95
        self.policy_net = self.model
        self.intermediate_loss_weight = 0.1

    def forward(self, x):
        return self.model(x)

    def add_episode(self, state_tensor, action_idx, intermediate_loss=0):
        self.episode_memory.append((state_tensor, action_idx, intermediate_loss))

    def replay_game(self, final_reward):
        # Monte Carlo update with discounting:
        for t, (state_tensor, action_idx, intermediate_loss) in enumerate(self.episode_memory):
            discounted_reward = final_reward * (self.gamma ** (len(self.episode_memory) - t - 1))
            discounted_reward += intermediate_loss * self.intermediate_loss_weight

            predicted_q = self.policy_net(state_tensor)[action_idx]
            target_q = torch.tensor(discounted_reward, dtype=torch.float32)

            loss = self.criterion(predicted_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.episode_memory = []

linear_vs_multi = {
    "dir" : "linear-vs-multi",
    "model1" : M,
    "model2" : MDQN,
}

linear_vs_linear = {
    "dir" : "linear-vs-linear",
    "model1" : M,
    "model2" : M,
}

multi_vs_multi = {
    "dir" : "multi-vs-multi",
    "model1" : MDQN,
    "model2" : MDQN,
}

model_options = {
    "linear-vs-multi": linear_vs_multi,
    "linear-vs-linear": linear_vs_linear,
    "multi-vs-multi": multi_vs_multi
}