import torch
import torch.nn as nn
import numpy as np
from game.globals import *


class MultipleLayers(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.path = "multilayer"
        self.linear1 = nn.Linear(input_size, input_size * 2)
        self.bn1 = nn.LayerNorm(input_size * 2)
        self.linear2 = nn.Linear(input_size * 2, input_size * 4)
        self.bn2 = nn.LayerNorm(input_size * 4)
        self.linear3 = nn.Linear(input_size * 4, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.out = nn.Linear(output_size, output_size)
        self.path = "linear"

    def forward(self, x):
        x = self.layer(x)
        x = self.relu(x)
        x = self.out(x)
        return x


class DQN1(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.path = "DQN"
        self.fc1 = nn.Linear(input_size, input_size * 2)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(4)
        self.relu = nn.ReLU()
        self.out = nn.Linear(4 * input_size * 2, output_size)

    def forward(self, x):
        x = self.fc1(x)                     # (batch_size, input_size * 2)
        x = x.unsqueeze(-1)                  # (batch_size, 1, input_size * 2)
        x = self.conv1(x)                   # (batch_size, 4, input_size * 2)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)          # (batch_size, 4 * input_size * 2)
        x = self.out(x)
        return x
