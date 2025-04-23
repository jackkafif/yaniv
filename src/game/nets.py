import torch
import torch.nn as nn
import numpy as np
from game.globals import *

class DQN1(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=input_size*2, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(input_size*2)
        self.relu = nn.ReLU()
        self.out = nn.Linear(input_size*2 * input_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        x = self.out(x)
        return x
