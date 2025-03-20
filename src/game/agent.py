# agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game_state import GameState


class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, state_size: int, action_size: int, epsilon: float = 1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.model = DQN(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
    
    def choose_action(self, state: np.ndarray, valid_actions: list[int]) -> int:
        """
        Epsilon-greedy policy: with probability epsilon, choose a random valid action.
        Otherwise, use the network to pick the action with the highest Q-value among valid actions.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor).detach().numpy().flatten()
        # Filter valid actions: set invalid ones to a very low value.
        q_values_filtered = np.full_like(q_values, -1e9)
        for a in valid_actions:
            q_values_filtered[a] = q_values[a]
        return int(np.argmax(q_values_filtered))

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch
        states_tensor = torch.FloatTensor(states)
        next_states_tensor = torch.FloatTensor(next_states)
        rewards_tensor = torch.FloatTensor(rewards)
        dones_tensor = torch.FloatTensor(dones)

        q_values = self.model(states_tensor)
        next_q_values = self.model(next_states_tensor).detach()
        target_q = rewards_tensor + self.gamma * (1 - dones_tensor) * torch.max(next_q_values, dim=1)[0]
        q_value_taken = q_values.gather(1, torch.LongTensor(actions).unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_value_taken, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def action_to_move(self, action: int) -> tuple[list[int], int]:
        """
        Convert an action index into a game move.
        This is a placeholder mapping; design a proper mapping for your action space.
        For demonstration: if the action is even, discard one card at (action // 2) and draw from deck (-1);
        if odd, discard one card at (action // 2) and draw from the top (0).
        """
        if action % 2 == 0:
            return ([action // 2], -1)
        else:
            return ([action // 2], 0)
    
    def valid_actions(self, game_state: GameState) -> list[int]:
        """
        Return a list of valid action indices given the current game state.
        For demonstration purposes, we assume all actions from 0 to action_size - 1 are valid.
        """
        return list(range(self.action_size))
