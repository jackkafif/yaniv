from __future__ import annotations
from game.helpers import *
from game.state import GameState
import numpy as np
import copy
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

POSSIBLE_MOVES = generate_combinations(5)
N_MOVES = len(POSSIBLE_MOVES)

class CombinedModel(nn.Module):
    def __init__(self, learning_rate=0.001, gamma=0.99, memory_size=10000):
        super(CombinedModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(111, 512),  # Assuming input state size of 111 for illustration
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 300)  # Assuming 300 different action outputs
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = gamma  # Discount factor for future rewards
        self.memory = deque(maxlen=memory_size)  # Experience replay memory
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        return self.network(x)
    
    def decide_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 299)
        else:
            with torch.no_grad():
                return self.forward(state).argmax().item()
            
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_experiences(self, batch_size):
        if len(self.memory) < batch_size:
            return  # Not enough experiences to sample a batch
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Current Q values
        current_q_values = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q values
        next_q_values = self.network(next_states).max(1)[0]
        next_q_values[dones] = 0.0  # Zero out the next Q values where the episode has ended

        # Expected Q values
        expected_q_values = rewards + self.gamma * next_q_values

        # Compute loss
        loss = self.loss_function(current_q_values, expected_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def features(self, state: GameState, hand : np.ndarray, other_hand : np.ndarray):
        other_player_num_cards, turn, top_cards = state.get_features(hand, other_hand)
        basic_features = np.array([hand, top_cards, other_player_num_cards, turn])
        return torch.FloatTensor(basic_features)

    def make_decision(self, output_vector):
        # Decode the output vector to choose an action
        # This will require additional logic to map indices to specific actions
        # For example, certain ranges of the output vector might correspond to specific card plays
        action_index = torch.argmax(output_vector).item()
        return action_index  # Placeholder for action decision logic

def play_game(agent, game_env, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, batch_size=32):
    epsilon = epsilon_start
    while not game_env.is_game_over():
        current_player = game_env.current_player
        current_state = game_env.get_game_state(current_player)  # Get the current state for the agent
        action = agent.decide_action(current_state, epsilon)  # Let the agent decide the best action using epsilon-greedy policy
        next_state, reward, done = game_env.execute_action(current_player, action)  # Execute the action in the environment
        
        agent.store_experience(current_state, action, reward, next_state, done)
        
        if done:
            game_env.reset()

        agent.replay_experiences(batch_size)  # Train the agent by replaying experiences

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

def train_agent(agent : CombinedModel, game_env : GameState, num_episodes : int, batch_size : int = 32, max_steps_per_episode : int = 1000):
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995

    for episode in range(num_episodes):
        current_state = game_env.reset()
        epsilon = epsilon_start  # Reset epsilon at the start of each episode

        for step in range(max_steps_per_episode):
            action = agent.decide_action(current_state, epsilon)
            game_env.play(action)
            next_state, reward, done = game_env.execute_action(action)
            
            agent.store_experience(current_state, action, reward, next_state, done)
            agent.replay_experiences(batch_size)

            current_state = next_state
            epsilon = max(epsilon_end, epsilon_decay * epsilon)

            if done:
                break

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Training ongoing.")


def main(args):

    game_env = GameState()
    agent = CombinedModel()

    # play_game(agent, game_env)
    train_agent(agent, game_env, num_episodes=1000)

if __name__ == "__main__":
    main(sys.argv)
