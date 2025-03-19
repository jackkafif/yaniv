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
            nn.Linear(106, 512),  # Assuming input state size of 111 for illustration
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 300)  # Assuming 300 different action outputs
        )
        self.n_moves = N_MOVES
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = gamma  # Discount factor for future rewards
        self.memory = deque(maxlen=memory_size)  # Experience replay memory
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        return self.network(x)
    
    def decide_action(self, state : np.ndarray, gameState : GameState, hand : np.ndarray, epsilon): # State should be changed to the feats vector
        with torch.no_grad():  # Ensure no gradients are computed for this operation
            output_vector = self.forward(state)

        valid_moves = gameState.valid_moves(hand)
        # print(gameState.hand_to_cards(hand))
        # print(valid_moves)
        
        # Splitting the output for different decisions
        play_decisions = output_vector[:self.n_moves] * valid_moves
        draw_decisions = output_vector[self.n_moves:self.n_moves + 2]
        yaniv_decision = output_vector[self.n_moves + 2]
        # print(torch.argmax(play_decisions).item())
        
        # Decide on play move
        if random.random() < epsilon:
            play_action = random.choice(valid_moves)  # Randomly choose a play
        else:
            play_action = torch.argmax(play_decisions).item()  # Choose the best play based on network output
        
        # Decide on draw move
        if random.random() < epsilon:
            draw_action = random.choice([0, 1])
        else:
            draw_action = torch.argmax(draw_decisions).item()
        
        # Decide whether to call Yaniv
        call_yaniv = random.random() < epsilon or torch.sigmoid(yaniv_decision) > 0.5

        # print(play_action, draw_action, call_yaniv)
        
        return play_action, draw_action, call_yaniv
            
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_experiences(self, batch_size):
        if len(self.memory) < batch_size:
            return  # Not enough experiences to sample a batch
        print("Replaying...")        
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
    
    def features(self, state: GameState, hand: np.ndarray, other_hand: np.ndarray):
        basic_features = state.get_features(hand, other_hand)
        # Ensure all arrays are properly flattened and concatenated to match the network input size
        if top_cards.ndim > 1:
            top_cards = top_cards.flatten()
        
        if basic_features.shape[0] != 106:
            raise ValueError(f"Feature vector size mismatch: Expected 111, got {basic_features.shape[0]}")
        
        return torch.FloatTensor(basic_features)
    
def play_step(hand : np.ndarry, other_hands : list[np.ndarray], state: GameState, actions: tuple[str, list[int], str]) -> tuple[GameState, int, bool, bool]:
    # print(actions)
    selected_play, draw_action, call_yaniv = actions
    done = False
    win = False
    if call_yaniv == "Yaniv":
        done = True
        win = state.yaniv(hand, other_hands)
        reward1 = 10 if win else -50
    else:
        reward1 = 1
    # Deck == 0 Discard == 1
    draw = draw_action
    initial_hand = state.get_hand_value(hand)
    state.play(hand, COMBINATIONS[5][selected_play], draw)
    delta_hand = initial_hand - state.get_hand_value(hand)
    # print(delta_hand)
    if delta_hand < 0:
        reward3 = 1
    else:
        reward3 = -2

    return state, reward1 + delta_hand + reward3, done, win

def train_agent(agent : CombinedModel, agent2 : CombinedModel, game_env : GameState, num_episodes : int, batch_size : int = 32, max_steps_per_episode : int = 1000):
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995

    for episode in range(num_episodes):
        current_state = game_env.reset()
        epsilon = epsilon_start  # Reset epsilon at the start of each episode

        ags = [agent, agent2]
        for step in range(max_steps_per_episode):
            for i in range(2):
                ag = ags[i]
                if i == 1:
                    hand = game_env.player_1_hand
                    other = [game_env.player_2_hand]
                else:
                    hand = game_env.player_2_hand
                    other = [game_env.player_1_hand]

                state_vector = ag.features(current_state, hand, other)
                actions = ag.decide_action(state_vector, current_state, hand, epsilon)
                # print(actions)
                next_state, reward, done, _ = play_step(hand, other, game_env, actions)
                next_state_vector = ag.features(next_state, other[0], [hand])

                print(state_vector, actions, reward, next_state_vector, done)
                ag.store_experience(state_vector, actions, reward, next_state_vector, done)
                ag.replay_experiences(batch_size)

                current_state = next_state
                epsilon = max(epsilon_end, epsilon_decay * epsilon)

                if done:
                    break
            if done:
                break

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Training ongoing.")


def main(args):

    game_env = GameState()
    agent = CombinedModel()
    agent2 = CombinedModel()

    # play_game(agent, game_env)
    train_agent(agent, agent2, game_env, num_episodes=1000)

if __name__ == "__main__":
    main(sys.argv)
