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
            nn.Linear(512, N_MOVES + 3 + 2)  # Assuming 300 different action outputs
        )
        self.n_moves = N_MOVES
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = gamma  # Discount factor for future rewards
        self.memory = deque(maxlen=memory_size)  # Experience replay memory
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        return self.network(x)
    
    def decompose_action(self, action : np.ndarray, gameState : GameState, hand : np.ndarray):

        valid_moves_mask = gameState.valid_move_indices(hand)
        valid_moves_mask[valid_moves_mask <= 0] = 0

        valid_draws = gameState.valid_draws()
        valid_draws_mask = np.zeros(3)
        for i in valid_draws:
            valid_draws_mask[i + 1] = 1
        valid_draws_mask[valid_draws_mask < 1] = 0

        can_yaniv = gameState.can_yaniv(hand)
        valid_yaniv_mask = np.array([1, 1]) if can_yaniv else np.array([1, 0])

        play_action, draw_action, yaniv_decision = action

        cards_to_play = POSSIBLE_MOVES[int(play_action)]

        return cards_to_play, draw_action, yaniv_decision
    
    def decide_action(self, state : np.ndarray, gameState : GameState, hand : np.ndarray, epsilon):
        with torch.no_grad():  # Ensure no gradients are computed for this operation
            output_vector = self.forward(state)

        valid_moves_mask = gameState.valid_move_indices(hand)
        valid_moves_mask[valid_moves_mask <= 0] = 0

        valid_draws = gameState.valid_draws()
        valid_draws_mask = np.zeros(3)
        for i in valid_draws:
            valid_draws_mask[i] = 1
        valid_draws_mask[valid_draws_mask < 1] = 0

        can_yaniv = gameState.can_yaniv(hand)
        valid_yaniv_mask = np.array([1, 1]) if can_yaniv else np.array([1, 0])
        
        # Splitting the output for different decisions
        play_decisions = output_vector[:self.n_moves] * valid_moves_mask
        draw_decisions = output_vector[self.n_moves:self.n_moves + 3] * valid_draws_mask
        yaniv_decision = output_vector[self.n_moves + 3:] * valid_yaniv_mask
        
        valid_moves = np.where(valid_moves_mask == 1)[0]
        # Decide on play move
        if random.random() < epsilon:
            play_action = random.choice(valid_moves)  # Randomly choose a play
            draw_action = random.choice(valid_draws)
            yaniv_decision = 0 if not can_yaniv else np.random.choice([0, 1])
        else:
            play_action = torch.argmax(play_decisions).item()  # Choose the best play based on network output
            draw_action = torch.argmax(draw_decisions).item()
            yaniv_decision = torch.argmax(yaniv_decision).item()
        
        return (play_action, draw_action, yaniv_decision), output_vector
            
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_experiences(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, action_vector, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)

        # Ensure actions are properly formatted
        actions = torch.stack([a.clone().detach().long() if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.long) for a in action_vector])

        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Fix Q-value extraction
        current_q_values = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # Ensure shape (32,)

        next_q_values = self.network(next_states).max(1)[0]
        next_q_values[dones.bool()] = 0.0  # Set next state Q-value to 0 for terminal states

        expected_q_values = rewards + self.gamma * next_q_values  # Shape (32,)

        # Ensure both tensors have the same shape
        assert current_q_values.shape == expected_q_values.shape, f"Shape mismatch: {current_q_values.shape} vs {expected_q_values.shape}"

        loss = self.loss_function(current_q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def features(self, state: GameState, hand: np.ndarray, other_hand: np.ndarray):
        basic_features = state.get_features(hand, other_hand)
        # Ensure all arrays are properly flattened and concatenated to match the network input size
        if basic_features.ndim > 1:
            basic_features = basic_features.flatten()
        
        if basic_features.shape[0] != 106:
            raise ValueError(f"Feature vector size mismatch: Expected 111, got {basic_features.shape[0]}")
        
        return torch.FloatTensor(basic_features)
    
def play_step(hand : np.ndarry, other_hands : list[np.ndarray], state: GameState, actions: tuple[list[int], int, int]) -> tuple[GameState, int, bool, bool]:
    cards_to_play, draw_action, yaniv = actions
    done = False
    win = False
    if yaniv:
        done = True
        win = state.yaniv(hand, other_hands)
        reward1 = 10 if win else -50
    else:
        reward1 = 1
    draw = draw_action
    initial_hand = state.get_hand_value(hand)
    cards_to_play = POSSIBLE_MOVES[int(cards_to_play)]
    state.play(hand, cards_to_play, draw)
    delta_hand = initial_hand - state.get_hand_value(hand)
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
                actions, output_vector = ag.decide_action(state_vector, current_state, hand, epsilon)

                next_state, reward, done, _ = play_step(hand, other, game_env, actions)
                next_state_vector = ag.features(next_state, other[0], [hand])


                ag.store_experience(state_vector, output_vector, reward, next_state_vector, done)
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
