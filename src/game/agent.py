import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from game.state import GameState
from game.globals import *
from game.nets import DQN1

class YanivAgent:
    def __init__(self, state_size = STATE_SIZE):
        self.state_size = state_size

        self.model_phase1 = DQN1(state_size, PHASE1_ACTION_SIZE)
        self.model_phase2 = DQN1(state_size, PHASE2_ACTION_SIZE)
        self.model_phase3 = DQN1(state_size, PHASE3_ACTION_SIZE)

        self.optimizer_phase1 = optim.Adam(self.model_phase1.parameters(), lr=1e-3)
        self.optimizer_phase2 = optim.Adam(self.model_phase2.parameters(), lr=1e-3)
        self.optimizer_phase3 = optim.Adam(self.model_phase3.parameters(), lr=1e-3)

        self.loss_fn = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32

    def state_to_tensor(self, state : GameState, hand : np.ndarray, other : np.ndarray):
        return torch.FloatTensor(state.get_features(hand, other))

    def choose_action_phase1(self, state : GameState, hand : np.ndarray, other : np.ndarray):
        if not state.can_yaniv(hand):
            return 0
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            q_vals = self.model_phase1(self.state_to_tensor(state, hand, other).unsqueeze(0))
        return int(torch.argmax(q_vals))

    # Phase 2: Choose which cards to play
    def choose_action_phase2(self, state : GameState, hand : np.ndarray, other : np.ndarray):
        valid_moves = state.valid_move_indices(hand)
        if random.random() < self.epsilon:
            return random.choice(np.where(valid_moves > 0)[0])
        with torch.no_grad():
            q_vals = self.model_phase2(self.state_to_tensor(state, hand, other).unsqueeze(0))
            q_vals = q_vals.squeeze(0)
            q_vals[valid_moves.T <= 0] = -np.inf
        return int(torch.argmax(q_vals))

    # Phase 3: Choose where to draw from
    def choose_action_phase3(self, state : GameState, hand : np.ndarray, other : np.ndarray):
        valid_draws = state.valid_draws()
        valid_draws_mask = np.zeros(3)
        for i in valid_draws:
            valid_draws_mask[i] = 1
        if random.random() < self.epsilon:
            return random.choice(valid_draws)
        with torch.no_grad():
            q_vals = self.model_phase3(self.state_to_tensor(state, hand, other).unsqueeze(0))
            q_vals = q_vals.squeeze(0)
            q_vals[valid_draws_mask < 1] = -np.inf
        return int(torch.argmax(q_vals))
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        for phase in [1, 2, 3]:
            phase_batch = [exp for exp in batch if exp['phase'] == phase]
            if not phase_batch:
                continue
            states = torch.stack([exp['state'] for exp in phase_batch])
            actions = torch.LongTensor([exp['action'] for exp in phase_batch]).unsqueeze(1)
            rewards = torch.FloatTensor([exp['reward'] for exp in phase_batch])
            next_states = torch.stack([exp['next_state'] for exp in phase_batch])
            dones = torch.FloatTensor([exp['done'] for exp in phase_batch])

            if phase == 1:
                model = self.model_phase1
                optimizer = self.optimizer_phase1
            elif phase == 2:
                model = self.model_phase2
                optimizer = self.optimizer_phase2
            else:
                model = self.model_phase3
                optimizer = self.optimizer_phase3

            q_values = model(states).gather(1, actions).squeeze()
            next_q_values = model(next_states).max(1)[0].detach()
            target = rewards + self.gamma * (1 - dones) * next_q_values

            loss = self.loss_fn(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def store_experience(self, phase, state, action, reward, next_state, done):
        self.replay_buffer.append({
            'phase': phase,
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

def run_training_episode(agent: YanivAgent, opponent: YanivAgent):
    game = GameState()
    done = False

    # Phase 1: Call Yaniv or Play
    state_tensor = agent.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
    action1 = agent.choose_action_phase1(game, game.player_1_hand, game.player_2_hand)
    if action1 == 1:  # Call Yaniv
        reward = 100 if game.yaniv(game.player_1_hand, [game.player_2_hand]) else -50
        next_state_tensor = agent.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
        agent.store_experience(1, state_tensor, action1, reward, next_state_tensor, True)
        return reward
    else:
        reward = 0
        next_state_tensor = agent.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
        agent.store_experience(1, state_tensor, action1, reward, next_state_tensor, False)

    # Phase 2: Play cards
    state_tensor = agent.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
    action2 = agent.choose_action_phase2(game, game.player_1_hand, game.player_2_hand)
    discard_indices = POSSIBLE_MOVES[int(action2)]
    hand_copy = game.player_1_hand.copy()
    game.play(game.player_1_hand, discard_indices)
    reward = -game.get_hand_value(game.player_1_hand)  # shaping
    next_state_tensor = agent.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
    agent.store_experience(2, state_tensor, action2, reward, next_state_tensor, False)

    # Phase 3: Draw card
    state_tensor = agent.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
    action3 = agent.choose_action_phase3(game, game.player_1_hand, game.player_2_hand)
    game.draw(hand_copy, discard_indices, action3) 
    next_state_tensor = agent.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
    reward = 0
    agent.store_experience(3, state_tensor, action3, reward, next_state_tensor, False)

    # Opponent plays
    done, opponent_won = game.playOpponentTurn()
    if done and opponent_won:
        return -50
    return 0

agent = YanivAgent(12)
opp = YanivAgent(12)

# for i in range(100):
#     x = run_training_episode(agent, opp)
#     # print(x)