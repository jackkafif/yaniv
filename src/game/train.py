import numpy as np
import random
from collections import deque
import torch
from game.state import GameState, N_MOVES
from agent import Agent

STATE_SIZE = 106  # Player hand (52) + Top cards (52) + Opponent hand value (1) + turn (1) = 106
ACTION_SIZE = N_MOVES
NUM_EPISODES = 1000
BATCH_SIZE = 32

agent = Agent(STATE_SIZE, ACTION_SIZE)
replay_buffer = deque(maxlen=10000)

def sample_batch():
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    return np.array(states), list(actions), list(rewards), np.array(next_states), list(dones)

for episode in range(NUM_EPISODES):
    game = GameState()  # Reset game state for a new episode
    done = False
    total_reward = 0

    while not done:
        print("Rounding")
        # Get the current state features
        state = game.get_features(game.player_1_hand, game.player_2_hand)
        
        # Get valid actions
        valid_actions = list(game.valid_moves(game.player_1_hand))

        if not valid_actions:
            raise EnvironmentError("THIS SHOULD NEVER HAPPEN")
        else:
            action = agent.choose_action(state, valid_actions)

        # Convert action index to a move
        move_indices = list(agent.action_to_move(action))

        # Get valid draw indices
        valid_draws = game.valid_draws()
        draw_idx = random.choice(valid_draws)  # Randomly pick a valid draw (can be improved)

        # Execute player's move
        game.play(game.player_1_hand, move_indices, draw_idx)

        # Compute reward
        reward = -game.get_hand_value(game.player_1_hand)
        if game.can_yaniv(game.player_1_hand):
            if game.yaniv(game.player_1_hand, [game.player_2_hand]):
                reward += 100
            done = True

        # Get new state after the move
        new_state = game.get_features(game.player_1_hand, game.player_2_hand)

        # Store experience in replay buffer
        replay_buffer.append((state, action, reward, new_state, done))

        # Train agent when buffer has enough samples
        if len(replay_buffer) >= BATCH_SIZE:
            batch = sample_batch()
            agent.train_step(batch)

        # Opponent plays
        if not done:
            done, opponent_won = game.playOpponentTurn()
            if opponent_won:
                total_reward -= 50  # Penalty for losing

    print(f"Episode {episode}, Total Reward: {total_reward}")

# Save trained model
torch.save(agent.model.state_dict(), "dqn_model.pth")
