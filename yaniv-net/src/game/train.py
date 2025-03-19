# train.py
import numpy as np
import random
from collections import deque
import torch
from game_state import GameState
from agent import Agent

STATE_SIZE = 52 + 52 + 1  # Player hand (52) + Top cards (52) + Opponent hand value (1) = 105
ACTION_SIZE = 20        # Placeholder for the number of discrete actions; adjust as needed.
NUM_EPISODES = 1000
BATCH_SIZE = 32

agent = Agent(STATE_SIZE, ACTION_SIZE)
replay_buffer = deque(maxlen=10000)


def sample_batch():
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    return np.array(states), list(actions), list(rewards), np.array(next_states), list(dones)


for episode in range(NUM_EPISODES):
    game = GameState()
    done = False
    total_reward = 0


    # Instead of using the agent's action selection, choose a random valid move:
    valid_moves = game.valid_move_values(game.player_2_hand)
    if not valid_moves:
        # If no valid moves (rare), draw from the deck
        game.play(game.player_2_hand, [], -1)
    else:
        # Pick a random move
        chosen_move_value, chosen_indices = random.choice(valid_moves)
        # Play the chosen move (always draw from deck = -1 here)
        game.play(game.player_2_hand, chosen_indices, -1)
        
    # In a full game simulation, you would loop until the game ends.
    # For demonstration, we execute one move per episode.
    state = game.get_state_vector()
    valid_actions = agent.valid_actions(game)
    action = agent.choose_action(state, valid_actions)
    discard_indices, draw_idx = agent.action_to_move(action)
    
    game.play(game.player_2_hand, discard_indices, draw_idx)
    
    new_state = game.get_state_vector()
    reward = -game.get_hand_value(game.player_2_hand)
    if game.can_yaniv(game.player_2_hand):
        if game.yaniv(game.player_2_hand, [game.player_1_hand]):
            reward += 100
        done = True

    total_reward += reward
    replay_buffer.append((state, action, reward, new_state, done))

    if len(replay_buffer) >= BATCH_SIZE:
        batch = sample_batch()
        agent.train_step(batch)
    
    print(f"Episode {episode}, Total Reward: {total_reward}")

# Save the trained model
torch.save(agent.model.state_dict(), "dqn_model.pth")
