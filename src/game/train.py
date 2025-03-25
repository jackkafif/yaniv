# train_yaniv_agent.py

import torch
from agent import YanivAgent
from state import GameState
from helpers import *
import numpy as np

STATE_SIZE = 106  # 52 (hand) + 52 (top cards) + 1 (opp hand size) + 1 (turn)
NUM_EPISODES = 10000
SAVE_EVERY = 500
MODEL_DIR = "checkpoints"
POSSIBLE_MOVES = generate_combinations(5)

import os
import random
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

agent1 = YanivAgent(state_size=STATE_SIZE)
agent2 = YanivAgent(state_size=STATE_SIZE)

def play_turn(p1 : YanivAgent, p2 : YanivAgent, hand : np.ndarray, other : np.ndarray, debug=False):
    # Phase 1: Decide to call Yaniv or play (Player 1)
    reward1, reward2 = 0, 0
    s1 = p1.state_to_tensor(game, hand, other)
    a1 = p1.choose_action_phase1(game, hand, other)
    if a1 == 1:
        won = game.yaniv(hand, [other])
        r1 = 100 if won else -50
        r2 = -r1
        s1_next = p1.state_to_tensor(game, hand, other)
        p1.store_experience(1, s1, a1, r1, s1_next, True)
        p2.store_experience(1, s1, 0, r2, s1_next, True)
        reward1 += r1
        reward2 += r2
        return reward1, reward2, True, won
    else:
        p1.store_experience(1, s1, a1, 0, s1, False)

    # Phase 2: Play cards
    s2 = p1.state_to_tensor(game, hand, other)
    a2 = p1.choose_action_phase2(game, hand, other)
    discard_indices = POSSIBLE_MOVES[int(a2)]

    if len(discard_indices) == 0:
        # must call Yaniv
        won = game.yaniv(hand, [other])
        r1 = 100 if won else -50
        r2 = -r1
        s1_next = p1.state_to_tensor(game, hand, other)
        p1.store_experience(1, s1, a1, r1, s1_next, True)
        p2.store_experience(1, s1, 0, r2, s1_next, True)
        reward1 += r1
        reward2 += r2
        return reward1, reward2, True, won

    if debug:
        print(game.hand_to_cards(hand), discard_indices)

    h_c = hand.copy()
    game.play(hand, list(discard_indices))
    r2_play = -game.get_hand_value(hand)
    s2_next = p1.state_to_tensor(game, hand, other)
    p1.store_experience(2, s2, a2, r2_play, s2_next, False)
    reward1 += r2_play

    # Phase 3: Draw card
    s3 = p1.state_to_tensor(game, hand, other)
    a3 = p1.choose_action_phase3(game, hand, other)
    hand[game.draw(h_c, list(discard_indices), a3)] += 1

    if debug:
        print(game.hand_to_cards(hand))

    s3_next = p1.state_to_tensor(game, hand, other)
    p1.store_experience(3, s3, a3, 0, s3_next, False)
    return reward1, reward2, False, False

# Updated training loop for self-play
for episode in range(1, NUM_EPISODES + 1):
    game = GameState()
    done = False
    reward1 = 0
    reward2 = 0

    p1, p2 = agent1, agent2

    print(f"Episode {episode}")
    
    while not done:
        r1a, r2a, done, won = play_turn(p1, p2, game.player_1_hand, game.player_2_hand, True)
        if done:
            print("Player 1 won")
            break
        r1b, r2b, done, won = play_turn(p2, p1, game.player_2_hand, game.player_1_hand)
        if done:
            print("Player 2 won")
            break
        agent1.train()
        agent2.train()

    print(game.hand_to_cards(game.player_1_hand), game.hand_to_cards(game.player_2_hand))

    print(f"Episode {episode}, Agent1 Reward: {reward1}, Agent2 Reward: {reward2}")
