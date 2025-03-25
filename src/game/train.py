# train_yaniv_agent.py

import torch
from agent import YanivAgent
from state import GameState
from helpers import *

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

# Updated training loop for self-play
for episode in range(1, NUM_EPISODES + 1):
    game = GameState()
    done = False
    reward1 = 0
    reward2 = 0

    p1, p2 = agent1, agent2

    # Phase 1: Decide to call Yaniv or play (Player 1)
    s1 = p1.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
    a1 = p1.choose_action_phase1(game, game.player_1_hand, game.player_2_hand)
    if a1 == 1:
        won = game.yaniv(game.player_1_hand, [game.player_2_hand])
        r1 = 100 if won else -50
        r2 = -r1
        s1_next = p1.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
        p1.store_experience(1, s1, a1, r1, s1_next, True)
        p2.store_experience(1, s1, 0, r2, s1_next, True)
        reward1 += r1
        reward2 += r2
        continue
    else:
        p1.store_experience(1, s1, a1, 0, s1, False)

    # Phase 2: Play cards
    s2 = p1.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
    a2 = p1.choose_action_phase2(game, game.player_1_hand, game.player_2_hand)
    discard_indices = POSSIBLE_MOVES[int(a2)]
    h_c = game.player_1_hand.copy()
    game.play(game.player_1_hand, list(discard_indices))
    r2_play = -game.get_hand_value(game.player_1_hand)
    s2_next = p1.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
    p1.store_experience(2, s2, a2, r2_play, s2_next, False)
    reward1 += r2_play

    # Phase 3: Draw card
    s3 = p1.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
    a3 = p1.choose_action_phase3(game, game.player_1_hand, game.player_2_hand)
    game.draw(h_c, list(discard_indices), a3)
    s3_next = p1.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
    p1.store_experience(3, s3, a3, 0, s3_next, False)

    # Phase 1: Decide to call Yaniv or play (Player 2)
    s2 = p2.state_to_tensor(game, game.player_2_hand, game.player_1_hand)
    a2 = p2.choose_action_phase1(game, game.player_2_hand, game.player_1_hand)
    if a2 == 1:
        won = game.yaniv(game.player_2_hand, [game.player_1_hand])
        r2 = 100 if won else -50
        r1 = -r2
        s2_next = p2.state_to_tensor(game, game.player_2_hand, game.player_1_hand)
        p2.store_experience(1, s2, a2, r2, s2_next, True)
        p1.store_experience(1, s2, 0, r1, s2_next, True)
        reward2 += r2
        reward1 += r1
        continue
    else:
        p2.store_experience(1, s2, a2, 0, s2, False)

    # Phase 2: Play cards
    s2 = p1.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
    a2 = p1.choose_action_phase2(game, game.player_1_hand, game.player_2_hand)
    discard_indices = POSSIBLE_MOVES[int(a2)]
    h_c = game.player_1_hand.copy()
    game.play(game.player_1_hand, list(discard_indices))
    r2_play = -game.get_hand_value(game.player_1_hand)
    s2_next = p1.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
    p1.store_experience(2, s2, a2, r2_play, s2_next, False)
    reward1 += r2_play

    # Phase 3: Draw card
    s3 = p1.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
    a3 = p1.choose_action_phase3(game, game.player_1_hand, game.player_2_hand)
    game.draw(h_c, list(discard_indices), a3)
    s3_next = p1.state_to_tensor(game, game.player_1_hand, game.player_2_hand)
    p1.store_experience(3, s3, a3, 0, s3_next, False)

    # Train both agents
    agent1.train()
    agent2.train()

    print(f"Episode {episode}, Agent1 Reward: {reward1}, Agent2 Reward: {reward2}")
