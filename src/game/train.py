# train_yaniv_agent.py

import torch
from game.agent import YanivAgent
from game.state import GameState
from game.globals import *
import numpy as np
import os
import random


def play_agent(game: GameState, p1: YanivAgent, hand: np.ndarray, other: np.ndarray):
    s1 = p1.state_to_tensor(game, hand, other)
    a1 = p1.choose_action_phase1(game, hand, other)
    if a1 == 1:
        won = game.yaniv(hand, [other])
        return True, won

    s2 = p1.state_to_tensor(game, hand, other)
    a2 = p1.choose_action_phase2(game, hand, other)
    discard_indices = POSSIBLE_MOVES[int(a2)]

    if len(discard_indices) == 0:
        # must call Yaniv
        won = game.yaniv(hand, [other])
        return True, won

    h_c = hand.copy()
    game.play(hand, list(discard_indices))
    r2_play = -game.get_hand_value(hand)
    s2_next = p1.state_to_tensor(game, hand, other)

    # Phase 3: Draw card
    s3 = p1.state_to_tensor(game, hand, other)
    a3 = p1.choose_action_phase3(game, hand, other)
    hand[game.draw(h_c, list(discard_indices), a3)] += 1

    return False, False


def play_turn(game: GameState, p1: YanivAgent, p2: YanivAgent, hand: np.ndarray, other: np.ndarray, debug=False):
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
        print(game.hand_to_cards(hand), discard_indices, [
              POSSIBLE_MOVES[move] for move in game.valid_moves(hand)])

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


# -> tuple[int, int, YanivAgent, YanivAgent]:
def train_models(NUM_EPISODES=1000):

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    agent1 = YanivAgent(state_size=STATE_SIZE)
    agent2 = YanivAgent(state_size=STATE_SIZE)

    for i, agent in enumerate([agent1, agent2], start=1):
        try:
            agent.model_phase1.load_state_dict(
                torch.load(f"{MODEL_DIR}/agent{i}_phase1.pt"))
            agent.model_phase2.load_state_dict(
                torch.load(f"{MODEL_DIR}/agent{i}_phase2.pt"))
            agent.model_phase3.load_state_dict(
                torch.load(f"{MODEL_DIR}/agent{i}_phase3.pt"))
            print(f"Loaded model checkpoints for Agent {i}.")
        except:
            print(
                f"No checkpoints found for Agent {i}. Training from scratch.")

    # Updated training loop for self-play
    w1 = 0
    w2 = 0
    for episode in range(1, NUM_EPISODES + 1):
        game = GameState()
        done = False

        p1, p2 = agent1, agent2

        print(episode)

        while not done:
            r1a, r2a, done, won = play_turn(
                game, p1, p2, game.player_1_hand, game.player_2_hand)
            if done:
                w1 += 1
                break
            r1b, r2b, done, won = play_turn(
                game, p2, p1, game.player_2_hand, game.player_1_hand)
            if done:
                w2 += 1
                break
        agent1.train()
        agent2.train()

        if episode % SAVE_EVERY == 0:
            # Flip agent roles
            agent1, agent2 = agent2, agent1
            w1, w2 = w2, w1
            for i, agent in enumerate([agent1, agent2], start=1):
                torch.save(agent.model_phase1.state_dict(),
                           f"{MODEL_DIR}/agent{i}_phase1.pt")
                torch.save(agent.model_phase2.state_dict(),
                           f"{MODEL_DIR}/agent{i}_phase2.pt")
                torch.save(agent.model_phase3.state_dict(),
                           f"{MODEL_DIR}/agent{i}_phase3.pt")
            print(f"Checkpoint saved at episode {episode}")
            print(
                f"Agent 1 win % is {w1 / episode} to Agent 2's {w2 / episode}")

    return w1, w2, agent1, agent2


if __name__ == "__main__":
    w1, w2, _, _ = train_models(1000)
    print(f"Agent 1 won {w1} to 2's {w2}")
