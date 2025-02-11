from __future__ import annotations
from game.model_a import ModelA
from game.model_b import ModelB
from game.model_c import ModelC
from game.helpers import *
from game.state import GameState
import numpy as np
import copy
import sys


class CombinedModel:
    def __init__(self):
        self.m1 = ModelA()
        self.m2 = ModelB()
        self.m3 = ModelC()

    # def features(self, state : GameState):
    #     our_hand_value, _, other_player_num_cards, turn, last_five, completes_set, completes_straight = state.get_features()
    #     last = np.array([i for i in last_five])
    #     completes_s = np.array([c for c in completes_set])
    #     completes_str = np.array([c for c in completes_straight])
    #     return np.concatenate(np.array([our_hand_value, turn, other_player_num_cards]), np.concatenate(np.concatenate(completes_s, completes_str), last))

    def features(self, state: GameState, hand : np.ndarray, other_hand : np.ndarray):
        our_hand_value, _, other_player_num_cards, turn, last_five, completes_set, completes_straight = state.get_features(hand, other_hand)
        last = np.array(last_five)
        completes_s = np.array(completes_set)
        completes_str = np.array(completes_straight)
        basic_features = np.array([our_hand_value, turn, other_player_num_cards])
        return np.concatenate([basic_features, completes_s, completes_str, last])

    def game_iteration(self, state: GameState):
        features = self.features(state)
        move1, move2, move3 = self.choose_actions(features, state)

    def choose_actions(self, hand : np.ndarray, features: np.ndarray[int], state: GameState) -> tuple:
        first_moves = ["Continue"] + \
            (["Yaniv"] if state.can_yaniv(hand) else [])
        move1 = self.m1.choose_action(first_moves, features)

        second_moves = state.valid_move_values(hand)
        move_value = self.m2.choose_action(
            [i[0] for i in second_moves], features)
        for i, move in second_moves:
            if i == move_value:
                move2 = (i, move)

        third_moves = state.valid_third_moves()
        move3 = self.m3.choose_action(third_moves, features)

        return move1, move2, move3, first_moves, second_moves, third_moves

    def play_step(self, hand : np.ndarry, other_hands : list[np.ndarray], state: GameState, actions: tuple[str, list[int], str]) -> tuple[GameState, int, bool, bool]:
        move1, move2, move3 = actions
        done = False
        win = False
        if move1 == "Yaniv":
            done = True
            win = state.yaniv(hand, other_hands)
            reward = state.get_hand_value(
                hand) * 5 if win else -5
        draw = -1 if move3 == "Deck" else 0 if move3 == "Draw 0" else 1
        initial_hand = state.get_hand_value(hand)
        state.play(hand, move2[1], draw)
        reward = initial_hand - state.get_hand_value(hand)

        return state, reward, done, win

    def update_weights(self, features: np.ndarray[int], actions: tuple[str, int, str],
                       possible_actions: tuple[list], reward: int) -> None:
        move1, move2, move3 = actions
        poss1, poss2, poss3 = possible_actions
        self.m1.update_weights(move1, poss1, reward, features)
        self.m2.update_weights(move2[0], [p[0]
                               for p in poss2], reward, features)
        self.m3.update_weights(move3, poss3, reward, features)

def sim_step(sim : CombinedModel, state : GameState, hand : np.ndarray, other_hand : np.ndarray):
    features = sim.features(state, hand, other_hand)
    m1, m2, m3, p1, p2, p3 = sim.choose_actions(hand, features, state)
    action = (m1, m2, m3)
    possible = (p1, p2, p3)
    next_state, reward, done, win = sim.play_step(hand, [other_hand], state, action)
    sim.update_weights(features, action, possible, reward)
    state = next_state
    return state, done, win

def main(args):
    
    won_games = 0
    num_episodes = int(args[1])
    sim = CombinedModel()
    sim2 = CombinedModel()
    for episode in range(num_episodes):
        # print(f"Running episode {episode}")
        state = GameState()
        done = False
        player1_hand = state.player_1_hand
        player2_hand = state.player_2_hand
        while not done:
            state, done, win = sim_step(sim, state, player1_hand, player2_hand)
            if done and win:
                break
            state, done, win = sim_step(sim2, state, player2_hand, player1_hand)
            if done and win:
                win = False
                break
        won_games += 1 if win else 0
        print(won_games / (episode + 1))

    print("---------")

    won_games = 0
    sim3 = CombinedModel()
    for episode in range(num_episodes):
        state = GameState()
        done = False
        player1_hand = state.player_1_hand
        player2_hand = state.player_2_hand
        while not done:
            state, done, win = sim_step(sim, state, player1_hand, player2_hand)
            if done and win:
                break
            state, done, win = sim_step(sim3, state, player2_hand, player1_hand)
            if done and win:
                win = False
                break
        won_games += 1 if win else 0
        print(won_games / (episode + 1))

        
if __name__ == "__main__":
    main(sys.argv)
