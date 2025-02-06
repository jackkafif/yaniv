from model_a import ModelA
from model_b import ModelB
from model_c import ModelC
from helpers import *
from state import GameState
import numpy as np
import copy

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
    
    def features(self, state: GameState):
        our_hand_value, _, other_player_num_cards, turn, last_five, completes_set, completes_straight = state.get_features()
        last = np.array(last_five)
        completes_s = np.array(completes_set)
        completes_str = np.array(completes_straight)
        basic_features = np.array([our_hand_value, turn, other_player_num_cards])
        return np.concatenate([basic_features, completes_s, completes_str, last])

    def game_iteration(self, state : GameState):
        features = self.features(state)
        move1, move2, move3 = self.choose_actions(features, state)
    
    def choose_actions(self, features : np.ndarray[int], state : GameState) -> tuple:
        first_moves = ["Continue"] + (["Yaniv"] if state.can_yaniv(state.player_1_hand) else [])
        move1 = self.m1.choose_action(first_moves, features)
        
        second_moves = state.valid_move_values(state.player_1_hand)
        move_value = self.m2.choose_action([i[0] for i in second_moves], features)
        for i, move in second_moves:
            if i == move_value:
                move2 = (i, move)

        third_moves = state.valid_third_moves()
        move3 = self.m3.choose_action(third_moves, features)

        return move1, move2, move3, first_moves, second_moves, third_moves
    
    def play_step(self, state : GameState, actions : tuple[str, list[int], str]) -> tuple[GameState, int, bool, bool]:
        move1, move2, move3 = actions
        done = False
        win = False
        if move1 == "Yaniv":
            done = True
            win = state.yaniv(state.player_1_hand, [state.player_2_hand])
            reward = state.get_hand_value(state.player_1_hand) * 5 if win else -5
        draw = -1 if move3 == "Deck" else 0 if move3 == "Draw 0" else 1
        initial_hand = state.get_hand_value(state.player_1_hand)
        state.play(state.player_1_hand, move2[1], draw)
        reward = initial_hand - state.get_hand_value(state.player_1_hand)
        state.playOpponentTurn()
        
        return state, reward, done, win
    
    def update_weights(self, features : np.ndarray[int], actions : tuple[str, int, str], 
        possible_actions : tuple[list], reward : int) -> None:
        move1, move2, move3 = actions
        poss1, poss2, poss3 = possible_actions
        self.m1.update_weights(move1, poss1, reward, features)
        self.m2.update_weights(move2[0], [p[0] for p in poss2], reward, features)
        self.m3.update_weights(move3, poss3, reward, features)

def main():
    won_games = 0
    for episode in range(10000):
        sim = CombinedModel()
        state = GameState()
        done = False
        while not done:
            features = sim.features(state)
            m1, m2, m3, p1, p2, p3 = sim.choose_actions(features, state)
            action = (m1, m2, m3)
            possible = (p1, p2, p3)
            next_state, reward, done, win = sim.play_step(state, action)
            sim.update_weights(features, action, possible, reward)
            state = next_state
            print(action)
            print(state.player_1_hand)
            # print(state.player_1_hand)
        won_games += 1 if win else 0
    print(won_games)
        
if __name__ == "__main__":
    main()