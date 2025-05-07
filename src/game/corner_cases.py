# corner_cases.py
# This module tests the models for specific cases of the game to test the models actions

from game.agent import YanivAgent
from game.state import GameState
from game.globals import *
from game.train import train_models
import numpy as np
import random
import torch

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

state = GameState()

def test_straight(agent: YanivAgent, top_card: str = "Ace of Diamonds"):
    """
    Test the model with a hand that has a straight over a pair
    """
    state.reset()
    opp_cards = ["Ace of Spades", "Two of Spades", "Three of Spades", "Four of Spades", "Five of Spades"]
    cards = ["Ten of Spades", "Ten of Hearts", "Jack of Hearts", "Queen of Hearts", "King of Hearts"]
    state.player_1_hand[state.player_1_hand != 0] = 0
    state.player_2_hand[state.player_2_hand != 0] = 0

    for card in cards:
        state.player_1_hand[state.name_to_card(card)] += 1

    for card in opp_cards:
        state.player_2_hand[state.name_to_card(card)] += 1

    state.top_cards = [state.name_to_card(top_card)]    

    a1, a2, a3 = agent.choose_turn_moves(state, state.player_1_hand, state.player_2_hand)
    print(a1, a2, a3)
    discard_indices = POSSIBLE_MOVES[int(a2)]
    print(f"Discard indices: {discard_indices}")


def test_straight_over_pair(agent: YanivAgent):
    """
    Test the model with a hand that has a straight over a pair
    """

if __name__ == "__main__":
    print("Testing straight over pair")
    agent = YanivAgent(state_size=STATE_SIZE)
    agent.epsilon = 0.0
    for i in range(10):
        top_card = state.card_to_name(i)
        print(f"Testing with top card: {top_card}")
        test_straight(agent, top_card)
