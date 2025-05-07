import torch
from game.agent import YanivAgent, run_game
from game.state import GameState
from game.globals import *
import numpy as np
import os
import random
from game.nets import *
import sys

# Create a class that takes in an agent and can run different corner cases
import numpy as np
import random
import torch

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CornerCases:
    def __init__(self, agent: YanivAgent):
        self.agent = agent

    def test_straight(agent: YanivAgent, top_card: str = "Ace of Diamonds"):
        """
        Test the model with a hand that has a straight over a pair
        """
        state = GameState()
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


    def run_should_yaniv(self, num_turns):
        # Test case where the player can call Yaniv
        state = GameState()

        # Player 1 hand: Ace of Clubs
        hand = np.zeros(52)
        hand[0] = 1  # Ace of Spades

        # Player 2 hand: Two of Clubs, Three of Clubs
        other = np.zeros(52)
        other[1] = 1
        other[2] = 1

        # Set the state
        state.player_1_hand = hand
        state.player_2_hand = other
        state.curr_idx = num_turns
        state.top_cards = [51]
        print(state.player_1_hand, state.player_2_hand, state.top_cards)

        return (self.agent.choose_action_phase1(state, hand, other))


for x in range(10):
    agent = YanivAgent()
    agent.epsilon = 0.0
    cornerCases = CornerCases(agent)
    result = (cornerCases.run_should_yaniv(0))
    print("Turn: ", x, "Result: ", result)
    for i in range(10):
        top_card = state.card_to_name(i)
        print(f"Testing with top card: {top_card}")
        CornerCases.test_straight(agent, top_card)
