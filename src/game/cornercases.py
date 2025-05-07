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


def set_seed():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CornerCases:
    def __init__(self):
        # Create an agent and set epsilon to 0.0
        set_seed()
        agent = YanivAgent()
        agent.epsilon = 0.0

        self.agent = agent

    def generate_state(self, your_cards, opp_cards, top_card):
        """
        Generate a game state with the given cards and top card.
        """
        state = GameState()
        state.player_1_hand[state.player_1_hand != 0] = 0
        state.player_2_hand[state.player_2_hand != 0] = 0

        for card in your_cards:
            state.player_1_hand[state.name_to_card(card)] += 1

        for card in opp_cards:
            state.player_2_hand[state.name_to_card(card)] += 1

        state.top_cards = [state.name_to_card(top_card)]
        return state

    def test_straight(self, top_card: str = "Ace of Diamonds"):
        """
        Test the model with a hand that has a straight over a pair
        """
        opp_cards = ["Ace of Spades", "Two of Spades",
                     "Three of Spades", "Four of Spades", "Five of Spades"]
        cards = ["Ten of Spades", "Ten of Hearts",
                 "Jack of Hearts", "Queen of Hearts", "King of Hearts"]
        state = self.generate_state(cards, opp_cards, top_card)

        a1, a2, a3 = self.agent.choose_turn_moves(
            state, state.player_1_hand, state.player_2_hand)
        print(a1, a2, a3)
        discard_indices = POSSIBLE_MOVES[int(a2)]
        print(f"Discard indices: {discard_indices}")

    def run_should_yaniv(self, your_card: str = "Ace of Spades", num_turns: int = 0):
        """
        Test the model with a hand where Yaniv should be called
        """
        # Set the cards for the player, opponent, and top card
        cards = [your_card]
        opp_cards = ["Queen of Hearts"]
        top_card = "Jack of Hearts"
        # Generate the state
        state = self.generate_state(cards, opp_cards, top_card)

        # Set the number of turns
        state.curr_idx = num_turns
        # print(state.player_1_hand, state.player_2_hand, state.top_cards)

        return (self.agent.choose_action_phase1(state, state.player_1_hand, state.player_2_hand))


your_cards = ["Ace of Spades", "Two of Spades", "Three of Spades",
              "Four of Spades", "Five of Spades", "Six of Spades", "Seven of Spades", "Eight of Spades"]
for card in your_cards:
    for x in range(30):
        set_seed()
        # Create the corner cases object
        cornerCases = CornerCases()
        result = (cornerCases.run_should_yaniv(your_card=card, num_turns=x))
        print("Card:", card, "Turn:", x, "Result:", result)


# for i in range(10):
#     state = GameState()
#     cornerCases = CornerCases(agent)

#     top_card = state.card_to_name(i)
#     print(f"Testing with top card: {top_card}")
#     cornerCases.test_straight(top_card)
