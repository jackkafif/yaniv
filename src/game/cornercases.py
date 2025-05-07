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


class CornerCases:
    def __init__(self, agent: YanivAgent):
        self.agent = agent

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
