import torch
from game.agent import YanivAgent, run_game
from game.state import GameState
from game.model import MDQN, M
from game.globals import *
import numpy as np
import os
import random
from game.nets import *
import sys

num_trials = 10


def set_seed():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

t1 = "linear-vs-linear"
t2 = "linear-vs-multi"
t3 = "multi-vs-multi"

class CornerCases:
    def __init__(self):
        # Create an agent and set epsilon to 0.0
        set_seed()
        # agent = YanivAgent(t1, STATE_SIZE, M, M, M)
        agent = YanivAgent(t3, STATE_SIZE, MDQN, MDQN, MDQN)
        agent.load_models(1)
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

        if len(top_card) == 1:
            state.top_cards = [state.name_to_card(top_card[0])]
        elif len(top_card) == 2:
            state.top_cards = [state.name_to_card(
                top_card[0]), state.name_to_card(top_card[1])]
        return state

    def test_straight(self, top_card: str = "Ace of Diamonds"):
        """
        Test the model with a hand that has a straight over a pair
        """
        opp_cards = ["Ace of Spades", "Two of Spades",
                     "Three of Spades", "Four of Spades", "Five of Spades"]
        cards = ["Ten of Spades", "Ten of Hearts",
                 "Jack of Hearts", "Queen of Hearts", "King of Hearts"]
        state = self.generate_state(cards, opp_cards, [top_card])

        a1, a2, a3 = self.agent.choose_turn_moves(
            state, state.player_1_hand, state.player_2_hand)
        # print(a1, a2, a3)
        discard_indices = POSSIBLE_MOVES[int(a2)]
        print(f"Discarding: {[cards[i] for i in discard_indices]}")

    def run_should_yaniv(self, your_card: str = "Ace of Spades", num_turns: int = 0):
        """
        Test the model with a hand where Yaniv should be called
        """

        # Set the cards for the player, opponent, and top card
        cards = [your_card]
        opp_cards = ["Queen of Hearts"]
        top_cards = ["Jack of Hearts"]

        # Generate the state
        state = self.generate_state(cards, opp_cards, top_cards)

        # Set the number of turns
        state.curr_idx = num_turns

        return (self.agent.choose_action_phase1(state, state.player_1_hand, state.player_2_hand, state.top_cards))

    def run_pick_up_higher(self, num_turns: int = 5):
        """
        Test the model with a hand where a higher card should be picked up
        """

        # Set the cards for the player, opponent, and top card
        cards = ["Seven of Hearts", "Seven of Diamonds"]
        opp_cards = ["Ace of Hearts"]
        # top_card = ["Five of Spades", "Seven of Spades"]
        top_card = ["Nine of Spades", "Seven of Spades"]

        # Generate the state
        state = self.generate_state(cards, opp_cards, top_card)
        state.tc_holder = state.top_cards

        # Set the number of turns
        state.curr_idx = num_turns

        a3 = (self.agent.choose_action_phase3(state, state.player_1_hand, state.player_2_hand, state.tc_holder))
        return a3



    def run_pick_up_card(self, top_card: str = "Ace of Spades", num_turns: int = 5):
        """
        Test the model with a hand where a card should be picked up (as opposed to a card from the deck)
        """
        # Set the cards for the player, opponent, and top card
        cards = ["Seven of Hearts", "Seven of Diamonds"]
        opp_cards = ["Queen of Hearts"]
        top_card = [top_card]
        # Generate the state
        state = self.generate_state(cards, opp_cards, top_card)
        state.tc_holder = state.top_cards

        # Set the number of turns
        state.curr_idx = num_turns
        # print(state.player_1_hand, state.player_2_hand, state.top_cards)

        a3 = (self.agent.choose_action_phase3(state, state.player_1_hand, state.player_2_hand, state.tc_holder))
        return a3

    def run_play_high_card(self, num_turns: int = 5):
        """
        Test the model to make sure high cards are played first
        """

        # Set the cards for the player, opponent, and top card
        # cards = ["Ace of Hearts", "Two of Hearts", "Seven of Diamonds"]
        # opp_cards = ["Queen of Hearts", "Jack of Clubs", "Ten of Diamonds"]
        # top_card = ["King of Spades"]

        # cards = ["Six of Clubs", "King of Clubs", "Ace of Hearts", "Three of Diamonds", "Queen of Diamonds"]
        # opp_cards = ["Four of Clubs", "Nine of Clubs", "Seven of Spades", "Nine of Diamonds"]
        # top_card = ["Seven of Hearts"]

        cards = ["Six of Clubs", "King of Clubs", "Ace of Spades", "Ace of Hearts"]
        opp_cards = ["Four of Clubs", "Seven of Clubs", "Nine of Clubs", "Three of Diamonds"]
        top_card = ["Three of Diamonds"]

        # Generate the state
        state = self.generate_state(cards, opp_cards, top_card)

        # Set the number of turns
        state.curr_idx = num_turns

        result = (self.agent.choose_action_phase2(
            state, state.player_1_hand, state.player_2_hand, state.top_cards, True))
        return [cards[i] for i in POSSIBLE_MOVES[int(result)]]

# Tests


def test_straight():
    for i in range(10):
        state = GameState()
        cornerCases = CornerCases()

        top_card = state.card_to_name(i)
        print(f"Testing with top card: {top_card}")
        cornerCases.test_straight(top_card)


def test_yaniv_calling():
    your_cards = ["Ace of Spades", "Two of Spades", "Three of Spades",
                  "Four of Spades", "Five of Spades", "Six of Spades", "Seven of Spades", "Eight of Spades"]
    for card in your_cards:
        for x in range(num_trials):
            set_seed()
            # Create the corner cases object
            cornerCases = CornerCases()
            result = (cornerCases.run_should_yaniv(
                your_card=card, num_turns=x))
            print("Card:", card, "Turn:", x, "Result:", "Y" if result == 1 else "N")


def test_pick_up_higher():
    state = GameState()
    for x in range(num_trials):
        # set_seed()
        # Create the corner cases object
        cornerCases = CornerCases()
        result = (cornerCases.run_pick_up_higher(num_turns=x))
        print("Turn:", x, "Result:", "Deck" if result == 52 else state.card_to_name(result))


def test_pick_up_card():
    top_cards = ["Ace of Spades", "Two of Spades", "Three of Spades",
                 "Four of Spades", "Five of Spades", "Six of Spades", "Seven of Spades", "Eight of Spades"]
    state = GameState()
    for card in top_cards:
        for x in range(num_trials):
            # set_seed()
            # Create the corner cases object
            cornerCases = CornerCases()
            result = (cornerCases.run_pick_up_card(top_card=card, num_turns=x))

            print("Top Card:", card, "Turn:", x, "Result:", "Deck" if result == 52 else state.card_to_name(result))


def test_play_high_card():
    # set_seed()
    cornerCases = CornerCases()
    result = (cornerCases.run_play_high_card())
    print("Result:", result)

if __name__ == "__main__":
    print("Testing Yaniv Calling...")
    test_yaniv_calling()
    print("Finished testing Yaniv Calling")
    print("Testing Pick Up Higher...")
    test_pick_up_higher()
    print("Finished testing Pick Up Higher")
    print("Testing Pick Up Card...")
    test_pick_up_card()
    print("Finished testing Pick Up Card")
    print("Testing Play High Card...")
    test_play_high_card()
    print("Finished testing Play High Card")
