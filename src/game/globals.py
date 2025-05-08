from game.helpers import *
import numpy as np

POSSIBLE_MOVES = generate_combinations(5)
N_MOVES = len(POSSIBLE_MOVES)

PHASE1_ACTION_SIZE = 2  # 0 = Play, 1 = Call Yaniv
PHASE2_ACTION_SIZE = len(POSSIBLE_MOVES)
# 0 = Draw from deck, 1 = Draw first from discard, 2 = Draw second from dicard
phase_3_arr = np.zeros(53)
PHASE3_ACTION_SIZE = phase_3_arr.shape[0] # index 52 is draw from deck, i=0-52 is draw card i from discard, masked by 0s at i where i is not in discard

# 52 (hand) + 52 (top cards) + 1 (opp hand size) + 1 (turn) + 32 (move values) + 6 (for each top card, if it completes, max move, and value)
STATE_SIZE = 104 + 2 + 32 + 6
SAVE_EVERY = 100
MODEL_DIR = "src/agent/"

SUITS = {
    0: "Clubs",
    1: "Spades",
    2: "Hearts",
    3: "Diamonds"
}

RANKS = {
    0: "Ace", 1: "Two", 2: "Three", 3: "Four",
    4: "Five", 5: "Six", 6: "Seven", 7: "Eight", 8: "Nine",
    9: "Ten", 10: "Jack", 11: "Queen", 12: "King"
}

SUITS_REVERSE = {value: key for key, value in SUITS.items()}
RANKS_REVERSE = {value: key for key, value in RANKS.items()}
