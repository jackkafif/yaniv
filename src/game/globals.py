from game.helpers import *

POSSIBLE_MOVES = generate_combinations(5)
N_MOVES = len(POSSIBLE_MOVES)

PHASE1_ACTION_SIZE = 2  # 0 = Play, 1 = Call Yaniv
PHASE2_ACTION_SIZE = len(POSSIBLE_MOVES)
PHASE3_ACTION_SIZE = 3  # 0 = Draw from deck, 1 = Draw first from discard, 2 = Draw second from dicard

STATE_SIZE = 106 + 32  # 52 (hand) + 52 (top cards) + 1 (opp hand size) + 1 (turn)
SAVE_EVERY = 50
MODEL_DIR = "src/agent/checkpoints"