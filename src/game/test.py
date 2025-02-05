import state

def main():
    s = state.GameState()
    hand = s.player_1_hand
    hand[0:3] = 1
    hand[12] = hand[12 + 13] = hand[12 + 2*13] = 1
    print(s.move_value(hand, 0))
    print(s.move_value(hand, 12))

if __name__ == "__main__":
    main()