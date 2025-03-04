from game.combined_model import CombinedModel, sim_step
from game.state import GameState
import time
import os

def trainModel() -> CombinedModel:
    won_games = 0
    num_episodes = 10000
    sim = CombinedModel()
    sim2 = CombinedModel()
    for episode in range(num_episodes):
        print(f"Running episode {episode}")
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
    print(sim.m1.weights)
    print(sim.m2.weights)
    print(sim.m3.weights)
    return sim

class Play:
    def __init__(self):
        self.model = trainModel()

    def playModelMove(self, state : GameState) -> tuple[bool, bool]:
        features = self.model.features(state)
        m1, m2, m3, p1, p2, p3 = self.model.choose_actions(features, state)
        action = (m1, m2, m3)
        possible = (p1, p2, p3)
        next_state, reward, done, win = self.model.play_step(state, action, False)
        self.model.update_weights(features, action, possible, reward)
        state = next_state
        return done, win
    
    def play(self):
        return
        os.system('clear')
        while input("Would you like to play Yaniv? Y/N: ") == "Y":
            state = GameState()
            done = False
            print("Dealing Cards...")
            # time.sleep(2)
            while done != True:
                print(f"Opponent has {state.hand_to_cards(state.player_1_hand)}")
                state, done, win = sim_step(self.model, state, state.player_1_hand, state.player_2_hand)
                if done:
                    print(f"Your opponent called Yaniv! \
                          They have {state.get_hand_value(state.player_1_hand)} \
                            to your {state.get_hand_value(state.player_2_hand)}")
                    print(f"Opponent has {state.hand_to_cards(state.player_1_hand)}")
                    message = "won!" if win == False else "lost!"
                    print(f"You have {message}")
                    break
                opp = [state.card_to_name(card) for card in state.top_cards]
                print(f"Your Opponent played {opp}")
                # print(f"Opponent has {state.hand_to_cards(state.player_1_hand)}")
                our_hand = state.hand_to_cards(state.player_2_hand)
                print(f"Your hand is: {our_hand}")
                print(f"The card(s) on the top is/are {[state.card_to_name(card) for card in state.top_cards]} ")
                move = input("Input the cards you want to play in their indices 0 1 2 ... ")  
                if "Y" == move and state.can_yaniv(state.player_2_hand):
                    if state.can_yaniv(state.player_2_hand):
                        done = True
                        if state.yaniv(state.player_2_hand, [state.player_1_hand]):
                            print("You won!")
                        else:
                            print("You lost!")
                        break
                move = [int(i) for i in move.split(" ")]
                vmvs = state.valid_move_values(state.player_2_hand)
                print(vmvs)
                valid_moves = [i[1] for i in vmvs]
                valid_moves_english = [[our_hand[j] for j in i[1]] for i in vmvs]
                while move not in valid_moves:
                    print(f"That move is invalid, please try again {valid_moves_english}")
                    move = input("Input the cards you want to play in their indices 0 1 2 ... ")   
                    move = [int(i) for i in move.split(" ")]
                draw = input("-1 to draw from deck, 0 to draw first card, 1 to draw second card from pile ")
                # print(f"Your hand was {state.player_2_hand}")
                state.play(state.player_2_hand, move, int(draw))
                # print(f"Your hand is now {state.player_2_hand}")

if __name__ == "__main__":
    Play().play()


