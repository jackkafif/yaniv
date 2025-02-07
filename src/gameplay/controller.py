from game.combined_model import CombinedModel
from game.state import GameState
import time

def trainModel() -> CombinedModel:
    won_games = 0
    num_episodes = 1
    sim = CombinedModel()
    for episode in range(num_episodes):
        # print(f"Running episode {episode}")
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
        won_games += 1 if win else 0
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
        while input("Would you like to play Yaniv? Y/N: ") == "Y":
            state = GameState()
            done = False
            print("Dealing Cards...")
            time.sleep(2)
            while done != True:
                print(f"Opponent has {state.hand_to_cards(state.player_1_hand)}")
                done, win = self.playModelMove(state)
                if done:
                    print(f"Opponent has {state.hand_to_cards(state.player_1_hand)}")

                    print(f"Your opponent called Yaniv! \
                          They have {state.get_hand_value(state.player_1_hand)} \
                            to your {state.get_hand_value(state.player_2_hand)}")
                    message = "won!" if win == False else "lost!"
                    print(f"You have {message}")
                    break
                opp = [state.card_to_name(card) for card in state.top_cards]
                print(f"Your Opponent played {opp}")
                print(f"Your hand is: {state.hand_to_cards(state.player_2_hand)}")
                print(f"The card(s) on the top is/are {state.top_cards} ")
                move = input("Input the cards you want to play in their indices 0 1 2 ... ")  
                move = tuple([int(i) for i in move.split(" ")])  
                valid_moves = [i[1] for i in state.valid_move_values(state.player_2_hand)]
                while move not in valid_moves:
                    print(f"That move is invalid, please try again {move}")
                    move = input("Input the cards you want to play in their indices 0 1 2 ... ")   
                    move = tuple([int(i) for i in move.split(" ")])
                draw = input("-1 to draw from deck, 0 to draw first card, 1 to draw second card from pile ")
                state.play(state.player_2_hand, move, int(draw))

if __name__ == "__main__":
    Play().play()


