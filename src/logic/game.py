from deck import Deck
from player import Player
from move import Move

class Game:
    def __init__(self, n_players = 2):
        self.deck = Deck()
        assert n_players >= 2
        self.players = [Player() for player in range(n_players)]
        for player in self.players:
            for _ in range(5):
                player.add_to_hand(self.deck.deal())

        self.turn = 0
        
    def play_turn(self, move : Move, pickup) -> bool:
        self.players[self.turn].play_turn(move, pickup)

    def game_over(self) -> int:
        pass
        