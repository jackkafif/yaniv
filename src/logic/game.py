from deck import Deck
from player import Player

class Game:
    def __init__(self, n_players):
        self.deck = Deck()
        assert n_players >= 2
        self.players = [Player() for player in range(n_players)]