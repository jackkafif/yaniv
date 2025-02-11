from game.superModel import SuperModel


class ModelC(SuperModel):

    def __init__(self, learning_rate=0.01, discount_factor=0.90, explore_prob=0.40):
        draw_or_from_discard = {
            'Deck': 0.0,
            'Card 0': 0.0,
            'Card 1': 0.0
        }
        super().__init__(draw_or_from_discard, learning_rate, discount_factor, explore_prob)
