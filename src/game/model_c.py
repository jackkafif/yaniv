from game.superModel import SuperModel

class ModelC(SuperModel):

    def __init__(self):
        draw_or_from_discard = {
            'Deck' : 0.0,
            'Card 0' : 0.0,
            'Card 1' : 0.0
        }        
        super().__init__(draw_or_from_discard)