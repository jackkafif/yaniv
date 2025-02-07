from game.superModel import SuperModel

class ModelA(SuperModel):

    def __init__(self):
        yaniv_weights = {
            'Yaniv': 0.0,
            'Continue': 0.0,
        }
        super().__init__(yaniv_weights)