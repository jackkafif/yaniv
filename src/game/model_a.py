from superModel import SuperModel


class ModelA(SuperModel):

    def __init__(self, learning_rate=0.01, discount_factor=0.90, explore_prob=0.40):
        yaniv_weights = {
            'Yaniv': 0.0,
            'Continue': 0.0,
        }
        super().__init__(yaniv_weights, learning_rate, discount_factor, explore_prob)
