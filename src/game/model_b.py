from superModel import SuperModel


class ModelB(SuperModel):

    def __init__(self, learning_rate=0.01, discount_factor=0.90, explore_prob=0.40):
        move_value = {
            i: 0.0 for i in range(1, 48)
        }
        super().__init__(move_value, learning_rate, discount_factor, explore_prob)
