from superModel import SuperModel

class ModelB(SuperModel):

    def __init__(self):
        move_value = {
            i : 0.0 for i in range(1, 48)
        }
        super().__init__(move_value)