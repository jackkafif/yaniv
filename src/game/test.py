import numpy as np
from game.model_a import *
from game.model_b import *
from game.model_c import *

def test_model_a():
    features = np.array([1, 2, 3])
    model = ModelA()
    assert model.weights == {"Yaniv" : 0.0, "Continue" : 0.0}
    assert model.predict_q("Yaniv", features) == 0.0
    model.update_weights("Yaniv", ["Yaniv", "Continue"], 10, features)
    assert model.predict_q("Continue", features) == 0.0
    print(model.predict_q("Yaniv", features))
    model.update_weights("Yaniv", ["Yaniv", "Continue"], -10, features)
    print(model.predict_q("Yaniv", features))

def test_model_b():
    move = 10
    possible_moves = [move, 2, 5]
    features = np.array((1, 2, 3))
    model = ModelB()
    assert model.predict_q(move, features) == 0.0
    model.update_weights(move, possible_moves, 10, features)
    assert model.predict_q(possible_moves[1], features) == 0.0
    print(model.predict_q(move, features))
    model.update_weights(move, possible_moves, -10, features)
    print(model.predict_q(move, features))

def test_model_c():
    move = "Deck"
    possible_moves = [move, "Card 0", "Card 1"]
    features = np.array((1, 2, 3))
    model = ModelC()
    assert model.predict_q(move, features) == 0.0
    model.update_weights(move, possible_moves, 10, features)
    assert model.predict_q(possible_moves[1], features) == 0.0
    print(model.predict_q(move, features))
    model.update_weights(move, possible_moves, -10, features)
    print(model.predict_q(move, features))


if __name__ == "__main__":
    test_model_a()
    test_model_b()
    test_model_c()