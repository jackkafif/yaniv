import numpy as np

class SuperModel:
    def __init__(self, initial_weight_dictionary):
        self.learning_rate = 0.01
        self.discount_factor = 0.90
        self.explore_prob = 0.40
        self.weights = initial_weight_dictionary
    
    def predict_q(self, action, features : np.ndarray) -> int:
        return (self.weights[action] * features).sum()
        
    def update_weights(self, action, possible_actions : list, reward : int, features : np.ndarray):
        possible_actions.sort(key = lambda x : self.predict_q(x, features))
        next_action = possible_actions[0]
        next_features = self.predict_q(next_action, features)

        best_future_q = 0 if next_action == -1 else self.predict_q(next_action, next_features)

        current_q = self.predict_q(action, features)
        
        self.weights[action] += (self.learning_rate * (reward + self.discount_factor * best_future_q - current_q) * features).sum()

    def choose_action(self, possible_actions, features):
        if len(possible_actions) == 1:
            return possible_actions[0]
        elif np.random.rand() < self.explore_prob:
            return np.random.choice(possible_actions)
        else:
            possible_actions.sort(key = lambda action : self.predict_q(action, features))
            return possible_actions[0]
