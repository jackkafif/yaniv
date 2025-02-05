import itertools

def generate_combinations(n = 5):
    all_combinations = []
    for length in range(1, n + 1):
        for combination in itertools.combinations(range(n), length):
            all_combinations.append(combination)
    return all_combinations

COMBINATIONS = {i : generate_combinations(i) for i in range(1, 5)}