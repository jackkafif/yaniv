import itertools

def generate_combinations(n = 5):
    all_combinations = []
    for length in range(1, n + 1):
        for combination in itertools.combinations(range(n), length):
            all_combinations.append(combination)
    return all_combinations

# COMBINATIONS = {i : {idx : comb for idx, comb in enumerate(generate_combinations(i))} for i in range(6)}
COMBINATIONS = {i + 1 : {} for i in range(6)}

five_comb = generate_combinations(5)
for idx, comb in enumerate(five_comb):
    for i in range(len(comb) - 1, 6):
        COMBINATIONS[i + 1][idx] = comb