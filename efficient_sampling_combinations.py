import math
import random

def unrank_combination(N, K, r):
    """
    Convert a combination rank r into the actual combination in lexicographic order.
    N: total number of elements
    K: number of elements in each combination
    r: 0-based rank
    """
    combination = []
    chosen = 0
    for x in range(N):
        if chosen == K:
            # If we've chosen K elements, no need to continue
            break
        # Count how many combinations start with x if we pick it now:
        c = math.comb(N - x - 1, K - chosen - 1)
        if r < c:
            # Our desired rank falls within these combinations, so we pick x.
            combination.append(x)
            chosen += 1
            # r stays the same because we are now looking at combos starting with x.
        else:
            # Skip these combinations
            r -= c
        # Move to the next element
    return combination

def sample_random_combinations(num_all_h, length, K):
    """
    Sample K random unique combinations from combinations(range(num_all_h), length)
    without enumerating them all.

    :param num_all_h: The size of the range (N)
    :param length: The combination length
    :param K: Number of combinations to sample
    :return: A list of K randomly selected combinations (each is a list of integers)
    """
    total = math.comb(num_all_h, length)
    if K > total:
        raise ValueError("Cannot sample more combinations than the total available.")

    used_ranks = set()
    sampled_ranks = []

    # Generate K unique random ranks first
    while len(sampled_ranks) < K:
        r = random.randrange(total)
        if r not in used_ranks:
            used_ranks.add(r)
            sampled_ranks.append(r)

    # Now unrank all combinations
    # Each combination will be a list of integers
    sampled_combos = [unrank_combination(num_all_h, length, r) for r in sampled_ranks]

    return sampled_combos

if __name__ == '__main__':
    #sampled = sample_random_combinations(4096, 4, 1024)
    #print(sampled)  # Each element is now a list of integers.
    N = 6
    K = 4
    for r in range(10):
        print(unrank_combination(N, K, r))