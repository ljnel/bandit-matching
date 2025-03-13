"Here we define the environment (the preference lists)."

import numpy as np
from simulation import simulate


def get_random_preferences(beta, N, L):
    """
    Generate the mean rewards randomly as in the paper, with N bandits and L arms.
    Returns an (N, L) array.
    As beta -> infinity, the player's rewards get more correlated.
    """
    x = np.random.rand(L)
    eps = np.random.logistic(size=(N, L))
    mu = beta * x + eps
    return np.argsort(mu)

bpref = get_random_preferences(1, 5, 5)
apref = np.vstack([np.random.permutation(range(5)) for _ in range(5)])

simulate(.1, bpref, apref, 10)