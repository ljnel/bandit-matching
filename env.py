"Here we define the environment (the preference lists)."

import numpy as np
import matplotlib.pyplot as plt
from simulation import simulate


def get_corr_preferences(beta, N, L):
    """
    Generate correlated preferences as in the paper.
    Returns an (N, L) array.
    As beta -> infinity, the player's rewards get more correlated.
    """
    x = np.random.rand(L)
    eps = np.random.logistic(size=(N, L))
    mu = beta * x + eps
    return np.argsort(mu)


def get_global_preferences(N):
    return np.vstack([np.array(range(N)) for _ in range(N)])


def corr_pref_experiment(lam, beta, N, T):
    bpref = get_corr_preferences(beta, N, N)
    apref = np.vstack([np.random.permutation(range(N)) for _ in range(N)])
    return simulate(lam, bpref, apref, T)


def global_pref_experiment(lam, N, T):
    bpref = get_global_preferences(N)
    apref = get_global_preferences(N)
    return simulate(lam, bpref, apref, T)


def cum_rew_plot(r):
    plt.plot(r.cumsum(0), alpha=.5)
    plt.savefig('plot')