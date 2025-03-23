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
    return np.vstack([np.arange(N) for _ in range(N)])


def corr_pref_experiment(lam, beta, N, T):
    bpref = get_corr_preferences(beta, N, N)
    apref = np.vstack([np.random.permutation(range(N)) for _ in range(N)])
    return simulate(lam, bpref, apref, T)


def global_pref_experiment(lam, N, T, multiple=False):
    bpref = np.vstack([np.arange(N)[::-1] for _ in range(N)])
    apref = get_global_preferences(N)
    rew = simulate(lam, bpref, apref, T, multiple=multiple)
    if not multiple:
        max_rew = np.vstack([np.arange(N)[::-1] for _ in range(T)])
    else:
        best = sorted(2 * list(range(N)), reverse=True)[:N]
        max_rew = np.vstack([best for _ in range(T)])
    return max_rew - rew


def cum_rew_plot(r):
    plt.plot(r.cumsum(0), alpha=.5)
    plt.savefig('plot')


def avg_rew_plot(r):
    t = np.arange(1, len(r) + 1)
    plt.plot(r.cumsum(0) / t[:, np.newaxis], alpha=.5)
    plt.savefig('plot')


def max_avg_rew_plot(r):
    max_r = r.max(1)
    t = np.arange(1, len(r) + 1)
    plt.plot(max_r.cumsum() / t)
    plt.savefig('max_plot')


