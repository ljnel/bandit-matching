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
    #print(bpref)
    #print(apref)
    #print(rew)
    return max_rew - rew


def cum_rew_plot(r):
    plt.plot(r.cumsum(0), alpha=.5)
    plt.savefig('plot')


def avg_rew_plot(r):
    t = np.arange(1, len(r) + 1)
    plt.plot(r.cumsum(0) / t[:, np.newaxis], alpha=.5)
    plt.savefig('plot')


def max_avg_rew_plot(r, ax=None, label=''):
    max_r = r.max(1)
    t = np.arange(1, len(r) + 1)
    if ax is None:
        plt.plot(max_r.cumsum() / t, label=label)
        plt.show()
    else:
        ax.plot(max_r.cumsum() / t, label=label)


def single_exp_plot():
    Ns = [5, 10, 15, 20]
    fig, ax = plt.subplots()
    for N in Ns:
        r = global_pref_experiment(.1, N, 5000, multiple=False)
        max_avg_rew_plot(r, ax, label=f'N={N}')
    ax.set_title('Regret in original setting', fontsize=20)
    ax.set_xlabel('T')
    ax.set_ylabel('Regret')
    fig.legend()
    fig.savefig('single_plot')


def multiple_exp_plot():
    Ns = [5, 10, 15, 20]
    fig, ax = plt.subplots()
    for N in Ns:
        r = global_pref_experiment(.1, N, 5000, multiple=True)
        max_avg_rew_plot(r, ax, label=f'N={N}')
    ax.set_title('Regret in modified setting', fontsize=20)
    ax.set_xlabel('T')
    ax.set_ylabel('Regret')
    fig.legend()
    fig.savefig('multiple_plot')

