from collections import defaultdict
import numpy as np
from ucb import CA_UCB

"""
Protocol:
    - Bandits don't know their true preferences
    - Each bandits knows which bandits are higher ranked than it on each arm
"""


def resolve_conflicts(choices, apref):
    """"
    Given a list of bandits' choices and the preferences of the arms
    (in decreasing order of preference), find the winners.
    """

    accepted = [False] * len(choices)

    best_proposal = {}

    for bandit, arm in enumerate(choices):
        if arm not in best_proposal:
            best_proposal[arm] = bandit
        else:
            # Compare preferences
            current_bandit = best_proposal[arm]
            if apref[arm, bandit] < apref[arm, current_bandit]:
                best_proposal[arm] = bandit
    # Mark accepted proposals
    for arm, bandit in best_proposal.items():
        accepted[bandit] = True

    return accepted


def resolve_conflicts2(choices, apref):
    """"
    In this variation, each arm accepts <= 2 bandits.
    """

    accepted = [False] * len(choices)

    proposals = defaultdict(list)

    for bandit, arm in enumerate(choices):
        proposals[arm].append(bandit)

    for arm, bandits in proposals.items():
        top = [bandit for bandit in apref[arm] if bandit in bandits][:2]
        for bandit in top:
            accepted[bandit] = True

    return accepted


def test_resolve_conflicts():
    aprefs = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    choices = [2, 2, 2]
    return resolve_conflicts(choices, aprefs)


def simulate(lam, bpref, apref, T, multiple=False):
    "Note: the regret is determined by the preferences, so it can be calculated outside this function."
    N = len(bpref)
    rewards = np.zeros((T, N))
    winners = []
    conflict_fn = resolve_conflicts if not multiple else resolve_conflicts2

    bandits = [CA_UCB(lam, apref, n) for n in range(N)]

    for t in range(T):
        choices = [bandits[n].choose(winners) for n in range(N)]
        outcome = conflict_fn(choices, apref)

        winners = []
        for bandit, won in enumerate(outcome):
            if won:
                winners.append((choices[bandit], bandit))

        for n in range(N):
            rewards[t, n] = bpref[n, choices[n]] + np.random.randn() \
                            if outcome[n] else 0
            bandits[n].update(choices[n], rewards[t, n])

    return rewards
