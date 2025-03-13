import numpy as np
from ucb import CA_UCB

"""
Protocol:
    - Bandits don't know their true preferences
    - Each bandits knows which bandits are higher ranked than it on each arm
"""


def resolve_conflicts(choices, apref):
    "Given a list of bandits' choices and the preferences of the arms, "
    "find the winners."

    accepted = [False] * len(choices)

    best_proposal = {}

    for bandit, arm in enumerate(choices):
        if arm is not None:
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


def simulate(lam, bpref, apref, T):
    N = len(bpref)
    rewards = np.zeros(T, N)
    winners = []

    bandits = [CA_UCB(lam, apref, n) for n in range(N)]

    for t in range(T):
        choices = [bandits[n].choose(winners) for n in range(N)]
        outcome = resolve_conflicts(choices, apref)

        winners = []
        for bandit, won in enumerate(outcome):
            if won:
                winners.append((choices[bandit], bandit))

        for n in range(N):
            if outcome[n]:
                rewards[t, n] = bpref[n, choices[n]] + np.random.randn()
            else:
                rewards[t, n] = 0
            bandits[n].update(choices[n], rewards[t, n])

    return rewards
