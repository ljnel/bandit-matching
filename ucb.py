import numpy as np

"""
Protocol:
    - bandits know the preferences of the arms
    - bandits know who got what on the prev round
"""


class CA_UCB:
    def __init__(self, lam, apref, id):
        self.n_arms = len(apref)
        self.mu = np.zeros(self.n_arms)
        self.n = np.zeros(self.n_arms)
        self.apref = apref
        self.t = 0
        self.id = id
        self.lam = lam
        self.ucb = np.full(self.n_arms, np.inf)

    def get_plausible(self, winners):
        "Based on the last round's winners, get the plausible set."
        "winners: list of (arm, bandit) pairs"
        plausible = set(self.n_arms)
        for arm, bandit in winners.items():
            this_pos = np.argwhere(self.apref[arm] == self.id)
            other_pos = np.argwhere(self.apref[arm] == bandit)
            if other_pos < this_pos:
                plausible.remove(arm)
        return np.array(plausible)

    def choose(self, winners):
        "Choose the plausible arm with highest UCB."
        if self.t == 0:
            choice = np.random.randint(self.n_arms)
            self.prev = choice
            return choice
        elif np.random.rand() < self.lam:  # randomization strategy
            return self.prev
        else:
            idx = self.get_plausible(winners)
            choice = idx[np.argmax(self.ucb[idx])]
            self.prev = choice
            return choice

    def update(self, x, r):
        if r != 0:
            self.n[x] += 1
        self.mu[x] = ((self.n[x]-1) * self.mu[x] + r) / self.n[x]
        self.t += 1
        self.ucb = self.mu + np.sqrt(np.log(self.t) / self.n)
