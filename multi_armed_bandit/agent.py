import numpy as np


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.N = np.zeros(10)
        self.Qn = np.zeros(10)

    def get_action(self):
        action_type = np.random.choice(
            ["explore", "exploit"], p=[self.epsilon, (1 - self.epsilon)]
        )

        actions = {"exploit": self.exploit(), "explore": self.explore()}
        return actions[action_type]

    def update(self, action, reward):
        n = self.N[action] + 1
        q = self.Qn[action]

        new_q = q + (1 / n) * (reward - q)
        self.N[action] = n
        self.Qn[action] = new_q

    def exploit(self):
        return np.random.choice(np.flatnonzero(self.Qn == self.Qn.max()))

    def explore(self):
        return np.random.randint(10)
