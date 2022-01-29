import numpy as np
import pandas as pd


class TestBedEnv:
    def __init__(self):
        self.mean = np.random.randn(10)

    def step(self, action):
        reward = np.random.normal(loc=self.mean)
        optimal = np.argmax(self.mean)
        return reward[action], optimal == action


class Agent:
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


def run(epsilon):
    stats = []
    env = TestBedEnv()
    agent = Agent(epsilon=epsilon)
    for i in range(1000):
        action = agent.get_action()
        (reward, is_optimal) = env.step(action)
        agent.update(action, reward)
        stats.append(
            {"step": i, "reward": reward, "is_optimal": is_optimal, "epsilon": epsilon}
        )
    return pd.DataFrame(stats)


def experiment():
    results = []
    for i in range(2000):
        epsilon = 0
        if i % 100 == 0:
            print(f"Iteration {i} for {epsilon}")
        results.append(run(epsilon))
    for i in range(2000):
        epsilon = 0.01
        if i % 100 == 0:
            print(f"Iteration {i} for {epsilon}")
        results.append(run(epsilon))
    for i in range(2000):
        epsilon = 0.1
        if i % 100 == 0:
            print(f"Iteration {i} for {epsilon}")
        results.append(run(epsilon))

    return pd.concat(results)


def main():
    results = experiment()
    results.to_csv("./results-testbed.csv")
