import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import logging


class AdsEnv:
    def __init__(self, file):
        self.df = pd.read_csv(file)
        self.generator = self.df.iterrows()

    def step(self, action):
        _n, row = next(self.generator)
        reward = row[action]
        optimal = np.argmax(row)
        return reward, optimal == action


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
    env = AdsEnv(file="Ads_Optimisation.csv")
    agent = Agent(epsilon=epsilon)
    for i in range(10000):
        # if i % 100 == 0:
        #     print(f"\t\tIteration {i} for epsilon {epsilon}")
        action = agent.get_action()
        (reward, is_optimal) = env.step(action)
        agent.update(action, reward)
        stats.append(
            {
                "step": i,
                "reward": reward,
                "is_optimal": is_optimal,
                "epsilon": epsilon,
                "action": action,
            }
        )
    return pd.DataFrame(stats)


def experiment(epsilon):
    results = []
    print(f"Starting experiment for epsilon {epsilon}")
    for i in range(1000):
        if i % 10 == 0:
            print(f"\tIteration {i} for epsilon {epsilon}")
        result = run(epsilon)
        result["run"] = i
        results.append(result)
    print(f"Done experiment for epsilon {epsilon}")
    return pd.concat(results)


def main():
    # generate epsilons from 0.01 to 1 with log scale
    epsilons = np.logspace(0.0, 2.0, 10) / 100
    epsilons = np.round(epsilons, 3)
    results = Parallel(n_jobs=10)(delayed(experiment)(e) for e in epsilons)
    results = pd.concat(results)
    results.to_csv("./results-ads.csv", index=False)
