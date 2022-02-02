import numpy as np
import pandas as pd
from .agent import EpsilonGreedy
from .env import TestBedEnv


def run(epsilon):
    stats = []
    env = TestBedEnv()
    agent = EpsilonGreedy(epsilon=epsilon)
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
