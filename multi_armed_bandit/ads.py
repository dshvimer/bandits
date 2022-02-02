import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .agent import EpsilonGreedy
from .env import AdsEnv


def run(epsilon):
    stats = []
    env = AdsEnv(file="Ads_Optimisation.csv")
    agent = EpsilonGreedy(epsilon=epsilon)
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
