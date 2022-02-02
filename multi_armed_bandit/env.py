import numpy as np
import pandas as pd


class TestBedEnv:
    def __init__(self):
        self.mean = np.random.randn(10)

    def step(self, action):
        reward = np.random.normal(loc=self.mean)
        optimal = np.argmax(self.mean)
        return reward[action], optimal == action


class AdsEnv:
    def __init__(self, file):
        self.df = pd.read_csv(file)
        self.generator = self.df.iterrows()

    def step(self, action):
        _n, row = next(self.generator)
        reward = row[action]
        optimal = np.argmax(row)
        return reward, optimal == action
