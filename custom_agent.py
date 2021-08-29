from numpy.core.numeric import isscalar
from sklearn.neural_network import MLPRegressor
import gym
import random
import numpy as np
from gym.spaces.discrete import Discrete


class CustomAgent:
    def __init__(self, env: gym.Env):
        self.env = env({})
        self.model = MLPRegressor(warm_start=True)
        self.avg_reward = 0
        self.avg_total = 0
        self.max_total = None
        self.explore = 1
        self.fitted = False

    def predict_action(self, x):
        action = self.model.predict(x)
        if type(self.env.action_space) == Discrete:
            if np.shape(action) == (1,):
                action = action[0]
            action = round(action)
            action = max(0, action)
            action = min(self.env.action_space.n - 1, action)
        return action

    def transform_x(self, state):
        return (
            state
            if np.isscalar(state)
            else [y for x in state for y in np.array(x).flatten()]
        )

    def transform_y(self, action):
        return action[0] if np.shape(action) == (1,) else action

    def train_episode(self):
        prev_state = self.env.reset()
        train_x = []
        train_y = []
        hist_x = []
        hist_y = []
        total = 0
        for _ in range(1000):
            self.niter += 1
            x = self.transform_x(prev_state)
            if random.random() < self.explore or not self.fitted:
                action = self.env.action_space.sample()
            else:
                action = self.predict_action([x])
            state, r, d, info = self.env.step(action)
            total += r
            y = self.transform_y(action)
            if r > self.avg_reward + 1 * abs(self.avg_reward):
                train_x.append(x)
                train_y.append(y)
            hist_x.append(x)
            hist_y.append(y)
            self.avg_reward = self.avg_reward * 0.99 + r * 0.01
            if d:
                break
            prev_state = state
        if total > self.avg_total + 1 * abs(self.avg_total):
            train_x.extend(hist_x)
            train_y.extend(hist_y)
        self.avg_total = self.avg_total * 0.9 + total * 0.1
        if self.max_total is None or total > self.max_total:
            self.max_total = total
        if train_x:
            nit = max(
                1,
                round(1 * (r - self.avg_total + 1) / (abs(self.avg_total) + 1) - 1000),
            )
            for _ in range(nit):
                self.model.fit(train_x, train_y)
            self.fitted = True
        self.explore = max(0.3, self.explore * 0.9999)

    def train(self):
        self.niter = 0
        self.max_total = None
        for _ in range(1000):
            self.train_episode()
            if self.niter > 100000:
                break
        print(
            "avg r:",
            self.avg_reward,
            ", avg total:",
            self.avg_total,
            "max:",
            self.max_total,
            "explore:",
            self.explore,
        )

    def evaluate(self):
        prev_state = self.env.reset()
        total = 0
        for _ in range(1000):
            if self.fitted:
                x = self.transform_x(prev_state)
                action = self.predict_action([x])
            else:
                action = self.env.action_space.sample()
            state, r, d, info = self.env.step(action)
            total += r
            if d:
                break
            prev_state = state
        return {"evaluation": {"episode_reward_min": total}}

    def stop(self):
        pass
