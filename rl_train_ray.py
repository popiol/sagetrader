import gym
import ray
from ray.rllib.agents import sac, ppo
from custom_agent import CustomAgent
import time
import numpy as np
import random
import math


class StocksSimulator(gym.Env):
    def __init__(self, config):
        self.max_steps = config.get("max_steps", 100)
        self.n_comps = config.get("n_comps", 500)
        self.max_quotes = config.get("max_quotes", 25)
        
        # action space: for each company: buy confidence, buy price, sell price
        self.action_space = [gym.spaces.Box(0.0, 1.0, (self.n_comps, 3))]
        
        # observation space: for each company: bid/ask offers, recent quotes
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (self.n_comps, self.max_quotes))
        
        self.state = None
        self.steps = None

    def reset(self):
        self.state = self.observation_space.sample()
        self.steps = 0
        return self.state

    def _step(self, action):
        return None, None

    def step(self, action):
        self.steps += 1
        reward, self.state = self._step(action)
        done = self.steps >= self.max_steps
        return self.state, reward, done, {}


class StocksHistSimulator(StocksSimulator):
    def __init__(self, config):
        StocksSimulator.__init__(self, config)
        data_size = 671
        self.max_steps = min(self.max_steps, data_size-200)
        start = random.randint(50, data_size-self.max_steps)
        self.file = open("data/all_hist.csv", "r")
        self.prices = {}
        for _ in range(start):
            self.next_data()
            
    def next_data(self):
        timestamp = self.file.readline().strip()
        companies = self.file.readline().strip().split(",")
        prices = self.file.readline().strip().split(",")
        for company, price in zip(companies, prices):
            if company not in self.prices:
                self.prices[company] = [price]
            else:
                self.prices[company].append(price)

    def _step(self, action):
        reward = None
        state = None
        self.next_data()
        for company in self.prices:
            prices = self.prices[company]
            compressed = []
            for group_i in range(math.ceil(self.max_quotes/5)):
                

        if self.steps >= self.max_steps:
            self.file.close()
        return reward, state

ray.init()

config = {
    **sac.DEFAULT_CONFIG,
    "num_workers": 0,
    "timesteps_per_iteration": 1000,
    "rollout_fragment_length": 10,
}

ppo_config = {
    **ppo.DEFAULT_CONFIG,
    "num_workers": 0,
}

#agent = sac.SACTrainer(config=config, env=StocksSimulator)
#agent = ppo.PPOTrainer(config=ppo_config, env=StocksSimulator)
agent = CustomAgent(env=StocksSimulator)
#agent = sac.SACTrainer(config=config, env=CartPoleEnvWrapper)
#agent = ppo.PPOTrainer(config=ppo_config, env=CartPoleEnvWrapper)
#agent = CustomAgent(env=CartPoleEnvWrapper)
#agent = sac.SACTrainer(config=config, env=WindyMazeEnv)
#agent = ppo.PPOTrainer(config=ppo_config, env=WindyMazeEnv)
#agent = CustomAgent(env=WindyMazeEnv)

timestamp1 = time.time()

for _ in range(100):
    info = agent.train()
    score = agent.evaluate()["evaluation"]["episode_reward_min"]
    print(score)
    if score > 70:
        break

timestamp2 = time.time()
print("Execution time:", timestamp2 - timestamp1)

agent.stop()
