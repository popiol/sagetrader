import gym
import ray
from ray.rllib.agents import sac, ppo
from custom_agent import CustomAgent
import time
from gym.envs.classic_control import CartPoleEnv, Continuous_MountainCarEnv
from ray.rllib.examples.env.windy_maze_env import WindyMazeEnv
import numpy as np

        
class MountainCarEnv(Continuous_MountainCarEnv):
    def __init__(self, config):
        Continuous_MountainCarEnv.__init__(self)
        self.max_steps = config.get("max_steps", 100)

class CartPoleEnvWrapper(CartPoleEnv):
    def __init__(self, config):
        CartPoleEnv.__init__(self)
        self.max_steps = config.get("max_steps", 100)

class StocksSimulator(gym.Env):
    def __init__(self, config):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,))
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,))
        self.max_steps = config.get("max_steps", 100)
        self.state = None
        self.steps = None

    def reset(self):
        self.state = self.observation_space.sample()
        self.steps = 0
        return self.state

    def step(self, action):
        if np.shape(action) == (1,):
            action = action[0]
        self.steps += 1
        r = 1-abs(action - pow(self.state[0],2))
        d = self.steps >= self.max_steps
        self.state = self.observation_space.sample()
        return self.state, r, d, {}


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
