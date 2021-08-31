import ray
from ray.rllib.agents import sac, ppo
from custom_agent import CustomAgent
import time
import numpy as np
from simulator import StocksHistSimulator
from gym.envs.classic_control import CartPoleEnv
from ray.rllib.examples.env.windy_maze_env import WindyMazeEnv
import glob


class CartPoleEnvWrapper(CartPoleEnv):
    def __init__(self, config):
        CartPoleEnv.__init__(self)
        self.max_steps = config.get("max_steps", 100)


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

# agent = sac.SACTrainer(config=config, env=StocksHistSimulator)
# agent = ppo.PPOTrainer(config=ppo_config, env=StocksHistSimulator)
agent = CustomAgent(env=StocksHistSimulator, env_config={"max_steps":10000})
# agent = sac.SACTrainer(config=config, env=CartPoleEnvWrapper)
# agent = ppo.PPOTrainer(config=ppo_config, env=CartPoleEnvWrapper)
# agent = CustomAgent(env=CartPoleEnvWrapper)
# agent = sac.SACTrainer(config=config, env=WindyMazeEnv)
# agent = ppo.PPOTrainer(config=ppo_config, env=WindyMazeEnv)
# agent = CustomAgent(env=WindyMazeEnv)

timestamp1 = time.time()
files = glob.glob("data/model*")
if files:
    filename = max(files)
    print("model:", filename)
    if filename:
        agent.load_checkpoint(filename)

for _ in range(100):
    info = agent.train()
    eval = agent.evaluate()
    score = eval["evaluation"]["episode_reward_min"]
    print(score)
    if score > 10:
        break

timestamp2 = time.time()
print("Execution time:", timestamp2 - timestamp1)

agent.stop()
