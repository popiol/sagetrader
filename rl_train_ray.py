import gym
import ray
from ray.rllib.agents import sac


class StocksSimulator(gym.Env):
    def __init__(self, env_config):
        self.action_space = gym.spaces.Box(0.0, 1.0, (1,))
        self.observation_space = gym.spaces.Box(0.0, 1.0, (1,))

    def reset(self):
        return obs

    def step(self, action):
        return obs, reward, done, info


ray.init()
config = sac.DEFAULT_CONFIG.copy()
config["Q_model"] = sac.DEFAULT_CONFIG["Q_model"].copy()
config["num_workers"] = 0  # Run locally.
config["twin_q"] = True
config["clip_actions"] = False
config["normalize_actions"] = True
config["learning_starts"] = 0
config["prioritized_replay"] = True
config["rollout_fragment_length"] = 10
config["train_batch_size"] = 10
config["buffer_size"] = 40000
num_iterations = 1
trainer = ppo.PPOTrainer(
    env=StocksSimulator,
    config={
        "env_config": {},
    },
)

while True:
    print(trainer.train())
