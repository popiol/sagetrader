import gym
import ray
from ray.rllib.agents import sac
import tensorflow_probability as tfp


class StocksSimulator(gym.Env):
    def __init__(self, config):
        self.action_space = gym.spaces.Box(0.0, 1.0, (1,))
        self.observation_space = gym.spaces.Box(0.0, 1.0, (1,))
        self.max_steps = config.get("max_steps", 100)
        self.state = None
        self.steps = None

    def reset(self):
        self.state = self.observation_space.sample()
        self.steps = 0
        return self.state

    def step(self, action):
        self.steps += 1
        r = action[0]
        d = self.steps >= self.max_steps
        self.state = self.observation_space.sample()
        return self.state, r, d, {}


ray.init()

config = {
    **sac.DEFAULT_CONFIG,
    "Q_model": sac.DEFAULT_CONFIG["Q_model"].copy(),
    "num_workers": 0,  # Run locally.
    "twin_q": True,
    "clip_actions": False,
    "normalize_actions": True,
    "learning_starts": 0,
    "prioritized_replay": True,
    "rollout_fragment_length": 10,
    "train_batch_size": 10,
    "buffer_size": 40000,
    "framework": "tfe",
}
trainer = sac.SACTrainer(config=config, env=StocksSimulator)

print(trainer.train())
trainer.stop()
