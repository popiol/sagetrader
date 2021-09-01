import ray
from ray.rllib.agents import sac, ppo
from custom_agent import CustomAgent
import time
import numpy as np
from simulator import StocksHistSimulator
from gym.envs.classic_control import CartPoleEnv
from ray.rllib.examples.env.windy_maze_env import WindyMazeEnv
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
import os
import common


def main():
    data_dir = "data"
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    filename = f"{data_dir}/all_hist.csv"
    if not os.path.isfile(filename):
        print("Download data")
        common.s3_download_file(filename)

    ray.init()

    agent = CustomAgent(env=StocksHistSimulator, env_config={"max_steps":10000})
    
    filename = f"{agent.model_dir}/model.tf"
    if os.path.isfile(filename):
        print("model:", filename)
        agent.load_checkpoint(filename)

    timestamp1 = time.time()

    for _ in range(1):
        agent.train()
        eval = agent.evaluate()
        score = eval["evaluation"]["episode_reward_min"]
        print(score)
        if score > 10:
            break

    timestamp2 = time.time()
    print("Execution time:", timestamp2 - timestamp1)

    agent.stop()

if __name__ == "__main__":
    main()
