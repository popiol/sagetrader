import ray
from custom_agent import CustomAgent
import time
import numpy as np
from simulator import StocksHistSimulator
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
    
    filename = f"{agent.model_dir}/agent.tf"
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

    common.s3_upload_file("data/agent.dat", "model/agent.dat")
    common.s3_upload_file("data/model.h5", "model/model.h5")

if __name__ == "__main__":
    main()
