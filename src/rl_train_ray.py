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
    common.s3_download_file("data/all_hist.csv", f"{data_dir}/all_hist.csv", if_not_exists=True)
    common.s3_download_file("model/agent.dat", f"{data_dir}/agent.dat", if_not_exists=True)
    common.s3_download_file("model/model.h5", f"{data_dir}/model.h5", if_not_exists=True)

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

    common.s3_upload_file(f"{data_dir}/agent.dat", "model/agent.dat")
    common.s3_upload_file(f"{data_dir}/model.h5", "model/model.h5")

if __name__ == "__main__":
    main()
