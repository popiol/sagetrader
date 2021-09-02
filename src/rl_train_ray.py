import ray
from custom_agent import CustomAgent
import time
import numpy as np
from simulator import StocksHistSimulator
import os
import common
import shutil


def main():
    data_dir = "data"
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    train_file = f"{data_dir}/train.csv"
    agent_file = f"{data_dir}/agent.dat"
    model_file = f"{data_dir}/model.h5"
    train_file_remote = "data/all_hist.csv"
    agent_file_remote = "data/agent.dat"
    model_file_remote = "data/model.h5"
    common.s3_download_file(train_file_remote, train_file, if_not_exists=True)
    common.s3_download_file(agent_file_remote, agent_file, if_not_exists=True)
    common.s3_download_file(model_file_remote, model_file, if_not_exists=True)

    ray.init()

    env_config = {"max_steps": 10000, "train_file": train_file}

    agent = CustomAgent(env=StocksHistSimulator, env_config=env_config)

    if os.path.isfile(agent_file):
        print("checkpoint:", agent_file)
        agent.load_checkpoint(agent_file)

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

    common.s3_upload_file(agent_file, agent_file_remote)
    common.s3_upload_file(model_file, model_file_remote)

    if os.getenv("SM_MODEL_DIR"):
        shutil.copy2(model_file, os.getenv("SM_MODEL_DIR"))


if __name__ == "__main__":
    main()
