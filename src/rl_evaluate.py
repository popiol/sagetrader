import os
import sys

if os.getcwd().endswith("/src"):
    os.chdir("..")

from custom_agent import CustomAgent
import time
from simulator import StocksRTSimulator
import common


def main(worker_id):
    timestamp1 = time.time()
    data_dir = "data"
    train_file = f"{data_dir}/train.csv"
    agent_file = f"{data_dir}/agent.dat"
    agent_file_worker = f"{data_dir}/agent-*.dat"

    if worker_id is not None:
        agent_file = agent_file_worker.replace("*", worker_id)

    env_config = {
        "train_file": train_file,
    }

    agent = CustomAgent(
        env=StocksRTSimulator, env_config=env_config, worker_id=worker_id
    )
    if os.path.isfile(agent_file):
        agent.load_checkpoint(agent_file)
    score = agent.evaluate(find_best=(worker_id is None))
    common.log("Score:", score)
    common.log("Bought:", agent.env.n_bought, "Sold:", agent.env.n_sold)

    timestamp2 = time.time()
    common.log("Execution time:", timestamp2 - timestamp1)


if __name__ == "__main__":
    worker_id = None
    for arg in sys.argv:
        if arg.startswith("--worker_id"):
            worker_id = arg.split("=")[1]
            break

    main(worker_id)
