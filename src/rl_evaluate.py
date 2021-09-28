import os

if os.getcwd().endswith("/src"):
    os.chdir("..")

from custom_agent import CustomAgent
import time
from simulator import StocksRTSimulator
import common


def main():
    timestamp1 = time.time()
    data_dir = "data"
    train_file = f"{data_dir}/train.csv"
    agent_file = f"{data_dir}/agent.dat"

    env_config = {
        "train_file": train_file,
    }

    agent = CustomAgent(
        env=StocksRTSimulator, env_config=env_config
    )
    if os.path.isfile(agent_file):
        agent.load_checkpoint(agent_file)
    score = agent.evaluate()
    common.log("Score:", score)
    common.log("Bought:", agent.env.n_bought, "Sold:", agent.env.n_sold)

    timestamp2 = time.time()
    common.log("Execution time:", timestamp2 - timestamp1)


if __name__ == "__main__":
    main()
