import os
import sys

if os.getcwd().endswith("/src"):
    os.chdir("..")

from custom_agent import CustomAgent
import time
from simulator import StocksRTSimulator
import common
import subprocess
import glob
import shutil


def main(worker_id, model, master):
    timestamp1 = time.time()
    data_dir = "data"
    best_models_dir = "models"
    train_file = f"{data_dir}/train.csv"
    agent_file = f"{data_dir}/agent.dat"
    agent_file_worker = f"{data_dir}/agent-*.dat"

    if master:
        agent_files = []
        workers = []
        scores = []
        for agent_file in glob.iglob(best_models_dir + "/agent*.dat"):
            cmd = [
                "/usr/bin/python3",
                "src/rl_evaluate.py",
                "--model=" + agent_file,
            ]
            worker = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            agent_files.append(agent_file)
            workers.append(worker)
            scores.append(None)

        for worker_i, worker in enumerate(workers):
            out, err = worker.communicate()
            common.log(out)
            try:
                score = None
                for line in out.split(b"\n"):
                    line = line.strip()
                    if line.startswith(b"score: "):
                        score = float(line.split(b" ")[1])
                        break
            except ValueError:
                score = None
            scores[worker_i] = score

        common.log("Scores:", scores)
        best_agent = None
        best_score = None
        for worker_i, score in enumerate(scores):
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_agent = agent_files[worker_i]
        if best_agent is not None:
            shutil.copyfile(
                best_agent, best_agent.replace(best_models_dir + "/", best_models_dir + "/winners/")
            )
    else:
        common.s3_download_file(train_file, if_not_exists=True)

        if worker_id is not None:
            agent_file = agent_file_worker.replace("*", worker_id)

        env_config = {
            "train_file": train_file
        }

        if model is not None:
            env_config["validate_max_steps"] = 500000
            agent_file = model
            path = model.split("/")
            if len(path) > 1:
                data_dir = "/".join(path[:-1])

        agent = CustomAgent(
            env=StocksRTSimulator, env_config=env_config, worker_id=worker_id
        )
        if model is not None:
            agent.model_dir = data_dir
        if os.path.isfile(agent_file):
            agent.load_checkpoint(agent_file)
        score = agent.evaluate(find_best=(worker_id is None))
        print("score:", score)
        common.log("Score:", score)
        common.log("Bought:", agent.env.n_bought, "Sold:", agent.env.n_sold)

    timestamp2 = time.time()
    common.log("Execution time:", timestamp2 - timestamp1)


if __name__ == "__main__":
    worker_id = None
    model = None
    master = False
    for arg in sys.argv:
        if arg.startswith("--worker_id"):
            worker_id = arg.split("=")[1]
        elif arg.startswith("--model"):
            model = arg.split("=")[1]
        elif arg.startswith("--master"):
            master = True

    main(worker_id, model, master)
