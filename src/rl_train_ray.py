import os
if os.getcwd().endswith("/src"):
    os.chdir("..")

from custom_agent import CustomAgent
import time
import numpy as np
from simulator import StocksRTSimulator
import common
import shutil
import sys
import glob
import subprocess


def main(rebuild, worker_id, n_workers, n_iterations, max_steps):
    data_dir = "data"
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    train_file = f"{data_dir}/train.csv"
    agent_file = f"{data_dir}/agent.dat"
    hist_model_file = f"{data_dir}/hist_model.h5"
    rt_model_file = f"{data_dir}/rt_model.h5"
    train_file_remote = "data/all_hist.csv"
    agent_file_remote = "data/agent.dat"
    hist_model_file_remote = "data/hist_model.h5"
    rt_model_file_remote = "data/rt_model.h5"
    agent_file_worker = f"{data_dir}/agent-*.dat"
    hist_model_file_worker = f"{data_dir}/hist_model-*.h5"
    rt_model_file_worker = f"{data_dir}/rt_model-*.h5"

    n_workers = n_workers or 1
    n_iterations = n_iterations or 1
    max_steps = max_steps or 10000
    model_changed = False

    env_config = {
        "train_file": train_file,
    }

    if worker_id is not None:
        agent = CustomAgent(
            env=StocksRTSimulator, env_config=env_config, worker_id=worker_id
        )
        if os.path.isfile(agent_file):
            agent.load_checkpoint(agent_file)
        agent.train()
        score = agent.evaluate(quick=rebuild)
        print("score:", score)
        return

    if not rebuild:
        common.s3_download_file(train_file_remote, train_file, if_not_exists=True)
        common.s3_download_file(agent_file_remote, agent_file, if_not_exists=True)
        common.s3_download_file(hist_model_file_remote, hist_model_file, if_not_exists=True)
        common.s3_download_file(rt_model_file_remote, rt_model_file, if_not_exists=True)
    else:
        if os.path.isfile(agent_file):
            os.remove(agent_file)
        if os.path.isfile(hist_model_file):
            os.remove(hist_model_file)
        if os.path.isfile(rt_model_file):
            os.remove(rt_model_file)

    for _ in range(n_iterations):
        timestamp1 = time.time()

        files = glob.glob(agent_file_worker)
        files.extend(glob.glob(hist_model_file_worker))
        files.extend(glob.glob(rt_model_file_worker))
        for file in files:
            os.remove(file)

        scores = [None] * n_workers

        workers = []
        for worker_id in range(n_workers):
            cmd = [
                "/usr/bin/python3",
                "src/rl_train_ray.py",
                "--worker",
                str(worker_id),
                "--max_steps",
                str(max_steps),
            ]
            if rebuild:
                cmd.append("--rebuild")
            worker = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            workers.append(worker)

        for worker_id, worker in enumerate(workers):
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
            scores[worker_id] = score

        common.log("Scores:", scores)
        best_worker = None
        best_score = None
        for worker_i, score in enumerate(scores):
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_worker = str(worker_i)
        if best_worker is not None:
            common.log(hist_model_file_worker.replace("*", best_worker), "->", hist_model_file)
            shutil.copyfile(hist_model_file_worker.replace("*", best_worker), hist_model_file)
            common.log(rt_model_file_worker.replace("*", best_worker), "->", rt_model_file)
            shutil.copyfile(rt_model_file_worker.replace("*", best_worker), rt_model_file)
            shutil.copyfile(agent_file_worker.replace("*", best_worker), agent_file)
            model_changed = True

        timestamp2 = time.time()
        common.log("Execution time:", timestamp2 - timestamp1)

        if model_changed and not rebuild:
            common.s3_upload_file(agent_file, agent_file_remote)
            common.s3_upload_file(hist_model_file, hist_model_file_remote)
            common.s3_upload_file(rt_model_file, rt_model_file_remote)

        if os.getenv("SM_MODEL_DIR"):
            agent = CustomAgent(
                env=StocksRTSimulator, env_config=env_config, worker_id=worker_id
            )
            agent.load_checkpoint(agent_file)


if __name__ == "__main__":
    rebuild = False
    params = {}
    add_to = []
    for arg in sys.argv:
        if add_to:
            params[add_to[0]] = arg
            add_to = add_to[1:]
        elif arg == "--rebuild":
            rebuild = True
        elif arg == "--worker":
            add_to = ["worker"]
        elif arg == "--n_workers":
            add_to = ["n_workers"]
        elif arg == "--n_iterations":
            add_to = ["n_iterations"]
        elif arg == "--max_steps":
            add_to = ["max_steps"]

    worker_id = params.get("worker")
    n_workers = int(params["n_workers"]) if "n_workers" in params else None
    n_iterations = int(params["n_iterations"]) if "n_iterations" in params else None
    max_steps = int(params["max_steps"]) if "max_steps" in params else None

    main(rebuild, worker_id, n_workers, n_iterations, max_steps)
