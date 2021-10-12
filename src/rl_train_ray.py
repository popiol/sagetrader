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
import datetime


def main(rebuild, worker_id, n_workers):
    data_dir = common.data_dir
    os.makedirs(data_dir, exist_ok=True)
    best_models_dir = common.best_models_dir
    os.makedirs(best_models_dir, exist_ok=True)
    winners_dir = common.winners_dir
    agent_file = common.agent_file
    hist_model_file = common.hist_model_file
    rt_model_file = common.rt_model_file
    agent_file_worker = common.agent_file_worker
    hist_model_file_worker = common.hist_model_file_worker
    rt_model_file_worker = common.rt_model_file_worker
    agent_file_best = common.agent_file_best
    hist_model_file_best = common.hist_model_file_best
    rt_model_file_best = common.rt_model_file_best

    n_workers = n_workers or 1
    
    env_config = {}

    agent = CustomAgent(
        env=StocksRTSimulator, env_config=env_config, worker_id=worker_id
    )
    if os.path.isfile(agent_file):
        agent.load_checkpoint(agent_file)
    if agent.explore < 0.1:
        rebuild = True

    if worker_id is not None:
        if not os.path.isfile(agent_file):
            max_timestamp = None
            for file in glob.iglob(winners_dir + "/agent*"):
                timestamp = os.path.getmtime(file)
                if max_timestamp is None or timestamp > max_timestamp:
                    max_timestamp = timestamp
                    last_winner = file
            if max_timestamp is not None:
                tmp_agent = CustomAgent(
                    env=StocksRTSimulator, env_config=env_config
                )
                tmp_agent.load_checkpoint(last_winner)
                agent.hist_model = agent.randomly_change_model(tmp_agent.hist_model)
                agent.rt_model = agent.randomly_change_model(tmp_agent.rt_model)
        agent.train()
        score = agent.evaluate(quick=True)
        print("score:", score)
        return

    if rebuild:
        if os.path.isfile(agent_file):
            os.remove(agent_file)
        if os.path.isfile(hist_model_file):
            os.remove(hist_model_file)
        if os.path.isfile(rt_model_file):
            os.remove(rt_model_file)
        dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        agent_file_best2 = agent_file_best.replace(data_dir + "/", best_models_dir + "/").replace("best", "best-" + dt)
        shutil.copyfile(
            agent_file_best, agent_file_best2
        )
        hist_model_file_best2 = hist_model_file_best.replace(data_dir + "/", best_models_dir + "/").replace("best", "best-" + dt)
        shutil.copyfile(
            hist_model_file_best, hist_model_file_best2
        )
        rt_model_file_best2 = rt_model_file_best.replace(data_dir + "/", best_models_dir + "/").replace("best", "best-" + dt)
        shutil.copyfile(
            rt_model_file_best, rt_model_file_best2
        )
        common.s3_upload_file(agent_file_best2)
        common.s3_upload_file(hist_model_file_best2)
        common.s3_upload_file(rt_model_file_best2)
        

    timestamp1 = time.time()

    files = glob.glob(agent_file_worker)
    files.extend(glob.glob(hist_model_file_worker))
    files.extend(glob.glob(rt_model_file_worker))
    for file in files:
        if not rebuild and len(file.split("-")[1].split(".")[0]) >= 4:
            continue
        os.remove(file)

    scores = [None] * n_workers

    workers = []
    for worker_id in range(n_workers):
        cmd = [
            "/usr/bin/python3",
            "src/rl_train_ray.py",
            "--worker",
            str(worker_id)
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
        common.log(
            hist_model_file_worker.replace("*", best_worker), "->", hist_model_file
        )
        shutil.copyfile(
            hist_model_file_worker.replace("*", best_worker), hist_model_file
        )
        common.log(
            rt_model_file_worker.replace("*", best_worker), "->", rt_model_file
        )
        shutil.copyfile(
            rt_model_file_worker.replace("*", best_worker), rt_model_file
        )
        shutil.copyfile(agent_file_worker.replace("*", best_worker), agent_file)

    timestamp2 = time.time()
    common.log("Execution time:", timestamp2 - timestamp1)


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
        
    worker_id = params.get("worker")
    n_workers = int(params["n_workers"]) if "n_workers" in params else None

    main(rebuild, worker_id, n_workers)
