import os
import sys

if os.getcwd().endswith("/src"):
    os.chdir("..")

import glob
import random
import shutil
import subprocess
import time

import common
from custom_agent import CustomAgent
from simulator import StocksRTSimulator


def main(worker_id, model, master):
    timestamp1 = time.time()
    best_models_dir = common.best_models_dir
    winners_dir = common.winners_dir
    archive_dir = common.archive_dir
    agent_file = common.agent_file
    agent_file_worker = common.agent_file_worker

    if master:
        os.makedirs(best_models_dir, exist_ok=True)
        winners = []
        files = glob.glob(winners_dir + "/*.dat")
        files.extend(glob.glob(winners_dir + "/*.h5"))
        for file in files:
            os.remove(file)
        files = common.s3_find_objects(winners_dir + "/agent")
        if files:
            last_files = random.sample(files, min(3, len(files)))
            for file in last_files:
                winners.append(
                    file.key.replace(winners_dir + "/", best_models_dir + "/")
                )
                model_id = common.model_id_from_filename(file.key)
                common.s3_download_file(
                    f"{winners_dir}/agent-{model_id}.dat",
                    f"{best_models_dir}/agent-{model_id}.dat",
                    if_not_exists=True,
                )
                common.s3_download_file(
                    f"{winners_dir}/hist_model-{model_id}.h5",
                    f"{best_models_dir}/hist_model-{model_id}.h5",
                    if_not_exists=True,
                )
                common.s3_download_file(
                    f"{winners_dir}/rt_model-{model_id}.h5",
                    f"{best_models_dir}/rt_model-{model_id}.h5",
                    if_not_exists=True,
                )
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
            common.log(agent_file)
            out, err = worker.communicate()
            err2 = b""
            for line in err.split(b"\n"):
                if (
                    b"Operation was cancelled" not in line
                    and b"tensorflow/" not in line
                    and b"rebuild TensorFlow" not in line
                ):
                    err2 += line + b"\n"
            if len(err2) > 0:
                common.log(err2)
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
            scores.append(score)

        common.log("Scores:", scores)
        best_agent = None
        best_score = None
        for worker_i, score in enumerate(scores):
            agent = CustomAgent()
            agent.load_checkpoint(agent_files[worker_i])
            agent.wins.append(0)
            agent.save_checkpoint()
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_agent = agent_files[worker_i]
        if best_agent is not None:
            agent = CustomAgent()
            agent.load_checkpoint(best_agent)
            agent.wins[-1] = 1
            agent.save_checkpoint()
        os.makedirs(winners_dir, exist_ok=True)
        os.makedirs(archive_dir, exist_ok=True)
        for file in glob.iglob(best_models_dir + "/agent*.dat"):
            if (file in winners) or (
                best_agent is not None and best_score > 0 and best_agent == file
            ):
                model_id = common.model_id_from_filename(file)
                for file in glob.iglob(f"{best_models_dir}/*{model_id}*"):
                    file2 = file.replace(best_models_dir + "/", winners_dir + "/")
                    common.log(file, "->", file2)
                    shutil.move(file, file2)
                    common.s3_upload_file(file2)
                    common.s3_delete_file(file)
        files = glob.glob(best_models_dir + "/*.dat")
        files.extend(glob.glob(best_models_dir + "/*.h5"))
        for file in files:
            file2 = file.replace(best_models_dir + "/", archive_dir + "/")
            common.log(file, "->", file2)
            shutil.move(file, file2)
            common.s3_upload_file(file2)
            common.s3_delete_file(file)
    else:
        if worker_id is not None:
            agent_file = agent_file_worker.replace("*", worker_id)

        env_config = {}

        if model is not None:
            env_config["validate_max_steps"] = 500000
            agent_file = model

        agent = CustomAgent(
            env=StocksRTSimulator, env_config=env_config, worker_id=worker_id
        )
        if os.path.isfile(agent_file):
            agent.load_checkpoint(agent_file)
        score = agent.evaluate(find_best=(worker_id is None and model is None))
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
            common.log("model:", model)
        elif arg.startswith("--master"):
            master = True

    main(worker_id, model, master)
