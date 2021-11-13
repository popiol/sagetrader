import common
import os
import glob
from custom_agent import CustomAgent
import numpy as np
import shutil
import math


def main():
    winners_dir = common.winners_dir
    processing_dir = common.processing_dir
    stable_dir = common.stable_dir
    os.makedirs(processing_dir, exist_ok=True)
    os.makedirs(stable_dir, exist_ok=True)
    files = glob.glob(processing_dir + "/*.dat")
    for file in files:
        os.remove(file)
    files = common.s3_find_objects(winners_dir + "/agent")
    for file in files:
        model_id = common.model_id_from_filename(file.key)
        common.s3_download_file(
            f"{winners_dir}/agent-{model_id}.dat",
            f"{processing_dir}/agent-{model_id}.dat",
            if_not_exists=True,
        )

    bad_losers = []
    scores = {}
    files = glob.glob(processing_dir + "/*.dat")
    for file in files:
        common.log(file)
        model_id = common.model_id_from_filename(file)
        agent = CustomAgent()
        agent.load_checkpoint(file, load_model=False)
        avg = np.average(agent.score_hist)
        std = np.std(agent.score_hist)
        min = np.min(agent.score_hist)
        max = np.max(agent.score_hist)
        n = len(agent.score_hist)
        common.log("n", n, "avg", avg, "std", std, "min", min, "max", max)
        if avg < -6000000 + 3000000 * math.log(n):
            common.log("Bad loser")
            bad_losers.append(model_id)
        elif avg > 200000 + 1000000 * math.log(n):
            common.log("Good winner")
            scores[model_id] = (avg + min) * math.log(n)
            common.log("Score:", scores[model_id])
        else:
            common.log("No action")

    if scores:
        best_score = None
        for model_id in scores:
            score = scores[model_id]
            if best_score is None or score > best_score:
                best_score = score
                best_model = model_id
        common.log("Stable winner:", best_model)
        files = glob.glob(stable_dir + "/*")
        for file in files:
            os.remove(file)
        shutil.copyfile(
            processing_dir + f"/agent-{model_id}.dat",
            stable_dir + f"/agent-{model_id}.dat",
        )
        common.s3_download_file(
            f"{winners_dir}/hist_model-{model_id}.h5",
            f"{stable_dir}/hist_model-{model_id}.h5"
        )
        common.s3_download_file(
            f"{winners_dir}/rt_model-{model_id}.h5",
            f"{stable_dir}/rt_model-{model_id}.h5"
        )
        for file in glob.iglob(stable_dir + "/*"):
            common.s3_upload_file(file)

    for model_id in bad_losers:
        common.log("delete", winners_dir + f"/agent-{model_id}.dat")
        common.s3_delete_file(winners_dir + f"/agent-{model_id}.dat")
        common.s3_delete_file(winners_dir + f"/hist_model-{model_id}.h5")
        common.s3_delete_file(winners_dir + f"/rt_model-{model_id}.h5")


if __name__ == "__main__":
    main()
