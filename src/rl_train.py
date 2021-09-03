import sagemaker
from sagemaker.rl import RLEstimator
import json
import common


with open("tfout.json", "r") as f:
    params = json.load(f)
    role = params["sagemaker_role_arn"]["value"]

with open("config.tfvars", "r") as f:
    for line in f:
        if "app_ver = " in line:
            branch = line.split("=")[1].split('"')[1]

estimator = RLEstimator(
    source_dir="src",
    entry_point="rl_train_ray.py",
    git_config={
        "repo": "https://gitlab.com/popiol/sagetrader.git",
        "branch": branch,
    },
    toolkit=sagemaker.rl.RLToolkit.RAY,
    toolkit_version=RLEstimator.RAY_LATEST_VERSION,
    framework=sagemaker.rl.RLFramework.TENSORFLOW,
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=f"s3://{common.bucket_name}/output",
    base_job_name="rltest",
    hyperparameters={
        "n_workers": 1,
        "n_iterations": 1,
        "max_steps": 10000,
    },
    environment={
        "AWS_DEFAULT_REGION": common.region,
        #"SMDEBUG_LOG_LEVEL": "warning",
    },
)

print("Fitting...")

estimator.fit()

print("Model created")
