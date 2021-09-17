import sagemaker
from sagemaker.rl import RLEstimator
import common


estimator = RLEstimator(
    source_dir="src",
    entry_point="rl_train_ray.py",
    git_config={
        "repo": "https://gitlab.com/popiol/sagetrader.git",
        "branch": common.branch,
    },
    toolkit=sagemaker.rl.RLToolkit.RAY,
    toolkit_version=RLEstimator.RAY_LATEST_VERSION,
    framework=sagemaker.rl.RLFramework.TENSORFLOW,
    role=common.role,
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=f"s3://{common.default_bucket_name}/output",
    base_job_name="rltest",
    hyperparameters={
        "n_workers": 1,
        "n_iterations": 10,
        "train_max_steps": 10000,
        "validate_max_steps": 100000,
    },
    environment={
        "AWS_DEFAULT_REGION": common.region,
        #"SMDEBUG_LOG_LEVEL": "warning",
    },
)

common.log("Fitting...")

estimator.fit()

common.log("Model created")
