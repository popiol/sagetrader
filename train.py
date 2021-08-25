import sagemaker
import boto3
import pandas as pd
import os

s3 = boto3.resource("s3")
sess = sagemaker.Session()
bucket_name2 = sess.default_bucket()
bucket2 = s3.Bucket(bucket_name2)
role = "arn:aws:iam::278088188282:role/service-role/AmazonSageMaker-ExecutionRole-20210714T012499"
region = sess.boto_region_name
bucket_name = "popiol.daytrader-master-quotes"
bucket = s3.Bucket(bucket_name)
files = [x.key for x in bucket.objects.filter(Prefix="csv_clean/date=202105")]
print(files[:5])
quotes = None

for file in files[:400]:
    quotes1 = pd.read_csv(bucket.Object(file).get()["Body"])
    quotes = quotes1 if quotes is None else pd.concat([quotes, quotes1])

quotes2 = quotes.rename(columns={"quote_dt": "start"}).drop(
    columns=["low_price", "high_price", "row_id"]
)
grouped = quotes2.groupby("comp_code")
start = grouped["start"].min()
target = grouped["price"].agg(lambda x: x.tolist())
train = pd.DataFrame({"start": start, "target": target})
print(train.head())

train_file_key = "train.jsonl"
train_file = f"s3://{bucket_name2}/{train_file_key}"
with open("tmp.jsonl", "w") as f:
    train.to_json(f, orient="records", lines=True)
bucket2.upload_file("tmp.jsonl", Key=train_file_key)
os.remove("tmp.jsonl")
# train_file_key = "train.parquet"
# train_file = f"s3://{bucket_name2}/{train_file_key}"
# with open("tmp.parquet", "bw") as f:
#    train.to_parquet(f)
# bucket2.upload_file("tmp.parquet", Key=train_file_key)
# os.remove("tmp.parquet")

deepar_img = sagemaker.image_uris.retrieve("forecasting-deepar", region)

estimator = sagemaker.estimator.Estimator(
    sagemaker_session=sess,
    image_uri=deepar_img,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    base_job_name="deepar",
    output_path=f"s3://{bucket_name2}/model",
)

hyperparameters = {
    "time_freq": "H",
    "context_length": "20",
    "prediction_length": "10",
    "epochs": "20",
}
estimator.set_hyperparameters(**hyperparameters)

print("Fitting...")

estimator.fit(train_file)

print("Model created")
