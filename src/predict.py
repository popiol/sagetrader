import sagemaker
import boto3
import json
import os

s3 = boto3.resource("s3")
sess = sagemaker.Session()
bucket_name = sess.default_bucket()
bucket = s3.Bucket(bucket_name)

data = bucket.download_file("train.jsonl", "tmp.jsonl")
recs = []
with open("tmp.jsonl", "r") as f:
    for line in f:
        recs.append(json.loads(line))
os.remove("tmp.jsonl")

endpoint_name = "deepar-quotes"

predictor = sagemaker.predictor.Predictor(
    endpoint_name=endpoint_name, serializer=sagemaker.serializers.JSONSerializer()
)

outputs = predictor.predict({"instances": recs})
bucket.put_object(Key="output/train.jsonl.out", Body=outputs)
