import sagemaker
import boto3
import pandas as pd
import json

s3 = boto3.resource("s3")
sess = sagemaker.Session()
bucket_name = sess.default_bucket()
bucket = s3.Bucket(bucket_name)

role = "arn:aws:iam::278088188282:role/service-role/AmazonSageMaker-ExecutionRole-20210714T012499"

region = sess.boto_region_name

deepar_img = sagemaker.image_uris.retrieve("forecasting-deepar", region)

data = bucket.download_file("train.jsonl", "tmp.jsonl")
recs = []
with open("tmp.jsonl", "r") as f:
    for line in f:
        recs.append(json.loads(line))

endpoint_name = "deepar-quotes"
model = sagemaker.model.Model(
    image_uri=deepar_img, 
    model_data="s3://sagemaker-us-east-2-278088188282/model/deepar-2021-07-16-19-47-44-405/output/model.tar.gz",
    role=role
)

model.deploy(initial_instance_count=1, instance_type="ml.m5.large", endpoint_name=endpoint_name)

predictor = sagemaker.predictor.Predictor(
    endpoint_name=endpoint_name,
    serializer=sagemaker.serializers.JSONSerializer()
)

outputs = predictor.predict({"instances": recs})
bucket.put_object(Key="output/train.jsonl.out", Body=outputs)

#predictor.delete_endpoint()
#sm = boto3.client("sagemaker")
#models = sm.list_models()["Models"]
#for model in models:
#    sm.delete_model(model["ModelName"])

