import sagemaker
import boto3
from botocore.exceptions import ClientError

s3 = boto3.resource("s3")
sess = sagemaker.Session()
bucket_name = sess.default_bucket()
bucket = s3.Bucket(bucket_name)
role = "arn:aws:iam::278088188282:role/service-role/AmazonSageMaker-ExecutionRole-20210714T012499"
region = sess.boto_region_name
deepar_img = sagemaker.image_uris.retrieve("forecasting-deepar", region)
endpoint_name = "rltest"

predictor = estimator.deploy(
    initial_instance_count=1, instance_type=instance_type, entry_point="deploy-mxnet-coach.py"
)

predictor = sagemaker.predictor.Predictor(
    endpoint_name=endpoint_name, serializer=sagemaker.serializers.JSONSerializer()
)

model = sagemaker.model.Model(
    image_uri=deepar_img,
    model_data="s3://sagemaker-us-east-2-278088188282/model/deepar-2021-07-16-19-47-44-405/output/model.tar.gz",
    role=role,
)

try:
    predictor = sagemaker.predictor.Predictor(
        endpoint_name=endpoint_name, serializer=sagemaker.serializers.JSONSerializer()
    )
    predictor.delete_endpoint()
except:    
    pass

model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name,
)

sm = boto3.client("sagemaker")
models = sm.list_models()["Models"]
configs = sm.list_endpoint_configs()["EndpointConfigs"]
used_models = []
for config in configs:
    config = sm.describe_endpoint_config(EndpointConfigName=config["EndpointConfigName"])
    for variant in config["ProductionVariants"]:
        used_models.append(variant["ModelName"])
for model in models:
    model_name = model["ModelName"]
    if model_name not in used_models:
        print("deleting", model_name)
        sm.delete_model(ModelName=model_name)

