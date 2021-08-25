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
endpoint_name = "deepar-quotes"

predictor = sagemaker.predictor.Predictor(
    endpoint_name=endpoint_name, serializer=sagemaker.serializers.JSONSerializer()
)

sm = boto3.client("sagemaker")
models = sm.list_models()["Models"]
for model in models:
    print(model.ModelName)
    # sm.delete_model(model["ModelName"])

exit()

model = sagemaker.model.Model(
    image_uri=deepar_img,
    model_data="s3://sagemaker-us-east-2-278088188282/model/deepar-2021-07-16-19-47-44-405/output/model.tar.gz",
    role=role,
)

try:
    model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=endpoint_name,
    )
except ClientError:
    predictor = sagemaker.predictor.Predictor(
        endpoint_name=endpoint_name, serializer=sagemaker.serializers.JSONSerializer()
    )
    predictor.update_endpoint(
        initial_instance_count=1, instance_type="ml.m5.large", model_name=model.name
    )
