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

predictor.delete_endpoint()
