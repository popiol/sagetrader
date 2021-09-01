import boto3
import sagemaker
import os
from botocore.exceptions import ClientError


s3 = boto3.resource("s3")
sess = sagemaker.Session()
bucket_name = sess.default_bucket()
bucket = s3.Bucket(bucket_name)
region = sess.boto_region_name

def s3_upload_file(filename, obj_key=None):
    if obj_key is None:
        obj_key = filename
    bucket.upload_file(filename, obj_key)

def s3_download_file(obj_key, filename=None, if_not_exists=False, fail_on_missing=False):
    if filename is None:
        filename = obj_key
    if if_not_exists and os.path.isfile(filename):
        return
    try:
        with open(filename, 'wb') as f:
            bucket.download_fileobj(obj_key, f)
    except ClientError:
        if fail_on_missing:
            raise
