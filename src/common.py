import boto3
import sagemaker
import os
from botocore.exceptions import ClientError
import sys
import csv
import json
import datetime


def log(*x):
    print(*x, file=sys.stderr)


with open("tfout.json", "r") as f:
    params = json.load(f)
    role = params["sagemaker_role_arn"]["value"]
    main_bucket_name = params["main_bucket"]["value"]

with open("config.tfvars", "r") as f:
    for line in f:
        if "app_ver = " in line:
            branch = line.split("=")[1].split('"')[1]

s3 = boto3.resource("s3")
sess = sagemaker.Session()
default_bucket_name = sess.default_bucket()
default_bucket = s3.Bucket(default_bucket_name)
region = sess.boto_region_name
main_bucket = s3.Bucket(main_bucket_name)

def s3_upload_file(filename, obj_key=None):
    if obj_key is None:
        obj_key = filename
    main_bucket.upload_file(filename, obj_key)

def s3_download_file(obj_key, filename=None, if_not_exists=False, fail_on_missing=False):
    if filename is None:
        filename = obj_key
    if if_not_exists and os.path.isfile(filename):
        return
    try:
        main_bucket.Object(obj_key).load()
        with open(filename, 'wb') as f:
            main_bucket.download_fileobj(obj_key, f)
    except ClientError:
        if fail_on_missing:
            raise


company_list_filename = "data/company_list.csv"
hist_quotes_filename = "data/hist/{}/{}/{}.csv"
rt_quotes_filename = "data/real_time/{}/{}/{}.csv"

def save_comp_list(companies):
    with open(company_list_filename, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(companies[0]))
        writer.writeheader()
        writer.writerows(companies)
    s3_upload_file(company_list_filename)

def load_comp_list():
    s3_download_file(company_list_filename, if_not_exists=True, fail_on_missing=True)
    with open(company_list_filename, "r") as f:
        reader = csv.DictReader(f)
        companies = list(reader)
    return companies

def save_hist_quotes(conid, month, quotes, append):
    if not quotes:
        raise SaveDataError(f"Missing data for conid {conid}")
    first_timestamp = quotes[0]["t"]
    year = month[:4]
    month = month[4:]
    filename = hist_quotes_filename.format(conid, year, month)
    dir = os.path.dirname(filename)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    mode = "a" if append else "w"
    if append and os.path.isfile(filename):
        with open(filename, "r") as f:
            header = next(f).strip().split(",")
            last = None
            for line in f:
                if "," in line:
                    last = line
            last = last.strip().split(",")
            timestamp = last[header.index("t")]
            if timestamp >= first_timestamp:
                raise AppendDataError(conid, timestamp, first_timestamp)
    with open(filename, mode) as f:
        writer = csv.DictWriter(f, fieldnames=list(quotes[0]))
        if not append:
            writer.writeheader()
        writer.writerows(quotes)
    s3_upload_file(filename)

class PullDataError(Exception):
    pass

class SaveDataError(Exception):
    pass

class AppendDataError(SaveDataError):
    def __init__(self, conid, last_timestamp, first_timestamp):
        SaveDataError.__init__(self, "conid:", conid, "last old timestamp:", last_timestamp, "first new timestamp:", first_timestamp)
