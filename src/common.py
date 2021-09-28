import boto3
import sagemaker
import os
from botocore.exceptions import ClientError
import sys
import csv
import json
import random
import logging
import logging.config
import datetime
import io
import psutil
import glob
import re


logfile_timestamp_format = "%Y%m%d%H%M%S"
log_timestamp_format = "%Y-%m-%d %H:%M:%S"
log_filename = None
prev_log_filename = None


def getLogger():
    global log_filename
    level = logging.DEBUG if "--debug" in sys.argv else logging.INFO
    name = os.path.basename(sys.argv[0]).replace(".py", "") or "console"
    formatter = logging.Formatter(
        fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt=log_timestamp_format,
    )
    logger = logging.getLogger(name)
    dt = datetime.datetime.now().strftime(logfile_timestamp_format)
    log_filename = f"logs/{name}_{dt}.log"
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    handler = logging.FileHandler(log_filename, "a")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = getLogger()


def log(*x, level=logging.INFO):
    if not x or (len(x) == 1 and not x[0]):
        return
    f = io.StringIO()
    print(*x, file=f)
    logger.log(level, f.getvalue().strip())


def log_debug(*x):
    log(*x, level=logging.DEBUG)


def log_warning(*x):
    log(*x, level=logging.WARNING)


def log_error(*x):
    log(*x, level=logging.ERROR)


class StreamToLogger:
    def __init__(self, level):
        self.level = level

    def write(self, x):
        log(x, level=self.level)


sys.stderr = StreamToLogger(logging.ERROR)


with open("tfout.json", "r") as f:
    tfout = json.load(f)
    role = tfout["sagemaker_role_arn"]["value"]
    main_bucket_name = tfout["main_bucket"]["value"]

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


def s3_download_file(
    obj_key, filename=None, if_not_exists=False, fail_on_missing=False
):
    if filename is None:
        filename = obj_key
    if if_not_exists and os.path.isfile(filename):
        return
    try:
        main_bucket.Object(obj_key).load()
        with open(filename, "wb") as f:
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


def get_last_timestamp(filename):
    try:
        with open(filename, "r") as f:
            header = next(f).strip().split(",")
            last = None
            for line in f:
                if "," in line:
                    last = line
        last = last.strip().split(",")
        timestamp = int(last[header.index("t")])
        return timestamp
    except Exception:
        return None


def get_first_timestamp(filename):
    try:
        with open(filename, "r") as f:
            header = next(f).strip().split(",")
            last = next(f).strip().split(",")
        timestamp = int(last[header.index("t")])
        return timestamp
    except Exception:
        return None


def assert_new_timestamp(filename, first_timestamp, conid):
    timestamp = get_last_timestamp(filename)
    if timestamp >= first_timestamp:
        raise AppendDataError(conid, timestamp, first_timestamp)


def save_hist_quotes(conid, month, quotes, append):
    if not quotes:
        raise SaveDataError(f"Missing data for conid {conid}")
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
        assert_new_timestamp(filename, quotes[0]["t"], conid)
    else:
        header = sorted(list(quotes[0]))
    add_header = not os.path.isfile(filename)
    with open(filename, mode) as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if add_header:
            writer.writeheader()
        writer.writerows(quotes)
    s3_upload_file(filename)


def save_rt_quotes(day, quotes):
    year = day[:4]
    month = day[4:6]
    day = day[6:]
    filename = rt_quotes_filename.format(year, month, day)
    dir = os.path.dirname(filename)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    if os.path.isfile(filename):
        assert_new_timestamp(filename, quotes[0]["t"], quotes[0]["conidex"])
        add_header = False
    else:
        add_header = True
    with open(filename, "a") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(list(quotes[0])))
        if add_header:
            writer.writeheader()
        writer.writerows(quotes)
    s3_upload_file(filename)


def find_hist_quotes(year, month):
    dir = hist_quotes_filename.split("{")[0]
    files = []
    for path in glob.iglob(dir + "/*"):
        filename = path + "/" + year + "/" + month + ".csv"
        if os.path.isfile(filename):
            files.append(filename)
    return files


def find_random_rt_quotes():
    path = rt_quotes_filename.split("{")[0]
    while not path.endswith(".csv"):
        files = glob.glob(path + "/*")
        last_file = max(files)
        files = [x for x in files if x != last_file]
        path = random.choice(files)
    return path


def get_watchlist():
    companies = load_comp_list()
    conidexs = None
    if prev_log_filename is not None:
        with open(prev_log_filename, "r") as f:
            for line in f:
                if "Companies:" in line:
                    conidexs = line.split(":")[-1].strip()
                    conidexs = json.loads(conidexs.replace("'", '"'))
                    break
    if conidexs is not None:
        watchlist = []
        for company in companies:
            if company["conidex"] in conidexs:
                watchlist.append(company)
    else:
        watchlist = random.sample(companies, 5)
    return watchlist


class PullDataError(Exception):
    pass


class SaveDataError(Exception):
    pass


class AppendDataError(SaveDataError):
    def __init__(self, conid, last_timestamp, first_timestamp):
        SaveDataError.__init__(
            self,
            "conid:",
            conid,
            "last old timestamp:",
            last_timestamp,
            "first new timestamp:",
            first_timestamp,
        )


def already_running():
    name = os.path.basename(sys.argv[0])
    if not name:
        return False
    pids = []
    for proc in psutil.process_iter():
        if proc.name() == "python3":
            for cmd in proc.cmdline():
                if name in cmd:
                    pids.append(f"{proc.pid} {proc.create_time()}")
                    if len(pids) > 1:
                        log("Already running:", pids)
                        return True
    return False


def already_finished():
    global prev_log_filename
    name = os.path.basename(sys.argv[0]).replace(".py", "")
    if not name:
        return False
    try:
        last_file = max(
            x.split("/")[-1]
            for x in glob.glob(f"logs/{name}_??????????????.log")
            if x != log_filename and os.path.getsize(x) > 10000
        )
    except ValueError:
        return False
    filename = "logs/" + last_file
    timestamp = last_file.split("_")[-1].split(".")[0]
    dt1 = datetime.datetime.strptime(timestamp, logfile_timestamp_format)
    dt2 = datetime.datetime.now()
    if dt2 - dt1 > datetime.timedelta(hours=3):
        return False
    prev_log_filename = filename
    finished = False
    with open(filename, "r") as f:
        for line in f:
            if "Finish script" in line:
                log("Already finished:", filename)
                finished = True
                break
    return finished


def row_to_datetime(row):
    return datetime.datetime.fromtimestamp(int(row["t"]) // 1000)


def price_to_float(x):
    x = re.sub(r"^[^0-9]", "", x)
    x = re.sub(r"[^0-9]$", "", x)
    x = x.replace(",", "")
    return float(x)
