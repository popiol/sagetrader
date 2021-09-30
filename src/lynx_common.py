import asyncio
import websockets
import requests
import json
import common
import datetime
import os
import glob
import shutil


ws_url = "wss://api.ibkr.com/v1/api/ws"

def get_session_id():
    resp = requests.get("https://localhost:5000/v1/api/tickle", verify=False)
    common.log(resp.text)

    if not resp.text:
        exit("Not connected")

    if resp.text == "error":
        exit("Unknown error")

    if not resp.json()["iserver"]["authStatus"]["authenticated"]:
        exit("Not authenticated")

    session_id = resp.json()["session"]

    resp = requests.get(
        "https://localhost:5000/v1/api/iserver/marketdata/unsubscribeall", verify=False
    )
    common.log(resp.text)

    return session_id
    

async def send(websocket, msg):
    common.log(">", msg)
    await websocket.send(msg)


async def send_session_id(websocket, session_id):
    req = json.dumps({"session": session_id})
    await send(websocket, req)

    async for msg in websocket:
        common.log("<", msg)
        resp = json.loads(msg)
        if resp.get("topic") == "sts":
            if resp["args"]["authenticated"]:
                break
            else:
                error = "Not authenticated"
                errorcode = 1
                return
        elif "error" in resp:
            error = resp["error"]
            errorcode = resp["code"]
            raise Exception(errorcode, error)

def main(companies, handler, start_conid=None, if_not_exists=False, remove_incomplete=False):
    n_failures = 0
    skip = start_conid is not None
    for company in companies:
        if skip and start_conid == company["conid"]:
            skip = False
        if skip:
            continue
        common.log("Start company", company["symbol"], company["conid"])
        path = common.hist_quotes_filename.split("{")[0] + company["conid"]
        if os.path.isdir(path):
            try:
                years = [x.split("/")[-1] for x in glob.glob(path + "/*")]
                min_path = path + "/" + min(years)
                max_path = path + "/" + max(years)
                filename = min(x.split("/")[-1] for x in glob.glob(min_path + "/*"))
                min_path = min_path + "/" + filename
                filename = max(x.split("/")[-1] for x in glob.glob(max_path + "/*"))
                max_path = max_path + "/" + filename
                first_timestamp = int(common.get_first_timestamp(min_path))
                last_timestamp = int(common.get_last_timestamp(max_path))
                dt0 = datetime.datetime.fromtimestamp(first_timestamp/1000)
                dt1 = datetime.datetime.fromtimestamp(last_timestamp/1000)
                dt2 = datetime.datetime.now()
                if dt2 - dt1 > datetime.timedelta(days=7):
                    common.log("Outdated data exists -- removing")
                    shutil.rmtree(path)
                elif dt2 - dt0 < datetime.timedelta(days=7) and remove_incomplete:
                    common.log("Incomplete data exists -- removing")
                    shutil.rmtree(path)
                elif if_not_exists:
                    common.log("Finish company", company["symbol"], company["conid"], f"- History already exists ({path})")
                    continue
            except Exception:
                common.log("Invalid data exists -- removing")
                shutil.rmtree(path)
        try:
            asyncio.run(handler(company))
            n_failures = 0
            common.log("Finish company", company["symbol"], company["conid"])
        except (common.PullDataError, common.SaveDataError) as e:
            if type(e) != common.AppendDataError:
                n_failures += 1
            common.log(type(e).__name__, e)
            if n_failures >= 5:
                raise
