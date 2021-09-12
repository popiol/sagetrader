import asyncio
import websockets
import requests
import json
import common
import datetime


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

def main(companies, handler, start_conid=None):    
    n_failures = 0
    skip = start_conid is not None
    for company in companies:
        common.log("Start company", company["symbol"], company["conid"])
        if skip and start_conid == company["conid"]:
            skip = False
        if skip:
            continue
        try:
            asyncio.run(handler(company))
            n_failures = 0
        except (common.PullDataError, common.SaveDataError):
            n_failures += 1
            if n_failures >= 5:
                raise
        common.log("Finish company", company["symbol"], company["conid"])
