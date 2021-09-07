import asyncio
import websockets
import requests
import json
import common


#resp = requests.get("https://localhost:5000/v1/api/iserver/scanner/params", verify=False)
#print(resp.text)
#exit()

# resp = requests.post("https://localhost:5000/v1/api/iserver/secdef/search", json={"symbol": "AAPL"}, verify=False)
# print(resp.text)
# exit()

resp = requests.get("https://localhost:5000/v1/api/tickle", verify=False)
print(resp.text)

if not resp.text:
    exit("Not connected")

if resp.text == "error":
    exit("Unknown error")

if not resp.json()["iserver"]["authStatus"]["authenticated"]:
    exit("Not authenticated")

session_id = resp.json()["session"]
uri = "wss://api.ibkr.com/v1/api/ws"
quotes = {}
companies = common.load_comp_list()


async def send(websocket, msg):
    print(">", msg)
    await websocket.send(msg)


async def handler(company):
    conid = company["conid"]
    
    async with websockets.connect(uri) as websocket:
        req = json.dumps({"session": session_id})
        await send(websocket, req)

        async for msg in websocket:
            print("<", msg)
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

        params = json.dumps({"period": "5d", "bar": "1d"})
        await send(websocket, f"smh+{conid}+{params}")

        async for msg in websocket:
            print("<", msg)
            resp = json.loads(msg)
            if "symbol" in resp:
                server_id = resp["serverId"]
                quotes[company["symbol"]] = resp["data"]
                break
            elif "error" in resp:
                error = resp["error"]
                errorcode = resp["code"]
                raise Exception(errorcode, error)

        await send(websocket, "umh+" + server_id)


for company in companies:
    asyncio.run(handler(company))
    break

print(quotes)
