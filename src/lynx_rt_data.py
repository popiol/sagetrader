import asyncio
import websockets
import requests
import json

#resp = requests.get("https://localhost:5000/v1/api/iserver/scanner/params", verify=False)
#print(resp.text)
#exit()

#resp = requests.post("https://localhost:5000/v1/api/iserver/secdef/search", json={"symbol": "AAPL"}, verify=False)
#print(resp.text)
#exit()

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
quotes = []
error = ""
errorcode = 0


async def send(websocket, msg):
    print(">", msg)
    await websocket.send(msg)


async def handler():
    global error, errorcode
    async with websockets.connect(uri) as websocket:
        req = json.dumps({"session": session_id})
        await send(websocket, req)

        async for msg in websocket:
            print("<", msg)
            resp = json.loads(msg)
            if resp.get("topic") == "sts":
                break
                if resp["args"]["authenticated"]:
                    break
                else:
                    error = "Not authenticated"
                    errorcode = 1
                    return
            elif "error" in resp:
                error = resp["error"]
                errorcode = resp["code"]
                return

        await send(websocket, 'smd+265598+{"fields":["0","1","2","3","4","5"]}')
        #await send(websocket, 'smh+265598+{"period":"4d","bar":"1d"}')
        
        async for msg in websocket:
            print("<", msg)
            resp = json.loads(msg)
            if "symbol" in resp:
                server_id = resp["serverId"]
                quotes.append(resp["data"])
                break
            elif "error" in resp:
                error = resp["error"]
                errorcode = resp["code"]
                return

        await send(websocket, 'umh+'+server_id)
                


asyncio.run(handler())
if errorcode > 0:
    print("error", errorcode, error)
