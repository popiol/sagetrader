import asyncio
import websockets
import requests
import json


#resp = requests.post("https://localhost:5000/v1/api/iserver/secdef/search", json={"symbol": "KGH"}, verify=False)
#print(resp.text)
#exit()

resp = requests.get("https://localhost:5000/v1/api/tickle", verify=False)
print(resp.text)

if not resp.text:
    exit("Not authenticated")

session_id = resp.json()["session"]
uri = "wss://api.ibkr.com/v1/api/ws"
quotes = []


async def send(websocket, msg):
    print(">", msg)
    await websocket.send(msg)


async def handler():
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
                    return

        await send(websocket, 'smh+13462029+{"source":"t"}')

        async for msg in websocket:
            print("<", msg)
            resp = json.loads(msg)
            if "symbol" in resp:
                quotes.append(resp)
                break

        await send(websocket, 'umh+'+quotes[-1]["serverId"])
                


asyncio.run(handler())

