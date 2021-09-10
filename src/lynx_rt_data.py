import asyncio
import websockets
import requests
import json
import common
import datetime
import lynx_common


def main():
    session_id = lynx_common.get_session_id()

    async def handler(companies):
        async with websockets.connect(lynx_common.ws_url) as websocket:
            await lynx_common.send_session_id(websocket, session_id)

            fields = [0, 1, 2, 3, 4, 5]
            topic_to_company = {}
            for company in companies:
                conidex = company["conidex"]
                params = json.dumps({"fields": fields})
                topic = f"smd+{conidex}+{params}"
                topic_to_company[topic] = company
                await lynx_common.send(websocket, topic)
            
            finally_raise = None

            async for msg in websocket:
                print("<", msg)
                resp = json.loads(msg)
                if resp.get("topic", "").startswith("smd+"):
                    company = topic_to_company[resp.get("topic")]
                    conidex = resp["conid"]
                    row = {}
                    for field in fields:
                        row[field] = resp[field] if field in resp else None
                    if not [x for x in row if x is not None]:
                        raise common.SaveDataError(f"Missing data for conidex {conidex}")

                    row["company"] = company["symbol"]
                    row["conidex"] = conidex
                    row["t"] = resp["_updated"]
                    dt = datetime.datetime.fromtimestamp(row["t"] / 1000)
                    day = dt.strftime("%Y%m%d")
                    try:
                        common.save_rt_quote(conidex, day, row)
                    except common.SaveDataError as e:
                        finally_raise = e
                    break
                elif "error" in resp:
                    error = resp["error"]
                    errorcode = resp["code"]
                    raise common.PullDataError(conidex, errorcode, error)

            await lynx_common.send(websocket, f"umd+{conidex}")

            if finally_raise:
                raise finally_raise

    companies = common.get_watchlist()
    asyncio.run(handler(companies))


if __name__ == "__main__":
    main()
