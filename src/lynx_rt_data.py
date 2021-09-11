import asyncio
import websockets
import requests
import json
import common
import datetime
import lynx_common
import time


def main():
    session_id = lynx_common.get_session_id()

    async def handler(companies):
        quotes = []
        timestamp0 = time.time()
        timestamp1 = timestamp0

        async with websockets.connect(lynx_common.ws_url) as websocket:
            await lynx_common.send_session_id(websocket, session_id)

            fields = ["31", "84", "85", "86", "87", "88"]
            topic_to_company = {}
            for company in companies:
                conidex = company["conidex"]
                params = json.dumps({"fields": fields}, separators=(",", ":"))
                topic = f"smd+{conidex}+{params}"
                topic_base = f"smd+{conidex}"
                topic_to_company[topic_base] = company
                await lynx_common.send(websocket, topic)

            finally_raise = None

            async for msg in websocket:
                print("<", msg)
                resp = json.loads(msg)
                if resp.get("topic", "").startswith("smd+"):
                    topic_base = "+".join(resp["topic"].split("+")[:2])
                    company = topic_to_company[topic_base]
                    conidex = resp["conid"]
                    row = {}
                    for field in fields:
                        row[field] = resp[field] if field in resp else None
                    if not [x for x in row if x is not None]:
                        raise common.SaveDataError(
                            f"Missing data for conidex {conidex}"
                        )
                    row["company"] = company["symbol"]
                    row["conidex"] = conidex
                    row["t"] = resp["_updated"]
                    quotes.append(row)
                elif "error" in resp:
                    error = resp["error"]
                    errorcode = resp["code"]
                    finally_raise = common.PullDataError(conidex, errorcode, error)
                    break

                timestamp2 = time.time()
                if timestamp2 - timestamp0 >= 500:
                    break
                elif timestamp2 - timestamp1 >= 60 and quotes:
                    dt = datetime.datetime.fromtimestamp(row["t"] / 1000)
                    day = dt.strftime("%Y%m%d")
                    try:
                        common.save_rt_quotes(day, quotes)
                    except common.SaveDataError as e:
                        finally_raise = e
                    timestamp1 = time.time()
                    quotes = []

            for company in companies:
                conidex = company["conidex"]
                await lynx_common.send(websocket, f"umd+{conidex}"+"+{}")

            if finally_raise:
                raise finally_raise

    companies = common.get_watchlist()
    asyncio.run(handler(companies))


if __name__ == "__main__":
    main()
