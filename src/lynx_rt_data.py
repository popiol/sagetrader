import asyncio
import websockets
import requests
import json
import common
import datetime
import lynx_common
import time
import copy


def main():
    session_id = lynx_common.get_session_id()

    async def handler(companies):
        quotes = []
        timestamp0 = time.time()
        timestamp1 = timestamp0

        async with websockets.connect(lynx_common.ws_url) as websocket:
            await lynx_common.send_session_id(websocket, session_id)

            """
            31 - Last Price
            84 - Bid Price
            85 - Ask Size
            86 - Ask Price
            7762 - Volume
            88 - Bid Size
            """
            fields = ["31", "84", "85", "86", "7762", "88"]
            topic_to_company = {}
            for company in companies:
                conidex = company["conidex"]
                params = json.dumps({"fields": fields}, separators=(",", ":"))
                topic = f"smd+{conidex}+{params}"
                topic_base = f"smd+{conidex}"
                topic_to_company[topic_base] = company
                await lynx_common.send(websocket, topic)

            finally_raise = None
            first_row = {x["conidex"]: {} for x in companies}
            last_row = copy.deepcopy(first_row)
            row = {}

            async for msg in websocket:
                common.log("<", msg)
                resp = json.loads(msg)
                if resp.get("topic", "").startswith("smd+"):
                    topic_base = "+".join(resp["topic"].split("+")[:2])
                    company = topic_to_company[topic_base]
                    conidex = resp["conid"]
                    for field in fields:
                        if field in resp:
                            row[field] = resp[field]
                            last_row[conidex][field] = resp[field]
                            if field not in first_row:
                                first_row[conidex][field] = resp[field]
                    if len(row) >= len(fields):
                        row["company"] = company["symbol"]
                        row["conidex"] = conidex
                        row["t"] = resp["_updated"]
                        quotes.append(row)
                        row = {}
                elif "error" in resp:
                    error = resp["error"]
                    errorcode = resp["code"]
                    finally_raise = common.PullDataError(conidex, errorcode, error)
                    break

                timestamp2 = time.time()
                if timestamp2 - timestamp0 >= 36000:
                    break
                elif timestamp2 - timestamp0 >= 300 and first_row == last_row:
                    break
                elif timestamp2 - timestamp1 >= 60:
                    if quotes and first_row != last_row:
                        dt = datetime.datetime.fromtimestamp(quotes[-1]["t"] / 1000)
                        day = dt.strftime("%Y%m%d")
                        common.log("save quotes", day)
                        try:
                            common.save_rt_quotes(day, quotes)
                        except common.SaveDataError as e:
                            finally_raise = e
                        quotes = []
                        first_row = copy.deepcopy(last_row)
                    timestamp1 = time.time()                    

            for company in companies:
                conidex = company["conidex"]
                await lynx_common.send(websocket, f"umd+{conidex}"+"+{}")

            if finally_raise:
                raise finally_raise

    companies = common.get_watchlist()
    common.log("Companies:", [x["conidex"] for x in companies])
    asyncio.run(handler(companies))


if __name__ == "__main__":
    if common.already_running():
        common.log("Already running")
        exit()
    
    if common.already_finished():
        common.log("Already finished")
        exit()

    common.log("Start script")
    main()
    common.log("Finish script")
