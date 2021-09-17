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
            empty_row = {x["conidex"]: {} for x in companies}
            first_row = copy.deepcopy(empty_row)
            last_row = copy.deepcopy(empty_row)
            row = copy.deepcopy(empty_row)
            last_added = copy.deepcopy(empty_row)

            async for msg in websocket:
                common.log("<", msg)
                resp = json.loads(msg)
                if resp.get("topic", "").startswith("smd+"):
                    topic_base = "+".join(resp["topic"].split("+")[:2])
                    company = topic_to_company[topic_base]
                    conidex = resp["conidEx"]
                    for field in fields:
                        if field in resp:
                            row[conidex][field] = resp[field]
                            last_row[conidex][field] = resp[field]
                            if field not in first_row[conidex]:
                                first_row[conidex][field] = resp[field]
                    for conidex in row:
                        if (
                            len(row[conidex]) >= len(fields)
                            and row[conidex] != last_added[conidex]
                        ):
                            last_added[conidex] = copy.deepcopy(row[conidex])
                            row[conidex]["company"] = company["symbol"]
                            row[conidex]["conidex"] = conidex
                            row[conidex]["t"] = resp["_updated"]
                            quotes.append(row[conidex])
                            row[conidex] = {}
                elif "error" in resp:
                    error = resp["error"]
                    errorcode = resp["code"]
                    finally_raise = common.PullDataError(conidex, errorcode, error)
                    break

                timestamp2 = time.time()
                if timestamp2 - timestamp0 >= 300 and timestamp2 - timestamp1 >= 60 and first_row == last_row:
                    if datetime.datetime.now().hour < 21:
                        finally_raise = Exception(
                            "The trading session hasn't started yet"
                        )
                    common.log("No changes in 5 min")
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
                    else:
                        common.log("nothing to save")
                    timestamp1 = time.time()

            for company in companies:
                conidex = company["conidex"]
                await lynx_common.send(websocket, f"umd+{conidex}" + "+{}")

            if finally_raise:
                raise finally_raise

    companies = common.get_watchlist()
    common.log("Companies:", [x["conidex"] for x in companies])
    asyncio.run(handler(companies))


if __name__ == "__main__":
    if common.already_running():
        exit()

    if common.already_finished():
        exit()

    common.log("Start script")
    main()
    common.log("Finish script")
