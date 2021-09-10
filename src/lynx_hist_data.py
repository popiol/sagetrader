import websockets
import json
import common
import datetime
import lynx_common


def main(period="2d", append=True, start_conid=None):
    session_id = lynx_common.get_session_id()

    async def handler(company):
        conid = company["conid"]
        
        async with websockets.connect(lynx_common.ws_url) as websocket:
            await lynx_common.send_session_id(websocket, session_id)

            params = json.dumps({"period": period, "bar": "1d"})
            await lynx_common.send(websocket, f"smh+{conid}+{params}")
            finally_raise = None

            async for msg in websocket:
                print("<", msg)
                resp = json.loads(msg)
                if "symbol" in resp:
                    server_id = resp["serverId"]
                    data = resp["data"]
                    quotes = {}
                    if not data:
                        finally_raise = common.SaveDataError(f"Missing data for conid {conid}")
                        break
                    
                    for row in data:
                        row["company"] = company["symbol"]
                        row["conid"] = conid
                        dt = datetime.datetime.fromtimestamp(row["t"]/1000)
                        month = dt.strftime("%Y%m")
                        if month not in quotes:
                            quotes[month] = []
                        quotes[month].append(row)
                    try:
                        for month in quotes:
                            common.save_hist_quotes(conid, month, quotes[month], append)
                    except common.SaveDataError as e:
                        finally_raise = e
                    break
                elif "error" in resp:
                    error = resp["error"]
                    errorcode = resp["code"]
                    raise common.PullDataError(conid, errorcode, error)

            await lynx_common.send(websocket, f"umh+{server_id}")

            if finally_raise:
                raise finally_raise

    companies = common.load_comp_list()
    lynx_common.main(companies, handler, start_conid)

if __name__ == "__main__":
    main()
    