import requests
import csv
import common
import lynx_common
import time


def main():
    try:
        comp_list = common.load_comp_list()
        companies = {}
        for company in comp_list:
            companies[company["conidex"]] = company
    except FileNotFoundError:
        companies = {}

    scan_types = [
        "TOP_PERC_GAIN",
        "TOP_PERC_LOSE",
        "MOST_ACTIVE",
        "HOT_BY_PRICE",
        "HOT_BY_VOLUME",
        "HIGH_DIVIDEND_YIELD_IB",
        "TOP_PRICE_RANGE",
        "HOT_BY_PRICE_RANGE",
        "TOP_VOLUME_RATE",
        "OPT_VOLUME_MOST_ACTIVE",
        "HIGH_VS_13W_HL",
        "LOW_VS_13W_HL",
        "HIGH_VS_26W_HL",
        "LOW_VS_26W_HL",
        "HIGH_VS_52W_HL",
        "LOW_VS_52W_HL",
        "SCAN_socialSentimentScore_ASC",
        "SCAN_socialSentimentScore_DESC",
        "SCAN_socialSentimentScoreChange_ASC",
        "SCAN_socialSentimentScoreChange_DESC",
    ]

    for scan_type in scan_types:
        params = {
            "instrument": "STK",
            "type": scan_type,
            "filter": [],
            "location": "STK.US.MAJOR",
        }

        resp = requests.post(
            "https://localhost:5000/v1/api/iserver/scanner/run", json=params, verify=False
        )
        resp = resp.json()

        for item in resp["contracts"]:
            companies[item["conidex"]] = {
                "symbol": item["symbol"],
                "conid": item["con_id"],
                "conidex": item["conidex"],
                "exchange": item["listing_exchange"],
            }

        common.log("size:", len(companies))

    session_id = lynx_common.get_session_id()
    min_volume = 100000

    companies2 = []
    for conid in companies:
        company = companies[conid]
        include = True
        for _ in range(6):
            resp = requests.get("https://localhost:5000/v1/api/iserver/marketdata/snapshot", params={"conids": conid, "fields": "7282"}, verify=False)
            common.log(resp.text)
            if "7282_raw" in resp.json()[0]:
                break
        if "7282_raw" in resp.json()[0]:
            volume = float(resp.json()[0]["7282_raw"])
            if volume < min_volume:
                include = False
        if include:
            companies2.append(company)

    with open(common.company_list_filename, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(companies2[0]))
        writer.writeheader()
        writer.writerows(companies2)

if __name__ == "__main__":
    main()
    