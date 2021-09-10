import requests
import csv
import common


try:
    comp_list = common.load_comp_list()
    companies = {}
    for company in comp_list:
        companies[company["conid"]] = company
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
        companies[item["conid"]] = {
            "symbol": item["symbol"],
            "conid": item["con_id"],
            "conidex": item["conidex"],
            "exchange": item["listing_exchange"]
        }

    print("size:", len(companies))

companies = list(companies.values())

with open(common.company_list_filename, "w") as f:
    writer = csv.DictWriter(f, fieldnames=list(companies[0]))
    writer.writeheader()
    writer.writerows(companies)
