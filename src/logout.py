import requests
import common

resp = requests.get("https://localhost:5000/v1/api/logout", verify=False)

common.log(resp.text)
