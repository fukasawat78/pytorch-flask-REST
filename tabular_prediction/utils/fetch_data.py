import requests
import json
import pandas as pd
import os

def fetch_data():
    
    fpath = "./input/raw.csv"
    
    if os.path.exists(fpath):
        print("The requested file already exists")
    else:
        url = "https://www.land.mlit.go.jp/webland/api/TradeListSearch"

        payload = {"area": 13, "from": 20053, "to": 20193}
        response = requests.get(url, params=payload)

        data = json.loads(response.text)
        df = pd.DataFrame(data["data"])

        os.mkdir("../input")
        df.to_csv("../input/raw.csv", index=False)