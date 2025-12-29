import pandas as pd
import time
import requests

df = pd.read_csv("dataset.csv")

for i, row in df.iterrows():
    data = row.drop("is_fraud").to_dict()

    res = requests.post("http://127.0.0.1:5000/predict", json=data)

    print(f"REQUEST {i+1}")
    print(data)
    print("RESPONSE:", res.json())
    print("----------------------")

    time.sleep(2)   # stream every 2 seconds
