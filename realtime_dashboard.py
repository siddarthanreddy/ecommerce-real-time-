import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import time
import random

st.set_page_config(page_title="Real-Time Fraud Dashboard", layout="wide")

API_URL = "http://127.0.0.1:5000/predict"

# Load base dataset
df = pd.read_csv("dataset.csv")

st.title("📡 Real-Time Refund Fraud Monitoring")

placeholder = st.empty()

while True:

    # pick random row to simulate new transaction
    row = df.sample(1).to_dict(orient="records")[0]

    res = requests.post(API_URL, json=row)
    result = res.json()

    prob = result["fraud_probability"]
    decision = result["decision"]

    row["probability"] = prob
    row["decision"] = decision

    df = pd.concat([df, pd.DataFrame([row])])

    with placeholder.container():

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Requests", len(df))
        col2.metric("Fraud Flags", len(df[df["decision"]=="HIGH RISK - BLOCK"]))
        col3.metric("Review Queue", len(df[df["decision"]=="REVIEW REQUIRED"]))

        st.markdown("---")

        st.subheader("🔔 Latest Event")
        st.json(row)

        st.subheader("Fraud Trend")
        line = px.line(df.tail(50), y="probability")
        st.plotly_chart(line, use_container_width=True)

    time.sleep(3)
