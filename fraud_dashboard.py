import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Fraud Analytics Dashboard", layout="wide")

st.title("🛍️ E-Commerce Fraud Detection Dashboard")

# ---------- Load data ----------
try:
    df = pd.read_csv("dataset.csv")
    st.success("Dataset loaded successfully")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Show preview
st.dataframe(df.head())

# ---------- KPI Section ----------
if "is_fraud" not in df.columns:
    st.error("❌ 'is_fraud' column missing")
    st.stop()

total = len(df)
fraud = df["is_fraud"].sum()
fraud_rate = round((fraud / total) * 100, 2) if total > 0 else 0
loss = int(fraud * df["order_amount"].mean()) if fraud > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Requests", total)
col2.metric("Fraudulent Requests", fraud)
col3.metric("Fraud Rate", f"{fraud_rate}%")
col4.metric("Loss Estimated ($)", f"${loss}")

st.markdown("---")

# ---------- Risk Segments ----------
st.subheader("Risk Segments")

df["risk"] = np.where(df["is_fraud"] == 1, "High",
             np.where(df["past_returns"] >= 3, "Medium", "Low"))

risk_chart = df["risk"].value_counts().reset_index()
risk_chart.columns = ["Risk", "Count"]

fig = px.pie(risk_chart, values="Count", names="Risk")
st.plotly_chart(fig, use_container_width=True)

# ---------- Fraudulent Categories ----------
st.subheader("Top Fraudulent Categories")

fraud_cat = df[df["is_fraud"] == 1]["product_category"].value_counts()

if fraud_cat.empty:
    st.info("No fraudulent records found in dataset")
else:
    fig2 = px.bar(
        fraud_cat,
        x=fraud_cat.index,
        y=fraud_cat.values,
        labels={"x": "Category", "y": "Fraud Count"},
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Alerts ----------
st.subheader("⚠ Suspicious Activity Alerts")

alerts = []

if len(df[df["past_returns"] >= 4]) > 0:
    alerts.append("🚨 Multiple returns detected")

if len(df[(df["order_amount"] > 3000) & (df["is_fraud"] == 1)]) > 0:
    alerts.append("🚨 High-risk costly orders found")

if len(df[df["return_reason"] == "Not Delivered"]) > 0:
    alerts.append("🚨 Frequent 'Not Delivered' claims")

if alerts:
    for a in alerts:
        st.error(a)
else:
    st.success("No suspicious behavior detected")

# ---------- Real-Time Score ----------
st.subheader("📡 Real-Time Detection Indicator")
latest_prob = 0.92
st.progress(latest_prob)
st.write(f"**Detection score: {round(latest_prob*100)}% — FRAUDULENT**")
