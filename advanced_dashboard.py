import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
import time
import plotly.express as px

st.set_page_config(page_title="Fraud Analytics", layout="wide")

API_URL = "http://127.0.0.1:5000"
LIVE_FILE = "live_stream.json"

st.title("🛍️ E-Commerce Fraud Detection & Real-Time Monitoring")

tabs = st.tabs([
    "🔍 Predict Fraud",
    "📡 Real-Time Dashboard",
    "📊 Advanced Analytics"
])

# -------------------------------------------------
# TAB 1 — PREDICTION
# -------------------------------------------------
with tabs[0]:

    st.subheader("Enter Refund Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        order_amount = st.number_input("Order Amount", min_value=0)
        payment_method = st.selectbox("Payment Method", ["UPI", "Card", "COD"])

    with col2:
        product_category = st.selectbox(
            "Product Category",
            ["Clothing", "Electronics", "Footwear", "Books", "Cosmetics"]
        )
        refund_type = st.selectbox("Refund Type", ["Instant", "Post"])

    with col3:
        past_returns = st.number_input("Past Returns", min_value=0)
        delivery_delay_days = st.number_input("Delivery Delay Days", min_value=0)
        return_reason = st.selectbox(
            "Return Reason",
            [
                "Wrong Size", "Item Damaged", "Not Delivered",
                "Used then returned", "Wrong Product",
                "No reason", "Claimed damaged"
            ]
        )

    if st.button("Predict Now"):

        data = {
            "order_amount": order_amount,
            "product_category": product_category,
            "payment_method": payment_method,
            "return_reason": return_reason,
            "past_returns": past_returns,
            "delivery_delay_days": delivery_delay_days,
            "refund_type": refund_type
        }

        try:
            res = requests.post(f"{API_URL}/predict", json=data, timeout=5).json()

            st.success(f"Fraud Probability: {res['fraud_probability']}")
            st.write("Decision:", res["decision"])

            if os.path.exists(LIVE_FILE):
                live = json.load(open(LIVE_FILE))
            else:
                live = []

            data["is_fraud"] = res["is_fraud"]
            live.append(data)

            json.dump(live, open(LIVE_FILE, "w"))

        except Exception:
            st.error("API error — make sure fraud_api.py is running")


# -------------------------------------------------
# TAB 2 — REAL-TIME DASHBOARD
# -------------------------------------------------
with tabs[1]:

    st.subheader("Real-Time Fraud Dashboard")

    refresh = st.checkbox("Auto Refresh", value=True)
    interval = st.slider("Refresh every (seconds)", 2, 10, 4)

    placeholder = st.empty()

    if os.path.exists(LIVE_FILE):
        live = json.load(open(LIVE_FILE))
        df = pd.DataFrame(live)

        if len(df):

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Requests", len(df))
            col2.metric("Fraud Cases", int(df['is_fraud'].sum()))
            col3.metric(
                "Fraud Rate",
                f"{round(df['is_fraud'].mean()*100, 2)} %"
            )

            st.markdown("---")

            df["risk"] = np.where(df["is_fraud"] == 1, "High", "Low")

            fig = px.pie(df, names="risk")
            st.plotly_chart(fig, use_container_width=True, key=f"risk_{time.time()}")

            st.subheader("Recent Transactions")
            st.dataframe(df.tail(12))

        else:
            st.info("Waiting for live data...")

    else:
        st.info("Make predictions to generate live data")

    if refresh:
        time.sleep(interval)
        st.experimental_rerun()


# -------------------------------------------------
# TAB 3 — ADVANCED ANALYTICS
# -------------------------------------------------
with tabs[2]:

    st.subheader("Advanced Fraud Analytics Dashboard")

    # ----- SIDEBAR CONTROLS -----
    st.sidebar.title("⚙ Controls")

    refresh_adv = st.sidebar.checkbox("Auto refresh (simulated real-time)")
    interval_adv = st.sidebar.slider("Refresh interval (seconds)", 2, 15, 5)

    st.sidebar.markdown("---")

    st.sidebar.subheader("Filters")

    risk_filter = st.sidebar.multiselect(
        "Risk category",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )

    category_filter = st.sidebar.multiselect(
        "Product Category",
        ["Clothing", "Electronics", "Footwear", "Books", "Cosmetics"],
        default=["Clothing", "Electronics", "Footwear", "Books", "Cosmetics"]
    )

    @st.cache_data
    def load_data():
        df = pd.read_csv("dataset_clean.csv")

        if "customer_id" not in df.columns:
            df["customer_id"] = np.random.randint(1000, 5000, len(df))

        df["risk"] = np.where(
            df["is_fraud"] == 1,
            "High",
            np.where(df["past_returns"] >= 3, "Medium", "Low")
        )

        return df

    df = load_data()
    df = df[df["risk"].isin(risk_filter)]
    df = df[df["product_category"].isin(category_filter)]

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requests", len(df))
    col2.metric("High Risk", len(df[df["risk"]=="High"]))
    col3.metric("Medium Risk", len(df[df["risk"]=="Medium"]))
    col4.metric("Low Risk", len(df[df["risk"]=="Low"]))

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        pie = px.pie(df, names="risk")
        st.plotly_chart(pie, use_container_width=True, key=f"adv1_{time.time()}")

    with c2:
        fraud_cat = df[df["is_fraud"]==1]["product_category"].value_counts()
        bar = px.bar(fraud_cat, title="Fraud by Category")
        st.plotly_chart(bar, use_container_width=True, key=f"adv2_{time.time()}")

    with c3:
        reasons = df["return_reason"].value_counts()
        reason_chart = px.bar(reasons, title="Most Common Refund Reasons")
        st.plotly_chart(reason_chart, use_container_width=True, key=f"adv3_{time.time()}")

    st.markdown("---")

    a1, a2 = st.columns(2)

    with a1:
        heat = pd.crosstab(df["past_returns"], df["is_fraud"])
        heatmap = px.imshow(heat, text_auto=True)
        st.plotly_chart(heatmap, use_container_width=True, key=f"adv4_{time.time()}")

    with a2:
        amount_trend = px.box(df, x="is_fraud", y="order_amount")
        st.plotly_chart(amount_trend, use_container_width=True, key=f"adv5_{time.time()}")

    st.markdown("---")

    st.subheader("🚨 Top Risky Customers")

    risky = (
        df.groupby("customer_id")["is_fraud"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    risky_fig = px.bar(risky, x="customer_id", y="is_fraud")
    st.plotly_chart(risky_fig, use_container_width=True, key=f"adv6_{time.time()}")

    st.subheader("📄 Recent Transactions")
    st.dataframe(df.tail(15))

    if refresh_adv:
        time.sleep(interval_adv)
        st.experimental_rerun()
