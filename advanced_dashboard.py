import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import requests

st.set_page_config(
    page_title="Advanced Fraud Analytics Dashboard",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙ Controls")

refresh = st.sidebar.checkbox("Auto refresh (simulated real-time)")
interval = st.sidebar.slider("Refresh interval (seconds)", 2, 15, 5)

st.sidebar.markdown("---")

st.sidebar.subheader("Filters")

risk_filter = st.sidebar.multiselect(
    "Risk category",
    ["High", "Medium", "Low"],
    default=["High","Medium","Low"]
)

category_filter = st.sidebar.multiselect(
    "Product Category",
    ["Clothing","Electronics","Footwear","Books","Cosmetics"],
    default=["Clothing","Electronics","Footwear","Books","Cosmetics"]
)

st.sidebar.info("Dashboard auto refresh simulates streaming data.")


# ---------------- DATA LOAD ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_clean.csv")

    required_cols = [
        "order_amount",
        "product_category",
        "payment_method",
        "return_reason",
        "past_returns",
        "delivery_delay_days",
        "refund_type",
        "is_fraud"
    ]

    # guarantee required structure
    for col in required_cols:
        if col not in df.columns:
            if col in ["order_amount","past_returns","delivery_delay_days"]:
                df[col] = 0
            elif col == "is_fraud":
                df[col] = 0
            elif col == "refund_type":
                df[col] = "Instant"
            elif col == "return_reason":
                df[col] = "Unknown"
            else:
                df[col] = "Unknown"

    # numeric safety
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce").fillna(0)
    df["past_returns"] = pd.to_numeric(df["past_returns"], errors="coerce").fillna(0)
    df["delivery_delay_days"] = pd.to_numeric(df["delivery_delay_days"], errors="coerce").fillna(0)
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0)

    # simulate id if missing
    if "customer_id" not in df.columns:
        df["customer_id"] = np.random.randint(1000, 5000, len(df))

    # final risk segmentation
    df["risk"] = np.where(
        df["is_fraud"] == 1,
        "High",
        np.where(df["past_returns"] >= 3, "Medium", "Low")
    )

    return df


placeholder = st.empty()

while True:

    df = load_data()

    # apply filters
    df = df[df["risk"].isin(risk_filter)]
    df = df[df["product_category"].isin(category_filter)]

    with placeholder.container():

        st.title("🛍️ Advanced Fraud Detection Dashboard")

        # ---------- KPIs ----------
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Requests", len(df))
        col2.metric("High Risk", len(df[df["risk"]=="High"]))
        col3.metric("Medium Risk", len(df[df["risk"]=="Medium"]))
        col4.metric("Low Risk", len(df[df["risk"]=="Low"]))

        st.markdown("---")

        # ---------- CHART ROW 1 ----------
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Fraud Risk Distribution")
            pie = px.pie(
                df,
                names="risk",
                color="risk",
                color_discrete_map={"High":"red","Medium":"orange","Low":"green"}
            )
            st.plotly_chart(pie, use_container_width=True)

        with c2:
            st.subheader("Fraud Category Pattern")
            fraud_cat = df[df["is_fraud"]==1]["product_category"].value_counts()
            bar = px.bar(fraud_cat, title="Fraud by Category")
            st.plotly_chart(bar, use_container_width=True)

        with c3:
            st.subheader("Refund Reason Pattern")
            reasons = df["return_reason"].value_counts()
            reason_chart = px.bar(reasons, title="Most Common Refund Reasons")
            st.plotly_chart(reason_chart, use_container_width=True)

        st.markdown("---")

        # ---------- CHART ROW 2 ----------
        a1, a2 = st.columns(2)

        with a1:
            st.subheader("Fraud Heatmap (Returns vs Fraud)")
            heat = pd.crosstab(df["past_returns"], df["is_fraud"])
            heatmap = px.imshow(
                heat,
                text_auto=True,
                labels=dict(x="Fraud", y="Past Returns")
            )
            st.plotly_chart(heatmap, use_container_width=True)

        with a2:
            st.subheader("Order Amount vs Fraud Trend")
            amount_trend = px.box(
                df,
                x="is_fraud",
                y="order_amount",
                labels={"is_fraud":"Fraud(1) / Genuine(0)"}
            )
            st.plotly_chart(amount_trend, use_container_width=True)

        st.markdown("---")

        # ---------- REAL TIME FRAUD SCORE ----------
        st.subheader("📡 Real-Time Fraud Detection Score")

        sample = df.sample(1).to_dict(orient="records")[0]

        try:
            res = requests.post(
                "http://127.0.0.1:5000/predict",
                json=sample,
                timeout=4
            ).json()

            live_prob = res["fraud_probability"]

            st.progress(live_prob)
            st.metric("Live Fraud Score", f"{round(live_prob*100)} %")

            if live_prob > 0.75:
                st.error("🚨 HIGH RISK — Possible Fraud Activity")
            elif live_prob > 0.40:
                st.warning("⚠ Review Needed")
            else:
                st.success("✔ Looks Safe")

        except:
            st.warning("API not running — using simulated score")

            live_prob = np.random.uniform(0.05, 0.98)
            st.progress(live_prob)
            st.metric("Live Fraud Score", f"{round(live_prob*100)} %")

        st.markdown("---")

        # ---------- TOP RISKY CUSTOMERS ----------
        st.subheader("🚨 Top Risky Customers")

        risky = (
            df.groupby("customer_id")["is_fraud"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        risky_fig = px.bar(
            risky,
            x="customer_id",
            y="is_fraud",
            title="Customers with Highest Fraud Attempts"
        )

        st.plotly_chart(risky_fig, use_container_width=True)

        # ---------- RECENT RECORDS ----------
        st.subheader("📄 Recent Transactions")
        st.dataframe(df.tail(15))

    if not refresh:
        break

    time.sleep(interval)
