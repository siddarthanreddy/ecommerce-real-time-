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
    "📊 Dataset Prediction & Analysis"
])


# -------------------------------------------------
# TAB 1 — PREDICT SINGLE ORDER
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

            fig = px.pie(df, names="risk", title="Fraud Risk Split")
            st.plotly_chart(fig, use_container_width=True, key=f"risk_{time.time()}")

            st.subheader("Recent Transactions")
            st.dataframe(df.tail(12))

        else:
            st.info("Waiting for live data...")

    else:
        st.info("Make predictions first — data will appear here.")

    if refresh:
        time.sleep(interval)
        st.rerun()



# -------------------------------------------------
# TAB 3 — DATASET PREDICTION + ANALYSIS
# -------------------------------------------------
with tabs[2]:

    st.subheader("📊 Predict Fraud on Existing Dataset")

    candidate_files = [
        "dataset.csv",
        "dataset_clean.csv",
        "ecommerce_returns_synthetic_data.csv"
    ]

    df = None
    loaded_file = None

    for f in candidate_files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                loaded_file = f
                break
            except:
                pass

    if df is None:
        st.error("❌ No dataset found. Place a CSV file in the project folder.")

    else:
        st.success(f"📄 Loaded file: **{loaded_file}**")

        st.write("### Columns detected:", list(df.columns))

        st.write("### Preview")
        st.dataframe(df.head())

        # ======================================================
        #   🔎 DATASET INSIGHTS (EDA)
        # ======================================================
        st.header("📊 Dataset Insights & Patterns")

        eda = df.copy()
        eda["Product_Price"] = pd.to_numeric(eda["Product_Price"], errors="coerce").fillna(0)

        # ---------- BASIC SUMMARY ----------
        st.subheader("📌 Basic Statistics")
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Orders", len(eda))
        col2.metric("Returned Orders", (eda["Return_Status"] == "Returned").sum())
        col3.metric("Not Returned", (eda["Return_Status"] == "Not Returned").sum())

        st.markdown("---")

        # ---------- RETURN RISK BUCKETS ----------
        st.subheader("📈 Return Risk Percentage")

        eda["risk_level"] = np.where(
            eda["Return_Status"] == "Returned",
            "High",
            np.where(eda["Days_to_Return"].fillna(0) > 20, "Medium", "Low")
        )

        risk_percent = (
            eda["risk_level"]
            .value_counts(normalize=True) * 100
        ).round(2)

        risk_chart = px.pie(
            values=risk_percent.values,
            names=risk_percent.index,
            title="Return Risk Distribution",
            color=risk_percent.index,
            color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"}
        )

        st.plotly_chart(risk_chart, use_container_width=True)

        st.markdown("---")

        # ---------- PRICE DISTRIBUTION ----------
        st.subheader("💰 Product Price Distribution")
        price_chart = px.histogram(
            eda,
            x="Product_Price",
            nbins=40,
            title="Distribution of Product Prices"
        )
        st.plotly_chart(price_chart, use_container_width=True)

        st.markdown("---")

        # ---------- CATEGORY SALES ----------
        st.subheader("🛍 Top Selling Categories")
        cat_sales = eda["Product_Category"].value_counts()
        cat_chart = px.bar(cat_sales, title="Most Purchased Categories")
        st.plotly_chart(cat_chart, use_container_width=True)

        st.markdown("---")

        # ---------- RETURNS BY CATEGORY ----------
        st.subheader("📦 Returns by Category")
        return_cat = (
            eda.groupby("Product_Category")["Return_Status"]
            .apply(lambda x: (x == "Returned").mean())
            .sort_values(ascending=False)
        )
        return_chart = px.bar(
            return_cat,
            title="Return Rate per Category",
            labels={"value": "Return Rate"}
        )
        st.plotly_chart(return_chart, use_container_width=True)

        st.markdown("---")

        # ---------- FRAUD MAPPING FOR MODEL ----------
        final_df = pd.DataFrame({
            "order_amount": df["Product_Price"].fillna(0),
            "product_category": df["Product_Category"].fillna("Unknown"),
            "payment_method": df["Payment_Method"].fillna("Unknown"),
            "return_reason": df["Return_Reason"].fillna("Unknown"),
            "past_returns": 0,
            "delivery_delay_days": df["Days_to_Return"].fillna(0),
            "refund_type": "Post"
        })

        st.success("Data mapped successfully — sending to API...")

        results = []

        for _, row in final_df.iterrows():

            payload = {
                "order_amount": float(row["order_amount"]),
                "product_category": row["product_category"],
                "payment_method": row["payment_method"],
                "return_reason": row["return_reason"],
                "past_returns": int(row["past_returns"]),
                "delivery_delay_days": float(row["delivery_delay_days"]),
                "refund_type": row["refund_type"]
            }

            try:
                res = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=5
                ).json()

                results.append(res)

            except:
                results.append({
                    "fraud_probability": None,
                    "is_fraud": None,
                    "decision": "API ERROR"
                })

        results_df = pd.DataFrame(results)
        final = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        st.write("### 📄 Predictions Added")
        st.dataframe(final.head(25))

        st.markdown("---")

        st.subheader("📌 Fraud Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Orders", len(final))
        col2.metric("Fraud Cases", int(final["is_fraud"].sum()))
        col3.metric("Fraud Rate", f"{round(final['is_fraud'].mean()*100,2)} %")

        st.markdown("---")

        st.subheader("Fraud vs Genuine")
        fraud_chart = px.pie(
            final,
            names="is_fraud",
            title="Fraud Distribution",
            color="is_fraud",
            color_discrete_map={0: "green", 1: "red"}
        )
        st.plotly_chart(fraud_chart, use_container_width=True)

        st.markdown("---")

        st.subheader("High Risk Product Categories")
        cat = final.groupby("Product_Category")["is_fraud"].sum()
        cat_chart = px.bar(cat, title="Fraud Cases by Category")
        st.plotly_chart(cat_chart, use_container_width=True)

        st.markdown("---")

        st.subheader("Return Reason Risk Pattern")
        reason_chart = px.bar(
            final.groupby("Return_Reason")["is_fraud"].sum(),
            title="Fraud by Return Reason"
        )
        st.plotly_chart(reason_chart, use_container_width=True)

        st.markdown("---")

        st.subheader("🚨 Highest Risk Orders")
        st.dataframe(
            final.sort_values("fraud_probability", ascending=False).head(20)
        )
