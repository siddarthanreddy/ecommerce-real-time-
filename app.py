import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
import time
import plotly.express as px


# -------------------------------------------------
# PAGE STYLE
# -------------------------------------------------
st.set_page_config(
    page_title="Fraud Analytics",
    layout="wide",
    page_icon="🚨"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(180deg,#0f172a,#020617);
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 18px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    transition: 0.3s;
}
.card:hover{
    border-color:#38bdf8;
    box-shadow:0 0 20px rgba(56,189,248,.25);
}

h1,h2,h3,h4 {
    color: #f3f4f6 !important;
}
p, span, label{
    color:#d1d5db !important;
}

.stButton button{
    background: linear-gradient(90deg,#6366f1,#38bdf8);
    border: none;
    border-radius: 999px;
    padding: 8px 22px;
    color:white;
    font-weight:600;
}
.stButton button:hover{
    filter:brightness(1.15);
}

</style>
""", unsafe_allow_html=True)



API_URL = "http://127.0.0.1:5000"
LIVE_FILE = "live_stream.json"


st.markdown(
"""
<div style="text-align:center; margin-bottom:10px">
    <h1>🚨 AI Fraud Detection Dashboard</h1>
    <p>Real-time monitoring • Refund risk • Dataset insights</p>
</div>
""", unsafe_allow_html=True)


tabs = st.tabs([
    "🔍 Predict Fraud",
    "📡 Real-Time Dashboard",
    "📊 Dataset Prediction & Analysis"
])


# -------------------------------------------------
# TAB 1 — PREDICT SINGLE ORDER
# -------------------------------------------------
with tabs[0]:

    st.markdown('<div class="card">', unsafe_allow_html=True)
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
                "Wrong Size","Item Damaged","Not Delivered",
                "Used then returned","Wrong Product",
                "No reason","Claimed damaged"
            ]
        )

    if st.button("🚀 Predict Now"):

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

        except:
            st.error("API error — make sure fraud_api.py is running")

    st.markdown('</div>', unsafe_allow_html=True)



# -------------------------------------------------
# TAB 2 — REAL-TIME DASHBOARD
# -------------------------------------------------
with tabs[1]:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Real-Time Fraud Dashboard")

    refresh = st.checkbox("Auto Refresh", value=True)
    interval = st.slider("Refresh every (seconds)", 2, 10, 4)

    if os.path.exists(LIVE_FILE):
        live = json.load(open(LIVE_FILE))
        df = pd.DataFrame(live)

        if len(df):

            col1,col2,col3 = st.columns(3)
            col1.metric("Total Requests", len(df))
            col2.metric("Fraud Cases", int(df["is_fraud"].sum()))
            col3.metric("Fraud Rate", f"{round(df['is_fraud'].mean()*100,2)} %")

            st.markdown("---")

            df["risk"] = np.where(df["is_fraud"] == 1, "High", "Low")

            fig = px.pie(
                df,
                names="risk",
                title="Fraud Risk Split",
                hole=0.4
            )
            fig.update_traces(pull=[0.1, 0])

            st.plotly_chart(fig, use_container_width=True, key=f"risk_{time.time()}")

            st.subheader("Recent Transactions")
            st.dataframe(df.tail(12))

        else:
            st.info("Waiting for live data...")

    else:
        st.info("Make predictions first")

    if refresh:
        time.sleep(interval)
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)



# -------------------------------------------------
# TAB 3 — DATASET PREDICTION + ANALYSIS
# -------------------------------------------------
with tabs[2]:

    st.markdown('<div class="card">', unsafe_allow_html=True)

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
            df = pd.read_csv(f)
            loaded_file = f
            break

    if df is None:
        st.error("No dataset found")

    else:
        st.success(f"Loaded file: {loaded_file}")

        st.dataframe(df.head())

        # ---------- INSIGHTS ----------
        eda = df.copy()
        eda["Product_Price"] = pd.to_numeric(eda["Product_Price"], errors="coerce").fillna(0)

        st.subheader("📈 Return Risk Percentage")

        eda["risk_level"] = np.where(
            eda["Return_Status"] == "Returned",
            "High",
            np.where(eda["Days_to_Return"].fillna(0) > 20, "Medium", "Low")
        )

        risk_percent = eda["risk_level"].value_counts(normalize=True) * 100

        risk_chart = px.pie(
            values=risk_percent.values,
            names=risk_percent.index,
            hole=0.45
        )

        st.plotly_chart(risk_chart, use_container_width=True)

        st.markdown("---")

        st.subheader("💰 Product Price Distribution")
        price_chart = px.histogram(eda, x="Product_Price", nbins=40)
        st.plotly_chart(price_chart, use_container_width=True)

        st.markdown("---")

        st.subheader("🛍 Top Categories")
        cat_chart = px.bar(eda["Product_Category"].value_counts())
        st.plotly_chart(cat_chart, use_container_width=True)

        st.markdown("---")

        # ---------- FRAUD PREDICTIONS ----------
        final_df = pd.DataFrame({
            "order_amount": df["Product_Price"].fillna(0),
            "product_category": df["Product_Category"].fillna("Unknown"),
            "payment_method": df["Payment_Method"].fillna("Unknown"),
            "return_reason": df["Return_Reason"].fillna("Unknown"),
            "past_returns": 0,
            "delivery_delay_days": df["Days_to_Return"].fillna(0),
            "refund_type": "Post"
        })

        results = []

        for _, row in final_df.iterrows():
            payload = dict(row)

            try:
                r = requests.post(f"{API_URL}/predict", json=payload).json()
                results.append(r)
            except:
                results.append({"fraud_probability":None,"is_fraud":None})

        results_df = pd.DataFrame(results)
        final = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        st.subheader("Fraud Summary")
        st.dataframe(final.head(25))

        fraud_chart = px.pie(final, names="is_fraud", hole=0.4)
        st.plotly_chart(fraud_chart, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
