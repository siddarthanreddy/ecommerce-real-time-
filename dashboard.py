import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

st.title("🛍️ E-Commerce Return & Refund Fraud Analytics")
st.write("Enter return / refund details and get fraud risk instantly.")

# ---------- LOAD + TRAIN MODEL ----------
@st.cache_resource
def load_model():

    # load dataset from repo
    df = pd.read_csv("dataset.csv")

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    categorical = ["product_category","payment_method","return_reason","refund_type"]
    numeric = ["order_amount","past_returns","delivery_delay_days"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric)
        ]
    )

    model = Pipeline(steps=[
        ("prep", preprocess),
        ("model", RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    model.fit(X, y)
    return model

model = load_model()

# ---------- USER INPUT ----------
order_amount = st.number_input("Order Amount (₹)", min_value=0)
product_category = st.selectbox(
    "Product Category",
    ["Clothing","Electronics","Footwear","Books","Cosmetics"]
)
payment_method = st.selectbox(
    "Payment Method",
    ["UPI","Card","COD"]
)
return_reason = st.selectbox(
    "Return Reason",
    ["Wrong Size","Item Damaged","Not Delivered","Used then returned","Wrong Product","No reason","Claimed damaged"]
)
past_returns = st.number_input("Number of Past Returns", min_value=0)
delivery_delay_days = st.number_input("Delivery Delay (days)", min_value=0)
refund_type = st.selectbox(
    "Refund Type",
    ["Instant","Post"]
)

# ---------- PREDICTION ----------
if st.button("Predict Fraud"):

    df = pd.DataFrame([{
        "order_amount": order_amount,
        "product_category": product_category,
        "payment_method": payment_method,
        "return_reason": return_reason,
        "past_returns": past_returns,
        "delivery_delay_days": delivery_delay_days,
        "refund_type": refund_type
    }])

    prob = model.predict_proba(df)[0][1]

    st.write("### Fraud Probability:", round(prob, 3))

    if prob > 0.7:
        st.error("🚨 HIGH RISK — BLOCK REFUND")
    elif prob > 0.3:
        st.warning("⚠ REVIEW REQUIRED")
    else:
        st.success("✔ SAFE — APPROVE")
