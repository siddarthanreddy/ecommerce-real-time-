import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("fraud_model.pkl","rb"))

st.title("E-Commerce Return & Refund Fraud Analytics (Real-Time)")

order_amount = st.number_input("Order Amount")
product_category = st.selectbox("Product Category",["Clothing","Electronics","Footwear","Books","Cosmetics"])
payment_method = st.selectbox("Payment Method",["UPI","Card","COD"])
return_reason = st.selectbox("Return Reason",["Wrong Size","Item Damaged","Not Delivered","Used then returned","Wrong Product","No reason"])
past_returns = st.number_input("Past Returns",0)
delivery_delay_days = st.number_input("Delivery Delay Days",0)
refund_type = st.selectbox("Refund Type",["Instant","Post"])

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

    st.write("Fraud Probability:", round(prob,3))

    if prob > 0.7:
        st.error("🚨 HIGH RISK — BLOCK")
    elif prob > 0.3:
        st.warning("⚠ REVIEW REQUIRED")
    else:
        st.success("✔ SAFE — APPROVE")
