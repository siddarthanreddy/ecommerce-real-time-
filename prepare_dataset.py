import pandas as pd

df = pd.read_csv("dataset.csv")

# ---- AUTO MAP COLUMNS ----
mapping = {}

for col in df.columns:
    c = col.lower()

    if "amount" in c or "price" in c or "value" in c or "purchase" in c:
        mapping[col] = "order_amount"

    elif "category" in c:
        mapping[col] = "product_category"

    elif "payment" in c:
        mapping[col] = "payment_method"

    elif "reason" in c:
        mapping[col] = "return_reason"

    elif "return" in c or "refund" in c:
        mapping[col] = "past_returns"

    elif "delay" in c or "ship" in c or "delivery" in c:
        mapping[col] = "delivery_delay_days"

    elif "fraud" in c or "label" in c or "target" in c:
        mapping[col] = "is_fraud"

df = df.rename(columns=mapping)

# ---- FORCE REQUIRED COLUMNS ----
required = [
    "order_amount",
    "product_category",
    "payment_method",
    "return_reason",
    "past_returns",
    "delivery_delay_days",
    "refund_type",
    "is_fraud"
]

for col in required:
    if col not in df.columns:

        # sensible defaults
        if col == "refund_type":
            df[col] = "Instant"

        elif col == "return_reason":
            df[col] = "Unknown"

        elif col == "is_fraud":
            df[col] = 0

        else:
            df[col] = 0

df = df[required]

df.to_csv("dataset_clean.csv", index=False)

print("✔ CLEAN DATA READY")
print(df.head())
