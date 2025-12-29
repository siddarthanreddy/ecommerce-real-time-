import pandas as pd
import numpy as np

df = pd.read_csv("dataset_clean.csv")

# numeric safety
df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce").fillna(0)
df["past_returns"] = pd.to_numeric(df["past_returns"], errors="coerce").fillna(0)
df["delivery_delay_days"] = pd.to_numeric(df["delivery_delay_days"], errors="coerce").fillna(0)

if "is_fraud" not in df.columns:
    df["is_fraud"] = 0

# -------------------------------------------------
# 1️⃣ CREATE HIGH RISK (fraud) — ~30%
# -------------------------------------------------
high_idx = df.sample(frac=0.30, random_state=1).index

df.loc[high_idx, "is_fraud"] = 1
df.loc[high_idx, "past_returns"] = np.random.randint(3, 7, len(high_idx))
df.loc[high_idx, "refund_type"] = "Instant"
df.loc[high_idx, "return_reason"] = np.random.choice(
    ["Not Delivered", "Used then returned", "Item Damaged"], len(high_idx)
)

# -------------------------------------------------
# 2️⃣ CREATE MEDIUM RISK — frequent returners, not fraud
# -------------------------------------------------
remaining = df[~df.index.isin(high_idx)]
medium_idx = remaining.sample(frac=0.30, random_state=2).index

df.loc[medium_idx, "is_fraud"] = 0
df.loc[medium_idx, "past_returns"] = np.random.randint(3, 6, len(medium_idx))
df.loc[medium_idx, "refund_type"] = "Post"

# -------------------------------------------------
# 3️⃣ LOW RISK — normal shoppers
# -------------------------------------------------
low_idx = df[~df.index.isin(high_idx.union(medium_idx))].index

df.loc[low_idx, "is_fraud"] = 0
df.loc[low_idx, "past_returns"] = np.random.randint(0, 2, len(low_idx))


# -------------------------------------------------
# FINAL RISK LABEL
# -------------------------------------------------
df["risk"] = np.where(
    df["is_fraud"] == 1,
    "High",
    np.where(df["past_returns"] >= 3, "Medium", "Low")
)

df.to_csv("dataset_clean.csv", index=False)

print("✔ Balanced risks created")
print(df["risk"].value_counts(normalize=True) * 100)
