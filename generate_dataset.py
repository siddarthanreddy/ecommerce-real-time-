import pandas as pd
import random

categories = ["Electronics","Clothing","Footwear","Books","Cosmetics"]
reasons = ["Wrong Size","Item Damaged","Not Delivered","Used then returned","Wrong Product","No reason"]
payments = ["UPI","Card","COD"]
refunds = ["Instant","Post"]

rows = []

for i in range(50000):
    amount = random.randint(300,7000)
    past = random.randint(0,8)
    delay = random.randint(0,5)

    fraud = 1 if (amount>3500 and past>3) or random.random() < 0.15 else 0

    rows.append([
        amount,
        random.choice(categories),
        random.choice(payments),
        random.choice(reasons),
        past,
        delay,
        random.choice(refunds),
        fraud
    ])

df = pd.DataFrame(rows, columns=[
    "order_amount","product_category","payment_method","return_reason",
    "past_returns","delivery_delay_days","refund_type","is_fraud"
])

df.to_csv("dataset.csv", index=False)
print("Dataset created ✔")
