import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

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

model = RandomForestClassifier(n_estimators=160, random_state=42)

pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

pipe.fit(X_train, y_train)

pickle.dump(pipe, open("fraud_model.pkl", "wb"))
print("MODEL SAVED ✔ fraud_model.pkl")
