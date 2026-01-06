from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    return "Fraud API is running. Use POST request on /predict."

# load trained model
model = pickle.load(open("fraud_model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # enforce correct column order
        columns = [
            "order_amount",
            "product_category",
            "payment_method",
            "return_reason",
            "past_returns",
            "delivery_delay_days",
            "refund_type"
        ]

        df = pd.DataFrame([[data.get(col) for col in columns]], columns=columns)

        prob = model.predict_proba(df)[0][1]
        label = int(prob > 0.5)

        decision = (
            "APPROVE"
            if prob < 0.30
            else "REVIEW REQUIRED"
            if prob < 0.70
            else "HIGH RISK - BLOCK"
        )

        return jsonify({
            "fraud_probability": round(prob, 3),
            "is_fraud": label,
            "decision": decision
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
