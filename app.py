import os
from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import joblib
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Atlas connection string from .env
MONGO_URI = os.getenv("MONGO_URI")

# Path to your trained model
MODEL_PATH = os.getenv("MODEL_PATH", "src/models/churn_pipeline.joblib")

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client["churndb"]
customers_collection = db["customers"]

# Load the trained churn prediction model
churn_model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_by_id", methods=["POST"])
def predict_by_id():
    data = request.get_json()
    customer_id = data.get("customer_id")
    if not customer_id:
        return jsonify({"error": "Please provide a customer_id"}), 400

    customer_doc = customers_collection.find_one({"customer_number": str(customer_id)})
    if not customer_doc:
        return jsonify({"error": f"No customer found with ID {customer_id}"}), 404

    customer_doc.pop("_id", None)
    customer_df = pd.DataFrame([customer_doc])

    probability = churn_model.predict_proba(customer_df)[:, 1][0]
    prediction = int(probability > 0.5)

    return jsonify({
        "customer_id": customer_id,
        "churn_probability": float(probability),
        "churn_prediction": prediction
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
