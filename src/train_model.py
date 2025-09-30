import os
from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import joblib
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://rukesh2507_db_user:Rukesh_123@cluster0.7522su2.mongodb.net/churndb?retryWrites=true&w=majority"
)
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_pipeline.joblib")

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client['churndb']
customers_collection = db['customers']

# Load trained churn prediction model
churn_model = joblib.load(MODEL_PATH)

# Features used in training
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@app.route("/")
def home():
    return render_template("index.html")  # your existing HTML


@app.route("/predict_by_id", methods=["POST"])
def predict_by_id():
    # Get customer_id from form
    customer_id = request.form.get("customer_id", "").strip()

    if not customer_id:
        return jsonify({"error": "Please provide a customer_id"}), 400

    # Fetch customer from Atlas
    customer_doc = customers_collection.find_one({"customerID": customer_id})
    if not customer_doc:
        return jsonify({"error": f"No customer found with ID {customer_id}"}), 404

    # Remove MongoDB _id field
    customer_doc.pop("_id", None)

    # Convert to DataFrame and align columns
    customer_df = pd.DataFrame([customer_doc])
    for col in ALL_FEATURES:
        if col not in customer_df.columns:
            customer_df[col] = 0 if col in NUMERIC_FEATURES else ""
    customer_df = customer_df[ALL_FEATURES]

    # Predict churn
    probability = churn_model.predict_proba(customer_df)[:, 1][0]
    prediction = int(probability > 0.5)

    return render_template(
        "index.html",
        customer_id=customer_id,
        churn_probability=round(probability, 4),
        churn_prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
