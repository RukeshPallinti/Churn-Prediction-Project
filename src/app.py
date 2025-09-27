
import os
from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import joblib
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_pipeline.joblib")

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client['churndb']
customers_collection = db['customers']

# Load the trained churn prediction model
churn_model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    """Render the homepage."""
    return render_template("index.html")

@app.route("/predict_by_id", methods=["POST"])
def predict_by_id():
    """
    Predict churn for a customer by their ID.
    Expects JSON: {"customer_id": "some_id"}
    """
    data = request.get_json()
    customer_id = data.get("customer_id")

    if not customer_id:
        return jsonify({"error": "Please provide a customer_id"}), 400

    # Fetch customer from MongoDB
    customer_doc = customers_collection.find_one({"customer_number": str(customer_id)})
    if not customer_doc:
        return jsonify({"error": f"No customer found with ID {customer_id}"}), 404

    # Convert Mongo document to DataFrame
    customer_doc.pop("_id", None)  # Remove MongoDB's internal ID
    customer_df = pd.DataFrame([customer_doc])

    # Predict churn probability
    probability = churn_model.predict_proba(customer_df)[:, 1][0]
    prediction = int(probability > 0.5)

    return jsonify({
        "customer_id": customer_id,
        "churn_probability": float(probability),
        "churn_prediction": prediction
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
