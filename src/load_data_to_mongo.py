import pandas as pd
from pymongo import MongoClient

# 1. Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['churndb']
customers = db['customers']

# 2. Read CSV
df = pd.read_csv("../data/Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 3. Create numeric-only version of customerID
df["customer_number"] = df["customerID"].str.extract(r"(\d+)")

# 4. Replace NaN with None (MongoDB doesn’t support NaN)
records = df.where(pd.notnull(df), None).to_dict(orient='records')

# 5. Insert into MongoDB
customers.delete_many({})   # clears existing data
customers.insert_many(records)

print(f"Inserted {len(records)} records into churndb.customers")
