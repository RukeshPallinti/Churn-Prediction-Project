
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['churndb']
customers = db['customers']

# Load data into pandas
df = pd.DataFrame(list(customers.find()))
df = df.drop(columns=["_id"])  # remove MongoDB’s _id column

# Basic info
print(df.head())
print(df.info())
print(df['Churn'].value_counts())

# Example EDA plots
# 1. Churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.savefig("src/eda_plots/churn_distribution.png")
plt.close()

# 2. Tenure distribution
plt.figure(figsize=(6,4))
sns.histplot(df["tenure"], kde=True)
plt.title("Tenure Distribution")
plt.savefig("src/eda_plots/tenure_distribution.png")
plt.close()

# 3. Tenure vs Churn
plt.figure(figsize=(6,4))
sns.barplot(x="tenure", y="Churn", data=df)
plt.title("Tenure vs Churn")
plt.savefig("src/eda_plots/tenure_vs_churn.png")
plt.close()

<<<<<<< HEAD
=======

>>>>>>> 4fcdd0f73eef92ad667d1bdee4695de4ee1184a2
# Load data into pandas
df = pd.DataFrame(list(customers.find()))
df = df.drop(columns=["_id"])  # remove MongoDB’s _id column
