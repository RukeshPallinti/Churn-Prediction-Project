#model training
import os
import joblib
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# load env
from dotenv import load_dotenv
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

# 1. Read data
client = MongoClient(MONGO_URI)
db = client['churndb']
df = pd.DataFrame(list(db.customers.find()))
df = df.drop(columns=['_id'], errors='ignore')

# 2. Basic cleaning
df['Churn'] = df['Churn'].map({'No':0, 'Yes':1})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 3. Features & target
target = 'Churn'
drop_cols = ['customerID']  # or any identifier
X = df.drop(columns=[target] + drop_cols, errors='ignore')
y = df[target].astype(int)

# 4. Define column lists
numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
# sometimes numeric may be strings
numeric_features = [c for c in numeric_features if c not in ['SeniorCitizen']]  # check
categorical_features = X.select_dtypes(include=['object','bool']).columns.tolist()

# 5. Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 6. Model pipelines to compare
pipelines = {
    'logreg': Pipeline([('preproc', preprocessor), ('clf', LogisticRegression(max_iter=1000))]),
    'rf': Pipeline([('preproc', preprocessor), ('clf', RandomForestClassifier(random_state=42))]),
    'gb': Pipeline([('preproc', preprocessor), ('clf', GradientBoostingClassifier(random_state=42))])
}

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# quick baseline training & evaluation
results = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    print(f"--- MODEL: {name} ---")
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))
    results[name] = {'pipeline': pipe, 'roc': roc_auc_score(y_test, probs)}

# Choose best (by ROC AUC)
best_name = max(results, key=lambda k: results[k]['roc'])
best_pipeline = results[best_name]['pipeline']
print("Best model:", best_name)

# Save pipeline
os.makedirs("models", exist_ok=True)
joblib.dump(best_pipeline, "models/churn_pipeline.joblib")
print("Saved model to models/churn_pipeline.joblib")

# Model Selection & High Parameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
  'clf__n_estimators': [100, 200],
  'clf__max_depth': [6, 10]
}
grid = GridSearchCV(pipelines['rf'], param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
best = grid.best_estimator_
joblib.dump(best, "models/churn_pipeline.joblib")
