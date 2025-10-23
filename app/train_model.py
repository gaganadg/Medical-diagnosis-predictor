# app/train_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load data
print("🔁 Loading data...")
df = pd.read_csv("data/dataset.csv")
print(f"✅ Data shape: {df.shape}")

# Replace NaNs with 0
df.fillna(0, inplace=True)

# Extract features and target
X = df.drop("Disease", axis=1)
y = df["Disease"]

# Convert symptoms to binary (1 if present, 0 if not)
symptom_cols = X.columns
for col in symptom_cols:
    X[col] = X[col].apply(lambda x: 0 if x == 0 else 1)

# Train model
print("🔧 Training model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Evaluate
y_pred = model.predict(X)
print(f"✅ Accuracy: {accuracy_score(y, y_pred):.2f}")
print("📊 Classification Report:\n")
print(classification_report(y, y_pred))

# Save model and symptom list
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(symptom_cols.tolist(), "models/symptoms.pkl")

print("✅ Model and symptoms list saved.")
