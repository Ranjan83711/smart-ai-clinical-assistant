import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("processed_cleveland.csv")

# ---------------------------
# Convert target
# num: 0 = no disease, 1-4 = disease
# ---------------------------
df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
df.drop(columns=["num"], inplace=True)

# ---------------------------
# Handle missing values
# (dataset sometimes has '?')
# ---------------------------
df.replace("?", np.nan, inplace=True)
df = df.apply(pd.to_numeric)
df.fillna(df.median(), inplace=True)

# ---------------------------
# Split features / labels
# ---------------------------
X = df.drop("target", axis=1)
y = df["target"]

# ---------------------------
# Train test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Scaling
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# Model (interpretable)
# ---------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# ---------------------------
# Evaluation
# ---------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:,1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nReport:\n", classification_report(y_test, y_pred))

# ---------------------------
# Save artifacts
# ---------------------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully.")