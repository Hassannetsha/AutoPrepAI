import os

import pandas as pd
import numpy as np
import dspy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Import your existing service
from services.feature_engineering_service import *


# ===============================
# Setup DSPy - change to your LLM
# ===============================
lm = dspy.LM(
    model="groq/llama-3.3-70b-versatile",
    api_key=("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")
)
dspy.configure(lm=lm)

# ===============================
# Load & Clean Titanic
# ===============================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
df = df.dropna().reset_index(drop=True)
target = "Survived"

# ===============================
# Evaluate Model
# ===============================
def evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds), f1_score(y_test, preds, average='weighted')

# ===============================
# 1. Baseline (without engineering)
# ===============================
X_base = pd.get_dummies(df.drop(columns=[target])).reset_index(drop=True)
y = df[target].reset_index(drop=True)

acc_base, f1_base = evaluate_model(X_base, y)
print(f"Baseline -> Acc: {acc_base:.3f}, F1: {f1_base:.3f}, Features: {X_base.shape[1]}")

# ===============================
# 2. Run Feature Engineering Agent
# ===============================
suggest = dspy.Predict(SuggestFeatures)

result = suggest(
    dataset_columns=", ".join(df.drop(columns=[target]).columns.tolist()),
    sample_rows=df.head(5).to_json(),
    top_n="5"
)

print("\n=== LLM Suggested Features ===")
print(result.suggested_features)

reviewed_features = review_features(result.suggested_features)

# ===============================
# 3. Apply Generated Features
# ===============================
fe = FeatureEngineeringService()
df_engineered, features_added = fe.engineer(df.drop(columns=[target]), reviewed_features)

# ===============================
# 4. Evaluate with Generated Features
# ===============================
X_eng = pd.get_dummies(df_engineered).reset_index(drop=True)
acc_eng, f1_eng = evaluate_model(X_eng, y)

# ===============================
# Results
# ===============================
print("\n=== Feature Engineering Ablation ===")
print(f"Without Engineering -> Acc: {acc_base:.3f}, F1: {f1_base:.3f}, Features: {X_base.shape[1]}")
print(f"With Engineering    -> Acc: {acc_eng:.3f},  F1: {f1_eng:.3f},  Features: {X_eng.shape[1]}")
print(f"Features Added: {features_added}")
print(f"F1 Delta: {f1_eng - f1_base:+.3f}")