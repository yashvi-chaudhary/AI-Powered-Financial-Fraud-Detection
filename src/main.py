# ============================================
# AI-Powered Financial Fraud Detection System
# ============================================

# Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
import joblib

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Print current working directory

print("Current Working Directory:")
print(os.getcwd())

# Show files in project folder

print("\nFiles in current folder:")
print(os.listdir())

# Show files inside dataset folder

print("\nFiles in dataset folder:")
print(os.listdir("dataset"))

# Load dataset

df = pd.read_csv("dataset/creditcard.csv")

print("\nDataset Loaded Successfully!\n")

# Display first 5 rows

print(df.head())

# Display dataset shape

print("\nDataset Shape:")
print(df.shape)

# Display column names

print("\nColumn Names:")
print(df.columns)

# Display dataset information

print("\nDataset Information:")
df.info()

# Check for missing values

print("\nMissing Values:")
print(df.isnull().sum())

# Display statistical summary

print("\nStatistical Summary:")
print(df.describe())

# ============================================
# Check duplicate values
# ============================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ============================================
# Check class distribution
# ============================================

print("\nClass Distribution:")
print(df["Class"].value_counts())

# ============================================
# Remove duplicate rows
# ============================================

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

print("\nDuplicate Rows After Cleaning:")
print(df.duplicated().sum())

# ============================================
# Class Distribution After Cleaning
# ============================================

print("\nClass Distribution After Cleaning:")
print(df["Class"].value_counts())

print("\nClass Distribution Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)

# ============================================
# Fraud vs Genuine Transactions Bar Chart
# ============================================

import matplotlib.pyplot as plt

class_counts = df["Class"].value_counts()

plt.figure(figsize=(6, 5))
plt.bar(["Genuine", "Fraud"], class_counts.values)

plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.savefig("static/fraud_distribution.png")

plt.show()

# ============================================
# Separate Features and Target
# ============================================

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# ============================================
# Split Dataset into Training and Testing Sets
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# ============================================
# Feature Scaling
# ============================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed Successfully!")

print("\nTraining Data Shape After Scaling:")
print(X_train.shape)

print("\nTesting Data Shape After Scaling:")
print(X_test.shape)

# ============================================
# Train Logistic Regression Model
# ============================================

model = LogisticRegression(random_state=42)

model.fit(X_train, y_train)

print("\n🎉 Logistic Regression Model Trained Successfully!")

# ============================================
# Make Predictions
# ============================================

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)

# ============================================
# Model Evaluation
# ============================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance")
print("-------------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

# ============================================
# Confusion Matrix
# ============================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ============================================
# Random Forest Classifier
# ============================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("\n🌲 Random Forest Model Trained Successfully!")

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\nRandom Forest Performance")
print("-------------------------")
print(f"Accuracy : {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall   : {rf_recall:.4f}")
print(f"F1-Score : {rf_f1:.4f}")

rf_cm = confusion_matrix(y_test, rf_pred)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)

# ============================================
# Feature Importance
# ============================================

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

# ============================================
# Feature Importance Graph
# ============================================

plt.figure(figsize=(12, 8))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance - Random Forest")

plt.gca().invert_yaxis()

plt.savefig("static/feature_importance.png")

plt.show()

# ============================================
# Save the Trained Random Forest Model
# ============================================

joblib.dump(rf_model, "models/fraud_detection_model.pkl")

# Save Scaler

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("✅ Scaler Saved Successfully!")

print("\n✅ Model Saved Successfully!")
print("Model Location: models/fraud_detection_model.pkl")

# -------------------------------
# Decision Tree Classifier
# -------------------------------

decision_tree = DecisionTreeClassifier(random_state=42)

decision_tree.fit(X_train, y_train)

print("\n🌳 Decision Tree Model Trained Successfully!")

# Make Predictions

dt_predictions = decision_tree.predict(X_test)

print("\nFirst 10 Decision Tree Predictions:")
print(dt_predictions[:10])

print("\nDecision Tree Performance")
print("-------------------------")

print("Accuracy :", round(accuracy_score(y_test, dt_predictions), 4))
print("Precision:", round(precision_score(y_test, dt_predictions), 4))
print("Recall   :", round(recall_score(y_test, dt_predictions), 4))
print("F1-Score :", round(f1_score(y_test, dt_predictions), 4))

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_predictions))

# ============================================
# Save Decision Tree Model
# ============================================

joblib.dump(decision_tree, "models/decision_tree_model.pkl")

print("\n✅ Decision Tree Model Saved Successfully!")
print("Model Location: models/decision_tree_model.pkl")

# ============================================
# XGBoost Classifier
# ============================================

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

print("\n🚀 XGBoost Model Trained Successfully!")

# Make Predictions

xgb_pred = xgb_model.predict(X_test)

print("\nFirst 10 XGBoost Predictions:")
print(xgb_pred[:10])

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)

print("\nXGBoost Performance")
print("-------------------------")
print(f"Accuracy : {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall   : {xgb_recall:.4f}")
print(f"F1-Score : {xgb_f1:.4f}")

xgb_cm = confusion_matrix(y_test, xgb_pred)

print("\nXGBoost Confusion Matrix:")
print(xgb_cm)

# ============================================
# XGBoost Feature Importance
# ============================================

xgb_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
})

xgb_importance = xgb_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nXGBoost Feature Importance:")
print(xgb_importance)

plt.figure(figsize=(12,8))

plt.barh(
    xgb_importance["Feature"],
    xgb_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance - XGBoost")

plt.gca().invert_yaxis()

plt.show()

# ============================================
# Save XGBoost Model
# ============================================

joblib.dump(xgb_model, "models/xgboost_model.pkl")

print("\n✅ XGBoost Model Saved Successfully!")
print("Model Location: models/xgboost_model.pkl")


# ============================================
# Model Comparison Graph
# ============================================

algorithms = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost"
]

accuracies = [
    accuracy,
    round(dt_predictions is not None and accuracy_score(y_test, dt_predictions), 4),
    rf_accuracy,
    xgb_accuracy
]

plt.figure(figsize=(10, 6))

plt.bar(algorithms, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")

for i, value in enumerate(accuracies):
    plt.text(i, value, f"{value:.4f}", ha='center', va='bottom')

    plt.savefig("static/model_comparison.png")

plt.show()