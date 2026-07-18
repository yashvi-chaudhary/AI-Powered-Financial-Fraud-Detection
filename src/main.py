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