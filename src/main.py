# ============================================
# AI-Powered Financial Fraud Detection System
# ============================================

# Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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