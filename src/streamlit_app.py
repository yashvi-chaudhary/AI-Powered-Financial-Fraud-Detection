import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load model and scaler

model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")


# Page configuration

st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="centered"
)


st.title("💳 AI-Powered Financial Fraud Detection System")

st.write(
    "Enter transaction details to predict whether the transaction is Fraud or Genuine."
)


st.subheader("Transaction Features")


# Input features

features = []

for i in range(30):

    value = st.number_input(
        f"Feature {i+1}",
        value=0.0
    )

    features.append(value)



if st.button("Predict Transaction"):

    input_data = np.array(features).reshape(1,-1)


    # Scaling

    input_scaled = scaler.transform(input_data)


    prediction = model.predict(input_scaled)


    if prediction[0] == 1:

        st.error(
            "⚠️ Fraud Transaction Detected"
        )

    else:

        st.success(
            "✅ Genuine Transaction"
        )