import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI-Powered Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:
    st.title("🛡 Fraud Shield")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔍 Prediction",
            "📊 Dashboard",
            "📈 Analytics",
            "📂 Batch Prediction",
            "🤖 Model Comparison",
            "ℹ About"
        ]
    )

    st.markdown("---")

    st.success("🟢 System Online")
    st.info("Model : XGBoost")
    st.caption("Version 1.0")

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

/* Hide Streamlit default menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main background */
.stApp{
    background-color:#F5F7FA;
}

/* Hero Box */
.hero{
    background: linear-gradient(135deg,#1E3C72,#2A5298);
    padding:35px;
    border-radius:20px;
    color:white;
    box-shadow:0px 10px 25px rgba(0,0,0,0.15);
    margin-bottom:25px;
}

/* Metric Cards */
.card{
    background:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 6px 18px rgba(0,0,0,0.08);
    border-left:6px solid #1E88E5;
}

.card h2{
    color:#1E3C72;
    margin-bottom:5px;
}

.card p{
    color:#555;
    font-size:16px;
}

section[data-testid="stSidebar"]{
    background-color:#262730;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div{
    color:white !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")


# ---------------- HERO SECTION ---------------- #
if page == "🏠 Home":
    st.markdown("""
<div class="hero">

<h1>🛡 AI-Powered Financial Fraud Detection</h1>

<h4>Secure Banking Transactions using Machine Learning</h4>

<p>
Detect fraudulent transactions instantly using trained Machine Learning models.
Analyze risks, monitor predictions, and improve financial security through an interactive dashboard.
</p>

</div>
""", unsafe_allow_html=True)

    st.info("💡 Tip: Use the navigation menu on the left to explore different modules.")

# ---------------- INFO CARDS ---------------- #

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
    <div class="card">
    <h2>3</h2>
    <p>ML Models</p>
    </div>
    """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
    <div class="card">
    <h2>99.95%</h2>
    <p>Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
    <div class="card">
    <h2>284,807</h2>
    <p>Transactions</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("📌 Project Overview")

    st.write("""
This project uses Machine Learning algorithms to identify fraudulent financial transactions.
It enables secure transaction monitoring by predicting whether a transaction is **Legitimate** or **Fraudulent** based on transaction features.
""")

elif page == "🔍 Prediction":

    st.title("🔍 Financial Fraud Prediction")

    st.write(
        "Enter the transaction details below to predict whether the transaction is **Fraudulent** or **Legitimate**."
    )

    st.divider()

    st.subheader("📌 Basic Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        time = st.number_input(
            "⏱ Time",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

    with col2:
        amount = st.number_input(
            "💰 Amount",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

    st.divider()

    st.subheader("⚙ Advanced Transaction Features")

    col1, col2 = st.columns(2)

    features = {}

    for i in range(1, 29):

     if i % 2 != 0:

        with col1:
            features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                key=f"V{i}"
            )

     else:

        with col2:
            features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                key=f"V{i}"
            )

elif page == "📊 Dashboard":

    df = pd.read_csv("dataset/creditcard.csv")

    st.title("📊 Financial Fraud Dashboard")

    st.write("Overview of the Credit Card Fraud Detection Dataset.")

    # ================= Metrics =================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("💳 Total Transactions", "284,807")

    with col2:
        st.metric("🚨 Fraud Cases", "492")

    with col3:
        st.metric("✅ Model Accuracy", "99.95%")

    with col4:
        st.metric("🤖 Models Compared", "3")

    st.divider()

    # ================= Project Summary =================
    st.subheader("📌 Project Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.success("""
**Machine Learning Models Used**

- Decision Tree
- Random Forest ✅
- XGBoost
""")

    with summary_col2:
        st.info("""
**Dataset Information**

- Records : 284,807
- Features : 30
- Fraud Cases : 492
- Genuine Cases : 284,315
""")

    st.divider()

    # ================= Class Distribution =================
    st.subheader("📈 Fraud Distribution")

    chart_data = pd.DataFrame({
        "Category": ["Genuine", "Fraud"],
        "Transactions": [284315, 492]
    })

    st.bar_chart(chart_data.set_index("Category"))

    st.divider()

    # ================= Dataset Preview =================
    st.subheader("📋 Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    # ================= Project Highlights =================
    st.subheader("⭐ Dashboard Highlights")

    st.markdown("""
- ✅ Fraud Detection using Machine Learning
- ✅ Random Forest selected as the best model
- ✅ Single Transaction Prediction
- ✅ Batch CSV Prediction
- ✅ Model Performance Comparison
- ✅ Analytics Dashboard
""")


elif page == "📈 Analytics":

    st.title("📈 Analytics Dashboard")

    st.write("Detailed analysis of the financial fraud dataset.")

    # Load Dataset
    df = pd.read_csv("dataset/creditcard.csv")

    st.success("✅ Dataset Loaded Successfully")

    # Basic Information
    total_transactions = len(df)
    fraud_transactions = len(df[df["Class"] == 1])
    genuine_transactions = len(df[df["Class"] == 0])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Transactions", total_transactions)

    with col2:
        st.metric("Fraud Transactions", fraud_transactions)

    with col3:
        st.metric("Genuine Transactions", genuine_transactions)

    st.divider()

    # Dataset Statistics
    st.subheader("📈 Dataset Statistics")

    fraud_percentage = (fraud_transactions / total_transactions) * 100
    genuine_percentage = (genuine_transactions / total_transactions) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Fraud Percentage", f"{fraud_percentage:.3f}%")

    with col2:
        st.metric("Genuine Percentage", f"{genuine_percentage:.3f}%")

    st.divider()

    # Dataset Preview
    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    # Dataset Information
    st.subheader("📌 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

elif page == "📂 Batch Prediction":

    st.title("📂 Batch Prediction")

    st.write("Upload a CSV file to predict multiple transactions at once.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.success(f"✅ File uploaded successfully! ({len(df)} records)")

        st.subheader("Preview")
        st.dataframe(df.head())

        

        if st.button("🚀 Predict All Transactions"):

            input_data = df.copy()

            if "Class" in input_data.columns:
                      input_data = input_data.drop(columns=["Class"])

            input_scaled = scaler.transform(input_data)

            predictions = model.predict(input_scaled)

            input_data["Prediction"] = [
                "Fraud" if p == 1 else "Genuine"
                for p in predictions
            ]

            fraud_count = (predictions == 1).sum()
            genuine_count = (predictions == 0).sum()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Records", len(input_data))

            with col2:
                st.metric("Fraud", fraud_count)

            with col3:
                st.metric("Genuine", genuine_count)

            st.subheader("Prediction Results")
            st.dataframe(input_data)

            csv = input_data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇ Download Results",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

elif page == "🤖 Model Comparison":

    # Page Title
    st.title("🤖 Model Comparison")

    st.write("Comparison of Machine Learning Models")

    # Read metrics file
    comparison_df = pd.read_csv("models/model_metrics.csv")

    # Display table
    st.subheader("📋 Performance Metrics")

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    # Find best model
    best_model = comparison_df.loc[
        comparison_df["Accuracy"].idxmax(),
        "Model"
    ]

    best_accuracy = comparison_df["Accuracy"].max()

    st.success(
        f"🏆 Best Model: {best_model} (Accuracy: {best_accuracy:.4f})"
    )

    # Accuracy Chart
    st.subheader("📊 Accuracy Comparison")

    st.bar_chart(
        comparison_df.set_index("Model")["Accuracy"]
    )

    # Precision Chart
    st.subheader("🎯 Precision Comparison")

    st.bar_chart(
        comparison_df.set_index("Model")["Precision"]
    )

    # Recall Chart
    st.subheader("📈 Recall Comparison")

    st.bar_chart(
        comparison_df.set_index("Model")["Recall"]
    )

    # F1 Score Chart
    st.subheader("⭐ F1-Score Comparison")

    st.bar_chart(
        comparison_df.set_index("Model")["F1 Score"]
    )

    st.info("""
### 📌 Conclusion

✔ Random Forest achieved the highest overall performance.

✔ It provides the highest Accuracy and Precision among all models.

✔ Therefore, Random Forest is selected as the final fraud detection model.
""")

elif page == "ℹ About":

    st.title("ℹ️ About Project")

    st.markdown("""
# 💳 AI-Powered Financial Fraud Detection System

This project uses Machine Learning algorithms to detect fraudulent
financial transactions. It helps identify suspicious transactions
quickly and accurately, reducing financial losses.

---
""")

    st.subheader("🎯 Project Objective")

    st.write("""
The main objective of this project is to detect fraudulent financial
transactions using Machine Learning models and provide predictions
through an interactive Streamlit dashboard.
""")

    st.divider()

    st.subheader("🛠 Technologies Used")

    st.write("""
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
""")

    st.divider()

    st.subheader("🤖 Machine Learning Models")

    st.write("""
• Logistic Regression

• Decision Tree

• Random Forest ✅ (Best Performing Model)

• XGBoost
""")

    st.divider()

    st.subheader("📂 Dataset Information")

    st.write("""
Dataset: Credit Card Fraud Detection

Features: 30 Input Features

Target Variable: Class

Class = 0 → Genuine Transaction

Class = 1 → Fraud Transaction
""")

    st.divider()

    st.subheader("✨ Project Features")

    st.write("""
✅ Single Transaction Prediction

✅ Batch Prediction using CSV

✅ Analytics Dashboard

✅ Model Comparison

✅ Download Prediction Results
""")

    st.divider()

    st.subheader("👩‍💻 Developer")

    st.write("""
**Yashvi Choudhary**


""")

st.divider()

predict_btn = st.button(
    "🚀 Predict Transaction",
    use_container_width=True
)

if predict_btn:

    time = st.number_input("Time")

    input_data = [time]

    for i in range(1, 29):
        input_data.append(features[f"V{i}"])

    input_data.append(amount)

    input_data = np.array(input_data).reshape(1, -1)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    fraud_probability = probability[0][1] * 100
    genuine_probability = probability[0][0] * 100

    st.divider()
    st.subheader("📋 Prediction Result")

    if prediction[0] == 1:

        st.error("🚨 Fraud Transaction Detected")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Risk Level", "HIGH")

        with col2:
            st.metric("Fraud Probability", f"{fraud_probability:.2f}%")

        st.warning("""
### Recommended Actions

- Block the transaction immediately
- Verify customer identity
- Notify the bank
- Monitor account activity
""")

    else:

        st.success("✅ Genuine Transaction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Risk Level", "LOW")

        with col2:
            st.metric("Confidence", f"{genuine_probability:.2f}%")

        st.info("""
### Recommendation

Transaction appears to be legitimate.

Proceed with normal processing.
""")

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        💳 AI-Powered Financial Fraud Detection System <br>
        Developed by <b>Yashvi Choudhary</b> | B.Tech CSE
    </div>
    """,
    unsafe_allow_html=True
)