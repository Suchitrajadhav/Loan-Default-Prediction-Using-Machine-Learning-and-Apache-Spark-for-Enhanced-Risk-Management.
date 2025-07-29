import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set Streamlit page config
st.set_page_config(page_title="Loan Risk Prediction", page_icon="🏦")

# Load the trained model
model = joblib.load("rf_model_project.pkl")  # Ensure this file is in your current working directory

# App Title
st.title("🏦 Loan Default Risk Predictor")
st.markdown("Use this tool to assess the risk of default before approving a loan.")

# Input Form
st.markdown("### 📋 Enter Applicant Information:")

age = st.slider("Age", 18, 70, 30)
income = st.number_input("Monthly Income (₹)", min_value=0, value=30000)
loan_amount = st.number_input("Loan Amount (₹)", min_value=1000, value=100000)
credit_score = st.slider("Credit Score", 300, 850, 650)
months_employed = st.slider("Months Employed", 0, 480, 60)
interest_rate = st.slider("Interest Rate (%)", 0.0, 30.0, 12.0)
dti_ratio = st.slider("DTI Ratio (%)", 0.0, 100.0, 35.0)
loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])

# Predict button
if st.button("Check Loan Risk"):
    # Create input DataFrame with column names
    input_df = pd.DataFrame([{
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'MonthsEmployed': months_employed,
        'InterestRate': interest_rate,
        'DTIRatio': dti_ratio,
        'LoanTerm': loan_term
    }])

    # Prediction using model
    prob = model.predict_proba(input_df)[0][1]  # Probability of default

    # Use custom threshold instead of model.predict()
    threshold = 0.35
    prediction = 1 if prob >= threshold else 0

    # Result
    st.markdown("### 🔍 Result:")
    if prediction == 1:
        st.error(f"🚫 **Risky**: {prob:.2%} chance of default. Avoid granting the loan.")
    else:
        st.success(f"✅ **Safe**: Only {prob:.2%} chance of default. Loan can be considered.")

    # Output Section
    st.markdown("### 📊 Model Output")
    st.write(f"Prediction Class (custom threshold @ {threshold:.2f}): `{prediction}` (1 = Risky, 0 = Safe)")
    st.write(f"Default Probability: `{prob:.4f}`")
