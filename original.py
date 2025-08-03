import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Streamlit page configuration
st.set_page_config(page_title="Loan Risk Predictor", page_icon="🏦")

# Load trained XGBoost model
model = joblib.load("best_xgboost_model.pkl")  # Replace with your actual model path

# App title
st.title("🏦 Loan Default Risk Predictor")
st.markdown("Use this tool to assess the risk of default before approving a loan.")

# Input form
st.markdown("### 📋 Enter Applicant Details")

age = st.slider("Age", 18, 75, 30)
income = st.number_input("Monthly Income (₹)", min_value=0, value=30000)
loan_amount = st.number_input("Loan Amount (₹)", min_value=1000, value=100000)
credit_score = st.slider("Credit Score", 300, 850, 650)
months_employed = st.slider("Months Employed", 0, 600, 36)
interest_rate = st.slider("Interest Rate (%)", 0.0, 30.0, 12.0)
loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
dti_ratio = st.slider("DTI Ratio (%)", 0.0, 100.0, 35.0)

# Prediction trigger
if st.button("Check Loan Risk"):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'MonthsEmployed': months_employed,
        'InterestRate': interest_rate,
        'LoanTerm': loan_term,
        'DTIRatio': dti_ratio
    }])

    # Get prediction probability
    prob = model.predict_proba(input_data)[0][1]

    # Custom threshold
    threshold = 0.35
    prediction = 1 if prob >= threshold else 0

    # Result output
    st.markdown("### 🔍 Prediction Result")
    if prediction == 1:
        st.error(f"🚫 **Risky**: {prob:.2%} chance of default. Loan should be avoided.")
    else:
        st.success(f"✅ **Safe**: Only {prob:.2%} chance of default. Loan may be granted.")

    # Model output
    st.markdown("### 📊 Model Output")
    st.write(f"Prediction Class (threshold @ {threshold:.2f}): `{prediction}` (1 = Risky, 0 = Safe)")
    st.write(f"Probability of Default: `{prob:.4f}`")
