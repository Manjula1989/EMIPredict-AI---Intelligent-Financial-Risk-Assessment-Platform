import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("models/best_regression_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

st.set_page_config(page_title="EMIPredict AI", layout="centered")
st.title("💰 EMIPredict AI - Financial Risk Assessment Platform")
st.markdown("Predict Maximum Monthly EMI based on applicant details")

# Collect user inputs for main features
age = st.number_input("Age", 18, 70, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
education = st.selectbox("Education", ["Graduate", "Post-Graduate", "PhD"])
monthly_salary = st.number_input("Monthly Salary (INR)", 10000, 500000, 50000)
employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
years_of_employment = st.number_input("Years of Employment", 0, 40, 5)
company_type = st.selectbox("Company Type", ["Private", "Government", "Startup"])
house_type = st.selectbox("House Type", ["Owned", "Rented"])

# Provide default values for other columns in dataset
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'marital_status': [marital_status],
    'education': [education],
    'monthly_salary': [monthly_salary],
    'employment_type': [employment_type],
    'years_of_employment': [years_of_employment],
    'company_type': [company_type],
    'house_type': [house_type],
    # Default/dummy values for remaining features
    'monthly_rent': [0],
    'family_size': [1],
    'dependents': [0],
    'school_fees': [0],
    'college_fees': [0],
    'travel_expenses': [0],
    'groceries_utilities': [0],
    'other_monthly_expenses': [0],
    'existing_loans': [0],
    'current_emi_amount': [0],
    'credit_score': [700],
    'bank_balance': [0],
    'emergency_fund': [0],
    'emi_scenario': [0],
    'requested_amount': [0],
    'requested_tenure': [12],
    'emi_eligibility': [0]
})

# Encode categorical inputs
for col in label_encoders.keys():
    if col in input_data.columns:
        le = label_encoders[col]
        val = input_data.at[0, col]
        if val in le.classes_:
            input_data[col] = le.transform([val])
        else:
            # If unseen, map to -1 or a safe default
            input_data[col] = -1

# Predict EMI
if st.button("🔍 Predict Maximum Monthly EMI"):
    prediction = model.predict(input_data)[0]
    st.subheader("💸 Estimated Maximum Monthly EMI")
    st.success(f"₹ {prediction:,.2f} per month")
