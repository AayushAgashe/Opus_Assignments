import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model files
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# Page config
st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title("Customer Churn Prediction System")
st.markdown("Predict customer churn and analyze business insights")

# ---------------- INPUT SECTION ----------------
st.sidebar.header("Customer Input")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

tenure = st.sidebar.slider("Tenure (months)", 0, 72)

phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

monthly = st.sidebar.number_input("Monthly Charges")
total = st.sidebar.number_input("Total Charges")

# ---------------- PREPARE INPUT ----------------
# Start with a dictionary for ALL columns from training
input_dict = {col: 0 for col in columns}

# Numeric
if 'tenure' in input_dict: input_dict['tenure'] = tenure
if 'MonthlyCharges' in input_dict: input_dict['MonthlyCharges'] = monthly
if 'TotalCharges' in input_dict: input_dict['TotalCharges'] = total
if 'SeniorCitizen' in input_dict: input_dict['SeniorCitizen'] = senior

# Gender
if 'gender_Male' in input_dict:
    input_dict['gender_Male'] = 1 if gender == "Male" else 0

# Partner
if 'Partner_Yes' in input_dict:
    input_dict['Partner_Yes'] = 1 if partner == "Yes" else 0

# Dependents
if 'Dependents_Yes' in input_dict:
    input_dict['Dependents_Yes'] = 1 if dependents == "Yes" else 0

# Phone
if 'PhoneService_Yes' in input_dict:
    input_dict['PhoneService_Yes'] = 1 if phone == "Yes" else 0

# Internet
if internet == "Fiber optic" and 'InternetService_Fiber optic' in input_dict:
    input_dict['InternetService_Fiber optic'] = 1
elif internet == "DSL" and 'InternetService_DSL' in input_dict:
    input_dict['InternetService_DSL'] = 1

# Contract
if contract == "One year" and 'Contract_One year' in input_dict:
    input_dict['Contract_One year'] = 1
elif contract == "Two year" and 'Contract_Two year' in input_dict:
    input_dict['Contract_Two year'] = 1

# Payment
if payment == "Electronic check" and 'PaymentMethod_Electronic check' in input_dict:
    input_dict['PaymentMethod_Electronic check'] = 1
elif payment == "Mailed check" and 'PaymentMethod_Mailed check' in input_dict:
    input_dict['PaymentMethod_Mailed check'] = 1
elif payment == "Bank transfer (automatic)" and 'PaymentMethod_Bank transfer (automatic)' in input_dict:
    input_dict['PaymentMethod_Bank transfer (automatic)'] = 1
elif payment == "Credit card (automatic)" and 'PaymentMethod_Credit card (automatic)' in input_dict:
    input_dict['PaymentMethod_Credit card (automatic)'] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# CRITICAL FIX: Align input to EXACT training column order
input_df = input_df.reindex(columns=columns, fill_value=0)

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("Customer is likely to churn")
        else:
            st.success("Customer will stay")

    with col2:
        st.metric("Churn Probability", f"{prob:.2f}")

    # Risk level
    if prob > 0.7:
        st.warning("High Risk Customer")
    elif prob > 0.4:
        st.info("Medium Risk Customer")
    else:
        st.success("Low Risk Customer")

# ---------------- BUSINESS INSIGHTS ----------------
st.subheader("Business Insights")

# Feature importance (only works for tree models like XGBoost / RF)
try:
    importance = pd.DataFrame({
        'Feature': columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(10)

    fig, ax = plt.subplots()
    ax.barh(importance['Feature'], importance['Importance'])
    ax.invert_yaxis()
    plt.title("Top Features Influencing Churn")

    st.pyplot(fig)

except:
    st.info("Feature importance available only for tree-based models.")

# ---------------- RECOMMENDATIONS ----------------

st.subheader("Business Recommendations")

st.write("""
- Customers with high monthly charges are more likely to churn  
- Long-term contracts reduce churn probability  
- Customers using electronic check show higher churn risk  
- Providing discounts or offers to high-risk customers can reduce churn  
""")