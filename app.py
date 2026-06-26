"""
Telco Customer Churn Prediction App
"""
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

# Title
st.title("📊 Telecom Customer Churn Prediction")
st.markdown("""
### Predict if a customer will churn with Explainable AI
Enter customer details below to get a prediction and explanation.
""")

st.markdown("---")

# Input columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Demographics")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

with col2:
    st.subheader("💳 Services & Payment")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

# Predict button
if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
    try:
        # Create input
        input_data = pd.DataFrame({
            'gender': ['Male'],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': ['No'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['No'],
            'StreamingMovies': ['No'],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [tenure * monthly_charges]
        })
        
        # Load and predict
        model = load_model()
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Display result
        st.markdown("---")
        st.subheader("📈 Prediction Results")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        if prediction == 1:
            with col_r1:
                st.metric("Risk Level", "HIGH", delta="⚠️")
            with col_r2:
                st.metric("Probability", f"{probability:.1%}", delta="Action Needed")
            with col_r3:
                st.metric("Recommendation", "Retention Offer", delta="Urgent")
            
            st.error(f"⚠️ High churn risk! Probability: {probability:.1%}")
            st.markdown("**Suggested Actions:**")
            st.markdown("""
            1. Offer loyalty discount (10-15% off)
            2. Upgrade to annual contract
            3. Schedule customer satisfaction call
            """)
        else:
            with col_r1:
                st.metric("Risk Level", "LOW", delta="✅")
            with col_r2:
                st.metric("Probability", f"{probability:.1%}", delta="Good")
            with col_r3:
                st.metric("Recommendation", "Monitor", delta="Standard")
            
            st.success(f"✅ Low churn risk! Probability: {probability:.1%}")
            st.markdown("**Suggested Actions:**")
            st.markdown("""
            1. Continue standard engagement
            2. Consider upsell opportunities
            3. Maintain regular communication
            """)
        
        # Progress bar
        st.progress(int(probability * 100))
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("🔬 Powered by XGBoost | Explainable AI with SHAP")
