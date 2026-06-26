"""
Telco Customer Churn Prediction App 
"""
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ==============================================
# PAGE CONFIGURATION
# ==============================================

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ==============================================
# LOAD MODEL WITH PICKLE
# ==============================================

@st.cache_resource
def load_model():
    """Load the trained model using pickle"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please make sure 'model.pkl' is in the same directory as this app.")
        return None

# ==============================================
# TITLE AND HEADER
# ==============================================

st.title("📊 Telco Customer Churn Prediction")
st.markdown("""
### Predict if a customer is likely to churn (cancel their service)
Fill in the customer details below and click **Predict** to get a churn risk assessment.
""")

st.markdown("---")

# ==============================================
# INPUT FORM
# ==============================================

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Demographics & Account")
    
    tenure = st.slider(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
        help="How long the customer has been with the company"
    )
    
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=150.0,
        value=70.0,
        step=5.0,
        help="Customer's monthly bill amount"
    )
    
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"],
        help="Is the customer 65 years or older?"
    )
    
    partner = st.selectbox(
        "Has Partner",
        ["No", "Yes"],
        help="Does the customer have a partner?"
    )
    
    dependents = st.selectbox(
        "Has Dependents",
        ["No", "Yes"],
        help="Does the customer have dependents?"
    )

with col2:
    st.subheader("💳 Services & Payment")
    
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"],
        help="Month-to-month customers typically have higher churn risk"
    )
    
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
        help="Type of internet service the customer uses"
    )
    
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        help="Electronic check and mailed check have higher churn risk"
    )
    
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"],
        help="Paperless billing customers often have higher churn risk"
    )
    
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"],
        help="Customers without online security have higher churn risk"
    )

# ==============================================
# ADDITIONAL OPTIONS (Expandable)
# ==============================================

with st.expander("🔧 Advanced Options (Optional)"):
    col3, col4 = st.columns(2)
    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    with col4:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.markdown("---")

# ==============================================
# PREDICT BUTTON
# ==============================================

predict_button = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

# ==============================================
# PREDICTION LOGIC
# ==============================================

if predict_button:
    try:
        # Load the model
        model = load_model()
        
        if model is None:
            st.error("❌ Model could not be loaded. Please check the file.")
            st.stop()
        
        # Create input DataFrame with all features
        input_data = pd.DataFrame({
            'gender': [gender.lower()],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': ['No'],
            'DeviceProtection': ['No'],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [tenure * monthly_charges]  # Calculate TotalCharges
        })
        
        # Show input data for debugging (optional)
        # st.write("Input Data:", input_data)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # ==========================================
        # DISPLAY RESULTS
        # ==========================================
        
        st.markdown("---")
        st.subheader("📈 Prediction Results")
        
        # Create 3 columns for metrics
        col_r1, col_r2, col_r3 = st.columns(3)
        
        if prediction == 1:
            # HIGH RISK
            with col_r1:
                st.metric("Risk Level", "⚠️ HIGH", delta="Risk")
            with col_r2:
                st.metric("Churn Probability", f"{probability:.1%}", delta="Action Required")
            with col_r3:
                st.metric("Recommendation", "Retention Offer", delta="Urgent")
            
            # Red warning box
            st.error(f"⚠️ **HIGH CHURN RISK!** This customer has a **{probability:.1%}** probability of churning.")
            
            # Progress bar (red)
            st.markdown("**Churn Risk Meter**")
            st.progress(int(probability * 100))
            
            # Suggested actions
            st.markdown("### 💡 Suggested Actions")
            st.markdown("""
            <div style='background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ff6b6b;'>
                <ol>
                    <li><b>Offer loyalty discount</b> - 10-15% off for 6 months</li>
                    <li><b>Upgrade contract</b> - Offer annual contract with incentives</li>
                    <li><b>Proactive support</b> - Schedule a customer satisfaction call</li>
                    <li><b>Service review</b> - Check for any service issues</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            # LOW RISK
            with col_r1:
                st.metric("Risk Level", "✅ LOW", delta="Good")
            with col_r2:
                st.metric("Churn Probability", f"{probability:.1%}", delta="Normal")
            with col_r3:
                st.metric("Recommendation", "Monitor", delta="Standard")
            
            # Green success box
            st.success(f"✅ **LOW CHURN RISK!** This customer has a **{probability:.1%}** probability of churning.")
            
            # Progress bar (green)
            st.markdown("**Churn Risk Meter**")
            st.progress(int(probability * 100))
            
            # Suggested actions
            st.markdown("### 💡 Suggested Actions")
            st.markdown("""
            <div style='background-color: #d4edda; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745;'>
                <ol>
                    <li><b>Continue standard engagement</b> - Maintain regular communication</li>
                    <li><b>Consider upsell opportunities</b> - Upgrade to premium services</li>
                    <li><b>Monitor satisfaction</b> - Regular check-ins</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        # ==========================================
        # DETAILED METRICS (Expander)
        # ==========================================
        
        with st.expander("📊 Detailed Prediction Breakdown"):
            st.markdown("**Model Output:**")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.metric("Prediction", "Churn" if prediction == 1 else "Stayed")
            with col_d2:
                st.metric("Confidence", f"{max(probability, 1-probability)*100:.1f}%")
            
            st.markdown("**Feature Values Used:**")
            st.dataframe(input_data.T, use_container_width=True)
            
            st.markdown("**Interpretation:**")
            st.info("""
            - **Probability > 50%**: Customer is likely to churn
            - **Probability < 50%**: Customer is likely to stay
            - **Higher probability**: Stronger risk of churning
            """)
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.info("Please check that all fields are filled correctly.")
        import traceback
        st.code(traceback.format_exc())

# ==============================================
# FOOTER
# ==============================================

st.markdown("---")
st.caption("🔬 Powered by XGBoost | Built with Streamlit")
st.caption("📊 Model trained on Telco Customer Churn dataset")

# ==============================================
# SIDEBAR INFO
# ==============================================

with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/examples/streamlit_logo.png", width=150)
    st.title("📊 About")
    st.markdown("---")
    st.markdown("""
    ### Model Details
    - **Algorithm:** XGBoost
    - **Training Data:** 5,634 customers
    - **Features:** 19 customer attributes
    
    ### Key Predictors
    1. Contract Type (Month-to-month → ⚠️ High Risk)
    2. Tenure (Shorter → ⚠️ High Risk)
    3. Monthly Charges (Higher → ⚠️ High Risk)
    4. Payment Method (Electronic check → ⚠️ High Risk)
    
    ### Business Impact
    - **Early identification** of at-risk customers
    - **Proactive retention** strategies
    - **Significant cost savings** vs. acquisition
    """)
    st.markdown("---")
    st.caption("Version 2.0 | Pickle Compatible")
