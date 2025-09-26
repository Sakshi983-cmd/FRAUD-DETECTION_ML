import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("🔒 Fraud Detection System")

@st.cache_resource
def load_model():
    try:
        with open('fraud_detection_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

model_data = load_model()

if model_data:
    model = model_data['model']
    scaler = model_data['scaler']
    st.success("✅ Model Loaded Successfully!")
else:
    model, scaler = None, None

# Simple Input Form
st.header("Enter Transaction Details")

amount = st.number_input("Amount", value=500.0)
txn_type = st.selectbox("Transaction Type", ["cash_out", "debit", "payment", "transfer"])
account_age = st.number_input("Account Age (Days)", value=365)
location = st.selectbox("Location", ["Delhi", "Mumbai", "Chennai", "Kolkata", "Lucknow"])

if st.button("Check Fraud") and model:
    try:
        # Create input data
        input_data = {
            'amount': [amount],
            'account_age_days': [account_age],
            'amount_to_age_ratio': [amount/(account_age+1)],
            'high_value_txn': [1 if amount > 500 else 0],
            'new_account': [1 if account_age < 365 else 0],
        }
        
        # Add transaction types
        for t in ["cash_out", "debit", "payment", "transfer"]:
            input_data[f'transaction_type_{t}'] = [1 if txn_type == t else 0]
        
        # Add locations
        for loc in ["Chennai", "Delhi", "Kolkata", "Lucknow", "Mumbai"]:
            input_data[f'location_{loc}'] = [1 if location == loc else 0]
        
        input_df = pd.DataFrame(input_data)
        
        # Ensure correct column order
        if hasattr(model, 'feature_names_in_'):
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        
        # Scale features
        if scaler:
            num_cols = ['amount', 'account_age_days', 'amount_to_age_ratio']
            input_df[num_cols] = scaler.transform(input_df[num_cols])
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Show result
        if prediction == 1:
            st.error(f"🚨 FRAUD ({probability:.1%} confidence)")
        else:
            st.success(f"✅ LEGITIMATE ({probability:.1%} confidence)")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
