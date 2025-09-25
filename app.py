import streamlit as st
import pandas as pd
import pickle

# Page setup
st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("🔒 Fraud Detection System")

# Load model
try:
    model = pickle.load(open('fraud_detection_model.pkl', 'rb'))
    st.success("Model loaded successfully!")
except:
    st.error("Model file not found!")
    model = None

# Input form
st.header("Enter Transaction Details")

amount = st.number_input("Amount", min_value=0.0, value=500.0)
transaction_type = st.selectbox("Transaction Type", 
                               ["cash_out", "debit", "payment", "transfer"])
account_age = st.number_input("Account Age (Days)", min_value=0, value=365)
location = st.selectbox("Location", 
                       ["Delhi", "Mumbai", "Chennai", "Kolkata", "Lucknow"])

# Predict button
if st.button("Check for Fraud") and model is not None:
    # Create input data
    input_data = {
        'amount': amount,
        'account_age_days': account_age,
        'transaction_type_' + transaction_type: 1,
        'location_' + location: 1
    }
    
    # Create DataFrame with all expected columns
    columns = ['amount', 'account_age_days', 
               'transaction_type_cash_out', 'transaction_type_debit', 
               'transaction_type_payment', 'transaction_type_transfer',
               'location_Chennai', 'location_Delhi', 'location_Kolkata', 
               'location_Lucknow', 'location_Mumbai']
    
    input_df = pd.DataFrame(0, index=[0], columns=columns)
    
    # Fill the values
    input_df['amount'] = amount
    input_df['account_age_days'] = account_age
    input_df['transaction_type_' + transaction_type] = 1
    input_df['location_' + location] = 1
    
    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # Display result
    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ LEGITIMATE TRANSACTION (Probability: {probability:.2%})")

st.markdown("---")
st.write("Built with ❤️ using Streamlit")
