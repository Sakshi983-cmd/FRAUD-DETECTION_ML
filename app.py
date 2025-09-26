import streamlit as st
import pickle

# Load trained model & encoders
model, le_type, le_loc = pickle.load(open("fraud_detection_model.pkl", "rb"))

st.title("💳 Fraud Detection System")

amount = st.number_input("Transaction Amount", min_value=1.0)
t_type = st.selectbox("Transaction Type", le_type.classes_)
age = st.number_input("Account Age (days)", min_value=1)
location = st.selectbox("Location", le_loc.classes_)

if st.button("Check Fraud"):
    t_type_enc = le_type.transform([t_type])[0]
    loc_enc = le_loc.transform([location])[0]

    features = [[amount, t_type_enc, age, loc_enc]]
    pred = model.predict(features)[0]

    if pred == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
