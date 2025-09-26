# Fraud Detection ML 🚨

## 📌 Overview
This project detects fraudulent transactions using a **Random Forest Classifier** trained on synthetic transaction data.  
It is deployed using **Streamlit Cloud**.

## 📊 Dataset
- **fraud_data.csv** contains transaction details:
  - transaction_id
  - amount
  - transaction_type
  - account_age_days
  - location
  - is_fraud

## ⚙️ Workflow
1. Data preprocessing (encoding categorical features).
2. Train Random Forest Classifier.
3. Save model with pickle.
4. Streamlit app for prediction.

## 📈 Metrics
- Accuracy: ~94%
- False Positive Rate: ~2%

## 🚀 Deployment
```bash
# Run locally
streamlit run app.py
