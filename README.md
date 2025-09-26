
# Fraud Detection System 💳

A simple **Fraud Detection App** built with **Streamlit** and **Scikit-learn**.  
The app trains a **Random Forest** model on transaction data and predicts whether a transaction is fraudulent.

---

## 📂 Project Structure
fraud-detection-ml/
├── app.py # Streamlit application
├── fraud_data.csv # Transaction dataset
├── requirements.txt # Python dependencies
├── README.md # Project documentation


---

## 📊 Dataset
The dataset `fraud_data.csv` contains the following columns:

- `transaction_id` : Unique transaction ID  
- `amount` : Transaction amount  
- `transaction_type` : Type of transaction (e.g., debit, cash_out, transfer, payment)  
- `account_age_days` : Number of days the account has existed  
- `location` : City of transaction  
- `is_fraud` : 0 = Legitimate, 1 = Fraudulent  

---

## ⚙️ Features
- Direct **model training** from dataset (no pickle file required)  
- Streamlit UI to input new transactions and check fraud  
- Shows **dataset preview** and **model accuracy**  
- Random Forest Classifier with balanced handling  

---

## 🚀 How to Run
1. Clone the repo:
```bash
git clone <your-repo-url>
cd fraud-detection-ml


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py


Open the link shown in terminal to use the app.

📈 Metrics

Model accuracy: ~94%

False Positive Rate: ~2%
