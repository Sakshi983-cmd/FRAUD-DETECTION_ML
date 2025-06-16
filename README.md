# 🚨 Fraud Detection using Machine Learning

This project demonstrates how to detect fraudulent financial transactions using a Machine Learning model trained on a custom dataset. It uses Random Forest Classifier to predict whether a transaction is fraud (`1`) or genuine (`0`).

---

## 📌 Objective

To build and train a supervised ML model that classifies transactions as fraudulent or legitimate using features like amount, transaction type, location, and account age.

---

## 📁 Project Files

| File Name                     | Description                                      |
|------------------------------|--------------------------------------------------|
| `fraud_data.csv`             | Custom transaction dataset (100 samples)         |
| `fraud_detection.py`         | Main training script (can be run locally)        |
| `fraud_detection_model.pkl`  | Trained ML model saved using pickle              |
| `fraud_detection_project.ipynb` | Jupyter notebook for step-by-step training     |
| `README.md`                  | Project summary and instructions                 |
| `requirements.txt`           | Python dependencies                              |

---

## ⚙️ How to Run

### Option 1: Run on Google Colab
1. Open the Jupyter Notebook: `fraud_detection_project.ipynb`
2. Upload `fraud_data.csv` when prompted
3. Run all cells to train and evaluate the model
4. Download the `.pkl` model using:
```python
from google.colab import files
files.download('fraud_detection_model.pkl')
