# 📦 Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# 📊 Load Dataset
df = pd.read_csv('fraud_data.csv')

# 🔧 Encode Categorical Features
df_encoded = pd.get_dummies(df, columns=['transaction_type', 'location'], drop_first=False)  # Change to False

# 🎯 Feature Selection (transaction_id हटाएं)
X = df_encoded.drop(['transaction_id', 'is_fraud'], axis=1)
y = df_encoded['is_fraud']

# 📤 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model Training Complete")

# 📈 Evaluate the Model
y_pred = model.predict(X_test)

print("📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred))

# 💾 Save the Model (Important for Streamlit)
with open('fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as 'fraud_detection_model.pkl'")

# Print columns for Streamlit app
print("\n📋 Columns used in model:")
print(X.columns.tolist())
