
# 📦 Importing Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# 📊 Load Dataset
df = pd.read_csv('fraud_data.csv')
print("✅ Dataset Loaded")

# 🔧 Encode Categorical Features
df_encoded = pd.get_dummies(df, columns=['transaction_type', 'location'], drop_first=True)

# 🎯 Feature Selection
X = df_encoded.drop(['transaction_id', 'is_fraud'], axis=1)
y = df_encoded['is_fraud']

# 📤 Train-Test Split with Stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🧠 Train Random Forest with Class Weighting
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
print("✅ Model Trained with Imbalance Handling")

# 📈 Evaluate the Model
y_pred = model.predict(X_test)
print("📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))

# 💾 Save the Model
with open('fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model Saved as 'fraud_detection_model.pkl'")

# 📊 Visualization: Histograms & Anomaly Detection
# Transaction Amount Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['amount'], bins=50, kde=True, color='skyblue')
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()

# Fraud vs Non-Fraud Comparison
plt.figure(figsize=(6,4))
sns.countplot(x='is_fraud', data=df, palette='Set2')
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Fraud Label (0=Legit, 1=Fraud)")
plt.ylabel("Count")
plt.show()

# Anomaly Visualization: High-Risk Transactions by Location
plt.figure(figsize=(10,6))
sns.boxplot(x='location', y='amount', hue='is_fraud', data=df, palette='coolwarm')
plt.title("High-Risk Transaction Patterns by Location")
plt.xticks(rotation=45)
plt.show()
