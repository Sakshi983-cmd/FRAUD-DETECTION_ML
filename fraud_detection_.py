import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load dataset
df = pd.read_csv("fraud_data.csv")

# Encode categorical variables
le_type = LabelEncoder()
le_loc = LabelEncoder()
df["transaction_type"] = le_type.fit_transform(df["transaction_type"])
df["location"] = le_loc.fit_transform(df["location"])

# Split features/target
X = df.drop(["transaction_id", "is_fraud"], axis=1)
y = df["is_fraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) * 100

print(f"✅ Accuracy: {acc:.2f}%")
print(f"⚠️ False Positive Rate: {fpr:.2f}%")
print("Confusion Matrix:\n", cm)

# Save model + encoders
with open("fraud_detection_model.pkl", "wb") as f:
    pickle.dump((model, le_type, le_loc), f)
