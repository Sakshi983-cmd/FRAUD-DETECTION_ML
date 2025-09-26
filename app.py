import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Fraud Detection App", page_icon="💳")

# ------------------ TRAIN MODEL ------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("fraud_data.csv")

    # Encode categorical
    le_type, le_loc = LabelEncoder(), LabelEncoder()
    df["transaction_type"] = le_type.fit_transform(df["transaction_type"])
    df["location"] = le_loc.fit_transform(df["location"])

    # Features / Target
    X = df.drop(["transaction_id", "is_fraud"], axis=1)
    y = df["is_fraud"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = (fp / (fp + tn)) * 100

    metrics = {
        "accuracy": acc,
        "fpr": fpr,
        "confusion_matrix": cm,
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    return model, le_type, le_loc, metrics

model, le_type, le_loc, metrics = train_model()

# ------------------ UI ------------------
st.title("💳 Fraud Detection System")

st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {metrics['accuracy']:.2f}%")
st.write(f"**False Positive Rate:** {metrics['fpr']:.2f}%")
st.write("**Confusion Matrix:**")
st.write(metrics["confusion_matrix"])

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
