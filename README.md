
### Overview

This project uses a **transaction-level dataset** designed for **fraud detection**. Each row represents a single financial transaction, with features describing the transaction context and a target label indicating whether the transaction is fraudulent.
### Data set - save on csv file
### Target Variable

* **`is_fraud`** *(binary)*:

  * `0` → Not Fraud
  * `1` → Fraud

This makes the problem a **supervised binary classification** task with **class imbalance** (fraud cases are fewer than non-fraud cases).

### Features

Typical features included in the dataset:

* **`transaction_id`**: Unique identifier for each transaction (dropped during modeling).
* **`amount`**: Monetary value of the transaction.
* **`transaction_type`** *(categorical)*: Type of transaction (e.g., transfer, payment, withdrawal).
* **`location`** *(categorical)*: Geographic or channel location of the transaction.
* **Additional numerical features** (if present): Balance-related or time-based attributes.

### Preprocessing Applied

* **Categorical Encoding**: One-hot encoding applied to `transaction_type` and `location` using `pd.get_dummies`.
* **Feature Selection**: Non-informative identifiers (e.g., `transaction_id`) removed.
* **Train–Test Split**: Stratified split to preserve fraud/non-fraud distribution.

### Data Characteristics & Challenges

* **Class Imbalance**: Fraud transactions are rare, requiring imbalance-aware techniques.
* **Non-linear Feature Interactions**: Fraud often depends on combinations of features (amount + location + time).
* **Noisy Real-world Data**: Presence of outliers and variability across transactions.

### Modeling Implications

Due to these characteristics, **tree-based ensemble models** (Random Forest / XGBoost) are well-suited, as they:

* Capture non-linear interactions
* Handle mixed feature types
* Are robust to noise and imbalance (with class weighting)

---

## 🛠️ Tools and Techniques

### Tools

* **Python**: Core programming language for data processing and model development
* **Pandas**: Data loading, cleaning, and manipulation
* **NumPy**: Numerical computations
* **Scikit-learn**:

  * `RandomForestClassifier` – Model training
  * `train_test_split` – Data splitting with stratification
  * `classification_report`, `confusion_matrix` – Model evaluation
* **Pickle**: Model serialization for deployment

### Techniques

* **Supervised Machine Learning**: Binary classification (Fraud vs Non-Fraud)
* **Feature Engineering**:

  * One-hot encoding for categorical variables
  * Removal of non-informative identifiers
* **Class Imbalance Handling**:

  * Stratified train-test split
  * Class weighting in Random Forest
* **Ensemble Learning**:

  * Random Forest to capture non-linear feature interactions and reduce overfitting
* **Model Evaluation**:

  * Precision, Recall, F1-score (Recall prioritized for fraud detection)
  * Confusion Matrix analysis
* **Overfitting Control**:

  * Ensemble averaging using Random Forest
* **Model Persistence**:

  * Saving trained model using `pickle` as `fraud_detection_model.pkl` for deployment

---

**Problem Type:** Supervised Machine Learning – Binary Classification

