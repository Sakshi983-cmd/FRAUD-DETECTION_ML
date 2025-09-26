Fraud Detection Using Random Forest

Project Overview:
This is a beginner-level project to detect fraudulent transactions using a Random Forest Classifier. The project is implemented in Python using Google Colab, and demonstrates the workflow from data preprocessing to model evaluation.

📂 Dataset

The dataset (fraud_data.csv) contains transaction information with features like transaction_type, location, and the target variable is_fraud.

The dataset is already uploaded in the Colab notebook.

🛠 Tools & Libraries Used

Python 3.x

pandas, numpy

matplotlib, seaborn

scikit-learn (RandomForestClassifier, train_test_split, metrics)

pickle (for saving the model)

Google Colab

⚙️ Implementation Steps

Import Required Libraries

Load Dataset and display first few rows

Encode Categorical Features using pd.get_dummies()

Feature Selection

Train-Test Split (80% training, 20% testing)

Train Random Forest Model

Evaluate Model using

Confusion Matrix

Classification Report

Save Model as fraud_detection_model.pkl using pickle

📈 Results

Confusion Matrix:

[[19  0]
 [ 1  0]]


Classification Report:

Accuracy = 95%

Recall for fraud class is 0% due to class imbalance

⚠️ Note: Since the dataset is small and imbalanced, the model performs well on non-fraud transactions but struggles with fraud detection.

💾 Model Download

The trained model is saved as fraud_detection_model.pkl and can be downloaded directly from Google Colab.

📌 How to Run

Open the Google Colab Notebook

Upload fraud_data.csv

Run all cells to train the model and see evaluation metrics

Download the saved model if needed

📚 Skills Learned

Data Preprocessing (encoding categorical variables)

Training and evaluating a Random Forest Classifier

Handling class imbalance issues



GitHub Repository Name Suggestion:
