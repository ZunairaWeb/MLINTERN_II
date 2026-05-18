# Customer Churn Prediction using Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv(
    r'D:\Pprojects\MLintern_II\End-to-End ML Pipeline\DataSet\WA_Fn-UseC_-Telco-Customer-Churn.csv'
)

# -------------------------------
# 2. Data Cleaning
# -------------------------------
# Drop customerID (not useful for prediction)
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric and drop rows with missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# -------------------------------
# 3. Feature and Target Separation
# -------------------------------
X = df.drop('Churn', axis=1)                     # Features
y = df['Churn'].map({'Yes': 1, 'No': 0})         # Target (binary labels)

# -------------------------------
# 4. Encode Categorical Features
# -------------------------------
X = pd.get_dummies(X, drop_first=True)

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Model Training
# -------------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# -------------------------------
# 7. Predictions & Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("Predictions:\n", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
