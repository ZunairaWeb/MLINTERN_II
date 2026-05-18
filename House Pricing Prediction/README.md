
#  House Price Prediction (Kaggle Project)

## 📌 Project Overview
This project is based on the **Kaggle House Prices: Advanced Regression Techniques** competition.  
The goal is to build a machine learning model that predicts the **SalePrice** of houses based on various features (location, size, quality, etc.).

---

## ⚙️ Workflow Steps
1. **Data Loading**
   - Load `train.csv` (with SalePrice) and `test.csv` (without SalePrice).
   - Load `sample_submission.csv` to prepare the final submission file.

2. **Data Cleaning**
   - Handle missing values (e.g., fill with mean/median or drop columns).
   - Fix inconsistent data types.

3. **Feature Engineering & Encoding**
   - Convert categorical features into numeric using **One-Hot Encoding** (`pd.get_dummies`).
   - Scale/normalize numerical features if needed.

4. **Train/Validation Split**
   - Split training data into:
     - `X_train, y_train` → used to train the model.
     - `X_val, y_val` → used to validate the model (practice quiz).

5. **Model Training**
   - Train regression models (e.g., Linear Regression, RandomForest, XGBoost).
   - Tune hyperparameters for better performance.

6. **Model Evaluation**
   - Use **RMSE (Root Mean Squared Error)** to measure prediction accuracy.
   - RMSE tells us how far off our predictions are from real prices on average.

7. **Prediction on Test Set**
   - Use the trained model to predict house prices for Kaggle’s `test.csv`.
   - Store predictions in `submission_df["SalePrice"]`.

8. **Submission**
   - Save predictions to `submission.csv`.
   - Upload to Kaggle for scoring.

---

## 📊 Example Code Snippet
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Validate
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print("Validation RMSE:", rmse)

# Predict on Kaggle test set
test_preds = model.predict(X_test)
submission_df["SalePrice"] = test_preds
submission_df.to_csv("submission.csv", index=False)
# Project Title

A brief description of what this project does and who it's for

