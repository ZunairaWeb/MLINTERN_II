import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


base_path = r"D:\Pprojects\MLintern_II\House Pricing Prediction\Data\house-prices-advanced-regression-techniques"
train_df = pd.read_csv(base_path + r"\train.csv")
test_df = pd.read_csv(base_path + r"\test.csv")
submission_df = pd.read_csv(base_path + r"\sample_submission.csv")
# print(train_df)
# print(test_df)
# print(submission_df)
# # check the shape and content of the datatype 
# print("Train shape:", train_df.info)
# print("Train dtypes:\n", train_df.dtypes)

# print("\nTest shape:", test_df.info)
# print("Test dtypes:\n", test_df.dtypes)

# print("\nSubmission shape:", submission_df.info)
# print("Submission dtypes:\n", submission_df.dtypes)

# print(train_df.isnull().sum().sort_values(ascending=False).head(20))

# try to fill the missing values 
# Drop columns with too many missing values

train_df = train_df.drop(["PoolQC", "MiscFeature", "Alley", "Fence"], axis=1)
test_df = test_df.drop(["PoolQC", "MiscFeature", "Alley", "Fence"], axis=1)

# Fill missing numeric values with median
# Select numeric columns
# Select numeric columns from train
nums_col = train_df.select_dtypes(include=["float64", "int64"]).columns

# Only keep columns that are also in test
nums_col = [col for col in nums_col if col in test_df.columns]

# Fill missing numeric values with median
for col in nums_col:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    test_df[col] = test_df[col].fillna(test_df[col].median())

cat_col = train_df.select_dtypes(include=["object"]).columns

# Fill missing categorical values with mode (most frequent value)
for col in cat_col:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col]  = test_df[col].fillna(test_df[col].mode()[0])

# One-hot encode categorical features
X = pd.get_dummies(train_df.drop("SalePrice", axis=1))
X_test = pd.get_dummies(test_df)
        


# Align train and test so they have the same columns
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

# Target column
y = train_df["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
# Validate
val_preds = model.predict(X_val)
# Calculate MSE
mse = mean_squared_error(y_val, val_preds)

# Take square root to get RMSE
rmse = np.sqrt(mse)

print("Validation RMSE:", rmse)

# Predict on test set
test_preds = model.predict(X_test)
submission_df["SalePrice"] = test_preds
print(submission_df.head())
 