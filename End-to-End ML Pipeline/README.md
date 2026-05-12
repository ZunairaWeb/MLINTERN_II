
# ProjectCustomer Churn Prediction using Logistic Regression

 ## Workflow
1. Data Loading
Load dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) using pandas.

2.Data Cleaning
Drop irrelevant identifier (customerID).

Convert TotalCharges column to numeric values.

Handle missing values by dropping incomplete rows.

3. Feature Engineering
Separate features (X) and target (y).

Encode categorical variables using one-hot encoding (pd.get_dummies).

Map target column Churn → Yes=1, No=0.

4. Model Training
Split dataset into training and testing sets.

Train a Logistic Regression model with max_iter=2000.

5. Evaluation
Generate predictions on test data.

Calculate accuracy score.

Display confusion matrix to analyze performance.

### Example Results
Accuracy: ~78–79%

Confusion Matrix: Shows distribution of true positives, false positives, true negatives, and false negatives.

## Technologies Used
Python 3.9+

Pandas

Scikit-learn

NumPy

## How to Run
Clone the repository:

bash
git clone https://github.com/ZunairaWeb/MLINTERN_II.git
Navigate to project folder:

bash
cd End-to-End ML Pipeline
Run the pipeline:

bash
python Src/Pipeline.py