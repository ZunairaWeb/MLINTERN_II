import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix


df=pd.read_csv('D:\Pprojects\MLintern_II\End-to-End ML Pipeline\DataSet\WA_Fn-UseC_-Telco-Customer-Churn.csv')
# print(df.head())
# print(df.describe())
# print(df.info())
# Data Cleanning 
# drop the customer id 
df = df.drop('customerID', axis=1)
# print(df)
# Convert the  total charges to the column 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
# print(df)
# Drop the column Churn 
X = df.drop('Churn', axis=1)   # features
y = df['Churn'].map({'Yes':1, 'No':0})   # target converted to 0/1
# print(df)

# Convert the data into categorial form 
X = pd.get_dummies(X, drop_first=True)
# print(X)
# Train the method 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
#  prediction 
y_predict = model.predict(X_test)
print(y_predict)
model_accuracy = accuracy_score(y_test, y_predict)
print(model_accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
