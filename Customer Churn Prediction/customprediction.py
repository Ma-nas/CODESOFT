# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
df = pd.read_csv('c:/Users/hp/my-python-app/Customer Churn Prediction/Churn_Modelling.csv')
print("Dataset Preview:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
plt.figure(figsize=(6, 4))
sns.countplot(x='Exited', data=df)
plt.title('Churn Distribution')

df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, errors='ignore')

le = LabelEncoder()
categorical_cols = ['Geography', 'Gender']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('Exited', axis=1)
y = df['Exited']
scaler = StandardScaler()
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')



joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved successfully.")
# Example: Predict churn for a new customer
new_customer = X_test.iloc[0].values.reshape(1, -1)  # Take one test sample
prediction = model.predict(new_customer)
print("\nSample Prediction (0 = No Churn, 1 = Churn):", prediction[0])
plt.show()
plt.show()