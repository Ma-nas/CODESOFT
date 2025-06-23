import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv(r"C:/Users/hp/my-python-app/credit card fraud detection/fraudTrain.csv")

print("Class Distribution:\n", df['is_fraud'].value_counts())

# Data Preprocessing
df = df.drop(columns=['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant',
                      'first', 'last', 'street', 'city', 'state', 'zip', 'job', 'dob',
                      'trans_num', 'unix_time', 'merch_lat', 'merch_long', 'category'])

df = df.dropna()

X= df.drop('is_fraud', axis=1)
y= df['is_fraud']

if 'gender' in X.columns:
    X['gender'] = X['gender'].map({'M':0, 'F':1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handling Imbalanced Dataset using SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", pd.Series(y_train_res).value_counts())

# Train Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_res, y_train_res)
log_preds = log_model.predict(X_test_scaled)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_res, y_train_res)
rf_preds = rf_model.predict(X_test_scaled)

# Evaluation - Logistic Regression
print("\n--- Logistic Regression Report ---")
print(classification_report(y_test, log_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_preds))
print("ROC AUC Score:", roc_auc_score(y_test, log_preds))

# Evaluation - Random Forest
print("\n--- Random Forest Report ---")
print(classification_report(y_test, rf_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))
print("ROC AUC Score:", roc_auc_score(y_test, rf_preds))

# Visualization - Random Forest Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

