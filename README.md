# CodSoft ML Internship - Manas Mishra

This repository contains all the tasks completed as part of the CodSoft Machine Learning Internship from June 10 – July 10, 2025.

## Tasks
- Task 1: Credit Card Fraud Detection
- Task 2: Spam SMS Detection
- Task 3: Customer Churn Prediction

Each task folder contains:
- Code
- Output visualizations
- Results summary

## Tools Used
- Python, Pandas, scikit-learn, Matplotlib, Seaborn, TF-IDF, Random Forest, Gradient Boosting, Naive Bayes

Task 1:
# Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using machine learning algorithms — **Logistic Regression** and **Random Forest Classifier**. The dataset is highly imbalanced, and we handle this using **SMOTE (Synthetic Minority Oversampling Technique)** to improve model performance.

---

## 📁 Dataset
- (https://www.kaggle.com/datasets/kartik2112/fraud-detection)

- Original Class Distribution:
  - 0 (Not Fraud): 928,074
  - 1 (Fraud): 5,274

- After SMOTE (Resampled):
  - 0 : 742,459
  - 1 : 742,459

# Logistic Regression
- Accuracy: 95%
- ROC AUC Score: 0.8558
# Evaluation Summary

| Metric             | Logistic Regression | Random Forest     |
|--------------------|---------------------|-------------------|
| Precision (Fraud)  | 0.08                | 0.13              |
| Recall (Fraud)     | 0.76                | 0.58              |
| F1-score (Fraud)   | 0.15                | 0.22              |
| ROC AUC Score      | 0.8558              | 0.7797            |

task 2:
# SMS Spam Detection using Machine Learning

This project focuses on classifying SMS messages as either **SPAM** or **HAM** using Natural Language Processing (NLP) and a machine learning classification model.

#Dataset
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- Dataset used: spam.csv
- Contains 5,572 labeled SMS messages.
- Two main categories:
  - `ham – Legitimate messages
  - `spam – Unsolicited promotional/fraudulent messages
    
Model & Approach

  - Preprocessing:
  - Tokenization
  - Stopwords removal (using NLTK)
  - Lowercasing and punctuation removal
  - Feature Extraction:
  - Bag of Words / TF-IDF Vectorizer
  - Classifier:
  - Multinomial Naive Bayes (or Logistic Regression if modified)
  - 
- Accuracy: 96%
#Classification Report

| Label | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Ham   | 0.96      | 1.00   | 0.98     | 965     |
| Spam  | 1.00      | 0.77   | 0.87     | 150     |
| Accuracy     | 0.97|1115 |

> 🔍 Note: The model performs excellently on legitimate (ham) messages, and detects most spam with a solid recall of 77%.



#Test Predictions (Examples)

- Message: "URGENT! Your account has been suspended. Click the link to verify."  
Prediction: SPAM

- Message: "Hey, are we still meeting for coffee at 5 PM?"  
  Prediction: HAM

- Message: "Win $1000 now by just replying YES to this message!"  
Prediction: HAM (false negative)

Task 3:
# Customer Churn Prediction (Bank Dataset)

This project predicts whether a customer is likely to **churn (leave the bank)** based on their profile, using logistic regression. It uses the "Churn_Modelling.csv" dataset and performs preprocessing, training, evaluation, and prediction.

---

#Dataset:https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

The dataset used is **Churn_Modelling.csv**, which contains customer details like:
- Credit Score
- Geography (country)
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Exited (Target Variable: 1 = Churned, 0 = Retained)

---

## ⚙️ Libraries Used

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

---

## Workflow

1. Load and preview the dataset
2. Check for missing values
3. Visualize class imbalance
4. Preprocess:
   - Drop irrelevant columns
   - Label encode categorical features
   - Scale numerical features
5. Train Logistic Regression model
6. Evaluate using confusion matrix and classification report
7. Save model and scaler for future predictions

