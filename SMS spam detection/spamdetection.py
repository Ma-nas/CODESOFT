
import pandas as pd
import numpy as np
import string
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load CSV (dataset file is in the same folder)
data = pd.read_csv('C:/Users/hp/my-python-app/SMS spam detection/spam.csv', encoding='latin-1')
                     

# Keep only the useful columns
data = data[['v1', 'v2']]
data = data.rename(columns={'v1': 'label', 'v2': 'text'})

# Map labels to binary (ham = 0, spam = 1)
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Apply cleaning
data['clean_text'] = data['text'].apply(clean_text)
X = data['clean_text']
y = data['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")


# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

def predict_spam(message):
    message_clean = clean_text(message)
    message_vec = vectorizer.transform([message_clean])
    prediction = model.predict(message_vec)[0]
    return "SPAM" if prediction == 1 else "HAM"

# Test example
'''print("\nTest Prediction:")
test_msg = "Congratulations! You've won a free ticket. Call now!"
result = predict_spam(test_msg)
print(f"Message: {test_msg}")
print(f"Prediction: {result}")'''
examples = [
    "URGENT! Your account has been suspended. Click the link to verify.",
    "Hey, are we still meeting for coffee at 5 PM?",
    "Hi dear,had your dinner?",
    "Double Dollar offer live on Timezone Fun App starting at Rs.3500. Offer valid till 22nd June. Reload Now"
]

print("\n Test Predictions")
for msg in examples:
    result = predict_spam(msg)
    print(f"\nMessage: {msg}")
    print(f"Prediction: {result}")
plt.show()