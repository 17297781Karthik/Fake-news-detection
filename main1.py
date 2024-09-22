import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import nltk
# nltk.download('stopwords')

print(stopwords.words('english'))

# Data preprocessing
news_dataset = pd.read_csv('train.csv/train.csv')
print(news_dataset.shape)
print(news_dataset.head())
print(news_dataset.isnull().sum())
news_dataset = news_dataset.fillna('')

# Merging author name and title
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

# Stemming: the process of reducing words to their root words
port_stem = PorterStemmer()
def stemming(content):
 stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
 stemmed_content = stemmed_content.lower()
 stemmed_content = stemmed_content.split()
 stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
 stemmed_content = ' '.join(stemmed_content)
 return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)
print(news_dataset['content'])

X = news_dataset['content'].values
Y = news_dataset['label'].values
print(X)
print(Y)

vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluation
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Training Accuracy Score:', training_data_accuracy)

# Testing
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Testing Accuracy Score:', testing_data_accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, X_test_prediction)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, y_prob)
roc_auc = roc_auc_score(Y_test, y_prob)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
