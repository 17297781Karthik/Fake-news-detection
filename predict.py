import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import os

def load_model_and_vectorizer(model_path, vectorizer_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Usage
model_path = 'logistic_regression_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'
model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)


# Initialize the stemmer
port_stem = PorterStemmer()

def preprocess_text(text):
    """Preprocess the input text by stemming and removing stopwords."""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def predict_fake_news(text):
    """Predict whether the input text is fake news."""
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    print(f"Preprocessed text: {preprocessed_text}")

    # Transform the text using the vectorizer
    transformed_text = vectorizer.transform([preprocessed_text])
    print(f"Transformed text shape: {transformed_text.shape}")

    # Make a prediction using the model
    prediction = model.predict(transformed_text)
    print(f"Prediction: {prediction}")

    # Return the prediction result
    return 'Fake News' if prediction[0] == 1 else 'Real News'

if __name__ == "__main__":
    # Example usage
    news_paragraph = input("Enter a news paragraph: ")
    result = predict_fake_news(news_paragraph)
    print('\n')
    print(f"The news is: {result}")