
# Fake News Detection AI Model

## Overview

This repository contains a Fake News Detection AI model built using [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). The model aims to classify news articles as either "Fake" or "Real" based on their content. Fake news has become a significant issue in the digital age, and this model provides a machine learning-based solution to help mitigate the spread of misinformation.

## Features

- **Binary Classification**: The model classifies news articles into two categories: Fake and Real.
- **Text Preprocessing**: Includes steps like tokenization, stop-word removal, and TF-IDF vectorization.
- **Model Training**: Utilizes Logistic Regression for training the classification model.
- **Evaluation Metrics**: Accuracy, Precision, Recall, and F1-Score are used to evaluate the model's performance.

## Dataset

The dataset used for training and testing the model is sourced from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset). It consists of two CSV files:
- `Test.csv`
- `Train.csv`

