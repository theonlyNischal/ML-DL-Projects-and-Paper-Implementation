import utils

import dill
import joblib
import pandas as pd
import numpy as np


# Custom character tokenizer
import unicodedata

def character_tokenize(word):
    char = []
    i = 0
    for char_unicode in word:
        char.append(char_unicode)
        i += 1

    return char


df = pd.read_csv("input/test.csv")

# Load the saved model
best_model = joblib.load("models/logistic_regression_0.bin")
# Load the saved vectorizer

with open('models/vectorizer_text_logistic_regression_0.bin', 'rb') as f:
    vectorizer_text = dill.load(f)

with open('models/vectorizer_last_letter_logistic_regression_0.bin', 'rb') as f:
    vectorizer_last_letter = dill.load(f)


import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer


def gender_prediction(name):
    tokenized_new_name = ' '.join(character_tokenize(name))
    last_letter_new_name = tokenized_new_name[-1]

    X_new_name_text = vectorizer_text.transform([tokenized_new_name])
    X_new_name_last_letter = vectorizer_last_letter.transform([last_letter_new_name])
    X_new_name_combined = scipy.sparse.hstack([X_new_name_text, X_new_name_last_letter])

    predicted_label = best_model.predict(X_new_name_combined)[0]
    predicted_gender = 'Male' if predicted_label == 1 else 'Female'

    # Get the confidence scores (probabilities)
    confidence_scores = best_model.predict_proba(X_new_name_combined)[0]

    print(
        f"{name}: {predicted_gender}, {confidence_scores}"

    )


if __name__ == "__main__":
    name = "Sunita"
    gender_prediction(name)

