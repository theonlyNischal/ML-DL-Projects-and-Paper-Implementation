import utils
from utils import create_folder_if_not_exists, plot_confusion_matrix, compute_metrics
from utils_logger import initialize_logger
import config
import model_dispatcher
from hyperparameter_search import perform_hyperparameter_search

import joblib
import argparse
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split



# Custom character tokenizer
import unicodedata

def character_tokenize(word):
    char = []
    i = 0
    for char_unicode in word:
        char.append(char_unicode)
        i += 1

    return char

def train(fold, model_name):
    # Load the data from train_folds.csv
    df = pd.read_csv(config.TRAINING_FILE)

    # Meta features
    # Tokenize names using the custom character tokenizer
    df['Tokenized_Name'] = df['Name'].apply(character_tokenize)
    # Join the characters back to form tokenized strings
    df['Tokenized_Name'] = df['Tokenized_Name'].apply(lambda tokens: ' '.join(tokens))
    df['Last_Letter'] = df['Tokenized_Name'].apply(lambda x: x.split(" ")[-1])
    # Encode sex labels
    # df['Sex_Label'] = df['Gender'].apply(lambda x: 0 if x == 'male' else 1)
    df["Sex_Label"] = df["Gender"].astype('int')

    # Select the observation for specified kfold
    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_valid = df[df["kfold"] == fold].reset_index(drop=True)
    # Initialize CountVectorizer with a simple identity function as the tokenizer
    vectorizer_text = CountVectorizer(tokenizer=lambda x: x)

    # Split dataset
    X_text_features = df['Tokenized_Name']
    X_last_letter = df['Last_Letter']
    y = df['Sex_Label']
    X_text_train, X_text_test, X_last_letter_train, X_last_letter_test, y_train, y_test = train_test_split(
        X_text_features, X_last_letter, y, test_size=0.2, random_state=42
    )



    # Vectorize text features
    X_text_train_vec = vectorizer_text.fit_transform(X_text_train)
    X_text_test_vec = vectorizer_text.transform(X_text_test)

    # Convert last letter to a DataFrame to be used as a feature
    X_last_letter_train_df = pd.DataFrame(X_last_letter_train, columns=['Last_Letter'])
    X_last_letter_test_df = pd.DataFrame(X_last_letter_test, columns=['Last_Letter'])

    # Initialize CountVectorizer for last letters with a simple identity function as the tokenizer
    vectorizer_last_letter = CountVectorizer(tokenizer=lambda x: x)

    # Vectorize last letters
    X_last_letter_train_vec = vectorizer_last_letter.fit_transform(X_last_letter_train_df['Last_Letter'])
    X_last_letter_test_vec = vectorizer_last_letter.transform(X_last_letter_test_df['Last_Letter'])

    # Combine both sets of features
    import scipy.sparse
    X_train_vec_combined = scipy.sparse.hstack([X_text_train_vec, X_last_letter_train_vec])
    X_test_vec_combined = scipy.sparse.hstack([X_text_test_vec, X_last_letter_test_vec])


    # Create an instance of logistic regression class
    model = perform_hyperparameter_search(X_train_vec_combined, y_train, model_name)
    # # Fit the model in the training data
    # model.fit(X_train, y_train)
    print("-----------------------------------------------")
    # # Make the prediction on the validation data
    y_pred = model.predict(X_test_vec_combined)
    # Save confusion matrix
    create_folder_if_not_exists(config.VISUAL_OUTPUT)
    plot_confusion_matrix(y_pred, y_test, np.unique(y_train), Path(config.VISUAL_OUTPUT) / f"cf_{model_name}", model_name)
    # Calculate the accuracy on the validation data
    accuracy = accuracy_score(y_pred, y_test)
    print(f"Fold: {fold}, Accuracy: {accuracy}")
    auc = roc_auc_score(y_pred, y_test)
    print(f"Fold: {fold}, AUC: {auc}")
    # Compute precision and recall
    metrics = compute_metrics(y_test, y_pred)
    logger.info(f"Model Name: {model_name}, Fold: {fold}")
    logger.info(metrics)
    print(classification_report(y_true = y_test, y_pred = y_pred))
    print("-----------------------------------------------")
    # Save the model
    create_folder_if_not_exists(config.MODEL_OUTPUT)
    joblib.dump(
        model,
        Path(config.MODEL_OUTPUT) / f"{model_name}_{fold}.bin"
    )
    # Save the vectorizer
    import dill
    with open(Path(config.MODEL_OUTPUT) / f"vectorizer_last_letter_{model_name}_{fold}.bin", 'wb') as f:
        dill.dump(vectorizer_last_letter, f)
    with open(Path(config.MODEL_OUTPUT) / f"vectorizer_text_{model_name}_{fold}.bin", 'wb') as f:
        dill.dump(vectorizer_text, f)

        
    
    return model, vectorizer_text, vectorizer_last_letter

if __name__ == "__main__":
    logger = initialize_logger()
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser()
    # Add different arguments we need
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    # read the arguments from command line
    args = parser.parse_args()
    # run the specified fold
    train(args.fold, args.model)




