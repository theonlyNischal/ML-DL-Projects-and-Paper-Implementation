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

def train(fold, model_name):
    # Load the data from train_folds.csv
    df = pd.read_csv(config.TRAINING_FILE)
    # Meta features
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['unique_word_count'] = df['text'].apply(lambda x: len(set(str(x).split())))
    # df['url_count'] = df['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
    df['mean_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    # Clean the text
    df["text"] = df["text"].apply(utils.clean_text)
    # Select the observation for specified kfold
    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_valid = df[df["kfold"] == fold].reset_index(drop=True)
    # Using TfidfVectorizer to transform text into numerical representation
    vectorizer = TfidfVectorizer(
        # stop_words="english",
        # tokenizer=nltk.word_tokenize,
        # token_pattern=None,
        # ngram_range=(1,2)
        min_df=3,
        max_features=None, 
        strip_accents='unicode', 
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 3),
        use_idf=1,smooth_idf=1,
        sublinear_tf=1,
        stop_words = 'english'
    )
    vectorizer = CountVectorizer(
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 3),
        stop_words = 'english'
    )
    df_train["features"] = df_train["text"] + " " + df_train["word_count"].astype(str) + " " + df_train["unique_word_count"].astype(str)
    df_valid["features"] = df_valid["text"] + " " + df_valid["word_count"].astype(str) + " " + df_valid["unique_word_count"].astype(str)
    # Fit the vectorizer and transform it
    X_train = vectorizer.fit_transform(df_train["features"])
    X_val = vectorizer.transform(df_valid["features"])
    # X_train = hstack((train_text_features, df_train["word_count"]))
    # X_val = hstack((val_text_features, df_valid["word_count"]))

    # Convert the labels into numpy array
    y_train = df_train["target"].values
    y_valid = df_valid["target"].values
    # Create an instance of logistic regression class
    model = perform_hyperparameter_search(X_train, y_train, model_name)
    # # Fit the model in the training data
    # model.fit(X_train, y_train)
    print("-----------------------------------------------")
    # # Make the prediction on the validation data
    y_pred = model.predict(X_val)
    # Save confusion matrix
    create_folder_if_not_exists(config.VISUAL_OUTPUT)
    plot_confusion_matrix(y_pred, y_valid, np.unique(y_train), Path(config.VISUAL_OUTPUT) / f"cf_{model_name}", model_name)
    # Calculate the accuracy on the validation data
    accuracy = accuracy_score(y_pred, y_valid)
    print(f"Fold: {fold}, Accuracy: {accuracy}")
    auc = roc_auc_score(y_pred, y_valid)
    print(f"Fold: {fold}, AUC: {auc}")
    # Compute precision and recall
    metrics = compute_metrics(y_valid, y_pred)
    logger.info(f"Model Name: {model_name}, Fold: {fold}")
    logger.info(metrics)
    print(classification_report(y_true = y_valid, y_pred = y_pred))
    print("-----------------------------------------------")
    # Save the model
    create_folder_if_not_exists(config.MODEL_OUTPUT)
    joblib.dump(
        model,
        Path(config.MODEL_OUTPUT) / f"{model_name}_{fold}.bin"
    )
    # Save the vectorizer
    joblib.dump(
        vectorizer,
        Path(config.MODEL_OUTPUT) / f"vectorizer_{model_name}_{fold}.bin")
    
    return model, vectorizer

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




