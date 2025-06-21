import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters (including emojis)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    # tokens = text.split()  # Split text on whitespace
    # stop_words = stopwords.words('english')
    # tokens = [token for token in tokens if token not in stop_words]  # Remove stop words
    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(token) for token in tokens]  # Stem the text

    return text


def create_folder_if_not_exists(folder_name: str):
    folder_path = Path(folder_name)
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

def plot_confusion_matrix(y_preds, y_true, labels, save_dir, model_name):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title(f"Normalized confusion matrix for {model_name}")
    plt.savefig(save_dir)

def mean_predictions(probas):
    """
    Create mean predictions
    :param probas: 2-d array of probability values
    :return: mean probability
    """
    return np.mean(probas, axis=1)

def max_voting(preds):
    """
    Create mean predictions
    :param probas: 2-d array of prediction values
    :return: max voted predictions
    """
    idxs = np.argmax(preds, axis=1)
    return np.take_along_axis(preds, idxs[:, None], axis=1)


from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
    return metrics


