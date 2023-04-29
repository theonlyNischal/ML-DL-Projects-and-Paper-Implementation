import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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
