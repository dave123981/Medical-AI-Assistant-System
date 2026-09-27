"""
Plain string -> string preprocessing, used identically in both this service
and the training notebook.
"""
import re

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def stem_and_clean(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = word_tokenize(text)
    stemmed = [_stemmer.stem(t) for t in tokens if len(t) > 1 and t not in _stop_words]
    return " ".join(stemmed)