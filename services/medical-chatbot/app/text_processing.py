"""
Plain string -> string preprocessing, used identically in both this service
and the training notebook.
"""
import re

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def _ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    """
    Auto-downloads the given NLTK resource if it isn't already present,
    """
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


_ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
_ensure_nltk_resource("corpora/stopwords", "stopwords")

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def stem_and_clean(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = word_tokenize(text)
    stemmed = [_stemmer.stem(t) for t in tokens if len(t) > 1 and t not in _stop_words]
    return " ".join(stemmed)