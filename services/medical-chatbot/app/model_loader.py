"""
Loads three artifacts, all produced together by the same notebook run
(never mixed from different runs — same lesson as every other service's
ArtifactMismatchError):

  data/tfidf_vectorizer.joblib   Fitted TfidfVectorizer. Contains ONLY
                                  built-in sklearn behavior — see
                                  text_processing.py for why a custom
                                  tokenizer callable must never be embedded
                                  in this object.
  data/tfidf_matrix.joblib       Precomputed TF-IDF vectors for every
                                  question in the corpus, row-aligned with:
  data/qa_corpus.json            [{"question", "answer", "sources"}, ...]
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.text_processing import stem_and_clean

DATA_DIR = Path(os.getenv("CHATBOT_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
CONFIDENCE_THRESHOLD = float(os.getenv("CHATBOT_CONFIDENCE_THRESHOLD", "0.55"))


class DataLoadError(Exception):
    """Raised when a required data artifact is missing, malformed, or mismatched."""
    pass


class ChatbotArtifacts:
    def __init__(self, vectorizer, tfidf_matrix, corpus: list, version: str):
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.corpus = corpus
        self.version = version

    def retrieve(self, question: str, context_disease: "str | None" = None):
        """
        Returns (matched_corpus_entry, similarity_score). context_disease,
        when given, is concatenated into the query text before retrieval —
        a cheap soft bias toward that condition's Q&A pairs, not a hard filter.
        """
        full_query = f"{context_disease} {question}" if context_disease else question
        cleaned = stem_and_clean(full_query)
        query_vec = self.vectorizer.transform([cleaned])

        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_idx = int(np.argmax(sims))
        score = float(sims[top_idx])

        return self.corpus[top_idx], score


def _require(path: Path):
    if not path.exists():
        raise DataLoadError(
            f"Missing required data file: {path}. Run the notebook in notebooks/ "
            f"to produce it from the MedQuAD dataset."
        )


@lru_cache(maxsize=1)
def get_artifacts() -> ChatbotArtifacts:
    vectorizer_path = DATA_DIR / "tfidf_vectorizer.joblib"
    matrix_path = DATA_DIR / "tfidf_matrix.joblib"
    corpus_path = DATA_DIR / "qa_corpus.json"

    _require(vectorizer_path)
    _require(matrix_path)
    _require(corpus_path)

    vectorizer = joblib.load(vectorizer_path)
    tfidf_matrix = joblib.load(matrix_path)
    with open(corpus_path) as f:
        corpus = json.load(f)

    if tfidf_matrix.shape[0] != len(corpus):
        raise DataLoadError(
            f"tfidf_matrix.joblib has {tfidf_matrix.shape[0]} rows but qa_corpus.json "
            f"has {len(corpus)} entries — these must come from the same notebook run, "
            f"not mixed from different exports."
        )

    version = os.getenv("CHATBOT_MODEL_VERSION", "v1-tfidf-retrieval")

    return ChatbotArtifacts(vectorizer, tfidf_matrix, corpus, version)