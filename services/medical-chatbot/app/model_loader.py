"""
Loads whichever retrieval artifacts are present in data/, auto-detecting
between two genuinely different techniques:

  v1 (TF-IDF):
    data/tfidf_vectorizer.joblib   Fitted TfidfVectorizer — built-in
                                     sklearn behavior only, see
                                     text_processing.py for why.
    data/tfidf_matrix.joblib       Precomputed TF-IDF vectors, row-aligned
                                     with qa_corpus.json.

  v2 (semantic embeddings):
    data/embeddings_matrix.npy     Precomputed dense embeddings (plain
                                     numpy array — no custom-object
                                     pickling concerns at all, simpler
                                     than v1 in this respect).
    data/embedding_model_name.json {"model_name": "all-MiniLM-L6-v2"} —
                                     records which sentence-transformers
                                     model produced the embeddings, so the
                                     service loads the SAME model to embed
                                     incoming queries into the same space.

  Both versions share:
    data/qa_corpus.json             [{"question", "answer", "sources"}, ...]

v2's artifacts take priority if both are present — same "drop a file,
get a new version" pattern as every other service, even though the
underlying technique is genuinely different, not just a swapped model file.
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
    def __init__(self, retrieval_mode: str, corpus: list, version: str,
                 vectorizer=None, tfidf_matrix=None,
                 embedding_model=None, embeddings_matrix=None):
        self.retrieval_mode = retrieval_mode  # "tfidf" or "embeddings"
        self.corpus = corpus
        self.version = version
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.embedding_model = embedding_model
        self.embeddings_matrix = embeddings_matrix

    def retrieve(self, question: str, context_disease: "str | None" = None):
        """
        Returns (matched_corpus_entry, similarity_score). context_disease,
        when given, is concatenated into the query text before retrieval —
        a soft bias, not a hard filter. Branches on retrieval_mode, but the
        contract (what it returns) is identical either way.
        """
        full_query = f"{context_disease} {question}" if context_disease else question

        if self.retrieval_mode == "embeddings":
            query_vec = self.embedding_model.encode([full_query])
            sims = cosine_similarity(query_vec, self.embeddings_matrix)[0]
        else:
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


def _load_embeddings_artifacts(corpus: list):
    embeddings_path = DATA_DIR / "embeddings_matrix.npy"
    model_name_path = DATA_DIR / "embedding_model_name.json"

    _require(embeddings_path)
    _require(model_name_path)

    embeddings_matrix = np.load(embeddings_path)

    if embeddings_matrix.shape[0] != len(corpus):
        raise DataLoadError(
            f"embeddings_matrix.npy has {embeddings_matrix.shape[0]} rows but "
            f"qa_corpus.json has {len(corpus)} entries — these must come from "
            f"the same notebook run, not mixed from different exports."
        )

    with open(model_name_path) as f:
        model_name = json.load(f)["model_name"]

    try:
        # Imported lazily so v1-only deployments don't need this heavy
        # dependency installed at all — same pattern as TensorFlow being
        # lazy-imported in the imaging service's model_loader.py.
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError:
        raise DataLoadError(
            "embeddings_matrix.npy is present but the 'sentence-transformers' "
            "package isn't installed. pip install sentence-transformers, then "
            "add it to requirements.txt."
        ) from None

    try:
        embedding_model = SentenceTransformer(model_name)
    except Exception as e:
        raise DataLoadError(
            f"Failed to load embedding model '{model_name}': {e}. This model is "
            f"downloaded from HuggingFace on first use — confirm internet access "
            f"is available the first time this service starts."
        ) from None

    return embedding_model, embeddings_matrix, model_name


def _load_tfidf_artifacts(corpus: list):
    vectorizer_path = DATA_DIR / "tfidf_vectorizer.joblib"
    matrix_path = DATA_DIR / "tfidf_matrix.joblib"

    _require(vectorizer_path)
    _require(matrix_path)

    vectorizer = joblib.load(vectorizer_path)
    tfidf_matrix = joblib.load(matrix_path)

    if tfidf_matrix.shape[0] != len(corpus):
        raise DataLoadError(
            f"tfidf_matrix.joblib has {tfidf_matrix.shape[0]} rows but "
            f"qa_corpus.json has {len(corpus)} entries — these must come from "
            f"the same notebook run, not mixed from different exports."
        )

    return vectorizer, tfidf_matrix


@lru_cache(maxsize=1)
def get_artifacts() -> ChatbotArtifacts:
    corpus_path = DATA_DIR / "qa_corpus.json"
    _require(corpus_path)
    with open(corpus_path) as f:
        corpus = json.load(f)

    embeddings_present = (DATA_DIR / "embeddings_matrix.npy").exists()

    if embeddings_present:
        embedding_model, embeddings_matrix, model_name = _load_embeddings_artifacts(corpus)
        version = os.getenv("CHATBOT_MODEL_VERSION", f"v2-embeddings-{model_name}")
        return ChatbotArtifacts(
            "embeddings", corpus, version,
            embedding_model=embedding_model, embeddings_matrix=embeddings_matrix,
        )

    vectorizer, tfidf_matrix = _load_tfidf_artifacts(corpus)
    version = os.getenv("CHATBOT_MODEL_VERSION", "v1-tfidf-retrieval")
    return ChatbotArtifacts(
        "tfidf", corpus, version,
        vectorizer=vectorizer, tfidf_matrix=tfidf_matrix,
    )