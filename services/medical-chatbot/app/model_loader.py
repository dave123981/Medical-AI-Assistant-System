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
    data/embeddings_matrix.npy     Precomputed dense embeddings.
    data/embedding_model_name.json {"model_name": "all-MiniLM-L6-v2"}

  Both versions share:
    data/qa_corpus.json             [{"question", "answer", "sources"}, ...]

  v3 (cross-encoder re-ranking) and v4 (RAG generation) need NO new
  artifacts — both are pure service-side additions on top of v2's exact
  same embeddings, each independently toggled by its own env var, so the
  same exported v2 files can be A/B compared across all four modes
  without any re-export:

    CHATBOT_ENABLE_RERANKING=true    -> v3: retrieve, then re-rank shortlist
    CHATBOT_ENABLE_GENERATION=true   -> v4: retrieve (+ optional rerank),
                                        then generate a response GROUNDED
                                        in the retrieved answer, rather
                                        than returning it verbatim
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
CONFIDENCE_THRESHOLD = float(os.getenv("CHATBOT_CONFIDENCE_THRESHOLD", "0.5"))

ENABLE_RERANKING = os.getenv("CHATBOT_ENABLE_RERANKING", "false").lower() == "true"
CROSS_ENCODER_MODEL_NAME = os.getenv("CHATBOT_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_SHORTLIST_SIZE = int(os.getenv("CHATBOT_RERANK_SHORTLIST_SIZE", "10"))

ENABLE_GENERATION = os.getenv("CHATBOT_ENABLE_GENERATION", "false").lower() == "true"
GENERATION_MODEL_NAME = os.getenv("CHATBOT_GENERATION_MODEL", "google/flan-t5-base")
GENERATION_MAX_NEW_TOKENS = int(os.getenv("CHATBOT_GENERATION_MAX_NEW_TOKENS", "200"))


class DataLoadError(Exception):
    """Raised when a required data artifact is missing, malformed, or mismatched."""
    pass


def _build_generation_prompt(question: str, retrieved_answer: str) -> str:
    """
    Deliberately instructs the model to use ONLY the retrieved answer as
    its source of truth. This is a safety-relevant design choice, not
    just a phrasing preference — an ungrounded small LLM could otherwise
    generate fluent-sounding but unsourced medical claims, which is
    exactly what this whole service exists to avoid.
    """
    return (
        "Based on the following medical information, answer the question "
        "clearly and helpfully. Only use the information provided — do not "
        "add anything not stated in it.\n\n"
        f"Information: {retrieved_answer}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


class ChatbotArtifacts:
    def __init__(self, retrieval_mode: str, corpus: list, version: str,
                 vectorizer=None, tfidf_matrix=None,
                 embedding_model=None, embeddings_matrix=None,
                 cross_encoder=None, generator=None):
        self.retrieval_mode = retrieval_mode  # "tfidf" or "embeddings"
        self.corpus = corpus
        self.version = version
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.embedding_model = embedding_model
        self.embeddings_matrix = embeddings_matrix
        self.cross_encoder = cross_encoder  # None unless v3 re-ranking is active
        self.generator = generator          # None unless v4 generation is active

    def retrieve(self, question: str, context_disease: "str | None" = None):
        """
        Returns (matched_corpus_entry, similarity_score, generated_answer).
        generated_answer is None unless v4 generation is active — callers
        that only care about retrieval (v1-v3 behavior) can ignore it.
        """
        full_query = f"{context_disease} {question}" if context_disease else question

        if self.retrieval_mode == "embeddings":
            if self.cross_encoder is not None:
                entry, score = self._retrieve_with_reranking(full_query)
            else:
                query_vec = self.embedding_model.encode([full_query])
                sims = cosine_similarity(query_vec, self.embeddings_matrix)[0]
                top_idx = int(np.argmax(sims))
                entry, score = self.corpus[top_idx], float(sims[top_idx])
        else:
            cleaned = stem_and_clean(full_query)
            query_vec = self.vectorizer.transform([cleaned])
            sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            top_idx = int(np.argmax(sims))
            entry, score = self.corpus[top_idx], float(sims[top_idx])

        generated_answer = None
        if self.generator is not None:
            # Generation quality is entirely dependent on retrieval quality —
            # a low-confidence retrieval STILL gets generated from (matching
            # v1-v3's "always return something, flag low confidence" pattern
            # rather than silently blocking), but a fluent-sounding generated
            # answer grounded in a bad match will itself be a bad answer.
            # This is a known, documented limitation, not something v4 fixes.
            prompt = _build_generation_prompt(question, entry["answer"])
            result = self.generator(prompt, max_new_tokens=GENERATION_MAX_NEW_TOKENS)
            generated_answer = result[0]["generated_text"].strip()

        return entry, score, generated_answer

    def _retrieve_with_reranking(self, full_query: str):
        query_vec = self.embedding_model.encode([full_query])
        sims = cosine_similarity(query_vec, self.embeddings_matrix)[0]
        shortlist_idx = np.argsort(sims)[::-1][:RERANK_SHORTLIST_SIZE]

        pairs = [[full_query, self.corpus[i]["question"]] for i in shortlist_idx]
        cross_scores = self.cross_encoder.predict(pairs)

        best_pos = int(np.argmax(cross_scores))
        best_idx = shortlist_idx[best_pos]
        score = float(1 / (1 + np.exp(-cross_scores[best_pos])))

        return self.corpus[best_idx], score


def _require(path: Path):
    if not path.exists():
        raise DataLoadError(
            f"Missing required data file: {path}. Run the notebook in notebooks/ "
            f"to produce it from the MedQuAD dataset."
        )


def _load_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
    except ModuleNotFoundError:
        raise DataLoadError(
            "CHATBOT_ENABLE_RERANKING is true but 'sentence-transformers' isn't "
            "installed. pip install sentence-transformers, then add it to requirements.txt."
        ) from None
    try:
        return CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    except Exception as e:
        raise DataLoadError(
            f"CHATBOT_ENABLE_RERANKING is true but failed to load cross-encoder "
            f"'{CROSS_ENCODER_MODEL_NAME}': {e}. This model downloads from "
            f"HuggingFace on first use — confirm internet access is available."
        ) from None


def _load_generator():
    try:
        from transformers import pipeline
    except ModuleNotFoundError:
        raise DataLoadError(
            "CHATBOT_ENABLE_GENERATION is true but 'transformers' isn't "
            "installed. pip install transformers, then add it to requirements.txt."
        ) from None
    try:
        return pipeline("text2text-generation", model=GENERATION_MODEL_NAME)
    except Exception as e:
        raise DataLoadError(
            f"CHATBOT_ENABLE_GENERATION is true but failed to load generation "
            f"model '{GENERATION_MODEL_NAME}': {e}. This model downloads from "
            f"HuggingFace on first use — confirm internet access is available."
        ) from None


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
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError:
        raise DataLoadError(
            "embeddings_matrix.npy is present but 'sentence-transformers' isn't "
            "installed. pip install sentence-transformers, then add it to requirements.txt."
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

        cross_encoder = None
        version_parts = [f"v2-embeddings-{model_name}"]
        if ENABLE_RERANKING:
            cross_encoder = _load_cross_encoder()
            version_parts = [f"v3-reranked-{model_name}"]

        generator = None
        if ENABLE_GENERATION:
            generator = _load_generator()
            version_parts.append(f"v4-generated-{GENERATION_MODEL_NAME}")

        version = os.getenv("CHATBOT_MODEL_VERSION", "+".join(version_parts))

        return ChatbotArtifacts(
            "embeddings", corpus, version,
            embedding_model=embedding_model, embeddings_matrix=embeddings_matrix,
            cross_encoder=cross_encoder, generator=generator,
        )

    vectorizer, tfidf_matrix = _load_tfidf_artifacts(corpus)
    version = os.getenv("CHATBOT_MODEL_VERSION", "v1-tfidf-retrieval")
    return ChatbotArtifacts(
        "tfidf", corpus, version,
        vectorizer=vectorizer, tfidf_matrix=tfidf_matrix,
    )