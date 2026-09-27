# Service 4 — Medical Chatbot

Educational Q&A only. **Does not diagnose or prescribe.**

## Run locally

```bash
cd services/medical-chatbot
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
uvicorn app.main:app --reload --port 8003
```

Docs at http://localhost:8003/docs

## Why this service is structured differently from Services 1-3

- **"Versions" here mean genuinely different techniques, not just swapped
  model files.** v1 (TF-IDF retrieval) and a future v2 (semantic
  embeddings) have completely different artifact shapes — unlike
  Services 1-3, where every version was interchangeable via
  auto-detection. There's no single generic `model_loader.py` pattern
  that spans all of Service 4's planned versions.
- **No fixed vocabulary endpoint.** Services 1-3 all expose a
  `/conditions`- or `/symptoms`-style endpoint because they need exact-match
  input. This service's entire point is handling arbitrary natural
  language — a dropdown of "known questions" would defeat the purpose.
  The confidence flag on every response is the equivalent safety
  mechanism instead of a fixed vocabulary gate.
- **A critical serialization detail:** `data/tfidf_vectorizer.joblib`
  must NEVER be fit with a custom Python callable (e.g.
  `tokenizer=some_function`) — `joblib` pickles such vectorizers by
  reference to the callable's module path, which breaks the moment the
  file is loaded in a different process (verified: fails with
  `AttributeError: Can't get attribute ... on <module '__main__'>`).
  Text is pre-stemmed into plain strings *before* fitting, using
  `app/text_processing.py`'s `stem_and_clean()` — the exact same
  function is duplicated (not imported) in the training notebook, since
  it never gets pickled itself, only its plain-string output does.

## Before it will actually answer anything

Drop into `data/`:
- `tfidf_vectorizer.joblib` — fitted on pre-stemmed corpus text only
- `tfidf_matrix.joblib` — precomputed TF-IDF vectors for every question
- `qa_corpus.json` — `[{"question", "answer", "sources"}, ...]`, row-aligned with the matrix

## Endpoints

- `GET /health`
- `POST /ask` — `{question, context_disease?, conversation_id?}`
  - `context_disease` softly biases retrieval (concatenated into the
    query before matching) — not a hard filter
  - `conversation_id` is accepted but unused in v1 (no conversation
    memory yet)
  - Low-confidence matches (`similarity_score` below
    `CHATBOT_CONFIDENCE_THRESHOLD`) still return an answer, flagged
    `confident: false` — unlike Service 3's hard "no match found",
    free-text similarity is a graded scale, not a clean exists/doesn't
    binary

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CHATBOT_DATA_DIR` | `./data` | Where the three artifacts are read from |
| `CHATBOT_CONFIDENCE_THRESHOLD` | `0.3` | Below this similarity score, `confident: false` |
| `CHATBOT_MODEL_VERSION` | `v1-tfidf-retrieval` | Reported version string |