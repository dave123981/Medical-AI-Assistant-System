# Service 4 — Medical Chatbot

Educational Q&A only. **Does not diagnose or prescribe.** Complete
through v4 — the only service in this project whose four versions are
genuinely different techniques (not just swapped model files), and the
only one that grew a fully self-contained RAG generation stage.

## Run locally

```bash
cd services/medical-chatbot
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8003
```

Docs at http://localhost:8003/docs

NLTK data (stopwords, tokenizer) downloads automatically on first
import — no manual setup step required.

## Dataset

**`pythonafroz/medquad-medical-question-answer-for-ai-research`** on
Kaggle (MedQuAD) — deduplicated to ~14,979 unique question/answer
pairs. PubMedQA (also named in the original project spec) was not used
— it's structured as yes/no/maybe classification over research
abstracts, a poor fit for "explain this condition/medication/procedure"
style Q&A.

## Version progression — genuinely different techniques, not swapped files

Unlike Services 1-3, v2 and v3/v4 use **completely different artifact
shapes** (a fitted TF-IDF vectorizer vs. precomputed dense embeddings),
so there's no single generic model-loading pattern spanning all four
versions the way there is elsewhere in this project.

| Version | Technique | Key finding |
|---|---|---|
| v1 | TF-IDF + cosine similarity | Confidence threshold calibrated to `0.55` after finding an off-topic query ("recommend a good science fiction movie") scored **0.970** against "What is (are) Good syndrome?" — higher than genuine medical question matches. Root cause: "Good" as a surname vs. adjective, combined with that question's very short token count after stopword removal. |
| v2 | Semantic embeddings (`all-MiniLM-L6-v2`) | Fixed v1's exact failure case — the same query now scores **0.217** against the same question, a 78% reduction. Off-topic query ceiling dropped from 0.51-0.97 (v1, overlapping with genuine matches) to 0.12-0.38 (v2, cleanly separated). Threshold recalibrated to `0.5`. |
| v3 | Cross-encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`), **opt-in** | Reuses v2's exact same embeddings — no re-export needed. Demonstrated correcting a real case: v2 matched a treatment question to a *prognosis* answer ("What is the outlook for Migraine?"); re-ranking correctly picked the treatment answer instead. |
| v4 | Local RAG generation (`google/flan-t5-base`), **opt-in** | Also reuses v2's embeddings. Generates a response grounded in the retrieved answer rather than returning it verbatim — see the grounding/safety design below. |

**v3 and v4 are both opt-in via environment variable**, not
auto-detected from files present — since the underlying data doesn't
change, only the inference-time code path does. This lets you A/B
compare all four behaviors using the exact same exported v2 artifacts.

## A critical serialization detail

`data/tfidf_vectorizer.joblib` must **never** be fit with a custom
Python callable (e.g. `tokenizer=some_function`) — `joblib` pickles
such vectorizers by reference to the callable's module path, which
breaks the moment the file is loaded in a different process (verified:
fails with `AttributeError: Can't get attribute ... on <module
'__main__'>`). Text is pre-stemmed into plain strings *before* fitting,
using `app/text_processing.py`'s `stem_and_clean()` — the exact same
function is duplicated (not imported) in the training notebook, since
it never gets pickled itself, only its plain-string output does.

## Why generation is grounded, not free-form

v4's prompt explicitly instructs the model: *"Only use the information
provided — do not add anything not stated in it."* This is a
safety-relevant design choice, not a phrasing preference — an
ungrounded small LLM could otherwise generate fluent-sounding but
unsourced medical claims, which is exactly what this whole service
exists to avoid. `source_answer` is always included in the response
alongside the generated `answer`, so a caller can verify the generated
text against its actual retrieved source.

**A local, open-source model was chosen over an external API
(OpenAI/Anthropic/etc.) deliberately** — every other part of this
project is fully self-contained with no per-request cost or external
API key requirement, and v4 preserves that rather than introducing the
one paid dependency in an otherwise free, offline-capable system.

## Why this service has no fixed vocabulary endpoint

Services 1-3 all expose a `/conditions`- or `/symptoms`-style endpoint
because they need exact-match input. This service's entire point is
handling arbitrary natural language — a dropdown of "known questions"
would defeat the purpose. The `confident` flag on every response is the
equivalent safety mechanism instead of a fixed vocabulary gate.

## Before it will actually answer anything

Drop into `data/`, depending on which version you want active:

**v1 (TF-IDF):**
- `tfidf_vectorizer.joblib`, `tfidf_matrix.joblib`, `qa_corpus.json`

**v2+ (embeddings — takes priority if both artifact sets are present):**
- `embeddings_matrix.npy`, `embedding_model_name.json`, `qa_corpus.json`

## Endpoints

- `GET /health`
- `POST /ask` — `{question, context_disease?, conversation_id?}`

### Response shape

```json
{
  "answer": "...",
  "generated": false,
  "source_answer": "...",
  "matched_question": "What is the treatment for migraine?",
  "similarity_score": 0.82,
  "confident": true,
  "sources": ["MedQuAD"],
  "disclaimer": "Educational information only...",
  "model_version": "v2-embeddings-all-MiniLM-L6-v2"
}
```

- `context_disease` softly biases retrieval (concatenated into the
  query before matching) — not a hard filter.
- `conversation_id` is accepted but unused (no conversation memory yet).
- `generated: true` only when `CHATBOT_ENABLE_GENERATION` is active;
  `source_answer` is always the raw retrieved text regardless, so a
  generated answer can always be checked against its source.
- Low-confidence matches still return an answer, flagged `confident:
  false` — unlike Service 3's hard "no match found," free-text
  similarity is a graded scale, not a clean exists/doesn't binary.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CHATBOT_DATA_DIR` | `./data` | Where artifacts are read from |
| `CHATBOT_CONFIDENCE_THRESHOLD` | `0.5` | Calibrated for v2's embedding distribution; v1 used `0.55` |
| `CHATBOT_MODEL_VERSION` | derived automatically | Override the reported version string |
| `CHATBOT_ENABLE_RERANKING` | `false` | Enables v3's cross-encoder re-ranking |
| `CHATBOT_CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Which cross-encoder to load |
| `CHATBOT_RERANK_SHORTLIST_SIZE` | `10` | How many bi-encoder candidates get re-scored |
| `CHATBOT_ENABLE_GENERATION` | `false` | Enables v4's RAG generation |
| `CHATBOT_GENERATION_MODEL` | `google/flan-t5-base` | Which local generation model to load |
| `CHATBOT_GENERATION_MAX_NEW_TOKENS` | `200` | Generation length cap |
