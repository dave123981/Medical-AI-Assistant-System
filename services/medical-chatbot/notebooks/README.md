# Training Notebooks

v1 and v2 each need their own notebook (genuinely different artifact
shapes). v3 and v4 need **no notebook at all** — both reuse v2's exact
exported artifacts and are pure service-side additions, toggled by
environment variable. This is the only service in the project where a
later "version" doesn't mean a new training run.

## Dataset

**`pythonafroz/medquad-medical-question-answer-for-ai-research`** on
Kaggle — this mirror's exact column names weren't confirmed from
outside Kaggle before downloading (the canonical MedQuAD source is
GitHub XML files, not a Kaggle CSV at all) — the notebook discovers the
question/answer columns by keyword match rather than assuming exact
names, and prints them for a manual sanity check before proceeding.

~14,979 unique question/answer pairs after deduplication (the raw
download has duplicate rows — confirmed by two identical top-2 retrieval
scores appearing for the same question text during v1 testing, which is
what surfaced the duplication in the first place).

PubMedQA (also named in the original project spec) was not used for
this service — see the main service README for why.

## v1 — TF-IDF

```python
# Text preprocessing MUST be duplicated here, identical to
# app/text_processing.py's stem_and_clean() — see below for why.
def stem_and_clean(text): ...

preprocessed_questions = [stem_and_clean(q) for q in questions]

vectorizer = TfidfVectorizer()  # NO custom tokenizer= argument
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)

joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
joblib.dump(tfidf_matrix, "tfidf_matrix.joblib")
json.dump(qa_corpus, open("qa_corpus.json", "w"))
```

**Critical, verified gotcha**: never pass `tokenizer=stem_and_clean` (or
any custom Python function) directly to `TfidfVectorizer`. `joblib`
pickles a fitted vectorizer by reference to that callable's module path
— which won't resolve when the file is loaded in the service's own
process (confirmed: fails with `AttributeError: Can't get attribute
'stem_and_clean' on <module '__main__'>`, reproduced in an isolated test
before this was designed around). The fix is to pre-stem all text into
plain strings *before* fitting, so the saved vectorizer only ever
contains `TfidfVectorizer`'s own built-in tokenizer — no custom object
ever gets pickled.

**Confidence threshold calibration** (done empirically, not guessed):
off-topic test queries were scored, and the threshold set safely above
the worst case seen. The first candidate off-topic query found something
important — "recommend a good science fiction movie" scored **0.970**
against "What is (are) Good syndrome?", *higher* than genuine medical
matches scored. Root cause, confirmed by inspecting the actual matched
document: "Good" is a surname (Dr. Robert Good) misread as the common
adjective once lowercased, combined with that question reducing to just
2 tokens after stopword removal — cosine similarity's length
normalization lets a short document spike on minimal real overlap. No
single threshold can fully separate this from genuine matches (verified:
a "require ≥2 shared tokens" rule would also reject a *genuine* match in
testing). Threshold set to `0.55` based on the broader 8-query off-topic
calibration set, with this residual risk documented rather than hidden.

## v2 — Semantic embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings_matrix = model.encode(questions, show_progress_bar=True)

np.save("embeddings_matrix.npy", embeddings_matrix)
json.dump({"model_name": "all-MiniLM-L6-v2"}, open("embedding_model_name.json", "w"))
json.dump(qa_corpus, open("qa_corpus.json", "w"))  # same corpus as v1
```

No stemming needed here — embeddings work on raw text directly.

**This directly, measurably fixed v1's documented failure**: the same
"Good syndrome" test case dropped from 0.970 to 0.217 similarity — a
concrete, re-tested improvement, not just a theoretical one. The
confidence threshold was **recalibrated from scratch** (not carried over
from v1) once the score distribution's shape changed: off-topic ceiling
dropped from 0.51-0.97 (v1) to 0.12-0.38 (v2), so the threshold moved to
`0.5`, chosen with real margin above the new worst case.

## v3 — Cross-encoder re-ranking (no notebook export — service-side only)

If you want to evaluate this before enabling it in the service, reuse
v2's already-loaded `model`, `embeddings_matrix`, and `questions` in the
same Colab session:

```python
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_v3(query, top_k_retrieve=10):
    shortlist = np.argsort(cosine_similarity(model.encode([query]), embeddings_matrix)[0])[::-1][:top_k_retrieve]
    pairs = [[query, questions[i]] for i in shortlist]
    scores = cross_encoder.predict(pairs)
    return shortlist[np.argmax(scores)]
```

Tested against 3 queries: 1 changed (a case where v2 matched a
*prognosis* question to a *treatment* query — re-ranking correctly
picked the treatment answer instead), 2 stayed the same (confirming
re-ranking doesn't destabilize already-correct matches). Small sample —
worth a larger comparison (10-20 queries) before treating this as
conclusively validated rather than mechanism-confirmed.

## v4 — RAG generation (no notebook export — service-side only)

Also purely a service-side addition (`CHATBOT_ENABLE_GENERATION=true`),
using a local `google/flan-t5-base` model rather than an external paid
API — see the main service README for why that choice was made
deliberately. The generation prompt explicitly restricts the model to
the retrieved answer's content only, for safety reasons documented
there, not just phrasing.
