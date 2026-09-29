# Training Notebooks

Not really "training" in the model sense — v1's ranking is a weighted
aggregation computed once from the dataset, not a fitted model. Runs
fine on Colab's free CPU-only runtime; no GPU needed.

## Dataset

**`jessicali9530/kuc-hackathon-winter-2018`** on Kaggle (UCI ML Drug
Review Dataset) — two files, `drugsComTrain_raw.csv` and
`drugsComTest_raw.csv` (despite some documentation elsewhere describing
this dataset as tab-separated `.tsv`, **this specific Kaggle mirror
ships genuine comma-separated `.csv` files** — verified directly by
reading the raw first line before assuming a delimiter; guessing wrong
here doesn't error, it silently produces one garbled column instead of
seven clean ones).

Columns: `uniqueID`, `drugName`, `condition`, `review`, `rating` (1-10),
`date` (format `%d-%b-%y` — **2-digit year**, not `%Y`), `usefulCount`.

## A real scraping artifact, found and handled

~1,171 rows have `condition` values like `"3</span> users found this
comment helpful."` (80 distinct numeric variants) — a scraping bug that
picked up a UI "helpful count" label instead of the actual condition for
certain review pages. These were **not** caught by checking the longest
condition strings (they're short, so they didn't show up in that
check) — they were only found by specifically searching for HTML-tag
characters (`<`, `>`) in the condition column. If re-running this
notebook from scratch, check for this pattern specifically rather than
assuming a length-based junk filter is sufficient.

```python
junk_mask = df["condition"].str.contains("<|>", regex=True, na=False)
df = df[~junk_mask]
```

## Ranking formula (v1)

For each `(condition, drug)` pair with at least `MIN_REVIEWS = 5`
reviews (rare pairs below this threshold are dropped — a single 10/10
review from one person should not be able to outrank a drug with
hundreds of consistent reviews):


This is a documented approach from existing public analysis of this
exact dataset, not an invented formula.

## What the notebook must export

```python
json.dump(rankings, open("drug_rankings.json", "w"))
# {condition: [{"drug", "score", "review_count"}, ...], top 10-15 per condition}

json.dump(contraindication_rules, open("contraindication_rules.json", "w"))
# hand-curated, NOT derived from the review data — see the service README
```

The review dataset contains no chemical, allergy, or interaction data
whatsoever — `contraindication_rules.json` must be written by hand, with
a small number of well-known, publicly-documented facts, explicitly
labeled illustrative rather than exhaustive or clinically authoritative.
