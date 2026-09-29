# Training Notebooks

Training happens in Google Colab, not in this repo's runtime.

## Dataset

**`dhivyeshrk/diseases-and-symptoms-dataset`** on Kaggle — download via
`kagglehub.dataset_download("dhivyeshrk/diseases-and-symptoms-dataset")`.
773 diseases, 377 symptoms, already one-hot encoded (no free-text
symptom parsing needed, unlike some other disease datasets).

**Do not use `itachi9604/disease-symptom-description-dataset`** — that
dataset's ~5,000 rows are almost entirely duplicates of ~10 unique
symptom combinations per disease, padded to a round row count. A model
trained on it will show ~100% accuracy from memorization, not real
generalization. This was discovered and documented during v1 training;
the project switched datasets mid-way as a result.

## What every notebook must export

```python
import json, joblib

joblib.dump(model, "vN_modelname.joblib")   # or model.save(...) for Keras
json.dump(symptom_vocab, open("symptom_vocab.json", "w"))
json.dump(label_classes, open("label_classes.json", "w"))
```

`label_classes` must be built from the model's own `.classes_` (sklearn)
or the dense 0..N-1 remapping actually used at training time (XGBoost,
Keras) — **not** the original `LabelEncoder`'s full class list. After
filtering rare diseases (`MIN_SAMPLES_PER_CLASS = 5`), the surviving
class set has gaps in the original encoding; exporting the wrong
ordering silently misaligns every prediction to the wrong disease name
without erroring. See the v1 notebook's export cell for the exact
pattern, and the v3 (XGBoost) notebook for how this gets remapped when
the model needs a dense label range.

## Version progression and results

| Version | Model | Result |
|---|---|---|
| v1 | Decision Tree | 81.6% accuracy |
| v2 | Random Forest | 85.2% accuracy — required `max_leaf_nodes` (not `max_depth`) to control memory usage on Colab's free tier; an earlier attempt using `max_depth`/`min_samples_leaf` caused a 15-point accuracy regression, since it depth-capped uniformly rather than only where memory demanded it |
| v3 | XGBoost | 83.6% accuracy — genuinely underperformed v2 here; documented as a real result, not re-tuned indefinitely to beat v2 |
| v4 | Neural Network | 86.4% accuracy, `class_weight="balanced"` — verified against train/val loss curves before trusting the number (mild overfitting gap present but val loss plateaued rather than climbed, confirming the result wasn't a training artifact) |

Each version's notebook includes a class-distribution sanity check
before training — with 773 classes and real-world data, several classes
have very few samples, and blindly train/test splitting without
checking this produces a `ValueError` (`least populated class has only
1 member`) rather than a silent problem.
