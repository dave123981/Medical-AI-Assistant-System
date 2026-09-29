# Training Notebooks

Training happens in Google Colab. Both image types' notebooks live here

## Datasets

| Image type | Dataset | Notes |
|---|---|---|
| Chest X-ray v1 | `nih-chest-xrays/sample` | Small (5,606 images) — used to prove the pipeline works before committing to the full download |
| Chest X-ray v2+ | `khanfashee/nih-chest-x-ray-14-224x224-resized` | Full 112,120 images, pre-resized. Labels (`Data_Entry_2017.csv`) may or may not be bundled with this mirror — the notebook validates the CSV's row count and columns before trusting it, falling back to `kagglehub.dataset_download("nih-chest-xrays/data", path="Data_Entry_2017.csv")` (a single-file fetch, not the full 42GB dataset) if the bundled file looks wrong |
| Skin lesion (all versions) | `kmader/skin-cancer-mnist-ham10000` | 10,015 images, 7 classes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`), metadata in `HAM10000_metadata.csv` |

## Chest X-ray is multi-label; skin lesion is multiclass

This is the single most important structural difference between the two
image types' notebooks:

- **Chest X-ray**: a scan can show multiple conditions simultaneously (or
  none). Output layer is `sigmoid`, loss is `binary_crossentropy`
  (or the custom weighted variant in v3), labels are multi-hot vectors.
- **Skin lesion**: each image has exactly one diagnosis. Output layer
  is `softmax`, loss is `sparse_categorical_crossentropy`, labels are
  single integers.


## Known dataset issues, found and handled

- **Chest X-ray labels CSV**: a bundled CSV that's actually a tiny
  unrelated metadata file (not the real ~112k-row label set) was
  encountered once — the notebook validates row count (`>= 50000`) and
  required columns before accepting any discovered CSV as the real
  labels file, rather than trusting the first `.csv` found.
- **Skin lesion image-level split**: the same physical lesion can appear
  multiple times (different photos) and land in both train and val — a
  known, documented limitation of HAM10000, not fixed here. A proper
  lesion-level split (grouping by `lesion_id`) is the natural v5
  direction if pursued further.
- **EfficientNetB0's input scale**: unlike DenseNet121, EfficientNet has
  its own `Rescaling` layer built in as literally its second layer — it
  expects raw `[0, 255]` pixel input, not `/255`-normalized `[0, 1]`.
  Verified directly: `efficientnet.preprocess_input` is a genuine
  pass-through no-op. Feeding it normalized input silently wrecks
  training (no error, just badly degraded results) — the skin lesion
  v2+ notebooks deliberately skip the `/255` step present in every other
  notebook in this project.

## Version results

### Chest X-ray

| Version | Approach | Result |
|---|---|---|
| v1 | Baseline CNN, 128×128, small sample | 0.608 mean AUC |
| v2 | DenseNet121 transfer learning, full dataset (25k subset), 224×224 | 0.734 mean AUC |
| v3 | CheXNet-style weighted binary cross-entropy | **Null result** — mean AUC unchanged; Pneumonia and Hernia (the specifically targeted rare classes) slightly regressed. Ran the full 15-epoch schedule per phase, ruling out early stopping as the cause. |
| v4 | Grad-CAM (re-uploads v2's weights, not v3's — v3 didn't improve on v2) + F1-optimal per-class thresholds | See findings below |

**v2's Random Forest memory lesson carried over conceptually**: v2's
initial `RandomForestClassifier` config caused Colab OOM crashes;
switching from `max_depth`/`min_samples_leaf` to `max_leaf_nodes` fixed
memory without sacrificing accuracy (the former depth-caps uniformly
even where unnecessary; the latter caps exactly the thing driving memory
use — total node count). The same *category* of lesson applies to deep
learning notebooks here: watch Colab's session memory limits, especially
during v2's DenseNet121 fine-tuning phase on the full 25k-image subset.

**v4 finding**: Grad-CAM across 40 test images showed "Infiltration"
predictions had border/marker attention (not lung tissue) 74% of the
time — dramatically higher than any other condition (0-29%). This
independently corroborates the same condition's threshold instability
found separately during calibration (see below) — two unrelated analyses
flagging the same condition as unreliable is a meaningful, convergent
result, not a coincidence to explain away.

**Threshold calibration** (v4): a flat `0.5` cutoff was unusable — median
top-1 prediction was only 0.174 (the model is systematically
under-confident, which is a calibration problem, not an accuracy
problem — v2's 0.734 mean AUC only measures ranking, not confidence
magnitude). The first fix attempted, Youden's J statistic from the ROC
curve, was itself *wrong* under this dataset's per-condition imbalance
(2-17% positive rate) — it flagged Pneumonia at 11.4x its true rate and
Hernia at 93x. Switching to F1-optimal thresholds (from the
precision-recall curve, which is the standard recommendation for
imbalanced problems) fixed most conditions to within 1.0-2.5x; five
conditions (Pneumonia, Consolidation, Edema, Fibrosis, Atelectasis)
remain elevated and are a documented, accepted limitation rather than
further chased.

### Skin lesion

| Version | Approach | Result |
|---|---|---|
| v1 | Baseline CNN, 128×128 | 75.0% accuracy (66.9% majority-class baseline); melanoma recall 0.35 |
| v2 | EfficientNetB0 transfer learning, 224×224 | macro-F1 0.65; melanoma recall 0.42; every class improved or held steady vs. v1 |
| v3 | `class_weight="balanced"` (Keras built-in — works cleanly here since this is single-label multiclass, unlike chest X-ray's multi-label case where a custom loss was required) | Melanoma recall 0.58, but overall accuracy dropped to 69.2% (from 80.5%) — `nv` recall fell from 0.94→0.72 as the direct cost of the model no longer defaulting to the dominant class. Every minority class's recall improved, some dramatically (Dermatofibroma 0.39→0.70). Dermatofibroma precision fell to 0.19 as the flip side of that same trade. |
| v4 | Grad-CAM (re-uploads v3's weights, since v3 *was* a genuine improvement here — unlike chest X-ray) | Lesion-focused attention across all 5 spot-checked images, including 3 real melanoma cases, despite visible ruler tick-marks and dense hair in the source images. **No shortcut-learning found** — a meaningfully different (better) result than chest X-ray's Infiltration finding. n=5 is a small spot-check, not a systematic n=40-style investigation; judged sufficient given the clean result, but not exhaustive proof. |

## What every notebook must export

**Chest X-ray / skin lesion, v1-v3:**
```python
model.save("vN_name.keras")
json.dump(CONDITIONS, open("condition_names.json", "w"))
```

**v4 additionally exports (or, since Grad-CAM needs no retraining,
re-uses the *previous* version's model file unchanged):**
```python
json.dump(per_class_thresholds, open("per_class_thresholds.json", "w"))
```

`condition_names.json`'s order must exactly match the model's output
index order — for skin lesion, this means building the list from a
fixed, sorted `dx` code order (`akiec, bcc, bkl, df, mel, nv, vasc`)
and mapping to full names in that same order, not the order labels
happen to appear in the raw CSV.
