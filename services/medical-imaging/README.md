# Service 2 — Medical Image Analysis

FastAPI microservice for multi-label (chest X-ray) and multiclass (skin
lesion) disease classification from medical images. Called by the Go
API Gateway. **Both image types complete through v4.**

## Run locally

```bash
cd services/medical-imaging
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

Docs at http://localhost:8001/docs

## Image types

| Image type | Status | Dataset |
|---|---|---|
| `chest_xray` | Complete, v1-v4 | `nih-chest-xrays/sample` (v1), `khanfashee/nih-chest-x-ray-14-224x224-resized` (v2+) |
| `skin_lesion` | Complete, v1-v4 | `kmader/skin-cancer-mnist-ham10000` |


Each image type has its own `models/<image_type>/` folder, loaded and
validated independently. Retinal was deliberately dropped from scope —
two image types carried through a full v1-v4 progression was judged a
stronger result than three shallow ones.

## Chest X-ray — version results

| Version | Approach | Result |
|---|---|---|
| v1 | Baseline CNN | 0.608 mean AUC across 14 conditions |
| v2 | DenseNet121 transfer learning | 0.734 mean AUC |
| v3 | CheXNet-style weighted loss | **Documented null result** — mean AUC unchanged (0.734→0.734), and the two conditions it specifically targeted (Pneumonia, Hernia) slightly regressed. Ran the full epoch schedule with no early-stopping truncation, ruling that out as the cause. Conclusion: at <1% positive rate, reweighting can't compensate for genuine sample scarcity. |
| v4 | Grad-CAM + per-class F1-optimal thresholds | See below |

**A significant finding from v4**: Grad-CAM analysis across 40 test
images showed 74% of "Infiltration" predictions had peak attention on
image borders/markers rather than lung tissue — versus 0-29% for other
conditions. This matches published critiques identifying Infiltration
as ChestX-ray14's most label-noisy, shortcut-learning-prone condition.
Infiltration predictions from this model should be treated with
substantially lower confidence than other findings.

**Threshold calibration was itself a two-step fix.** A flat `0.5`
threshold was unusable (median top-1 prediction was 0.174 — the model
is systematically under-confident). The first fix, Youden's J statistic
(standard ROC-based threshold selection), was *itself wrong* under this
dataset's class imbalance — in one case flagging 93x the true positive
rate for a condition. Switching to F1-optimal thresholds (precision-recall
curve based, better suited to imbalanced data) fixed most conditions to
within 1.0-2.5x their true rate; five conditions remain elevated and are
documented as a known limitation.

## Skin lesion — version results

| Version | Approach | Result |
|---|---|---|
| v1 | Baseline CNN | 75.0% accuracy (vs. 66.9% majority-class baseline); melanoma recall only 0.35 |
| v2 | EfficientNetB0 transfer learning | macro-F1 0.65; melanoma recall 0.42 |
| v3 | Class-weighted training | Melanoma recall 0.58 — traded 11.3 points of overall accuracy for higher recall across **every** minority class. Dermatofibroma precision fell to 0.19 as the cost of that trade. For a screening tool, prioritizing recall over precision on malignant/rare conditions is generally appropriate, but it's a real cost, not a free win. |
| v4 | Grad-CAM | Lesion-focused attention confirmed across all 5 spot-checked images (including 3 real melanoma cases), even with ruler tick-marks and dense hair present in the source images — **no shortcut-learning found**, unlike chest X-ray's Infiltration issue |

**EfficientNetB0 has a critical, non-obvious input requirement**:
its own first layer (`Rescaling`) expects raw `[0,255]` pixel values,
not the `/255`-normalized `[0,1]` input every other model in this
project uses. Feeding it normalized input silently double-rescales
everything to `~[0, 0.004]` — no error, just badly degraded training.
The v2+ notebooks handle this correctly; watch for this if adapting the
code elsewhere.

## Grad-CAM — how it works across two different architectures

`app/gradcam.py`'s layer-detection logic is architecture-agnostic: it
looks for a layer literally named `relu` (DenseNet121's convention)
first, and falls back to searching for the last layer with 4D output
otherwise. This was verified against EfficientNetB0 specifically (whose
equivalent layer is named `top_activation`, not `relu`) — confirmed the
fallback correctly finds it with zero code changes needed for the new
architecture.

## Why this service is structured differently from Service 1

- **Multi-label vs. multiclass, in the same generic response shape.**
  Chest X-ray's model outputs independent sigmoid probabilities (a scan
  can show multiple conditions); skin lesion's outputs a single softmax
  distribution (mutually exclusive diagnosis). The API doesn't
  special-case this — it just evaluates N per-class probabilities
  against N thresholds either way. `positive_findings` naturally
  contains 0-1 entries for skin lesion (softmax sums to 1) vs.
  potentially several for chest X-ray.
- **One model slot per image type**, each with independent auto-detection,
  Grad-CAM capability detection, and threshold validation.
- **Always Keras** — no sklearn-tree option for raw pixels, so
  `model_loader.py` here is simpler than Service 1's in that respect.

## Before it will actually predict anything

Drop into `models/<image_type>/`:
- the trained `.keras` model
- `condition_names.json` — ordered condition/disease labels, index-aligned to the model's output
- `per_class_thresholds.json` — **optional**. If absent, every condition falls back to a flat `0.5` (known to be a poor cutoff for chest X-ray specifically — see above).

## Endpoints

- `GET /health`
- `GET /conditions?image_type=chest_xray` — the condition list the active model for that image type predicts
- `POST /analyze` — multipart form: `image` (file), `image_type` (default `chest_xray`), `threshold` (optional — omit to use per-class tuned thresholds, the recommended default; provide to force one value across all conditions)

### Response shape

```json
{
  "image_type": "chest_xray",
  "findings": [
    {"condition": "Effusion", "probability": 0.51, "positive": true, "threshold_used": 0.153}
  ],
  "positive_findings": ["Effusion"],
  "threshold_mode": "per_class_tuned",
  "global_threshold": null,
  "heatmap_base64": "...",
  "model_version": "v4-densenet-transfer"
}
```

`heatmap_base64` is generated for whichever condition scored highest
overall, even if it doesn't clear its threshold — useful diagnostic
information about what the model focused on for its best guess. It's
`null` for models that don't match Grad-CAM's expected architecture
shape (e.g. a hypothetical future from-scratch CNN without a
GlobalAveragePooling2D + nested pretrained backbone).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `IMAGING_MODEL_DIR` | `./models` | Where model subfolders are read from |
| `<IMAGE_TYPE>_MODEL_FILENAME` | auto-detected | Force a specific file, e.g. `CHEST_XRAY_MODEL_FILENAME` |
| `<IMAGE_TYPE>_MODEL_VERSION` | derived from filename | Override the reported version string |
