"""
Placeholder symptom vocabulary.

The real vocabulary MUST come from training: when you train v1 in Colab
on the Kaggle "Disease Symptom Prediction" dataset, export the exact,
ordered list of symptom column names the model was fit on to
`models/symptom_vocab.json`. The order matters — it defines the index
each symptom maps to in the multi-hot feature vector, and it must match
between training and inference exactly.

This file only exists so the service can start up and be exercised
locally before you've trained anything. Do not ship this list as your
real vocabulary.
"""
import re

PLACEHOLDER_VOCAB = [
    "itching",
    "skin_rash",
    "headache",
    "fatigue",
    "nausea",
    "vomiting",
    "high_fever",
    "cough",
    "chest_pain",
    "joint_pain",
]


def normalize_symptom(symptom: str) -> str:
    """
    Matches the Kaggle dataset's convention: lowercase, spaces -> underscores,
    strip stray whitespace. Apply the same normalization at train time and
    at inference time or lookups will silently fail.
    """
    s = symptom.strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s
