"""
Turns a PatientInput into the numeric feature vector the model expects.

v1 (Decision Tree) only uses `symptoms`, encoded as a multi-hot vector
over the training vocabulary, because that's all the Kaggle dataset
provides. age/gender/vital_signs/lab_values/medical_history are accepted
by the API and validated, but intentionally unused until a later model
version is trained on data rich enough to support them — don't silently
fold them into the feature vector without also retraining, or you'll
get a shape mismatch or, worse, a model that ignores them silently.
"""
from typing import List
import numpy as np

from app.symptom_vocab import normalize_symptom


def encode_symptoms(symptoms: List[str], vocab: List[str]) -> np.ndarray:
    """
    Returns a (1, len(vocab)) multi-hot row vector suitable for
    sklearn's .predict_proba(). Symptoms not found in the vocab are
    ignored (not an error) so the API stays usable as the vocab evolves,
    but you may want to log/track drops in production.
    """
    vocab_index = {normalize_symptom(v): i for i, v in enumerate(vocab)}
    vector = np.zeros((1, len(vocab)), dtype=np.float32)

    unmatched = []
    for symptom in symptoms:
        key = normalize_symptom(symptom)
        if key in vocab_index:
            vector[0, vocab_index[key]] = 1.0
        else:
            unmatched.append(symptom)

    return vector, unmatched
