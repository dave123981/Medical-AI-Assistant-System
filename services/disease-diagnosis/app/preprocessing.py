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
def encode_symptoms(symptoms: List[str], vocab: List[str]) -> np.ndarray:
    """
    Returns a (1, len(vocab)) multi-hot row vector. Symptoms not found
    in the vocab are ignored (not an error) so the API stays usable as
    the vocab evolves.
    """
    vocab_index = {v: i for i, v in enumerate(vocab)}
    vector = np.zeros((1, len(vocab)), dtype=np.float32)

    unmatched = []
    for symptom in symptoms:
        if symptom in vocab_index:
            vector[0, vocab_index[symptom]] = 1.0
        else:
            unmatched.append(symptom)

    return vector, unmatched
