"""
Loads two data artifacts (not a trained ML model, for v1):

  data/drug_rankings.json         REQUIRED. {condition: [{drug, score, review_count}, ...]},
                                   produced by the notebook from the UCI Drug Review dataset.
  data/contraindication_rules.json OPTIONAL. Small, hand-curated, illustrative-only
                                   allergy/interaction rules — NOT derived from the review
                                   dataset (which contains no chemical/interaction data at all).

v1's "ML ranking" is a weighted aggregation formula computed once in the notebook, not a
model retrained here — consistent with the project's pattern of starting simple and adding
sophistication (an actual learned ranker) in a later version.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(os.getenv("DRUG_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


class DataLoadError(Exception):
    """Raised when a required data artifact is missing or malformed."""
    pass


class DrugRecommendationArtifacts:
    def __init__(self, rankings: dict, display_names: dict, allergy_rules: dict,
                 interaction_rules: dict, version: str):
        self.rankings = rankings              # {condition_lower: [{"drug","score","review_count"}, ...]}
        self.condition_display_names = display_names  # {condition_lower: "Original Casing"}
        self.allergy_rules = allergy_rules    # {allergy_lower: [drug_lower, ...]}
        self.interaction_rules = interaction_rules  # {medication_lower: [drug_lower, ...]}
        self.version = version
        # False when no rules file was found — callers must know this means
        # "not checked," not "checked and found nothing," since silently
        # treating everything as safe would be actively misleading here.
        self.rules_loaded = bool(allergy_rules or interaction_rules)

    def find_condition(self, query: str):
        """Case-insensitive EXACT match only for v1 — no fuzzy matching, per the
        decision to keep condition-vocabulary bridging out of scope for now.
        Returns (display_name_or_None, list_of_drug_entries)."""
        key = query.strip().lower()
        if key in self.rankings:
            return self.condition_display_names.get(key, query), self.rankings[key]
        return None, []

    def check_contraindications(self, drug: str, allergies, current_medications) -> list:
        reasons = []
        drug_lower = drug.lower()
        for allergy in (allergies or []):
            if drug_lower in self.allergy_rules.get(allergy.strip().lower(), []):
                reasons.append(f"Patient allergy: {allergy}")
        for med in (current_medications or []):
            if drug_lower in self.interaction_rules.get(med.strip().lower(), []):
                reasons.append(f"Potential interaction with current medication: {med}")
        return reasons


def _load_json(path: Path, required: bool):
    if not path.exists():
        if required:
            raise DataLoadError(
                f"Missing required data file: {path}. Run the notebook in notebooks/ "
                f"to produce it from the UCI Drug Review dataset."
            )
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Malformed JSON in {path}: {e}") from None


@lru_cache(maxsize=1)
def get_artifacts() -> DrugRecommendationArtifacts:
    raw_rankings = _load_json(DATA_DIR / "drug_rankings.json", required=True)
    rules = _load_json(DATA_DIR / "contraindication_rules.json", required=False)

    allergy_rules = {
        k.strip().lower(): [d.strip().lower() for d in v]
        for k, v in rules.get("allergy_rules", {}).items()
    }
    interaction_rules = {
        k.strip().lower(): [d.strip().lower() for d in v]
        for k, v in rules.get("interaction_rules", {}).items()
    }

    rankings, display_names = {}, {}
    for condition, drugs in raw_rankings.items():
        key = condition.strip().lower()
        rankings[key] = drugs
        display_names[key] = condition

    version = os.getenv("DRUG_MODEL_VERSION", "v1-aggregated-ranking")

    return DrugRecommendationArtifacts(rankings, display_names, allergy_rules, interaction_rules, version)
