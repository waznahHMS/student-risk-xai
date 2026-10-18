"""
Finds the *smallest single-feature change* that flips the prediction.
This gives the teacher a concrete, actionable intervention instead of
just an explanation.

Example output:
   "Reducing Absences from 20 to 6 would flip the prediction to Low-Risk."
"""
import numpy as np
from src.config import COUNTERFACTUAL_RANGE, COUNTERFACTUAL_MAX_STEPS


def find_counterfactual(model, instance, original_prediction):
    """
    instance: pandas DataFrame with 1 row
    original_prediction: int (0 or 1)
    Returns: dict or None
    """
    best = None
    for feature, (lo, hi, step) in COUNTERFACTUAL_RANGE.items():
        for value in np.arange(lo, hi + step, step):
            trial = instance.copy()
            trial[feature] = value
            new_pred = model.predict(trial)[0]
            if new_pred != original_prediction:
                distance = abs(value - instance[feature].values[0])
                if best is None or distance < best["distance"]:
                    best = {
                        "feature": feature,
                        "original": instance[feature].values[0],
                        "suggested": value,
                        "distance": distance,
                    }
                break  # smallest change for this feature
    return best


def humanize_counterfactual(cf):
    if cf is None:
        return None
    feature = cf["feature"].replace('_', ' ').title()
    return (
        f"Changing **{feature}** from **{cf['original']}** to "
        f"**{cf['suggested']}** would flip the prediction."
    )
