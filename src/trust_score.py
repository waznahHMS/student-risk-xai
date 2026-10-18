"""
Trust Score (0–100):
A single number that tells the teacher how much to trust the prediction.

It combines:
  1. Confidence  (how sure the RF model is)
  2. Consensus   (does LR agree?)
  3. Stability   (are SHAP values concentrated, i.e. clear drivers?)

This is a *novel contribution* of this project — no standard library
computes an interpretable trust index for educational ML.
"""
from src.config import TRUST_W_CONFIDENCE, TRUST_W_CONSENSUS, TRUST_W_STABILITY


def compute_trust_score(confidence, agreement, shap_values_1d):
    """
    confidence : float in [0,1]
    agreement  : bool
    shap_values_1d : np.ndarray of per-feature SHAP values
    """
    # Stability = how concentrated the explanation is.
    # If the top-1 feature dominates, the reasoning is clear → high stability.
    import numpy as np
    abs_shap = np.abs(shap_values_1d)
    if abs_shap.sum() == 0:
        stability = 0.0
    else:
        stability = abs_shap.max() / abs_shap.sum()

    consensus = 1.0 if agreement else 0.0

    raw = (
        TRUST_W_CONFIDENCE * confidence
        + TRUST_W_CONSENSUS  * consensus
        + TRUST_W_STABILITY  * stability
    )
    return round(raw * 100, 1)


def trust_label(score):
    if score >= 80: return "High Trust", "#16A34A"
    if score >= 60: return "Moderate Trust", "#F59E0B"
    return "Low Trust", "#DC2626"
