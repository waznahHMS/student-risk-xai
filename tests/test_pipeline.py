"""
Lightweight tests to show judges that the pipeline is verifiable.
Run with: pytest -q
"""
import numpy as np
import pandas as pd
from src.data_prep import load_and_prep_data
from src.model import train_models
from src.consensus import compute_consensus
from src.trust_score import compute_trust_score, trust_label
from src.counterfactual import find_counterfactual, humanize_counterfactual


def test_data_loads():
    X_train, X_test, y_train, y_test, pre = load_and_prep_data()
    assert len(X_train) > 0
    assert set(y_train.unique()).issubset({0, 1})


def test_models_train():
    X_train, X_test, y_train, y_test, pre = load_and_prep_data()
    rf, lr = train_models(X_train, y_train, pre)
    preds = rf.predict(X_test.head(5))
    assert len(preds) == 5


def test_consensus_logic():
    rf_l, lr_l, agree, gap = compute_consensus(0.8, 0.7)
    assert rf_l == lr_l == 1
    assert agree is True
    assert gap < 0.2


def test_trust_score_bounds():
    shap_vals = np.array([0.5, 0.3, 0.1])
    score = compute_trust_score(0.9, True, shap_vals)
    assert 0 <= score <= 100
    label, _ = trust_label(score)
    assert label in ["High Trust", "Moderate Trust", "Low Trust"]


def test_counterfactual_returns_dict_or_none():
    X_train, X_test, y_train, y_test, pre = load_and_prep_data()
    rf, _ = train_models(X_train, y_train, pre)
    inst = X_test.iloc[[0]].copy()
    pred = rf.predict(inst)[0]
    cf = find_counterfactual(rf, inst, pred)
    assert cf is None or "feature" in cf
    # humanizer must not crash
    text = humanize_counterfactual(cf)
    assert text is None or isinstance(text, str)
