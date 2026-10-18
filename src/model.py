from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from src.config import RF_N_ESTIMATORS, RF_MAX_DEPTH, RANDOM_STATE


def train_models(X_train, y_train, preprocessor):
    """
    Trains TWO models with different inductive biases:
      - Random Forest   → captures non-linear interactions.
      - Logistic Regr.  → a transparent linear baseline.
    Agreement between them powers the trust score.
    """
    rf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_STATE,
        )),
    ])

    # A *fresh* preprocessor is needed for the second pipeline
    from sklearn.base import clone
    lr = Pipeline([
        ('preprocessor', clone(preprocessor)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    return rf, lr


def evaluate_model(model_pipeline, X_test, y_test, name="Model"):
    y_pred  = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    print(f"=== {name} Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.2f}")
    print(classification_report(y_test, y_pred))
    return y_pred, y_proba
