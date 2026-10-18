"""
Central configuration for the Student Risk XAI system.
Keeping magic numbers in one place makes the code auditable and reproducible.
"""
from pathlib import Path

# ── Paths
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_PATH     = BASE_DIR / "data" / "student-por.csv"
LOG_DIR       = BASE_DIR / "logs"
LOG_FILE      = LOG_DIR / "predictions.jsonl"

# ── Model
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
RISK_THRESHOLD_GRADE = 10          # G3 < 10 → at-risk
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH    = 5

# ── Trust score weights (must sum to 1.0)
TRUST_W_CONFIDENCE  = 0.5
TRUST_W_CONSENSUS   = 0.3
TRUST_W_STABILITY   = 0.2

# ── Counterfactual search
COUNTERFACTUAL_MAX_STEPS = 12
COUNTERFACTUAL_RANGE = {
    "absences":    (0, 30, 1),
    "failures":    (0, 3, 1),
    "studytime":   (1, 4, 1),
    "goout":       (1, 5, 1),
}

# ── UI theme
PRIMARY = "#4F46E5"
ACCENT  = "#F97316"
CREAM   = "#FDF8F3"
NAVY    = "#1E1B4B"
CARD    = "#FFFFFF"
