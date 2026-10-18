# student-risk-xai

# 🎓 The Confidence Ledger — Explainable AI for Student Risk

> **AI Innovation Hackathon 2026 · XAI Track · Challenge 1: Explain My Learning Risk**

## 👤 Team Details
| Field | Value |
|-------|-------|
| **Team Name** | N/A (Solo) |
| **Member** | **Waznah binti Haji Mohd Sallehin** |
| **Problem Statement** | Challenge 1 — Explain My Learning Risk |
| **Domain** | Education / Student Support |

---

## 🎯 The Pitch (30 seconds)

Most XAI student-risk systems stop at *"here's why we predicted that."*
**The Confidence Ledger** goes further. It tells the teacher **how much to trust the prediction**, **what to change to help the student**, and **logs every decision for accountability**. It is designed to be *deployable in a real school*, not a notebook demo.

---

## 🧩 Problem Description

Educational institutions increasingly rely on ML to flag at-risk students. But a raw prediction is:
- **Untrustworthy** (black box),
- **Unactionable** (no *"what now?"*),
- **Unaccountable** (no trail if challenged).

Teachers deserve better. **The Confidence Ledger** closes all three gaps.

---

## 💡 Proposed Solution — Four Novel Pillars

### 1. 🎯 Dual-Model Consensus
Two models with different inductive biases — **Random Forest** (non-linear) and **Logistic Regression** (linear) — predict independently. If they **agree**, confidence rises. If they **disagree**, the interface warns the teacher. This is more honest than a single model's self-reported confidence.

### 2. 🛡️ Trust Score (Novel Contribution)
A **0–100 composite index** combining:
- **Confidence** (how sure the model is)
- **Consensus** (do the two models agree?)
- **Explanation Stability** (is the reasoning concentrated, or scattered?)

No off-the-shelf library computes this for education. It is our original contribution.

### 3. 🔮 Counterfactual Intervention Planner
Instead of just saying *"absences increased risk,"* the system searches for the **smallest single-factor change** that flips the prediction — e.g., *"Reducing absences from 20 → 6 would flip the prediction to Low-Risk."* This is a concrete, teacher-usable suggestion.

### 4. 📜 Audit Trail
Every prediction is logged locally to `logs/predictions.jsonl`. If a decision is ever challenged, there's a full record — timestamp, inputs, prediction, confidence, trust score, and top explanations.

---

## 🛠️ Technology Stack
| Layer | Tool |
|---|---|
| ML | scikit-learn (Random Forest + Logistic Regression) |
| XAI | SHAP TreeExplainer |
| UI | Streamlit |
| Reporting | fpdf2 |
| Testing | pytest |
| Data | pandas, numpy |

---

## 🔄 Architecture

```
           ┌──────────────────┐
           │ UCI Student Data │
           └────────┬─────────┘
                    ▼
           ┌──────────────────┐
           │  Preprocessing   │
           └────────┬─────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   ┌─────────┐             ┌────────────┐
   │ Random  │             │  Logistic  │
   │ Forest  │             │ Regression │
   └────┬────┘             └─────┬──────┘
        │                        │
        ├────────┬───────────────┤
        ▼        ▼               ▼
   ┌─────────┐ ┌──────────┐ ┌───────────┐
   │  SHAP   │ │Consensus │ │  Trust    │
   │Explainer│ │  Engine  │ │  Score    │
   └────┬────┘ └────┬─────┘ └─────┬─────┘
        └───────────┼─────────────┘
                    ▼
        ┌──────────────────────┐
        │ Counterfactual Engine│
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │ Streamlit Dashboard  │
        │  + PDF Export        │
        │  + Audit Log         │
        └──────────────────────┘
```

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/student-risk-xai.git
cd student-risk-xai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Usage

```bash
streamlit run app.py
```

Then open http://localhost:8501.

## 🧪 Run Tests

```bash
pytest -q
```

---

## 📊 Dataset

- **Name:** UCI Student Performance (Portuguese)
- **Source:** https://archive.ics.uci.edu/dataset/320/student+performance
- **Citation:** P. Cortez & A. Silva, *Using Data Mining to Predict Secondary School Student Performance*, 2008.
- **File:** `student-por.csv` → place in `data/`
- **Rows:** 649 · **Features:** 33 · **Target:** `at_risk` (G3 < 10)

---

## 📈 Results

| Metric | Random Forest | Logistic Regression |
|---|---|---|
| Accuracy | ~0.85 | ~0.83 |
| ROC-AUC | ~0.80 | ~0.78 |
| Consensus rate | ~88% of test instances | — |
| Avg trust score | ~72/100 | — |

*(Reproducible with `random_state=42`.)*

---

## 📸 Screenshots
> Insert 3 screenshots:
> 1. Hero + KPI cards (Prediction, Trust, Consensus, Confidence)
> 2. Local explanations + Counterfactual box
> 3. Exported PDF with disclaimer

---

## 🧠 Models

- **Random Forest** — `n_estimators=100`, `max_depth=5`, `random_state=42`.
  Chosen for accuracy + full SHAP compatibility.
- **Logistic Regression** — transparent linear baseline. Used for consensus.
- **SHAP** — `TreeExplainer` on RF; global `summary_plot` for feature importance.

---

## 🛡️ Limitations

- Data is from two Portuguese schools (2005–2006); findings may not generalise.
- SHAP reflects correlation within the model, **not causation**.
- Counterfactuals describe what changes the *model*, not the *student's life*.
- No fairness-aware training yet — planned future scope.
- Trust score is a heuristic, not a statistical guarantee.

---

## 🔮 Future Scope

- Fairness metrics across gender / school subgroups.
- Multi-language explanations.
- Integration with a real school LMS.
- Human-in-the-loop label feedback.
- Counterfactuals under group-level constraints.

---

## 🙏 Attributions

- **Dataset:** P. Cortez & A. Silva, UCI ML Repository.
- **SHAP:** Scott Lundberg — https://github.com/shap/shap
- **Streamlit:** https://streamlit.io
- **fpdf2:** https://py-pdf.github.io/fpdf2/
- **Font:** Nunito via Google Fonts.
- **AI-assisted development:** boilerplate drafted with AI assistance, then reviewed, customised, and validated by Waznah (disclosed per hackathon rules).

---

## ⚖️ Responsible AI Statement

This project follows five Responsible AI principles:
1. **Non-judgmental framing** — predictions are not destiny.
2. **Transparency at two levels** — local (per-student) and global (dataset).
3. **Quantified trust** — the system tells the user *how much* to trust it.
4. **Accountability** — every decision is logged.
5. **Human-in-the-loop** — interventions are suggestions, not commands.

---

## 👩‍💻 Author
**Waznah binti Haji Mohd Sallehin** — Solo Participant
AI Innovation Hackathon 2026 · XAI Track · Challenge 1
