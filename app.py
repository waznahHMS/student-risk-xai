import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from src.config import (PRIMARY, ACCENT, CREAM, NAVY, CARD)
from src.data_prep import load_and_prep_data, get_feature_names
from src.model import train_models
from src.explainer import get_shap_values, get_human_readable_explanation
from src.consensus import compute_consensus
from src.trust_score import compute_trust_score, trust_label
from src.counterfactual import find_counterfactual, humanize_counterfactual
from src.audit_log import log_prediction, load_recent
from src.pdf_report import build_pdf_report


st.set_page_config(page_title="Student Risk XAI", page_icon="🎓", layout="wide")


# ───────────────────── CSS ─────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Nunito', sans-serif !important; }}
.stApp {{ background: {CREAM}; }}
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid #EEE7DF; }}
h1,h2,h3,h4 {{ font-family:'Nunito',sans-serif !important; color:{NAVY}; font-weight:800; }}

.hero {{
  background: linear-gradient(135deg, {PRIMARY} 0%, #6366F1 100%);
  padding: 40px 48px; border-radius: 24px; color: white; margin-bottom: 32px;
  box-shadow: 0 12px 32px rgba(79,70,229,0.18);
}}
.hero h1 {{ color: white; margin:0 0 8px 0; font-size:2.1rem; }}
.hero p  {{ color:#E0E7FF; margin:0; font-size:1.05rem; line-height:1.6; }}

.card {{
  background:{CARD}; border-radius:20px; padding:26px 30px;
  box-shadow:0 4px 20px rgba(30,27,75,0.06); border:1px solid #F1EAE0;
  margin-bottom:20px;
}}
.card-title {{ font-size:0.75rem; font-weight:700; letter-spacing:1.2px;
               text-transform:uppercase; color:{PRIMARY}; margin-bottom:8px; }}
.card-value {{ font-size:1.8rem; font-weight:800; color:{NAVY}; line-height:1.2; }}
.card-sub   {{ font-size:0.9rem; color:#6B7280; margin-top:6px; }}

.pill {{ display:inline-block; padding:8px 20px; border-radius:999px;
         font-weight:700; font-size:0.95rem; }}
.pill-high {{ background:#FEE2E2; color:#991B1B; }}
.pill-low  {{ background:#DCFCE7; color:#166534; }}

.exp-row {{
  display:flex; align-items:center; padding:14px 18px; background:#FFFBF6;
  border-radius:14px; margin-bottom:10px; border-left:4px solid {ACCENT};
}}
.exp-row.up   {{ border-left-color:#DC2626; }}
.exp-row.down {{ border-left-color:#16A34A; }}
.exp-text {{ font-size:1rem; color:{NAVY}; font-weight:500; }}

.cf-box {{
  background:#EEF2FF; border-left:4px solid {PRIMARY};
  padding:18px 22px; border-radius:14px; color:{NAVY};
  font-size:1rem; margin-bottom:16px;
}}

.stButton > button {{
  background:{PRIMARY}; color:white; border:none; border-radius:12px;
  padding:12px 26px; font-weight:700; font-family:'Nunito',sans-serif;
  box-shadow:0 4px 14px rgba(79,70,229,0.25);
}}
.stButton > button:hover {{ background:{ACCENT}; }}
.stDownloadButton > button {{
  background:{ACCENT}; color:white; border:none; border-radius:12px;
  padding:12px 26px; font-weight:700;
}}
.disclaimer {{
  background:#FFFBEB; border-left:4px solid {ACCENT};
  padding:20px 24px; border-radius:12px;
  font-size:0.9rem; color:#78350F; line-height:1.7;
}}
hr {{ border:none; border-top:1px solid #EEE7DF; margin:36px 0; }}
#MainMenu, footer {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)


# ───────────────────── Init ─────────────────────
@st.cache_resource
def init_system():
    X_train, X_test, y_train, y_test, pre = load_and_prep_data()
    rf, lr = train_models(X_train, y_train, pre)
    feats  = get_feature_names(rf.named_steps['preprocessor'], X_train)
    return X_train, X_test, y_train, y_test, rf, lr, feats

with st.spinner("Booting dual-model XAI engine..."):
    X_train, X_test, y_train, y_test, rf_model, lr_model, feature_names = init_system()


# ───────────────────── Hero ─────────────────────
st.markdown("""
<div class="hero">
  <h1>🎓 The Confidence Ledger</h1>
  <p>An Explainable AI system that predicts student learning risk, <b>quantifies how much to trust the prediction</b>, and proposes a concrete intervention — with a full audit trail for accountability.</p>
</div>
""", unsafe_allow_html=True)


# ───────────────────── Sidebar ─────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#4F46E5;'>Student Profile</h2>", unsafe_allow_html=True)
    st.caption("Adjust inputs — everything updates live.")
    st.markdown("---")

    inputs = {
        'school':    st.selectbox("School", ["GP", "MS"]),
        'sex':       st.selectbox("Sex", ["F", "M"]),
        'age':       st.slider("Age", 15, 22, 17),
        'studytime': st.slider("Weekly Study Time (1=<2h, 4=>10h)", 1, 4, 2),
        'failures':  st.slider("Past Failures", 0, 3, 0),
        'absences':  st.slider("Absences", 0, 30, 4),
        'goout':     st.slider("Going Out (1=low, 5=high)", 1, 5, 3),
    }
    st.markdown("---")
    st.caption("🔒 Predictions are logged locally for audit.")


# ───────────────────── Predict ─────────────────────
inst = X_test.iloc[[0]].copy()
for k, v in inputs.items():
    inst[k] = v

rf_proba = rf_model.predict_proba(inst)[0][1]
lr_proba = lr_model.predict_proba(inst)[0][1]
prediction = int(rf_proba >= 0.5)
confidence = rf_proba if prediction == 1 else (1 - rf_proba)

rf_label, lr_label, agreement, gap = compute_consensus(rf_proba, lr_proba)

explainer, shap_values, X_train_t, X_test_t = get_shap_values(rf_model, X_train, inst)
explanations = get_human_readable_explanation(shap_values, feature_names, 0)

trust = compute_trust_score(confidence, agreement, shap_values[1][0])
trust_text, trust_color = trust_label(trust)

cf = find_counterfactual(rf_model, inst, prediction)
cf_text = humanize_counterfactual(cf)

# Log it (audit trail)
log_prediction(inputs, prediction, confidence, trust, explanations)


# ───────────────────── KPI row ─────────────────────
c1, c2, c3, c4 = st.columns(4, gap="large")

with c1:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Prediction</div>
      <div class="card-value"><span class="pill {'pill-high' if prediction==1 else 'pill-low'}">
        {'⚠️ At-Risk' if prediction==1 else '✅ Low-Risk'}</span></div>
      <div class="card-sub">Random Forest</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Trust Score</div>
      <div class="card-value" style="color:{trust_color};">{trust}/100</div>
      <div class="card-sub">{trust_text}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    verdict = "✅ Agree" if agreement else "⚠️ Disagree"
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Model Consensus</div>
      <div class="card-value" style="font-size:1.4rem;">{verdict}</div>
      <div class="card-sub">RF vs Logistic ({gap:.2f} gap)</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Confidence</div>
      <div class="card-value">{confidence:.1%}</div>
      <div class="card-sub">Model certainty</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ───────────────────── Local explanations ─────────────────────
st.markdown("<h2>Why This Prediction?</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#6B7280;'>Top contributing factors for this student.</p>",
            unsafe_allow_html=True)

for e in explanations:
    text  = e.replace("**","")
    is_up = "increased" in text
    cls   = "up" if is_up else "down"
    icon  = "▲" if is_up else "▼"
    color = "#DC2626" if is_up else "#16A34A"
    st.markdown(f"""
    <div class="exp-row {cls}">
      <span style="color:{color}; font-weight:800; margin-right:14px;">{icon}</span>
      <span class="exp-text">{text}</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ───────────────────── Counterfactual ─────────────────────
st.markdown("<h2>💡 Suggested Intervention</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#6B7280;'>The smallest single-factor change that would flip this prediction.</p>",
            unsafe_allow_html=True)
if cf_text:
    st.markdown(f'<div class="cf-box">{cf_text}</div>', unsafe_allow_html=True)
else:
    st.info("No single-factor change within safe ranges flips this prediction.")

st.markdown("<br>", unsafe_allow_html=True)


# ───────────────────── Global SHAP ─────────────────────
st.markdown("<h2>Global Feature Importance</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#6B7280;'>What drives the model across the whole dataset?</p>",
            unsafe_allow_html=True)

fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor(CREAM)
shap.summary_plot(shap_values[1], X_test_t, feature_names=feature_names, show=False)
plt.gcf().set_facecolor(CREAM)
plt.tight_layout()
st.pyplot(plt.gcf(), use_container_width=True)
plt.close()

mean_abs = np.abs(shap_values[1]).mean(axis=0)
global_imp = (pd.DataFrame({"feature": feature_names, "importance": mean_abs})
              .sort_values("importance", ascending=False).reset_index(drop=True))

st.markdown("<br><br>", unsafe_allow_html=True)


# ───────────────────── Export ─────────────────────
st.markdown("<h2>Export</h2>", unsafe_allow_html=True)
if st.button("📄 Generate PDF Report"):
    pdf_bytes = build_pdf_report(
        inputs, prediction, confidence,
        trust, trust_text,
        explanations, cf_text, global_imp,
    )
    st.download_button(
        "⬇️ Download Report",
        data=pdf_bytes,
        file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
    )

st.markdown("<br>", unsafe_allow_html=True)


# ───────────────────── Audit trail ─────────────────────
with st.expander("📜 Audit Trail (last 20 predictions)"):
    logs = load_recent(20)
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    else:
        st.caption("No predictions logged yet.")

st.markdown("<br>", unsafe_allow_html=True)


# ───────────────────── Disclaimer ─────────────────────
st.markdown("""
<div class="disclaimer">
<b>⚠️ Responsible AI Disclaimer</b><br>
This prototype is for research and educational demonstration only. It is not a
judgment about any student's ability, potential, or future. Predictions reflect
statistical patterns in a public historical dataset and may contain bias. Explanations
reflect <b>model behavior</b>, not causal facts. Trust scores are heuristic. The system
is designed to <b>support</b> — never replace — human educational judgment. Do not use
for admissions, grading, or any high-stakes decision.
</div>
""", unsafe_allow_html=True)
