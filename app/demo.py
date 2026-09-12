"""Streamlit Demo Application for RAMMA NLP Service."""

import os
import sys
import pandas as pd
import streamlit as st

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ramma_nlp.contributors import rank_contributors
from ramma_nlp.explain import explain_violation
from ramma_nlp.interpreter import interpret_requirement
from ramma_nlp.schema import InterpretedRequirement

st.set_page_config(
    page_title="RAMMA NLP — ML Requirement Monitoring",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAMMA NLP — ML Requirement Interpreter & Monitor")
st.markdown(
    "Interpret natural language monitoring requirements, detect ambiguities, evaluate violations, "
    "and identify suspected drift contributors."
)

st.markdown("---")

# SECTION 1: Requirement Interpreter
st.header("1. Requirement Interpreter & Ambiguity Detector")

default_req = "Recall must remain above 93%"
req_text = st.text_input("Enter a business requirement", value=default_req)

col_btn, _ = st.columns([1, 4])
with col_btn:
    interpret_clicked = st.button("Interpret Requirement", type="primary")

if interpret_clicked or "interpreted_req" in st.session_state:
    if interpret_clicked:
        try:
            with st.spinner("Interpreting requirement with Gemini..."):
                interpreted = interpret_requirement(req_text)
                st.session_state["interpreted_req"] = interpreted
        except Exception as e:
            st.error(f"Interpretation failed: {e}")
            st.stop()

    interpreted: InterpretedRequirement = st.session_state.get("interpreted_req")

    if interpreted:
        st.subheader("Structured Interpretation Output")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Metric", interpreted.metric.upper())
        m2.metric("Operator", interpreted.operator)
        m3.metric("Threshold", f"{interpreted.threshold:.2%}" if interpreted.threshold <= 1.0 else f"{interpreted.threshold}")
        m4.metric("Severity", interpreted.severity.capitalize())
        m5.metric("Ambiguous", "YES" if interpreted.is_ambiguous else "NO")

        if interpreted.is_ambiguous:
            st.warning(
                f"⚠️ **Requirement is Ambiguous**: {interpreted.ambiguity_reason or 'No measurable metric or threshold found.'}"
            )

st.markdown("---")

# SECTION 2: Simulate Production Data
st.header("2. Simulate Production Data & Monitoring Check")


def compute_metric(df: pd.DataFrame, metric_name: str) -> float:
    """Helper function to compute or simulate observed metric value from production data."""
    # Stubbed values for demo simulation
    metric_stubs = {
        "recall": 0.865,
        "precision": 0.820,
        "f1": 0.840,
        "fnr": 0.08,
        "fpr": 0.065,
        "accuracy": 0.890,
        "auc": 0.720,
    }
    return metric_stubs.get(metric_name.lower(), 0.85)


check_clicked = st.button("Run Monitoring Check", type="secondary")

if check_clicked:
    if "interpreted_req" not in st.session_state:
        st.info("Please interpret a requirement first before running monitoring check.")
    else:
        req: InterpretedRequirement = st.session_state["interpreted_req"]

        ref_path = "data/reference.csv"
        prod_path = "data/production.csv"

        if not os.path.exists(ref_path) or not os.path.exists(prod_path):
            st.error(f"Missing data files ({ref_path} / {prod_path}). Please ensure sample data is generated.")
        else:
            ref_df = pd.read_csv(ref_path)
            prod_df = pd.read_csv(prod_path)

            observed_val = compute_metric(prod_df, req.metric)
            explanation_res = explain_violation(req, observed_val)

            st.subheader("Violation Status")
            if not explanation_res["is_violation"]:
                st.success(
                    f"✅ **No Violation Detected**\n\n{explanation_res['explanation']}"
                )
            else:
                st.error(
                    f"🚨 **VIOLATION ALERT** ({req.severity.upper()} SEVERITY)\n\n"
                    f"{explanation_res['explanation']}"
                )

                st.subheader("Top 3 Suspected Drift Contributors (PSI)")
                st.caption(
                    "Note: PSI measures distribution drift. This is a correlational signal, not a proven causal relationship."
                )

                contributors = rank_contributors(ref_df, prod_df, top_n=3)
                if contributors:
                    contrib_df = pd.DataFrame(contributors)
                    st.table(contrib_df)
                else:
                    st.info("No shared numerical features found for PSI calculation.")
