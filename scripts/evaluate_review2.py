"""Evaluation script for Review 2: Alerting, Explanations, and Suspected Drift Contributors."""

import os
import sys
import pandas as pd

# Ensure src directory and project root are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ramma_nlp.contributors import rank_contributors
from ramma_nlp.explain import explain_violation
from ramma_nlp.schema import InterpretedRequirement
from scripts.evaluate_interpreter import run_evaluation
from scripts.generate_sample_data import ref_df, prod_df  # ensuring data exists if needed


def run_review2_evaluation():
    print("\n" + "=" * 70)
    print("                      RAMMA NLP - REVIEW 2 EVALUATION REPORT")
    print("=" * 70 + "\n")

    # 1. Re-run Review 1 interpreter evaluation
    print("[SECTION 1] Interpreter Accuracy & Ambiguity Detection Evaluation")
    print("-" * 70)
    eval_df = run_evaluation()

    # 2. Load reference and production datasets
    ref_path = "data/reference.csv"
    prod_path = "data/production.csv"

    if not os.path.exists(ref_path) or not os.path.exists(prod_path):
        print(f"Creating sample reference and production datasets...")
        ref_df.to_csv(ref_path, index=False)
        prod_df.to_csv(prod_path, index=False)

    ref_data = pd.read_csv(ref_path)
    prod_data = pd.read_csv(prod_path)

    # 3. Run rank_contributors
    print("\n[SECTION 2] Suspected Drift Contributors (PSI Ranking)")
    print("-" * 70)
    contributors = rank_contributors(ref_data, prod_data, top_n=3)
    for idx, c in enumerate(contributors, 1):
        print(f"  Rank {idx}: Feature = {c['feature']:<25} | PSI = {c['psi']:.4f} | ({c['note']})")

    # 4. Violation Explanation Example
    print("\n[SECTION 3] Violation Explanation Engine Example")
    print("-" * 70)
    sample_req = InterpretedRequirement(
        metric="recall",
        operator=">=",
        threshold=0.93,
        severity="critical",
        is_ambiguous=False,
        raw_requirement="Fraud detection recall must remain above 93%",
    )
    observed_value = 0.865

    explanation_result = explain_violation(sample_req, observed_value)
    print(f"Input Requirement:  \"{sample_req.raw_requirement}\"")
    print(f"Metric:             {sample_req.metric}")
    print(f"Target:             {sample_req.operator} {sample_req.threshold}")
    print(f"Observed Value:     {observed_value}")
    print(f"Is Violation:       {explanation_result['is_violation']}")
    print(f"Generated Explanation:\n  -> \"{explanation_result['explanation']}\"")

    print("\n" + "=" * 70)
    print("                       END OF REVIEW 2 REPORT")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_review2_evaluation()
