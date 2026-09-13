"""RAMMA Command Line Interface for end-to-end ML Monitoring demo."""

import argparse
import os
import sys
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from sklearn.model_selection import StratifiedKFold
from data.prepare_dataset import FraudClassifier, prepare_datasets_and_train_baseline
from ramma_backend.model_adapter import ModelAdapter, get_model_adapter
from ramma_backend.monitoring_runtime import compute_dataset_metric, generate_monitor_config, run_monitor
from ramma_backend.threshold_engine import calibrate_threshold, compute_reference_metric_values
from ramma_nlp.contributors import rank_contributors
from ramma_nlp.explain import explain_violation
from ramma_nlp.interpreter import interpret_requirement


def prompt_model_selection() -> tuple[str, str]:
    """Prompts user interactively to select a model architecture and returns (model_source, explanation)."""
    print("\nWhich model would you like RAMMA to monitor?")
    print("1) RandomForest classifier (local file) — an ensemble model, generally more robust to noisy features, tuned for this fraud-detection scenario.")
    print("2) LogisticRegression classifier (local file) — a simpler linear model, faster and more interpretable, but less robust to complex patterns; included to prove RAMMA's pipeline works identically regardless of model type.")
    print("3) Remote model (HTTP endpoint) — connects to a model hosted on a separate server via REST API, simulating how RAMMA would integrate with a real enterprise's hosted model, e.g. an internal inference service or a cloud endpoint.")
    
    try:
        choice = input("Enter 1, 2, or 3: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "2":
        model_source = "local:models/alt_classifier_logreg.pkl"
        explanation = (
            "Selected: Local LogisticRegression Classifier (models/alt_classifier_logreg.pkl).\n"
            "Loaded directly via joblib in-process, representing an alternative linear baseline architecture."
        )
    elif choice == "3":
        try:
            url_input = input("Enter the model endpoint URL (default: http://localhost:5000): ").strip()
        except (EOFError, KeyboardInterrupt):
            url_input = ""
        url = url_input if url_input else "http://localhost:5000"
        model_source = f"remote:{url}"
        clean_url = url.rstrip("/predict").rstrip("/")
        explanation = (
            f"Connected via HTTP POST to {clean_url}/predict — this is the same standard your browser and every REST API use.\n"
            f"In a real deployment, this URL would point to an enterprise's actual model-serving endpoint (e.g., AWS SageMaker, an internal inference service), not just this local mock server."
        )
    else:
        model_source = "local:models/baseline_classifier.pkl"
        explanation = (
            "Selected: Local RandomForest Classifier (models/baseline_classifier.pkl).\n"
            "Loaded directly via joblib in-process, representing a local embedded model runtime."
        )

    return model_source, explanation


def run_demo(
    requirements: Optional[List[str]] = None,
    model_source: str = "local",
    interactive: bool = False,
    interactive_model: bool = False,
):
    print("=" * 80)
    print("                 RAMMA NLP & BACKEND — END-TO-END DEMO")
    print("=" * 80)

    # Ensure baseline data files and trained model exist
    ref_path = "data/reference.csv"
    clean_path = "data/production_clean.csv"
    drifted_path = "data/production_drifted.csv"
    model_path = "models/baseline_classifier.pkl"

    if not (
        os.path.exists(ref_path)
        and os.path.exists(clean_path)
        and os.path.exists(drifted_path)
        and os.path.exists(model_path)
    ):
        print("\n[SETUP] Preparing synthetic datasets and baseline classifier model...")
        prepare_datasets_and_train_baseline()

    # Interactive Model Selection Menu
    if interactive_model:
        model_source, model_explanation = prompt_model_selection()
        print(f"\n[MODEL SELECTION DETAILS]\n{model_explanation}")

    # Collect requirements list
    req_list = []
    if interactive or (interactive_model and not requirements):
        print("\n[INTERACTIVE MODE] Enter business requirements one by one. Type 'done' to finish:")
        while True:
            try:
                line = input("Enter requirement (or 'done'): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line or line.lower() == "done":
                break
            req_list.append(line)
        if not req_list:
            req_list = ["Recall must remain above 93%"]
    elif requirements:
        req_list = [r.strip() for r in requirements if r.strip()]
    else:
        req_list = ["Recall must remain above 93%"]

    # Instantiate ModelAdapter once
    print(f"\n[MODEL ADAPTER INITIALIZATION] Model Source: '{model_source}'")
    adapter = get_model_adapter(model_source)

    total_reqs = len(req_list)

    for idx, raw_text in enumerate(req_list, 1):
        if total_reqs > 1:
            print("\n" + "#" * 80)
            print(f"   EVALUATING REQUIREMENT {idx}/{total_reqs}: \"{raw_text}\"")
            print("#" * 80)
        else:
            print(f"\n[STEP 1: INPUT BUSINESS REQUIREMENT]")
            print("-" * 80)
            print(f"  Requirement Text: \"{raw_text}\"")

        # 2. Interpret Requirement
        print(f"\n[STEP 2: REQUIREMENT INTERPRETER & AMBIGUITY DETECTOR]")
        print("-" * 80)
        interpreted = interpret_requirement(raw_text)
        print(f"  Extracted Metric:    {interpreted.metric.upper()}")
        print(f"  Extracted Operator:  {interpreted.operator}")
        print(f"  Extracted Threshold: {interpreted.threshold:.2%}")
        print(f"  Extracted Severity:  {interpreted.severity.upper()}")
        print(f"  Is Ambiguous:        {interpreted.is_ambiguous}")
        if interpreted.is_ambiguous:
            print(f"  Ambiguity Reason:    {interpreted.ambiguity_reason}")

        # 3. Load reference dataset and calibrate empirical threshold per metric
        print(f"\n[STEP 3: THRESHOLD CALIBRATION ENGINE]")
        print("-" * 80)
        ref_df = pd.read_csv(ref_path)

        reference_metric_values = compute_reference_metric_values(ref_df, interpreted.metric, adapter)

        calib_res = calibrate_threshold(reference_metric_values, method="percentile", operator=interpreted.operator)
        print(f"  Target Metric:       {interpreted.metric.upper()}")
        print(f"  Calibration Method:  {calib_res['method']}")
        print(f"  Reference Samples:   {calib_res['sample_size']} calibration evaluation slices")
        print(f"  Baseline Mean:       {calib_res['mean']:.4f}")
        print(f"  Baseline Std Dev:    {calib_res['std']:.4f}")
        print(f"  Raw Threshold:       {calib_res['raw_threshold']:.4f}")
        print(f"  Calibrated Threshold:{calib_res['calibrated_threshold']:.4f} ({calib_res['calibrated_threshold']:.2%})")

        # Optional threshold override prompt in interactive modes
        auto_threshold = calib_res["calibrated_threshold"]
        op_threshold = auto_threshold
        threshold_src = "calibrated"

        if interactive or interactive_model:
            try:
                override_input = input(
                    f"\nAuto-calibrated operational threshold for {interpreted.metric.upper()}: {auto_threshold:.2%}. "
                    f"Press Enter to accept this, or type your own threshold value to override it: "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                override_input = ""

            if override_input:
                try:
                    val_str = override_input.rstrip("%").strip()
                    parsed_val = float(val_str)
                    if parsed_val > 1.0:
                        parsed_val /= 100.0
                    op_threshold = parsed_val
                    threshold_src = "user_override"
                    print(f"  -> User override applied: operational baseline threshold set to {op_threshold:.2%} (threshold_source: user_override)")
                except ValueError:
                    print(f"  -> Invalid threshold input '{override_input}'. Retaining calibrated threshold {auto_threshold:.2%}.")

        # 4. Generate Monitor Config using dual thresholds (business & empirical/user baseline)
        print(f"\n[STEP 4: MONITOR CONFIGURATION GENERATION]")
        print("-" * 80)
        monitor_config = generate_monitor_config(
            interpreted,
            calibrated_threshold=op_threshold,
            baseline_performance=calib_res["mean"],
            threshold_source=threshold_src,
        )
        print(f"  Generated Config:")
        print(f"    - Metric:                           {monitor_config['metric'].upper()}")
        print(f"    - Operator:                         {monitor_config['operator']}")
        print(f"    - Business Requirement Threshold:   {monitor_config['operator']} {monitor_config['business_requirement_threshold']:.2%}")
        print(f"    - Operational Baseline Threshold:  {monitor_config['operator']} {monitor_config['operational_baseline_threshold']:.2%}")
        print(f"    - Threshold Source:                 {monitor_config.get('threshold_source', 'calibrated')}")
        print(f"    - Requirement Met by Baseline:      {monitor_config['requirement_met_by_baseline']}")

        # 5. Path A: Run Monitor on Production Drifted Data (VIOLATION PATH)
        print(f"\n" + "=" * 80)
        print("  [RUN 1: PRODUCTION DRIFTED DATA — VIOLATION PATH]")
        print("=" * 80)
        prod_drifted_df = pd.read_csv(drifted_path)
        drift_result = run_monitor(monitor_config, prod_drifted_df, adapter)

        print(f"  Observed Metric Value:          {drift_result['observed_value']:.2%}")
        print(f"  Business Requirement Violation: {drift_result['business_requirement_violation']}")
        print(f"  Operational Drift Violation:     {drift_result['operational_drift_violation']}")
        print(f"  Requirement Met by Baseline:    {drift_result['requirement_met_by_baseline']}")

        if drift_result["business_requirement_violation"] or drift_result["operational_drift_violation"]:
            print(f"\n  --- VIOLATION EXPLANATION ENGINE ---")
            exp_res = explain_violation(monitor_config, drift_result["observed_value"])
            print(f"  Generated Alert:\n  -> \"{exp_res['explanation']}\"")

            print(f"\n  --- SUSPECTED DRIFT CONTRIBUTORS (PSI RANKING) ---")
            contributors = rank_contributors(ref_df, prod_drifted_df, top_n=3)
            for rank, c in enumerate(contributors, 1):
                print(f"  Rank {rank}: Feature = {c['feature']:<25} | PSI = {c['psi']:.4f} | ({c['note']})")

        # 6. Path B: Run Monitor on Production Clean Data (REUSING SAME CONFIG)
        print(f"\n" + "=" * 80)
        print("  [RUN 2: PRODUCTION CLEAN DATA — REUSING SAME MONITOR CONFIG]")
        print("=" * 80)
        prod_clean_df = pd.read_csv(clean_path)

        clean_result = run_monitor(monitor_config, prod_clean_df, adapter)

        print(f"  Observed Metric Value:          {clean_result['observed_value']:.2%}")
        print(f"  Business Requirement Violation: {clean_result['business_requirement_violation']}")
        print(f"  Operational Drift Violation:     {clean_result['operational_drift_violation']}")
        print(f"  Requirement Met by Baseline:    {clean_result['requirement_met_by_baseline']}")

        exp_clean = explain_violation(monitor_config, clean_result["observed_value"])
        print(f"  Status Summary:\n  -> \"{exp_clean['explanation']}\"")

    print("\n" + "=" * 80)
    print("                 END OF DEMO EXECUTION")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="RAMMA NLP & Backend CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available CLI commands")

    demo_parser = subparsers.add_parser("demo", help="Run full end-to-end ML monitoring demo")
    demo_parser.add_argument(
        "--model-source",
        type=str,
        default=None,
        help="Model inference source: 'local', 'local:<path>', or 'remote:<url>'",
    )
    demo_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom path to local model pickle file (e.g. models/alt_classifier_logreg.pkl)",
    )
    demo_parser.add_argument(
        "--requirements",
        type=str,
        default=None,
        help="Comma-separated list of business requirements to evaluate",
    )
    demo_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt interactively for multiple requirements until 'done'",
    )
    demo_parser.add_argument(
        "--interactive-model",
        action="store_true",
        help="Interactively select model source and threshold overrides",
    )

    args = parser.parse_args()

    if args.command == "demo" or len(sys.argv) == 1:
        req_list = None
        if hasattr(args, "requirements") and args.requirements:
            req_list = [r.strip() for r in args.requirements.split(",") if r.strip()]

        model_path_arg = getattr(args, "model_path", None)
        model_source_arg = getattr(args, "model_source", None)

        interactive_model_flag = getattr(args, "interactive_model", False)
        interactive_flag = getattr(args, "interactive", False)

        has_explicit_model_arg = (model_path_arg is not None) or (model_source_arg is not None)

        if interactive_model_flag:
            is_interactive_model = True
            model_src = "local"
        elif not has_explicit_model_arg and not req_list and not interactive_flag:
            # Default when cli.py demo is executed without arguments
            is_interactive_model = True
            model_src = "local"
        else:
            is_interactive_model = False
            if model_path_arg:
                model_src = f"local:{model_path_arg}"
            elif model_source_arg:
                model_src = model_source_arg
            else:
                model_src = "local"

        run_demo(
            requirements=req_list,
            model_source=model_src,
            interactive=interactive_flag,
            interactive_model=is_interactive_model,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
