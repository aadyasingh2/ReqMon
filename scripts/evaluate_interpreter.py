"""Evaluation harness for the requirement interpreter."""

import os
import sys
import pandas as pd
import math

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ramma_nlp.interpreter import interpret_requirement


def run_evaluation(
    csv_path: str = "data/labeled_requirements.csv",
    output_path: str | None = None,
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labeled requirements file not found at: {csv_path}")

    if output_path is None:
        filename = os.path.basename(csv_path)
        output_path = os.path.join("data", f"eval_results_{filename}")

    df = pd.read_csv(csv_path)

    predicted_metrics = []
    predicted_operators = []
    predicted_thresholds = []
    predicted_ambiguities = []
    correct_flags = []

    tp = 0  # Expected ambiguous, predicted ambiguous
    fp = 0  # Expected unambiguous, predicted ambiguous
    fn = 0  # Expected ambiguous, predicted unambiguous
    tn = 0  # Expected unambiguous, predicted unambiguous

    total_rows = len(df)
    correct_count = 0

    for idx, row in df.iterrows():
        text = str(row["requirement_text"])
        exp_metric = str(row["expected_metric"]).strip().lower() if pd.notna(row.get("expected_metric")) else ""
        exp_operator = str(row["expected_operator"]).strip() if pd.notna(row.get("expected_operator")) else ""
        exp_threshold = float(row["expected_threshold"]) if pd.notna(row.get("expected_threshold")) else None
        exp_ambiguous = str(row["expected_is_ambiguous"]).strip().lower() == "true"

        try:
            interp = interpret_requirement(text)
            pred_metric = interp.metric
            pred_operator = interp.operator
            pred_threshold = interp.threshold
            pred_ambiguous = interp.is_ambiguous
        except Exception as e:
            print(f"Error interpreting row {idx+1} ('{text}'): {e}")
            pred_metric = "error"
            pred_operator = "error"
            pred_threshold = -1.0
            pred_ambiguous = False

        predicted_metrics.append(pred_metric)
        predicted_operators.append(pred_operator)
        predicted_thresholds.append(pred_threshold)
        predicted_ambiguities.append(pred_ambiguous)

        # Ambiguity confusion matrix
        if exp_ambiguous and pred_ambiguous:
            tp += 1
        elif not exp_ambiguous and pred_ambiguous:
            fp += 1
        elif exp_ambiguous and not pred_ambiguous:
            fn += 1
        else:
            tn += 1

        # Correctness logic
        if exp_ambiguous:
            is_correct = pred_ambiguous is True
        else:
            is_correct = (
                (pred_ambiguous is False)
                and (pred_metric == exp_metric)
                and (pred_operator == exp_operator)
                and (exp_threshold is not None and math.isclose(pred_threshold, exp_threshold, abs_tol=1e-3))
            )

        if is_correct:
            correct_count += 1

        correct_flags.append(is_correct)

    df["predicted_metric"] = predicted_metrics
    df["predicted_operator"] = predicted_operators
    df["predicted_threshold"] = predicted_thresholds
    df["predicted_is_ambiguous"] = predicted_ambiguities
    df["correct"] = correct_flags

    accuracy = (correct_count / total_rows * 100) if total_rows > 0 else 0.0
    ambiguity_precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    ambiguity_recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    is_stress_test = "stress_test" in csv_path.lower()
    accuracy_label = "Stress-Test Accuracy (Hand-Written, Unseen)" if is_stress_test else "Overall Accuracy"

    print("=" * 60)
    print(f"REQUIREMENT INTERPRETER EVALUATION ({os.path.basename(csv_path)})")
    print("=" * 60)
    print(f"Total Evaluated Requirements: {total_rows}")
    print(f"{accuracy_label:<40}: {accuracy:.2f}% ({correct_count}/{total_rows})")
    print(f"Ambiguity Detection Precision:            {ambiguity_precision:.2%}")
    print(f"Ambiguity Detection Recall:               {ambiguity_recall:.2%}")
    print("=" * 60)

    # Save output CSV
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}\n")

    return df


if __name__ == "__main__":
    target_csv = sys.argv[1] if len(sys.argv) > 1 else "data/labeled_requirements.csv"
    run_evaluation(target_csv)
