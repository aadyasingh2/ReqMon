"""Alert and Explanation Engine for ML monitoring requirement violations."""

import math
import os
from typing import Any, Dict, Union
import google.generativeai as genai
from ramma_nlp.schema import InterpretedRequirement


def check_violation(operator: str, threshold: float, observed_value: float) -> bool:
    """Determines whether an observed metric value violates the requirement.

    Deterministic Python logic:
    - '>=': violated if observed < threshold
    - '<=': violated if observed > threshold
    - '>':  violated if observed <= threshold
    - '<':  violated if observed >= threshold
    - '==': violated if not math.isclose(observed, threshold, abs_tol=1e-5)
    """
    op = operator.strip()
    if op == ">=":
        return observed_value < threshold
    elif op == "<=":
        return observed_value > threshold
    elif op == ">":
        return observed_value <= threshold
    elif op == "<":
        return observed_value >= threshold
    elif op == "==":
        return not math.isclose(observed_value, threshold, abs_tol=1e-5)
    else:
        raise ValueError(f"Unsupported operator: '{operator}'")


def generate_llm_explanation(
    config_dict: Dict[str, Any],
    observed_value: float,
    biz_violation: bool = False,
    drift_violation: bool = False,
) -> str:
    """Generates a plain-English explanation of violations using Gemini."""
    metric = config_dict.get("metric", "metric")
    operator = config_dict.get("operator", ">=")
    biz_thresh = config_dict.get("business_requirement_threshold", config_dict.get("threshold", 0.0))
    op_thresh = config_dict.get("operational_baseline_threshold", config_dict.get("threshold", 0.0))
    req_met = config_dict.get("requirement_met_by_baseline", False)
    baseline_perf = config_dict.get("baseline_performance")
    if baseline_perf is not None:
        baseline_str = f"{float(baseline_perf):.2%}"
    else:
        baseline_str = f"{float(op_thresh):.2%}"

    explanations = []
    if biz_violation:
        if not req_met:
            explanations.append(
                f"Business Requirement Not Met: {metric} observed value ({observed_value:.2%}) is below business requirement of {operator} {biz_thresh:.2%} "
                f"(note: baseline model performance of ~{baseline_str} never satisfied this target)."
            )
        else:
            explanations.append(
                f"Business Requirement Violated: {metric} observed value ({observed_value:.2%}) failed business target of {operator} {biz_thresh:.2%}."
            )

    if drift_violation:
        explanations.append(
            f"Operational Drift Violated: {metric} observed value ({observed_value:.2%}) dropped below operational baseline threshold of {operator} {op_thresh:.2%}."
        )

    base_explanation = " | ".join(explanations) if explanations else f"Requirement violated for {metric}."

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        return base_explanation

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        prompt = (
            "You are an AI ML monitoring assistant. Rephrase the following violation alert into a clear, "
            "concise, professional plain-English explanation for an engineering team:\n\n"
            f"Metric: {metric}\n"
            f"Business Target: {operator} {biz_thresh}\n"
            f"Operational Baseline: {operator} {op_thresh}\n"
            f"Observed Value: {observed_value}\n"
            f"Standard style to follow:\n\"{base_explanation}\""
        )
        response = model.generate_content(prompt)
        text = response.text.strip()
        return text if text else base_explanation
    except Exception:
        return base_explanation


def explain_violation(
    config: Union[Dict[str, Any], InterpretedRequirement], observed_value: float
) -> Dict[str, Any]:
    """Evaluates dual violation conditions deterministically and returns structured result with explanation.

    Args:
        config: Dict or InterpretedRequirement specifying metric, operator, and thresholds.
        observed_value: Empirically computed metric value from production data.

    Returns:
        Dict: {is_violation, business_requirement_violation, operational_drift_violation, requirement_met_by_baseline, explanation, config, observed_value}
    """
    if isinstance(config, InterpretedRequirement):
        cfg = config.model_dump()
    else:
        cfg = dict(config)

    operator = str(cfg.get("operator", ">="))
    biz_thresh = float(cfg.get("business_requirement_threshold", cfg.get("threshold", 0.0)))
    op_thresh = float(cfg.get("operational_baseline_threshold", cfg.get("threshold", 0.0)))
    metric = str(cfg.get("metric", "metric"))
    req_met_by_baseline = bool(cfg.get("requirement_met_by_baseline", False))

    biz_violation = check_violation(operator, biz_thresh, observed_value)
    drift_violation = check_violation(operator, op_thresh, observed_value)

    if not biz_violation and not drift_violation:
        explanation = (
            f"No violation detected: {metric} observed value ({observed_value:.2%}) satisfies "
            f"business requirement ({operator} {biz_thresh:.2%}) and operational baseline ({operator} {op_thresh:.2%})."
        )
    else:
        explanation = generate_llm_explanation(
            cfg, observed_value, biz_violation=biz_violation, drift_violation=drift_violation
        )

    res = {
        "is_violation": biz_violation or drift_violation,
        "business_requirement_violation": biz_violation,
        "operational_drift_violation": drift_violation,
        "requirement_met_by_baseline": req_met_by_baseline,
        "explanation": explanation,
        "config": cfg,
        "observed_value": observed_value,
    }

    # Backward compatibility for code expecting 'requirement' key
    if isinstance(config, InterpretedRequirement):
        res["requirement"] = config

    return res
