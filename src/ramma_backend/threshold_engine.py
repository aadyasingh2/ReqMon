"""Threshold calibration engine for ML monitoring requirements."""

from typing import Any, Dict, List
import numpy as np


def calibrate_threshold(
    reference_metric_values: List[float], method: str = "percentile"
) -> Dict[str, Any]:
    """Calibrates an empirical monitoring threshold from reference metric values.

    Args:
        reference_metric_values: List of metric values evaluated on reference/baseline data.
        method: Calibration technique (default: 'percentile' - 2 std deviations below mean).

    Returns:
        Dict containing auditable calculation components:
        - method: calibration method name
        - sample_size: number of reference values provided
        - mean: mean of reference metric values
        - std: standard deviation of reference metric values
        - raw_threshold: calculated mean - 2 * std
        - calibrated_threshold: final threshold clipped to [0.0, 1.0]

    Raises:
        ValueError: If fewer than 10 reference values are provided.
    """
    sample_size = len(reference_metric_values)
    if sample_size < 10:
        raise ValueError(
            f"Insufficient data for calibration: provided {sample_size} reference values, "
            "but at least 10 reference values are required for reliable threshold calibration."
        )

    vals = np.array(reference_metric_values, dtype=float)
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))

    if method == "percentile":
        raw_threshold = mean_val - 2.0 * std_val
    else:
        raise ValueError(f"Unsupported calibration method: '{method}'. Supported methods: ['percentile']")

    # Clip threshold strictly to [0.0, 1.0]
    calibrated_threshold = max(0.0, min(1.0, raw_threshold))

    result = {
        "method": method,
        "sample_size": sample_size,
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "raw_threshold": round(raw_threshold, 6),
        "calibrated_threshold": round(calibrated_threshold, 6),
    }

    return result
