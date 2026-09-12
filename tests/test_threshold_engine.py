"""Unit tests for threshold calibration engine (threshold_engine.py)."""

import math
import numpy as np
import pytest
from ramma_backend.threshold_engine import calibrate_threshold


def test_calibrate_threshold_valid_synthetic():
    """Test threshold calibration on a synthetic normal distribution."""
    np.random.seed(42)
    # Synthetic metric values centered around mean 0.92 with std 0.02
    synthetic_ref_values = list(np.random.normal(loc=0.92, scale=0.02, size=100))

    result = calibrate_threshold(synthetic_ref_values, method="percentile")

    assert isinstance(result, dict)
    assert result["method"] == "percentile"
    assert result["sample_size"] == 100
    assert math.isclose(result["mean"], np.mean(synthetic_ref_values), abs_tol=1e-4)
    assert math.isclose(result["std"], np.std(synthetic_ref_values), abs_tol=1e-4)

    expected_raw = result["mean"] - 2 * result["std"]
    assert math.isclose(result["raw_threshold"], expected_raw, abs_tol=1e-4)
    assert math.isclose(result["calibrated_threshold"], max(0.0, min(1.0, expected_raw)), abs_tol=1e-4)


def test_calibrate_threshold_clipping_lower_bound():
    """Test that thresholds falling below 0.0 are clipped to 0.0."""
    np.random.seed(42)
    # Synthetic values with low mean and large variance yielding negative raw threshold
    synthetic_ref_values = list(np.random.normal(loc=0.15, scale=0.10, size=20))

    result = calibrate_threshold(synthetic_ref_values, method="percentile")

    assert result["raw_threshold"] < 0.0
    assert result["calibrated_threshold"] == 0.0


def test_calibrate_threshold_insufficient_samples_error():
    """Test that providing fewer than 10 reference values raises a ValueError."""
    few_values = [0.90, 0.92, 0.91, 0.88, 0.89, 0.93, 0.87, 0.91, 0.90]  # 9 values
    assert len(few_values) == 9

    with pytest.raises(ValueError) as exc_info:
        calibrate_threshold(few_values)

    assert "provided 9 reference values" in str(exc_info.value)
    assert "at least 10 reference values are required" in str(exc_info.value)


def test_calibrate_threshold_unsupported_method():
    """Test that an invalid calibration method name raises a ValueError."""
    valid_values = [0.90] * 15

    with pytest.raises(ValueError) as exc_info:
        calibrate_threshold(valid_values, method="unknown_method")

    assert "Unsupported calibration method" in str(exc_info.value)
