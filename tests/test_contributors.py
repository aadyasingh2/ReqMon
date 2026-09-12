"""Unit tests for suspected contributor module (contributors.py)."""

import numpy as np
import pandas as pd
from ramma_nlp.contributors import rank_contributors


def test_no_drift_psi_near_zero():
    """Test that features with identical/similar distributions yield low PSI (< 0.1)."""
    np.random.seed(42)
    ref_data = {
        "age": np.random.normal(35, 5, 1000),
        "income": np.random.normal(50000, 10000, 1000),
    }
    curr_data = {
        "age": np.random.normal(35, 5, 1000),
        "income": np.random.normal(50000, 10000, 1000),
    }

    ref_df = pd.DataFrame(ref_data)
    curr_df = pd.DataFrame(curr_data)

    contributors = rank_contributors(ref_df, curr_df, top_n=2)
    assert len(contributors) == 2
    for item in contributors:
        assert item["psi"] < 0.1
        assert item["note"] == "possible contributor, not a proven cause"


def test_obvious_drift_psi_high():
    """Test that a shifted feature is ranked first with high PSI (> 0.25)."""
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "stable_feature": np.random.normal(10, 2, 1000),
        "drifted_feature": np.random.normal(100, 15, 1000),
    })
    curr_df = pd.DataFrame({
        "stable_feature": np.random.normal(10, 2, 1000),
        "drifted_feature": np.random.normal(140, 25, 1000),  # significant mean & std shift
    })

    contributors = rank_contributors(ref_df, curr_df, top_n=1)
    assert len(contributors) == 1
    assert contributors[0]["feature"] == "drifted_feature"
    assert contributors[0]["psi"] > 0.25


def test_missing_column_handled_gracefully():
    """Test that extra or missing columns between reference and current are handled smoothly."""
    ref_df = pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5] * 20,
        "only_in_ref": [10, 20, 30, 40, 50] * 20,
    })
    curr_df = pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5] * 20,
        "only_in_curr": [100, 200, 300, 400, 500] * 20,
    })

    contributors = rank_contributors(ref_df, curr_df, top_n=3)
    assert len(contributors) == 1
    assert contributors[0]["feature"] == "feature_a"
