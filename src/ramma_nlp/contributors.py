"""Suspected Contributor module using Population Stability Index (PSI)."""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


def calculate_feature_psi(
    ref_col: pd.Series, curr_col: pd.Series, num_bins: int = 10, eps: float = 1e-4
) -> float:
    """Calculates Population Stability Index (PSI) for a single numerical feature.

    Uses quantile bins defined by the reference distribution.
    """
    ref_clean = ref_col.dropna().values
    curr_clean = curr_col.dropna().values

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    # Determine bin edges from reference distribution quantiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    # Ensure unique non-decreasing bin edges
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) <= 1:
        return 0.0

    # Extend boundary limits slightly to include min and max values
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    # Compute counts per bin
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    # Convert to proportions
    ref_props = ref_counts / len(ref_clean)
    curr_props = curr_counts / len(curr_clean)

    # Replace zeros with epsilon smoothing
    ref_props = np.where(ref_props == 0, eps, ref_props)
    curr_props = np.where(curr_props == 0, eps, curr_props)

    # Calculate PSI per bin and sum
    psi_value = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
    return float(max(0.0, psi_value))


def rank_contributors(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, top_n: int = 3
) -> List[Dict[str, Any]]:
    """Ranks shared numeric features by Population Stability Index (PSI) drift.

    Note: PSI measures distribution drift between reference and production datasets.
    This is a correlational signal indicating potential distribution shifts, NOT a proven causal relationship.

    Args:
        reference_df: Baseline reference pandas DataFrame.
        current_df: Current production pandas DataFrame.
        top_n: Number of top feature contributors to return.

    Returns:
        List of dicts formatted as:
        [{"feature": name, "psi": round(value, 4), "note": "possible contributor, not a proven cause"}, ...]
    """
    if reference_df.empty or current_df.empty:
        return []

    # Identify shared numeric columns
    ref_numeric = set(reference_df.select_dtypes(include=[np.number]).columns)
    curr_numeric = set(current_df.select_dtypes(include=[np.number]).columns)
    shared_cols = sorted(list(ref_numeric.intersection(curr_numeric)))

    psi_results = []
    for col in shared_cols:
        try:
            psi_val = calculate_feature_psi(reference_df[col], current_df[col])
            psi_results.append((col, psi_val))
        except Exception:
            continue

    # Sort descending by PSI score
    psi_results.sort(key=lambda x: x[1], reverse=True)

    ranked_contributors = []
    for col, psi_val in psi_results[:top_n]:
        ranked_contributors.append(
            {
                "feature": col,
                "psi": round(psi_val, 4),
                "note": "possible contributor, not a proven cause",
            }
        )

    return ranked_contributors
