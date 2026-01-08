"""
summary.py

Functions to summarize features and recHits and flag potential outliers.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew

from .preprocessing import masked_flatten

def summarize_recHits(
    feat3d: np.ndarray,
    mask: np.ndarray,
    feature_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Summarize statistics for recHit features.

    Parameters
    ----------
    feat3d : np.ndarray
        Array of shape (n_tracks, n_hits, n_features)
    mask : np.ndarray
        Boolean mask indicating valid hits (same shape as first two dims of feat3d)
    feature_names : list of str
        Names of the features

    Returns
    -------
    summary : list of dict
        Each dict contains feature statistics: n, min, max, mean, std, zeros(%), skew
    """
    
    n_tracks, n_hits, n_feats = feat3d.shape
    summary = []

    # global per-feature
    flat = masked_flatten(feat3d, mask)
    for f, arr in enumerate(flat):
        if arr.size == 0:
            continue
        summary.append({
            "feature": feature_names[f],
            "n": arr.size,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "zeros(%)": 100.0 * np.sum(np.isclose(arr, 0.0)) / arr.size,
            "skew": float(skew(arr))
        })
    return summary

def summarize_features(
    X: np.ndarray,
    feature_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Summarize statistics for a 2D feature array.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples, n_features)
    feature_names : list of str
        Names of the features

    Returns
    -------
    summary : list of dict
        Each dict contains feature statistics: min, max, mean, std, zeros(%), NaNs(%), skew
    """
    summary = []
    for i, name in enumerate(feature_names):
        x = X[:, i]
        x_clean = x[~np.isnan(x)]
        min_val = np.min(x_clean)
        max_val = np.max(x_clean)
        mean_val = np.mean(x_clean)
        std_val = np.std(x_clean)
        zero_frac = np.sum(np.isclose(x_clean, 0)) / len(x_clean)
        nan_frac = np.sum(np.isnan(x)) / len(x)
        s = skew(x_clean)
        summary.append({
            "feature": name,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "zeros(%)": 100*zero_frac,
            "NaNs(%)": 100*nan_frac,
            "skew": s
        })
    return summary

def print_summary_table(
    summary: List[Dict[str, Any]],
    sort_by: str = "skew",
    top: int = 10
) -> pd.DataFrame:
    """
    Print a summary table sorted by a specific metric.

    Parameters
    ----------
    summary : list of dict
        Output from summarize_features or summarize_recHits
    sort_by : str, optional
        Column name to sort by (default: 'skew')
    top : int, optional
        Number of top features to display (default: 10)

    Returns
    -------
    df_sorted : pd.DataFrame
        Sorted pandas DataFrame
    """
    df = pd.DataFrame(summary)
    df_sorted = df.sort_values(by=sort_by, key=lambda x: np.abs(x), ascending=False)
    print(df_sorted.head(top).to_string(index=False))
    return df_sorted

def flag_outliers(
    summary: List[Dict[str, Any]],
    std_threshold: float = 100,
    skew_threshold: float = 5,
    zero_frac_threshold: float = 80
) -> List[Tuple[str, str]]:
    """
    Flag features that may be problematic based on thresholds.

    Parameters
    ----------
    summary : list of dict
        Output from summarize_features or summarize_recHits
    std_threshold : float, optional
        Maximum allowed ratio of (max-min)/std to flag extreme outliers (default: 100)
    skew_threshold : float, optional
        Maximum allowed skewness (default: 5)
    zero_frac_threshold : float, optional
        Maximum fraction of zeros before flagging (default: 80%)

    Returns
    -------
    issues : list of tuple
        Each tuple contains (feature_name, reason)
    """
    issues = []
    for row in summary:
        if np.isnan(row["std"]) or row["std"] == 0:
            issues.append((row["feature"], "Constant or NaN std"))
        if abs(row["skew"]) > skew_threshold:
            issues.append((row["feature"], f"Highly skewed ({row['skew']:.2f})"))
        if row["zeros(%)"] > zero_frac_threshold:
            issues.append((row["feature"], f"Mostly zeros ({row['zeros(%)']:.1f}%)"))
        if abs(row["max"] - row["min"]) > std_threshold * row["std"]:
            issues.append((row["feature"], "Extreme outlier range"))
    print("\n Potentially problematic features:")
    for f, msg in issues:
        print(f" - {f:30s}: {msg}")
    return issues