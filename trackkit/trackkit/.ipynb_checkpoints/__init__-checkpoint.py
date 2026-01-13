"""
track_utils package
==================
Utilities for track feature preprocessing, summarization, and plotting.

Submodules:
- preprocessing
- plotting
- summary
"""

# Expose submodules
from . import preprocessing
from . import plotting
from . import summary

# Expose most-used functions at package level
from .preprocessing import (
    normalize_features,
    normalize_2d,
    normalize_features_masked,
    clip_outliers,
    clip_outliers_3d,
    masked_log_transform,
    compute_percentile_bounds,
    masked_flatten
)

from .plotting import (
    plot_feature_distributions,
    plot_single_feature,
    plot_cms_tracker_background
)

from .summary import (
    summarize_features,
    summarize_recHits,
    print_summary_table,
    flag_outliers
)

# Public API
__all__ = [
    "preprocessing",
    "plotting",
    "summary",
    "normalize_features",
    "normalize_2d",
    "normalize_features_masked",
    "clip_outliers",
    "clip_outliers_3d",
    "masked_log_transform",
    "compute_percentile_bounds",
    "masked_flatten",
    "plot_feature_distributions",
    "plot_single_feature",
    "plot_cms_tracker_background",
    "summarize_features",
    "summarize_recHits",
    "print_summary_table",
    "flag_outliers"
]
