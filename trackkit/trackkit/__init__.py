"""
track_utils package
===================
Utilities for track feature preprocessing, summarization, and plotting.

Submodules:
- preprocessing
- plotting
- summary
"""

# -----------------------------------------------------------------------------
# Submodules
# -----------------------------------------------------------------------------
from . import plotting
from . import preprocessing
from . import summary

# -----------------------------------------------------------------------------
# Preprocessing utilities
# -----------------------------------------------------------------------------
from .preprocessing import (
    clip_outliers,
    clip_outliers_3d,
    compute_percentile_bounds,
    masked_flatten,
    masked_log_transform,
    normalize_2d,
    normalize_features,
    normalize_features_masked,
)

# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------
from .plotting import (
    plot_cms_tracker_background,
    plot_feature_distributions,
    plot_single_feature,
)

# -----------------------------------------------------------------------------
# Summary utilities
# -----------------------------------------------------------------------------
from .summary import (
    flag_outliers,
    print_summary_table,
    summarize_features,
    summarize_recHits,
)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    # submodules
    "preprocessing",
    "plotting",
    "summary",

    # preprocessing
    "clip_outliers",
    "clip_outliers_3d",
    "compute_percentile_bounds",
    "masked_flatten",
    "masked_log_transform",
    "normalize_2d",
    "normalize_features",
    "normalize_features_masked",

    # plotting
    "plot_cms_tracker_background",
    "plot_feature_distributions",
    "plot_single_feature",

    # summary
    "flag_outliers",
    "print_summary_table",
    "summarize_features",
    "summarize_recHits",
]
