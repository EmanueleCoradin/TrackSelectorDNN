"""
preprocessing.py

A collection of utilities for preprocessing 3D track hit data and 2D track features.
Includes normalization, masked operations, outlier clipping, and log transforms.

Data shapes conventions:
- feat3d: (n_tracks, n_hits, n_features)
- mask:   (n_tracks, n_hits), dtype=bool, True = real hit, False = padded
"""

import numpy as np

def normalize_features(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize a 3D array of features.
    
    Parameters
    ----------
    x : np.ndarray
        Input feature array of shape (n_tracks, n_hits, n_features)
    eps : float, optional
        Small value added to standard deviation to avoid division by zero (default: 1e-8)
    
    Returns
    -------
    x_norm : np.ndarray
        Normalized array of the same shape as x
    mean : np.ndarray
        Mean values per feature, shape (1, 1, n_features)
    std : np.ndarray
        Standard deviation per feature, shape (1, 1, n_features)
    
    Notes
    -----
    Normalization is applied over axes 0 and 1 (tracks and hits). 
    
    """
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    std = np.std(x, axis=(0, 1), keepdims=True)
    x_norm = (x - mean) / (std + eps)
    return x_norm, mean, std

def normalize_2d(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize a 2D array of features across samples.

    Parameters
    ----------
    x : np.ndarray
        Input feature array of shape (n_samples, n_features)
    eps : float, optional
        Small value added to standard deviation to avoid division by zero (default: 1e-8)

    Returns
    -------
    x_norm : np.ndarray
        Normalized array of same shape as x
    mean : np.ndarray
        Mean per feature, shape (1, n_features)
    std : np.ndarray
        Standard deviation per feature, shape (1, n_features)
    """
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    return (x - mean) / (std + eps), mean, std

def masked_flatten(feat3d: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """
    Flatten a 3D feature array into a list of 1D arrays for each feature, 
    keeping only real (mask==True) hits.

    Parameters
    ----------
    feat3d : np.ndarray
        Input array of shape (n_tracks, n_hits, n_features)
    mask : np.ndarray
        Boolean mask of shape (n_tracks, n_hits), True = real hit, False = padded

    Returns
    -------
    flat_per_feature : list of np.ndarray
        List of length n_features, each array containing all valid values for that feature.
    """
    n_features = feat3d.shape[2]
    flat_per_feature = []
    for f in range(n_features):
        vals = feat3d[:,:,f][mask]      # selects only real hits
        flat_per_feature.append(vals)
    return flat_per_feature

def compute_percentile_bounds(
    feat3d: np.ndarray,
    mask: np.ndarray,
    feature_idx: int,
    low: float = 0.001,
    high: float = 0.999
) -> tuple[float, float]:
    """
    Compute lower and upper percentile bounds for a single feature in a masked 3D array.

    Parameters
    ----------
    feat3d : np.ndarray
        Feature array of shape (n_tracks, n_hits, n_features)
    mask : np.ndarray
        Boolean mask (n_tracks, n_hits), True = real hit
    feature_idx : int
        Index of the feature to compute percentiles for
    low : float, optional
        Lower percentile (default: 0.001)
    high : float, optional
        Upper percentile (default: 0.999)

    Returns
    -------
    lo : float
        Lower percentile value
    hi : float
        Upper percentile value
    """

    vals = feat3d[:,:,feature_idx][mask]
    lo, hi = np.nanpercentile(vals, [100*low, 100*high])
    return (lo, hi)


def clip_outliers(
    x: np.ndarray,
    low: float = 0.001,
    high: float = 0.999
) -> tuple[np.ndarray, float, float]:
    """
    Clip outliers of a 1D or 2D array based on percentile bounds.

    Parameters
    ----------
    x : np.ndarray
        Input array of any shape
    low : float, optional
        Lower percentile bound (default: 0.001)
    high : float, optional
        Upper percentile bound (default: 0.999)

    Returns
    -------
    x_clipped : np.ndarray
        Array with values clipped to [lo, hi], same shape as x
    lo : float
        Computed lower percentile value
    hi : float
        Computed upper percentile value
    """
    lo, hi = np.percentile(x, [100 * low, 100 * high])
    x_clipped = np.clip(x, lo, hi)
    return x_clipped, lo, hi
    
def clip_outliers_3d(
    feat3d: np.ndarray,
    mask: np.ndarray,
    feature_idx: int,
    low: float = 0.001,
    high: float = 0.999
) -> tuple[np.ndarray, float, float]:
    """
    Clip outliers of a single feature in a 3D array based on percentile bounds.

    Parameters
    ----------
    feat3d : np.ndarray
        Input array (n_tracks, n_hits, n_features)
    mask : np.ndarray
        Boolean mask of shape (n_tracks, n_hits)
    feature_idx : int
        Feature index to clip
    low : float, optional
        Lower percentile bound (default: 0.001)
    high : float, optional
        Upper percentile bound (default: 0.999)

    Returns
    -------
    out : np.ndarray
        Copy of the input array with outliers clipped
    lo : float
        Lower clipping value
    hi : float
        Upper clipping value
    """
    out = feat3d.copy()
    lo, hi = compute_percentile_bounds(feat3d, mask, feature_idx, low, high)
    sel = mask
    vals = out[:,:,feature_idx]
    vals[sel] = np.clip(vals[sel], lo, hi)
    out[:,:,feature_idx] = vals
    return out, lo, hi
    
    
def masked_log_transform(
    feat3d: np.ndarray,
    mask: np.ndarray,
    feature_idx: int,
    eps: float = 1e-8,
    method: str = 'log1p'
) -> np.ndarray:
    """
    Apply a log-based transformation to masked elements of a 3D array.

    Parameters
    ----------
    feat3d : np.ndarray
        Input array of shape (n_tracks, n_hits, n_features)
    mask : np.ndarray
        Boolean mask of shape (n_tracks, n_hits)
    feature_idx : int
        Index of the feature to transform
    eps : float, optional
        Small value added to avoid log(0) (default: 1e-8)
    method : str, optional
        Method of transformation:
        - 'log1p' : log1p(x)
        - 'log_eps' : log10(x + eps)
        - 'signed_log_eps' : sign(x) * log10(|x| + eps)

    Returns
    -------
    out : np.ndarray
        Copy of input array with transformed values
    """
    out = feat3d.copy()
    sel = mask
    vals = out[:,:,feature_idx]
    if method == 'log1p':
        vals[sel] = np.log1p(vals[sel])
    elif method == 'log_eps':
        vals[sel] = np.log10(vals[sel] + eps)
    elif method == 'signed_log_eps':
        vals[sel] = np.sign(vals[sel])*np.log10(np.abs(vals[sel]) + eps)
    else:
        raise ValueError("unknown method")
    out[:,:,feature_idx] = vals
    return out


def normalize_features_masked(
    feat3d: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize a 3D feature array along the mask, leaving padded values zeroed.

    Parameters
    ----------
    feat3d : np.ndarray
        Input array of shape (n_tracks, n_hits, n_features)
    mask : np.ndarray
        Boolean mask of shape (n_tracks, n_hits)
    eps : float, optional
        Small value added to standard deviation to avoid division by zero (default: 1e-8)

    Returns
    -------
    out : np.ndarray
        Normalized array, same shape as input
    mean : np.ndarray
        Per-feature mean, shape (1, 1, n_features)
    std : np.ndarray
        Per-feature std, shape (1, 1, n_features)

    Notes
    -----
    - Normalization applied only to mask==True entries
    - Padded entries (mask==False) are set to 0
    - If no valid entries exist for a feature, mean=0 and std=1
    """
    out = feat3d.copy().astype(float)
    n_tracks, n_hits, n_feats = out.shape
    mean = np.zeros((1,1,n_feats), dtype=float)
    std  = np.ones((1,1,n_feats), dtype=float)
    for f in range(n_feats):
        vals = out[:,:,f][mask]
        if vals.size == 0:
            mean[0,0,f] = 0.0
            std[0,0,f] = 1.0
        else:
            m = np.nanmean(vals)
            s = np.nanstd(vals)
            mean[0,0,f] = m
            std[0,0,f] = s
            tmp = out[:,:,f]
            sel = mask
            tmp[sel] = (tmp[sel] - m) / (s + eps)
            out[:,:,f] = tmp
    out[~mask] = 0.0
    return out, mean, std
