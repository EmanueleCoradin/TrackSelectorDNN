"""
plotting.py

Utilities for plotting feature distributions and CMS tracker hit visualizations.

Data conventions:
-----------------
- X_ref, X_cmp: np.ndarray of shape (n_samples, n_features)
- y, y_cmp: np.ndarray of shape (n_samples,), binary labels (0/1 or False/True)
- true_hits: np.ndarray of shape (n_tracks, n_hits, n_features)
- isRecHit_true: np.ndarray of shape (n_tracks, n_hits), boolean mask
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_distributions(
    X_ref: np.ndarray,
    feature_names: list[str],
    y: np.ndarray | None = None,
    X_cmp: np.ndarray | None = None,
    labels: tuple[str, str] = ("Reference", "Comparison"),
    n_cols: int = 4,
    bins: int = 50,
    figsize: tuple[int, int] = (16, 12),
    density: bool = True,
    alpha_ref: float = 0.6,
    alpha_cmp: float = 0.4,
) -> None:
    """
    Plot feature distributions for one or two datasets, optionally separated by class labels.

    Parameters
    ----------
    X_ref : np.ndarray
        Reference dataset of shape (n_samples, n_features).
    feature_names : list of str
        Names of the features to plot.
    y : np.ndarray, optional
        Binary labels (True/False or 1/0). If given, separate histograms by class.
    X_cmp : np.ndarray, optional
        Comparison dataset (same shape as X_ref) to overlay.
    labels : tuple(str, str)
        Labels for the legend corresponding to (X_ref, X_cmp).
    n_cols : int
        Number of subplot columns.
    bins : int
        Number of histogram bins.
    figsize : tuple
        Overall figure size.
    density : bool
        Whether to normalize histograms to density.
    alpha_ref : float
        Transparency for reference histograms.
    alpha_cmp : float
        Transparency for comparison histograms.
    """

    n_features = X_ref.shape[1]
    n_rows = int(np.ceil(n_features / n_cols))
    plt.figure(figsize=figsize)

    for i, name in enumerate(feature_names):
        plt.subplot(n_rows, n_cols, i + 1)

        # Determine plotting mode
        if y is not None:
            mask_true = (y == 1)
            mask_fake = (y == 0)
            plt.hist(
                X_ref[mask_true, i], bins=bins, alpha=alpha_ref,
                label=f"{labels[0]} True", density=density
            )
            plt.hist(
                X_ref[mask_fake, i], bins=bins, alpha=alpha_ref,
                label=f"{labels[0]} Fake", density=density
            )
            if X_cmp is not None:
                plt.hist(
                    X_cmp[mask_true, i], bins=bins, alpha=alpha_cmp,
                    label=f"{labels[1]} True", density=density, linestyle="dashed"
                )
                plt.hist(
                    X_cmp[mask_fake, i], bins=bins, alpha=alpha_cmp,
                    label=f"{labels[1]} Fake", density=density, linestyle="dashed"
                )
        else:
            plt.hist(X_ref[:, i], bins=bins, alpha=alpha_ref, label=labels[0], density=density)
            if X_cmp is not None:
                plt.hist(X_cmp[:, i], bins=bins, alpha=alpha_cmp, label=labels[1], density=density)

        plt.title(name, fontsize=8)
        plt.tick_params(axis="x", labelsize=7)
        plt.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    plt.legend(fontsize=7)
    plt.show()

def plot_single_feature(
    X_ref: np.ndarray,
    feature_index: int,
    feature_name: str | None = None,
    y: np.ndarray | None = None,
    X_cmp: np.ndarray | None = None,
    y_cmp: np.ndarray | None = None,
    labels: tuple[str, str] = ("Reference", "Comparison"),
    bins: int = 60,
    figsize: tuple[int, int] = (8, 6),
    density: bool = True,
    alpha_ref: float = 0.6,
    alpha_cmp: float = 0.4,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """
    Plot the distribution of a single feature from one or two datasets,
    optionally separated by class labels, and optionally comparing a third dataset.

    Parameters
    ----------
    X_ref : np.ndarray
        Reference dataset (n_samples, n_features)
    feature_index : int
        Column index of the feature to plot
    feature_name : str, optional
        Feature name for titles and labels
    y : np.ndarray, optional
        Binary labels (0/1) for X_ref
    X_cmp : np.ndarray, optional
        Comparison dataset
    y_cmp : np.ndarray, optional
        Binary labels for X_cmp (if provided, separates True/False)
    labels : tuple of str, optional
        Labels for reference and comparison datasets
    bins : int, optional
        Number of histogram bins
    figsize : tuple, optional
        Figure size
    density : bool, optional
        Whether to normalize histograms to density
    alpha_ref : float, optional
        Transparency for reference dataset
    alpha_cmp : float, optional
        Transparency for comparison dataset
    title : str, optional
        Plot title
    xlabel : str, optional
        Label for x-axis
    ylabel : str, optional
        Label for y-axis
    """

    plt.figure(figsize=figsize)

    # Reference only
    if y is None:
        plt.hist(
            X_ref[:, feature_index], bins=bins, density=density,
            alpha=alpha_ref, label=labels[0], color="C0"
        )
        if X_cmp is not None:
            plt.hist(
                X_cmp[:, feature_index], bins=bins, density=density,
                alpha=alpha_cmp, label=labels[1], color="C1"
            )

    # Split by True / Fake
    else:
        mask_true = (y == 1)
        mask_fake = (y == 0)

        plt.hist(
            X_ref[mask_true, feature_index], bins=bins, density=density,
            alpha=alpha_ref, label=f"{labels[0]}: True", color="#022773"
        )
        plt.hist(
            X_ref[mask_fake, feature_index], bins=bins, density=density,
            alpha=alpha_ref, label=f"{labels[0]}: Fake", color="#FF0000"
        )

        if X_cmp is not None:
            if y_cmp is not None:
                mask_true_cmp = (y_cmp == 1)
                mask_fake_cmp = (y_cmp == 0)
                plt.hist(
                    X_cmp[mask_true_cmp, feature_index], bins=bins, density=density,
                    alpha=alpha_cmp, label=f"{labels[1]}: True", linestyle="--", color="C2"
                )
                plt.hist(
                    X_cmp[mask_fake_cmp, feature_index], bins=bins, density=density,
                    alpha=alpha_cmp, label=f"{labels[1]}: Fake", linestyle="--", color="C3"
                )
            else:
                plt.hist(
                    X_cmp[:, feature_index], bins=bins, density=density,
                    alpha=alpha_cmp, label=labels[1], color="C1"
                )
    
    # --- Titles & Labels ---
    if title is None:
        title = f"Distribution of {feature_name}"
    if xlabel is None:
        xlabel = feature_name
    if ylabel is None:
        ylabel = "Density" if density else "Counts"

    # Larger fonts
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    # Larger tick labels
    plt.tick_params(axis='both', labelsize=14)

    plt.grid(alpha=0.25)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_cms_tracker_background(
    true_hits: np.ndarray,
    isRecHit_true: np.ndarray,
    r_idx: int,
    z_idx: int,
    gridsize: int = 500,
    cmap: str = "inferno",
    save_path: str | None = None,
) -> None:
    """
    Plot a hexbin of CMS tracker hits in the r-z plane.

    Parameters
    ----------
    true_hits : np.ndarray
        Array of shape (n_tracks, n_hits, n_features)
    isRecHit_true : np.ndarray
        Boolean mask for valid hits, shape (n_tracks, n_hits)
    r_idx : int
        Index of the r coordinate
    z_idx : int
        Index of the z coordinate
    gridsize : int, optional
        Hexbin resolution
    cmap : str, optional
        Colormap
    save_path : str or None, optional
        If given, save figure to this path
    """
    
    # Flatten hits and apply mask
    r_hits = true_hits[..., r_idx][isRecHit_true]
    z_hits = true_hits[..., z_idx][isRecHit_true]

    # Remove zeros (padding)
    valid = r_hits > 1e-3
    r_hits = r_hits[valid]
    z_hits = z_hits[valid]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    hb = ax.hexbin(
        z_hits,
        r_hits,
        gridsize=gridsize,
        bins="log",
        cmap=cmap,
        mincnt=1
    )

    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("log10(N hits)")

    ax.set_xlabel("z [cm]")
    ax.set_ylabel("r [cm]")
    ax.set_title("CMS tracker hits")
    ax.grid(alpha=0.2)
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close()