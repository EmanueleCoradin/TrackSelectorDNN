'''
Module defining datasets for track selection using DNNs.
'''

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset

# -------------------------------------------------------------------------------------

@dataclass
class FeatureBundle:
    """
    Abstract bundle of features in input to the different models.
    """
    hit_features: Optional[torch.Tensor] = None         # (N_hits, hit_input_dim)
    track_features: Optional[torch.Tensor] = None       # (track_feat_dim)
    edge_features: Optional[torch.Tensor] = None        # (N_edges, edge_feat_dim)
    hit_extra_features: Optional[torch.Tensor] = None   # (N_hits, recHit_extra_dim)
    preselect_features: Optional[torch.Tensor] = None   # (F_pre)
    global_features: Optional[torch.Tensor] = None      # (global_feat_dim)
    mask: Optional[torch.Tensor] = None                 # (N_hits)

    def to(self, device: torch.device) -> "FeatureBundle":
        """
        Move all tensors in FeatureBundle to specified device.
        """

        for name, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self

# -------------------------------------------------------------------------------------

class TrackDatasetFromFile(Dataset):
    """
    PyTorch Dataset for loading track-level data from a serialized ``.pt`` file.

    This dataset represents the *maximal* view of the track data: it loads all
    available per-track, per-hit, per-edge, and preselector features, together
    with labels, metadata, and preprocessing statistics. Specialized model
    inputs (e.g. preselector-only or full DNN views) should be implemented as
    lightweight dataset views on top of this class.

    The dataset assumes that all tensors are already preprocessed and aligned
    at save time (padding, masking, normalization).

    Attributes
    ----------
    recHitFeatures : torch.Tensor
        Tensor of shape ``(N_tracks, MAX_HITS, hit_input_dim)`` containing the
        primary features of reconstructed hits associated to each track.

    recoPixelTrackFeatures : torch.Tensor
        Tensor of shape ``(N_tracks, track_feat_dim)`` containing per-track
        features of reconstructed pixel tracks.

    mask : torch.Tensor
        Boolean tensor of shape ``(N_tracks, MAX_HITS)`` indicating which hit
        slots correspond to valid reconstructed hits.

    edgeFeatures : torch.Tensor or None
        Optional tensor of shape ``(N_edges, edge_feat_dim)`` containing features
        associated to edges in a graph-based track representation.

    recHitExtraFeatures : torch.Tensor or None
        Optional tensor of shape ``(N_tracks, MAX_HITS, recHit_extra_dim)``
        containing additional per-hit features not included in
        ``recHitFeatures``.

    preselect_features : torch.Tensor or None
        Optional tensor of shape ``(N_tracks, F_pre)`` containing the flattened
        feature representation used by fast preselector models.

    labels : torch.Tensor or None
        Optional tensor of shape ``(N_tracks,)`` containing labels
        for each track.

    isHighPurity : torch.Tensor or None
        Optional boolean tensor of shape ``(N_tracks,)`` indicating whether a
        reconstructed track is flagged as high-purity.

    recHitBranches : list[str] or None
        Optional list of feature names corresponding to the last dimension of
        ``recHitFeatures``.

    recoPixelTrackBranches : list[str] or None
        Optional list of feature names corresponding to the last dimension of
        ``recoPixelTrackFeatures``.

    preselect_feature_names : list[str] or None
        Optional list of feature names corresponding to ``preselect_features``.

    preselect_is_onehot : list[bool] or None
        Optional boolean mask indicating which preselector features are
        one-hot encoded.

    MAX_HITS : int or None
        Maximum number of reconstructed hits per track used for padding.

    recHit_mean : torch.Tensor or None
        Optional tensor containing per-feature mean values used to normalize
        ``recHitFeatures``.

    recHit_std : torch.Tensor or None
        Optional tensor containing per-feature standard deviation values used
        to normalize ``recHitFeatures``.

    reco_mean : torch.Tensor or None
        Optional tensor containing per-feature mean values used to normalize
        ``recoPixelTrackFeatures``.

    reco_std : torch.Tensor or None
        Optional tensor containing per-feature standard deviation values used
        to normalize ``recoPixelTrackFeatures``.

    edge_mean : torch.Tensor or None
        Optional tensor containing per-feature mean values used to normalize
        ``edgeFeatures``.

    edge_std : torch.Tensor or None
        Optional tensor containing per-feature standard deviation values used
        to normalize ``edgeFeatures``.

    recHitExtra_mean : torch.Tensor or None
        Optional tensor containing per-feature mean values used to normalize
        ``recHitExtraFeatures``.

    recHitExtra_std : torch.Tensor or None
        Optional tensor containing per-feature standard deviation values used
        to normalize ``recHitExtraFeatures``.

    log_vars : list[str]
        List of feature names for which a logarithmic transformation was applied.

    clip_vars : list[str]
        List of feature names that were clipped to a percentile range during
        preprocessing.

    log_recHit_vars : list[str]
        List of reconstructed hit feature names that were log-transformed.

    EPSILON : float
        Small numerical constant used to avoid division by zero or ``log(0)``
        during preprocessing.

    LOW_PERCENTILE : float
        Lower percentile used for feature clipping.

    HIGH_PERCENTILE : float
        Upper percentile used for feature clipping.

    do_log_hit : torch.Tensor or None
        Boolean flag indicating whether logarithmic scaling was applied to
        hit-level features.

    clip_min_hit : torch.Tensor or None
        Minimum clipping values applied to hit-level features.

    clip_max_hit : torch.Tensor or None
        Maximum clipping values applied to hit-level features.

    do_log_track : torch.Tensor or None
        Boolean flag indicating whether logarithmic scaling was applied to
        track-level features.

    clip_min_track : torch.Tensor or None
        Minimum clipping values applied to track-level features.

    clip_max_track : torch.Tensor or None
        Maximum clipping values applied to track-level features.

    Parameters
    ----------
    path : str
        Path to the serialized ``.pt`` file containing the dataset.

    Examples
    --------
    >>> dataset = TrackDatasetFromFile("path/to/data.pt")
    >>> print(dataset.recHitFeatures.shape)
    """

    def __init__(self, path):
        data = torch.load(path)

        # --- Core tensors ---
        self.recHitFeatures = data["recHitFeatures"]                        # (N_tracks, max_hits, hit_input_dim)
        self.recoPixelTrackFeatures = data["recoPixelTrackFeatures"]        # (N_tracks, track_feat_dim)
        self.edgeFeatures = data.get("edgeFeatures", None)                  # (N_edges, edge_feat_dim) or None
        self.recHitExtraFeatures = data.get("recHitExtraFeatures", None)    # (N_tracks, max_hits, recHit_extra_dim) or None
        self.preselect_features = data.get(                                 # (N_tracks, pre_track_feat_dim)
            "recoPixelTrackFeatures_pre", None
        )
        self.mask = data["isRecHit"]                                        # (N_tracks, max_hits) boolean
        self.labels = data.get("labels", None)                              # (N_tracks,)
        self.isHighPurity = data.get("isHighPurity", None)                  # (N_tracks,)
        
        # --- Metadata ---
        self.recHitBranches = data.get("recHitBranches", None)
        self.recoPixelTrackBranches = data.get(
            "recoPixelTrackBranches", None
        )
        self.preselect_feature_names = data.get(
            "recoPixelTrackFeatures_pre_names", None
        )
        self.preselect_is_onehot = data.get(
            "recoPixelTrackFeatures_pre_is_onehot", None
        )
        self.MAX_HITS = data.get("MAX_HITS", None)

        # --- Normalization statistics (optional) ---
        self.recHit_mean = data.get("recHit_mean", None)
        self.recHit_std = data.get("recHit_std", None)
        self.recoPixelTrack_mean = data.get("recoPixelTrack_mean", None)
        self.recoPixelTrack_std = data.get("recoPixelTrack_std", None)
        self.edge_mean = data.get("edge_mean", None)
        self.edge_std = data.get("edge_std", None)
        self.recHitExtra_mean = data.get("recHitExtra_mean", None)
        self.recHitExtra_std = data.get("recHitExtra_std", None)

        # --- Preprocessing parameters (optional) ---
        self.log_vars = data.get("log_vars", [])
        self.clip_vars = data.get("clip_vars", [])
        self.log_recHit_vars = data.get("log_recHit_vars", [])
        self.EPSILON = data.get("EPSILON", 1e-8)
        self.LOW_PERCENTILE = data.get("LOW_PERCENTILE", 0.001)
        self.HIGH_PERCENTILE = data.get("HIGH_PERCENTILE", 0.999)

        self.do_log_hit = data.get("do_log_hit", None)
        self.clip_min_hit = data.get("clip_min_hit", None)
        self.clip_max_hit = data.get("clip_max_hit", None)
        self.do_log_track = data.get("do_log_track", None)
        self.clip_min_track = data.get("clip_min_track", None)
        self.clip_max_track = data.get("clip_max_track", None)

    def __len__(self):
        return self.recHitFeatures.shape[0]

    def __getitem__(self, idx):
        # Return in order: hit_features, track_features, mask, label
        features = FeatureBundle(
            hit_features=self.recHitFeatures[idx],
            track_features=self.recoPixelTrackFeatures[idx],
            edge_features=(
                self.edgeFeatures[idx]
                if self.edgeFeatures is not None
                else None
            ),
            hit_extra_features=(
                self.recHitExtraFeatures[idx]
                if self.recHitExtraFeatures is not None
                else None
            ),
            preselect_features=(
                self.preselect_features[idx]
                if self.preselect_features is not None
                else None
            ),
            mask=self.mask[idx],
        )
        
        item = {"features": features}
        
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        
        return item
    
    def get_feature_names(self):
        return {
            "recHit": self.recHitBranches,
            "recoTrack": self.recoPixelTrackBranches,
            "preselect": self.preselect_feature_names,
        }

    def get_normalization_stats(self):
        return {
            "recHit_mean": self.recHit_mean,
            "recHit_std": self.recHit_std,
            "reco_mean": self.recoPixelTrack_mean,
            "reco_std": self.recoPixelTrack_std,
            "edge_mean": self.edge_mean,
            "edge_std": self.edge_std,
            "recHitExtra_mean": self.recHitExtra_mean,
            "recHitExtra_std": self.recHitExtra_std,
        }
    # sampler = train_ds.get_reweighting_sampler(feature=FEATURE, n_bins=N_BINS, limit=LIMIT, do_class_norm=DO_CLASS_NORM)
    def get_reweighting_sampler(self, feature: str, n_bins: int, do_class_norm: bool, true_power: float = -1.0, fake_power: float = -0.5, normalize_fake: bool = False) -> torch.utils.data.WeightedRandomSampler:
        """
        Get a weighted sampler for reweighting the dataset based on a specified feature.

        Parameters
        ----------
        feature : str
            Name of the feature to base the reweighting on. Should be one of the feature names in the dataset.

        n_bins : int
            Number of bins to use for histogramming the feature distribution.

        do_class_norm : bool
            Whether to apply class normalization to the weights (i.e., normalize weights separately for each class).
        
        true_power : float
            Power to which the inverse of the true histogram is raised when computing weights. Default is -1.0 (i.e., inverse weighting).
        
        fake_power : float
            Power to which the inverse of the fake histogram is raised when computing weights. Default is -0.5 (i.e., inverse sqrt weighting).

        Returns
        -------
        sampler : torch.utils.data.WeightedRandomSampler
            Weighted random sampler that can be used with a DataLoader to sample tracks according to the specified reweighting.
        """
        # Get feature values
        if feature in self.recoPixelTrackBranches:
            feat_idx = self.recoPixelTrackBranches.index(feature)
            feat_values = self.recoPixelTrackFeatures[..., feat_idx]

        elif feature in self.preselect_feature_names:
            feat_idx = self.preselect_feature_names.index(feature)
            feat_values = self.preselect_features[..., feat_idx]
        
        elif feature in self.recHitBranches:
            raise ValueError(f"Feature '{feature}' in recHitBranches not supported.")

        else:
            raise ValueError(f"Feature '{feature}' not found in dataset.")
        
        bin_edges = torch.linspace(feat_values.min(), feat_values.max(), n_bins+1)
        if self.labels is None:
            raise ValueError("Labels are required for reweighting but not found in dataset.")

        labels = self.labels
        mask = labels.bool()
        true_hist, _ = torch.histogram(feat_values[mask], bins=bin_edges, density=False)
        fake_hist, _ = torch.histogram(feat_values[~mask], bins=bin_edges, density=False)
        true_hist = torch.clamp(true_hist, min=1)
        fake_hist = torch.clamp(fake_hist, min=1)
        true_hist = true_hist.pow(true_power)
        fake_hist = fake_hist.pow(fake_power)

        bin_edges[0] = -float("inf")
        bin_edges[-1] = float("inf")

        true_bucket_id = torch.bucketize(feat_values[mask], bin_edges) - 1
        fake_bucket_id = torch.bucketize(feat_values[~mask], bin_edges) - 1
        true_weights = true_hist[true_bucket_id]
        if normalize_fake:
            fake_weights = fake_hist[fake_bucket_id]
        else:
            fake_weights = torch.ones((~mask).sum(), device=feat_values.device)

        if do_class_norm:
            true_target = 0.5
            fake_target = 1 - true_target
            true_weights = true_weights / true_weights.sum() * true_target
            fake_weights = fake_weights / fake_weights.sum() * fake_target
        else:   
            true_weights = true_weights / true_weights.mean()
            fake_weights = fake_weights / fake_weights.mean()

        weights = torch.zeros(feat_values.shape[0], device=feat_values.device)
        weights[mask] = true_weights
        weights[~mask] = fake_weights 
        weights = weights.cpu()  # Ensure weights are on CPU for the sampler
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=feat_values.shape[0], replacement=True)

        return sampler

# -------------------------------------------------------------------------------------

# =============================================================================
# Dataset views
# =============================================================================

class DatasetView(Dataset):
    """
    Abstract dataset view.
    """

    def __init__(self, base: TrackDatasetFromFile):
        self.base = base

    def __len__(self):
        return len(self.base)


class TrackFullDNNView(DatasetView):
    """
    Full hit + track DNN view.
    """

    def __getitem__(self, idx):
        base_item = self.base[idx]
        bf: FeatureBundle = base_item["features"]

        features = FeatureBundle(
            hit_features=bf.hit_features,
            track_features=bf.track_features,
            edge_features=bf.edge_features,
            hit_extra_features=bf.hit_extra_features,
            mask=bf.mask,
        )

        item = {"features": features}

        if "labels" in base_item:
            item["labels"] = base_item["labels"]

        return item


class TrackPreselectorView(DatasetView):
    """
    Preselector-only view.
    """

    def __getitem__(self, idx):
        base_item = self.base[idx]
        bf: FeatureBundle = base_item["features"]

        features = FeatureBundle(
            preselect_features=bf.preselect_features
        )

        item = {"features": features}

        if "labels" in base_item:
            item["labels"] = base_item["labels"]

        return item

    def get_feature_names(self):
        return self.base.preselect_feature_names

    def get_onehot_mask(self):
        return self.base.preselect_is_onehot

class GNNTrackView(DatasetView):
    """
    GNN track view with edge features.
    """

    def __getitem__(self, idx):
        base_item = self.base[idx]
        bf: FeatureBundle = base_item["features"]

        global_feats = None
        if bf.track_features is not None and self.base.recoPixelTrackBranches is not None:
            branches = self.base.recoPixelTrackBranches
            indices = [
                branches.index(name)
                for name in ["hltPixelTrack_pt", "hltPixelTrack_eta", "hltPixelTrack_phi"]
            ]   

            global_feats = bf.track_features[indices]

        features = FeatureBundle(
            track_features=bf.track_features,
            hit_features=bf.hit_features,
            edge_features=bf.edge_features,
            hit_extra_features=bf.hit_extra_features,
            global_features=global_feats,
            mask=bf.mask,
        )

        item = {"features": features}

        if "labels" in base_item:
            item["labels"] = base_item["labels"]

        return item

# =============================================================================

def collate_fn(batch):
    '''
    Collate function for variable-length tracks.
    Each element of batch: dict with keys "features" (FeatureBundle) and optional "labels"
    '''
    features_out = {}
    labels = []

    for key in vars(batch[0]["features"]).keys():
        values = [
            getattr(item["features"], key)
            for item in batch
            if getattr(item["features"], key) is not None
        ]
        if len(values) > 0:
            features_out[key] = torch.stack(values)

    for item in batch:
        if "labels" in item:
            labels.append(item["labels"])

    out = {
        "features": FeatureBundle(**features_out)
    }

    if len(labels) > 0:
        out["labels"] = torch.stack(labels)

    return out


#old implementation for reference
'''
def collate_fn(batch):
    """
    Collate function for fixed-length tracks.
    Each element of batch: (hit_features, track_features, mask, label)
    """
    hit_features, track_features, mask, labels = [], [], [], []

    for item in batch:
        features: FeatureBundle = item["features"]
        hit_features.append(features.hit_features)
        track_features.append(features.track_features)
        mask.append(features.mask)
        if "labels" in item:
            labels.append(item["labels"])

    return {
        "features": FeatureBundle(
            hit_features=torch.stack(hit_features),
            track_features=torch.stack(track_features),
            mask=torch.stack(mask)
        ),
        "labels": torch.stack(labels) if len(labels)!=0 else None
    }
'''
