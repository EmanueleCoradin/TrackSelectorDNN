import torch
from torch.utils.data import Dataset

class TrackDatasetFromFile(Dataset):
    """
    PyTorch Dataset for loading track data from a pt file.
    It loads preprocessed features for 
    reconstructed hits and pixel tracks, along with optional metadata, labels, 
    and normalization statistics.

    Attributes:
        recHitFeatures (torch.Tensor): Tensor of shape (N_tracks, max_hits, hit_input_dim)
            containing features for each reconstructed hit in the tracks.
        recoPixelTrackFeatures (torch.Tensor): Tensor of shape (N_tracks, track_feat_dim)
            containing features for each reconstructed pixel track.
        mask (torch.Tensor): Boolean tensor of shape (N_tracks, max_hits) indicating valid hits.
        labels (torch.Tensor or None): Optional tensor of shape (N_tracks,) with track labels.
        isHighPurity (torch.Tensor or None): Optional tensor of shape (N_tracks,)
            indicating if tracks are high-purity.
        
        recHitBranches (list or None): Optional list of feature names for recHitFeatures.
        recoPixelTrackBranches (list or None): Optional list of feature names for recoPixelTrackFeatures.
        MAX_HITS (int or None): Optional maximum number of hits per track.
        
        recHit_mean (torch.Tensor or None): Optional mean values for recHitFeatures normalization.
        recHit_std (torch.Tensor or None): Optional standard deviation values for recHitFeatures normalization.
        recoPixelTrack_mean (torch.Tensor or None): Optional mean values for recoPixelTrackFeatures normalization.
        recoPixelTrack_std (torch.Tensor or None): Optional standard deviation values for recoPixelTrackFeatures normalization.
        
        log_vars (list): Optional list of variables to apply logarithmic transformation.
        clip_vars (list): Optional list of variables to clip to a specified percentile range.
        log_recHit_vars (list): Optional list of recHit variables to apply logarithmic transformation.
        EPSILON (float): Small constant to avoid division by zero or log(0) errors.
        LOW_PERCENTILE (float): Lower percentile for clipping variables.
        HIGH_PERCENTILE (float): Upper percentile for clipping variables.
    do_log: do_log,
    clip_min: clip_min,
    climp_max: clip_max

    Args:
        path (str): Path to the serialized .pt file containing the dataset.
    
    Example:
        dataset = TrackDatasetFromFile("path/to/data.pt")
        print(dataset.recHitFeatures.shape)
    """
    def __init__(self, path):
        data = torch.load(path)
        self.recHitFeatures = data["recHitFeatures"]           # (N_tracks, max_hits, hit_input_dim)
        self.recoPixelTrackFeatures = data["recoPixelTrackFeatures"]  # (N_tracks, track_feat_dim)
        self.mask = data["isRecHit"]                           # (N_tracks, max_hits) boolean
        self.labels = data["labels"] if "labels" in data else None # (N_tracks,)
        self.isHighPurity = data["isHighPurity"] if "isHighPurity" in data else None # (N_tracks,)
        
        # --- Metadata ---
        self.recHitBranches = data.get("recHitBranches", None)
        self.recoPixelTrackBranches = data.get("recoPixelTrackBranches", None)
        self.MAX_HITS = data.get("MAX_HITS", None)

        # --- Normalization statistics (optional) ---
        self.recHit_mean = data.get("recHit_mean", None)
        self.recHit_std = data.get("recHit_std", None)
        self.recoPixelTrack_mean = data.get("recoPixelTrack_mean", None)
        self.recoPixelTrack_std = data.get("recoPixelTrack_std", None)

        # --- Preprocessing parameters (optional) ---
        self.log_vars = data.get("log_vars", [])
        self.clip_vars = data.get("clip_vars", [])
        self.log_recHit_vars = data.get("log_recHit_vars", [])
        self.EPSILON = data.get("EPSILON", 1e-8)
        self.LOW_PERCENTILE = data.get("LOW_PERCENTILE", 0.001)
        self.HIGH_PERCENTILE = data.get("HIGH_PERCENTILE", 0.999)

        self.do_log_hit = data.get("do_log_hit", None),
        self.clip_min_hit = data.get("clip_min_hit", None),
        self.clip_max_hit = data.get("clip_max_hit", None),
        self.do_log_track = data.get("do_log_track", None),
        self.clip_min_track = data.get("clip_min_track", None),
        self.clip_max_track = data.get("clip_max_track", None),
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return in order: hit_features, track_features, mask, label
        item = {
            "hit_features": self.recHitFeatures[idx],
            "track_features": self.recoPixelTrackFeatures[idx],
            "mask": self.mask[idx]
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item
    
    def get_feature_names(self):
        return {
            "recHit": self.recHitBranches,
            "recoTrack": self.recoPixelTrackBranches
        }

    def get_normalization_stats(self):
        return {
            "recHit_mean": self.recHit_mean,
            "recHit_std": self.recHit_std,
            "reco_mean": self.recoPixelTrack_mean,
            "reco_std": self.recoPixelTrack_std
        }

def collate_fn(batch):
    """
    Collate function for fixed-length tracks.
    Each element of batch: (hit_features, track_features, mask, label)
    """
    hit_features, track_features, mask, labels = [], [], [], []

    for item in batch:
        hit_features.append(item["hit_features"])# (B, max_hits, hit_input_dim)
        track_features.append(item["track_features"])# (B, track_feat_dim)
        mask.append(item["mask"])# (B, max_hits)
        labels.append(item.get("labels", torch.tensor(0.0)))  # dummy if missing; (B,)

    return {
        "hit_features": torch.stack(hit_features),
        "track_features": torch.stack(track_features),
        "mask": torch.stack(mask),
        "labels": torch.stack(labels).float()
    }

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

class TrackPreselectorDatasetFromFile(Dataset):
    """
    Dataset for fast preselector models.
    """

    def __init__(self, path):
        data = torch.load(path, map_location="cpu")

        # --- Required ---
        self.X = data["recoPixelTrackFeatures_pre"]  # (N, F_pre)
        self.labels = data.get("labels", None)

        # --- Optional metadata ---
        self.feature_names = data.get(
            "recoPixelTrackFeatures_pre_names", None
        )
        self.is_onehot = data.get(
            "recoPixelTrackFeatures_pre_is_onehot", None
        )
        self.isHighPurity = data.get("isHighPurity", None)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        item = {
            "features": self.X[idx]
        }

        if self.labels is not None:
            item["labels"] = self.labels[idx]

        if self.isHighPurity is not None:
            item["isHighPurity"] = self.isHighPurity[idx]

        return item

    def get_feature_names(self):
        return self.feature_names

    def get_onehot_mask(self):
        """
        Boolean mask of one-hot encoded features.
        """
        return self.is_onehot

def preselector_collate_fn(batch):
    X = torch.stack([b["features"] for b in batch])
    y = torch.stack([b["labels"] for b in batch]) if "labels" in batch[0] else None

    out = {"features": X}
    if y is not None:
        out["labels"] = y.float()
    return out
