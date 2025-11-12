from torch.utils.data import Dataset, DataLoader
import torch

class TrackDatasetFromFile(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.recHitFeatures = data["recHitFeatures"]           # (N_tracks, max_hits, hit_input_dim)
        self.recoPixelTrackFeatures = data["recoPixelTrackFeatures"]  # (N_tracks, track_feat_dim)
        self.mask = data["isRecHit"]                           # (N_tracks, max_hits) boolean
        self.labels = data["labels"] if "labels" in data else None # (N_tracks,)

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
        self.clip_recHit_vars = data.get("clip_recHit_vars", [])
        self.EPSILON = data.get("EPSILON", 1e-8)
        self.LOW_PERCENTILE = data.get("LOW_PERCENTILE", 0.001)
        self.HIGH_PERCENTILE = data.get("HIGH_PERCENTILE", 0.999)
        
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