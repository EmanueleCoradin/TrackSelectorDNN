import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DummyTrackDataset(Dataset):
    """
    Generates synthetic tracks with a variable number of hits.
    Each track:
        - has 5–15 hits, each with `hit_input_dim` features
        - has `track_feat_dim` track-level features
        - has a binary label (1 = real, 0 = fake)
    """
    def __init__(self, n_tracks=2000, hit_input_dim=8, track_feat_dim=4, max_hits=15, seed=42):
        super().__init__()
        rng = np.random.default_rng(seed)

        self.hit_input_dim = hit_input_dim
        self.track_feat_dim = track_feat_dim
        self.tracks = []
        self.hits = []
        self.batch_idx = []
        self.labels = []

        for track_id in range(n_tracks):
            n_hits = rng.integers(5, max_hits)
            
            # --- simulate a "true" correlation between hits and label ---
            # for true tracks, hits are more consistent (less noise)
            label = rng.integers(0, 2)
            hit_spread = 0.5 if label == 1 else 2.0
            
            # random "true" trajectory center in feature space
            center = rng.normal(0, 1, size=(hit_input_dim,))
            hits = center + rng.normal(0, hit_spread, size=(n_hits, hit_input_dim)).astype(np.float32)

            # per-track features loosely correlated with the label
            track_feats = np.concatenate([
                center[:track_feat_dim] + rng.normal(0, 0.2, size=(track_feat_dim,)),
            ]).astype(np.float32)

            self.hits.append(hits)
            self.tracks.append(track_feats)
            self.labels.append(label)
            self.batch_idx.append(np.full(n_hits, track_id, dtype=np.int64))

        # Flatten for convenience
        self.all_hits = np.concatenate(self.hits, axis=0)
        self.all_batch_idx = np.concatenate(self.batch_idx, axis=0)
        self.all_tracks = np.stack(self.tracks, axis=0)
        self.all_labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        mask = self.all_batch_idx == idx
        return (
            torch.tensor(self.all_hits[mask], dtype=torch.float32),
            torch.tensor(self.all_tracks[idx], dtype=torch.float32),
            torch.tensor(idx, dtype=torch.long),
            torch.tensor(self.all_labels[idx], dtype=torch.float32)
        )

def collate_fn(batch):
    hits, tracks, batch_ids, labels = zip(*batch)
    hit_features = torch.cat(hits, dim=0)
    track_features = torch.stack(tracks, dim=0)
    batch_indices = torch.cat([
        torch.full((len(h),), i, dtype=torch.long)
        for i, h in enumerate(hits)
    ])
    labels = torch.stack(labels).float()
    return {
        "hit_features": hit_features,
        "track_features": track_features,
        "batch_indices": batch_indices,
        "labels": labels,
    }
