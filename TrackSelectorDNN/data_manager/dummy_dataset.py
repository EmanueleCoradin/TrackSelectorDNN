'''
Module defining a dummy dataset for testing track selection DNNs. 
Currently retained for reference but no longer used.
'''
import numpy as np
import torch
from torch.utils.data import Dataset

class DummyTrackDataset(Dataset):
    '''
    A dummy dataset generating synthetic track and hit features for testing.
    '''
    def __init__(self, 
                 n_tracks=1, 
                 hit_input_dim=1, 
                 track_feat_dim=1, 
                 max_hits=1, 
                 seed=42,
                 save_path=None,
                 load_path=None):

        super().__init__()

        # --------------------
        # LOAD FROM FILE
        # --------------------
        if load_path is not None:
            data = torch.load(load_path)
            self.all_hits = data["all_hits"]
            self.all_batch_idx = data["all_batch_idx"]
            self.all_tracks = data["all_tracks"]
            self.all_labels = data["all_labels"]
            return

        # --------------------
        # GENERATE SYNTHETIC
        # --------------------
        rng = np.random.default_rng(seed)
        self.hit_input_dim = hit_input_dim
        self.track_feat_dim = track_feat_dim

        hits = []
        tracks = []
        labels = []
        batch_idx = []

        for track_id in range(n_tracks):
            n_hits = rng.integers(5, max_hits)
            label = rng.integers(0, 2)
            hit_spread   = 0.5 if label == 1 else 2.0
            track_spread = 0.7 if label == 1 else 2.5
            
            center_hits  = rng.normal(0, 1, size=(hit_input_dim,))
            center_feats = rng.normal(0, 1, size=(track_feat_dim,))
            track_hits  = center_hits + rng.normal(0, hit_spread, size=(n_hits, hit_input_dim)).astype(np.float32)
            track_feats = center_feats + rng.normal(0, 0.2, size=(track_feat_dim,))

            hits.append(track_hits)
            tracks.append(track_feats.astype(np.float32))
            labels.append(label)
            batch_idx.append(np.full(n_hits, track_id, dtype=np.int64))

        self.all_hits = torch.tensor(np.concatenate(hits, axis=0), dtype=torch.float32)
        self.all_batch_idx = torch.tensor(np.concatenate(batch_idx, axis=0))
        self.all_tracks = torch.tensor(np.stack(tracks, axis=0), dtype=torch.float32)
        self.all_labels = torch.tensor(labels, dtype=torch.float32)

        # --------------------
        # SAVE TO FILE
        # --------------------
        if save_path is not None:
            torch.save({
                "all_hits": self.all_hits,
                "all_batch_idx": self.all_batch_idx,
                "all_tracks": self.all_tracks,
                "all_labels": self.all_labels,
            }, save_path)

    def __len__(self):
        return len(self.all_tracks)

    def __getitem__(self, idx):
        mask = self.all_batch_idx == idx
        return (
            self.all_hits[mask],
            self.all_tracks[idx],
            idx,
            self.all_labels[idx]
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
