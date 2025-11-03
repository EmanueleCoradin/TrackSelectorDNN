from torch.utils.data import Dataset, DataLoader
import torch

class TrackDatasetFromFile(Dataset):
    """Loads pre-saved hits and track features from a .pt file"""
    def __init__(self, path):
        data = torch.load(path)
        self.all_hits = data["all_hits"]
        self.all_batch_idx = data["all_batch_idx"]
        self.all_tracks = data["all_tracks"]
        self.all_labels = data["all_labels"]
        self.n_tracks = self.all_tracks.shape[0]

    def __len__(self):
        return self.n_tracks

    def __getitem__(self, idx):
        mask = self.all_batch_idx == idx
        return (
            self.all_hits[mask],
            self.all_tracks[idx],
            torch.tensor(idx),
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
    return hit_features, track_features, batch_indices, labels
