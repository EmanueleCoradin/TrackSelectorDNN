import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Attention-weighted pooling
# -----------------------------
class SoftmaxPooling(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1)
        )

    def forward(self, h, mask=None, batch_indices=None):
        """
        h: (N_hits_total, latent_dim)
        batch_indices: (N_hits_total,) tensor of integers labeling track membership
        mask: optional boolean mask for padded hits

        returns: (N_tracks, latent_dim)
        """
        scores = self.score_net(h)  # (N_hits_total, 1)
        scores = scores.squeeze(-1)

        # softmax within each track
        if batch_indices is None:
            # assume one track
            weights = F.softmax(scores, dim=0)
            pooled = torch.sum(weights.unsqueeze(-1) * h, dim=0, keepdim=True)
        else:
            # batch-wise pooling
            num_tracks = batch_indices.max().item() + 1
            pooled = []
            for i in range(num_tracks):
                mask_i = batch_indices == i
                s_i = scores[mask_i]
                h_i = h[mask_i]
                w_i = F.softmax(s_i, dim=0).unsqueeze(-1)
                pooled_i = torch.sum(w_i * h_i, dim=0, keepdim=True)
                pooled.append(pooled_i)
            pooled = torch.cat(pooled, dim=0)

        return pooled  # (N_tracks, latent_dim)

# -----------------------------
# Sum Pooling
# -----------------------------
class SumPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h, batch_indices=None):
        """
        h: (N_hits_total, latent_dim)
        batch_indices: (N_hits_total,) integers mapping hits to track IDs
        """
        if batch_indices is None:
            return h.sum(dim=0, keepdim=True)

        num_tracks = batch_indices.max().item() + 1
        pooled = torch.zeros(num_tracks, h.size(1), device=h.device)
        pooled.index_add_(0, batch_indices, h)
        return pooled


# -----------------------------
# Mean Pooling
# -----------------------------
class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h, batch_indices=None):
        """
        h: (N_hits_total, latent_dim)
        batch_indices: (N_hits_total,)
        """
        if batch_indices is None:
            return h.mean(dim=0, keepdim=True)

        num_tracks = batch_indices.max().item() + 1
        pooled = torch.zeros(num_tracks, h.size(1), device=h.device)
        counts = torch.zeros(num_tracks, device=h.device)

        pooled.index_add_(0, batch_indices, h)
        counts.index_add_(0, batch_indices, torch.ones_like(batch_indices, dtype=torch.float))
        counts = counts.clamp(min=1.0).unsqueeze(-1)

        return pooled / counts
