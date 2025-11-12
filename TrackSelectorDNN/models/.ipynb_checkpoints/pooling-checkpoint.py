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

    def forward(self, h, mask=None):
        """
        h: (N_tracks, N_hits, latent_dim)
        mask: (N_tracks, N_hits) boolean, True for real hits, False for padding

        returns: (N_tracks, latent_dim)
        """
        scores = self.score_net(h).squeeze(-1)  # (N_tracks, N_hits)

        if mask is not None:
            # Mask padded hits by large negative value to zero out in softmax
            scores = scores.masked_fill(~mask, float('-inf'))

        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (N_tracks, N_hits, 1)
        pooled = (weights * h).sum(dim=1)                # (N_tracks, latent_dim)
        return pooled

# -----------------------------
# Sum Pooling
# -----------------------------
class SumPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h, mask=None):
        """
        h: (N_tracks, max_hits, latent_dim)
        mask: (N_tracks, max_hits) boolean, True for real hits, False for padding
        """
        if mask is not None:
            h = torch.where(mask.unsqueeze(-1), h, torch.zeros_like(h)) 
        pooled = h.sum(dim=1)
        return pooled


# -----------------------------
# Mean Pooling
# -----------------------------
class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h, mask=None):
        """
        h: (N_tracks, N_hits, latent_dim)
        mask: (N_tracks, N_hits) boolean, True for real hits, False for padding

        returns: (N_tracks, latent_dim)
        """
        if mask is not None:
            mask = mask.unsqueeze(-1)
            h = torch.where(mask, h, torch.zeros_like(h)) 
            counts = mask.float().sum(dim=1).clamp(min=1e-6)  # avoid division by zero
        else:
            counts = h.size(1)

        pooled = h.sum(dim=1) / counts
        return pooled