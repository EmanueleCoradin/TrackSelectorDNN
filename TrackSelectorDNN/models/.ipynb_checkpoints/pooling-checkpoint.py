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

# -----------------------------
# Sum Pooling
# -----------------------------
class SumPoolingInference(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h):
        """
        h: (N_tracks, max_hits, latent_dim)
        NaNs signal padding hits
        returns: (N_tracks, latent_dim)
        """
        mask = ~torch.isnan(h).any(dim=-1)
        h = torch.where(mask.unsqueeze(-1), h, torch.zeros_like(h)) 
        pooled = h.sum(dim=1)
        return pooled


# -----------------------------
# Mean Pooling
# -----------------------------
class MeanPoolingInference(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h, mask=None):
        """
        h: (N_tracks, N_hits, latent_dim)
        NaNs signal padding hits

        returns: (N_tracks, latent_dim)
        """
        
        hit_mask = ~torch.isnan(h).any(dim=-1)           # (N_tracks, N_hits)
        hit_mask_exp = hit_mask.unsqueeze(-1)            # (N_tracks, N_hits, 1)

        # Replace NaNs with zeros so they don't propagate
        h = torch.nan_to_num(h, nan=0.0)

        # Zero out invalid hits explicitly
        h = h * hit_mask_exp

        # Count valid hits per track
        counts = hit_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (N_tracks, 1)

        pooled = h.sum(dim=1) / counts

        return pooled
