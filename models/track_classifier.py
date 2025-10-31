import torch
import torch.nn as nn
import torch.nn.functional as F
from .netA    import NetA 
from .netB    import NetB
from .registry import get_activation, get_pooling

class TrackClassifier(nn.Module):
    def __init__(self,
                 hit_input_dim,
                 track_feat_dim,
                 latent_dim=16,
                 pooling_type="softmax",
                 # --- NetA parameters ---
                 netA_hidden_dim=32,
                 netA_hidden_layers=2,
                 netA_batchnorm=True,
                 netA_activation="silu",
                 # --- NetB parameters ---
                 netB_hidden_dim=64,
                 netB_hidden_layers=2,
                 netB_batchnorm=True,
                 netB_activation="silu"):
        """
        TrackClassifier combining NetA, pooling, and NetB.

        Args:
            hit_input_dim (int):   Input dimension per hit.
            track_feat_dim (int):  Per-track feature dimension.
            latent_dim (int):      Latent embedding size.
            pooling_type (str):    One among POOLING_TYPES.
            netA_*:                Architecture parameters for NetA.
            netB_*:                Architecture parameters for NetB.
        """
        super().__init__()
        
        # Load the optional parameters
        actA = get_activation(netA_activation)
        actB = get_activation(netB_activation)
        self.pool = get_pooling(pooling_type, latent_dim)
        
        # Build NetA
        self.netA = NetA(
            input_dim=hit_input_dim,
            hidden_dim=netA_hidden_dim,
            latent_dim=latent_dim,
            hidden_layers=netA_hidden_layers,
            use_batchnorm=netA_batchnorm,
            activation=actA
        )

        # Build NetB
        self.netB = NetB(
            latent_dim=latent_dim,
            track_feat_dim=track_feat_dim,
            hidden_dim=netB_hidden_dim,
            hidden_layers=netB_hidden_layers,
            use_batchnorm=netB_batchnorm,
            activation=actB
        )

    def forward(self, hit_features, track_features, batch_indices):
        """
        Args:
            hit_features:   (N_hits_total, hit_input_dim)
            track_features: (N_tracks, track_feat_dim)
            batch_indices:  (N_hits_total,) integers mapping each hit to its track
        """
        h = self.netA(hit_features)                 # (N_hits_total, latent_dim)
        pooled = self.pool(h, batch_indices=batch_indices)  # (N_tracks, latent_dim)
        out = self.netB(pooled, track_features)     # (N_tracks,)
        return out
