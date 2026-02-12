'''
Module defining TrackClassifier and related classes.
'''

import torch
import torch.nn as nn

from TrackSelectorDNN.data_manager.dataset import FeatureBundle
from TrackSelectorDNN.configs.schema import TrackClassifierConfig, TrackOnlyClassifierConfig
from TrackSelectorDNN.models.factory import build_netA, build_netB, build_pooling

#-------------------------------------------------------------------------------------

class TrackClassifier(nn.Module):
    """
    TrackClassifier combining NetA, pooling, and NetB.
    """
    def __init__(self, cfg: TrackClassifierConfig):
        """
        TrackClassifier combining NetA, pooling, and NetB.

        Args:
             cfg (TrackOnlyClassifierConfig): Validated Pydantic config.
        """
        super().__init__()
        
        # Build NetA
        self.netA = build_netA(cfg.netA, cfg.hit_input_dim, cfg.latent_dim)

        # Build pooling
        self.pool = build_pooling(cfg)

        # Build NetB
        self.netB = build_netB(cfg.netB, cfg.latent_dim, cfg.track_feat_dim)

    def forward(self, hit_features, track_features, mask):
        """
        Args:
            hit_features:   (N_tracks, N_hits_total, hit_input_dim)
            track_features: (N_tracks, track_feat_dim)
            mask:           (N_tracks, N_hits_total) boolean to drop padding hits
                            TorchScript-safe: default empty tensor means "no mask"
        """
        N_tracks, N_hits, _ = hit_features.shape
        
        # Pass hits through NetA
        h = self.netA(hit_features.reshape(-1, hit_features.size(-1)))  # flatten hits
        h = h.view(N_tracks, N_hits, -1)                             # restore track dim
        
        # Pooling
        pooled = self.pool(h, mask)  # (N_tracks, latent_dim)
        
        # NetB
        out = self.netB(pooled, track_features)  # (N_tracks,)
        return out

    def forward_bundle(self, features: FeatureBundle):
        """
        Forward method accepting a FeatureBundle.

        Args:
            features (FeatureBundle): Input features bundle.
        """
        return self.forward(
            hit_features=features.hit_features,
            track_features=features.track_features,
            mask=features.mask
        )

# ------------------------------------------------------------------------------

class TrackClassifierInference(nn.Module):
    """
    Wrapper of TrackClassifier to incorporate the preprocessing and final sigmoid.
    """
    def __init__(self, base_model: TrackClassifier, 
                 pre_hit: nn.Module, pre_track: nn.Module):
        super().__init__()

        # Keep trained modules
        self.base = base_model

        # Preprocessing modules
        self.pre_hit = pre_hit
        self.pre_track = pre_track

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
            
    def forward(self, hit_features, track_features):
         # 1. Preprocessing
        hit_features  = self.pre_hit(hit_features)
        track_features = self.pre_track(track_features)
    
        # 2. Build valid track mask
        valid_mask = ~torch.isnan(track_features).any(dim=-1)
        hit_mask = ~torch.isnan(hit_features).any(dim=-1)
    
        # 3. Replace NaNs with zeros
        track_features = torch.nan_to_num_(track_features)
        hit_features = torch.nan_to_num_(hit_features)
        
        # Forward full batch
        logits = self.base(hit_features, track_features, hit_mask)
    
        probs = torch.sigmoid(logits)
    
        # Zero out invalid tracks
        probs = probs.unsqueeze(-1) * valid_mask.unsqueeze(-1)
    
        return probs.reshape(-1, 1)


    def train(self, mode: bool = True):
        super().train(False)
        return self

# ------------------------------------------------------------------------------

class TrackOnlyClassifier(nn.Module):
    """
    Lightweight classifier processing just track input features.
    """
    def __init__(self, cfg: TrackOnlyClassifierConfig):
        """
        Lightweight classifier processing just track input features.

        Args:
            cfg (TrackOnlyClassifierConfig): Validated Pydantic config.
        """
        super().__init__()

        self.netB = build_netB(cfg.netB, latent_dim=None, track_feat_dim=cfg.track_feat_dim)

    def forward(self, track_features):
        return self.netB(track_features)

    def forward_bundle(self, features: FeatureBundle):
        """
        Forward method accepting a FeatureBundle.

        Args:
            features (FeatureBundle): Input features bundle.
        """
        return self.forward(
            track_features=features.track_features
        )

# ------------------------------------------------------------------------------

class PreselectorClassifier(nn.Module):
    """
    Lightweight classifier processing just track input features.
    """
    def __init__(self, cfg: TrackOnlyClassifierConfig):
        """
        Lightweight classifier processing just track input features.

        Args:
            cfg (TrackOnlyClassifierConfig): Validated Pydantic config.
        """
        super().__init__()

        self.netB = build_netB(cfg.netB, latent_dim=None, track_feat_dim=cfg.preselector_feat_dim)

    def forward(self, preselect_features):
        return self.netB(preselect_features)

    def forward_bundle(self, features: FeatureBundle):
        """
        Forward method accepting a FeatureBundle.

        Args:
            features (FeatureBundle): Input features bundle.
        """
        return self.forward(
            preselect_features=features.preselect_features
        )

# ------------------------------------------------------------------------------
