import torch
import torch.nn as nn
from TrackSelectorDNN.models.netA    import NetA
from TrackSelectorDNN.models.netB    import NetB, NetBTrackOnly
from TrackSelectorDNN.models.registry import get_activation, get_pooling

class TrackClassifier(nn.Module):
    def __init__(self,
                 hit_input_dim,
                 track_feat_dim,
                 latent_dim,
                 pooling_type,
                 # --- NetA parameters ---
                 netA_hidden_dim,
                 netA_hidden_layers,
                 netA_batchnorm,
                 netA_activation,
                 # --- NetB parameters ---
                 netB_hidden_dim,
                 netB_hidden_layers,
                 netB_batchnorm,
                 netB_activation):
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

    def forward(self, hit_features, track_features, mask=None):
        """
        Args:
            hit_features:   (N_tracks, N_hits_total, hit_input_dim)
            track_features: (N_tracks, track_feat_dim)
            mask:  (N_tracks, N_hits_total) boolean to drop padding hits
        """
        N_tracks, N_hits, _ = hit_features.shape
    
        # Pass hits through NetA
        h = self.netA(hit_features.view(-1, hit_features.size(-1)))  # flatten hits
        h = h.view(N_tracks, N_hits, -1)                             # restore track dim
    
        # Pooling
        pooled = self.pool(h, mask)  # (N_tracks, latent_dim)
    
        # NetB
        out = self.netB(pooled, track_features)  # (N_tracks,)
        return out

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
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

    def forward(self, hit_features, track_features):

        # 1. Preprocessing
        hit_features  = self.pre_hit(hit_features)      # (N,T,F)
        track_features = self.pre_track(track_features) # (N,F)
        # Sanitize nan for safe execution
        track_features = torch.nan_to_num(track_features, nan=0.0)
        
        # 2. Compute mask from NaN locations
        mask = ~torch.isnan(hit_features).any(dim=-1)   # (N,T)
        hit_features = torch.nan_to_num(hit_features, nan=0.0)
       
        # 3. Reuse the original trained forward
        logits = self.base(hit_features, track_features, mask)

        #4. Compute probabilities applying the sigmoid function
        probs = torch.sigmoid(logits)
        return probs

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class TrackOnlyClassifier(nn.Module):
    def __init__(self,
                 track_feat_dim,
                 hidden_dim,
                 hidden_layers,
                 use_batchnorm,
                 activation)
        """
        Lightweight classifier processing just track input features.

        Args:
            track_feat_dim (int):   Dimension of per-track features.
            hidden_dim (int):       Width of each hidden layer.
            hidden_layers (int):    Number of hidden layers before the output.
            use_batchnorm (bool):   Whether to include BatchNorm1d.
            activation (nn.Module): Activation function.
        """
        super().__init__()
        
        # Load the optional parameters
        act = get_activation(activation)
        
        # Build NetB
        self.netB = NetBTrackOnly(
            track_feat_dim,
            hidden_dim,
            hidden_layers,
            use_batchnorm,
            activation
        )

    def forward(self, track_features):
        return self.netB(track_features)


