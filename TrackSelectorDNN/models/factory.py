'''
Module to build network models from config objects.
'''

from TrackSelectorDNN.configs.schema import NetAConfig, NetBConfig, ModelConfig
from TrackSelectorDNN.models.netA import *
from TrackSelectorDNN.models.netB import *
from TrackSelectorDNN.models.registry import get_activation, get_pooling

# -------------------------------------------------------------------------------------

def build_netA(cfg: NetAConfig, input_dim: int, latent_dim: int) -> nn.Module:
    """
    Returns a netA instance.
    Selection depends on config.ModelConfig.netA.kind
    """
    if cfg.kind == "mlp":
        activation = get_activation(cfg.activation)
        return NetA(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            latent_dim=latent_dim,
            hidden_layers=cfg.hidden_layers,
            use_batchnorm=cfg.batchnorm,
            activation=activation,
        )

    elif cfg.kind == "transformer":
        return NetATransformer(
            input_dim=input_dim,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            latent_dim=latent_dim,
            dropout=cfg.dropout,
        )

    else:
        raise ValueError(f"Unknown NetA kind: {cfg.kind}")

# -------------------------------------------------------------------------------------

def build_pooling(cfg):
    """
    Build a pooling module from a config object.
    """
    return get_pooling(cfg.pooling_type, cfg.latent_dim)

# -------------------------------------------------------------------------------------

def build_netB(cfg: NetBConfig, latent_dim: int, track_feat_dim: int) -> nn.Module:
    activation = get_activation(cfg.activation)
    if cfg.kind == "mlp":
        return NetB(
            latent_dim=latent_dim,
            track_feat_dim=track_feat_dim,
            hidden_dim=cfg.hidden_dim,
            hidden_layers=cfg.hidden_layers,
            use_batchnorm=cfg.batchnorm,
            activation=activation,
        )
    elif cfg.kind == "track_only":
        return NetBTrackOnly(
            track_feat_dim=track_feat_dim,
            hidden_dim=cfg.hidden_dim,
            hidden_layers=cfg.hidden_layers,
            use_batchnorm=cfg.batchnorm,
            activation=activation,
        )
    else:
        raise ValueError(f"Unknown NetB kind: {cfg.kind}")

# -------------------------------------------------------------------------------------

def build_model(cfg: ModelConfig) -> nn.Module:
    """
    Build the correct torch.nn.Module from a validated ModelConfig.
    """
    from TrackSelectorDNN.models.track_classifier import TrackClassifier, TrackOnlyClassifier
    if cfg.type == "track_classifier":
        return TrackClassifier(cfg)

    elif cfg.type == "track_only":
        return TrackOnlyClassifier(cfg)

    else:
        raise ValueError(f"Unknown model type: {cfg.type}")

# ------------------------------------------------------------------------------------- 
