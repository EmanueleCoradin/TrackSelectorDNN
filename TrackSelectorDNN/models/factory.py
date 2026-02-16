'''
Module to build network models from config objects.
'''

import torch.nn as nn

from TrackSelectorDNN.configs.schema import  ModelConfig, NetAConfig, NetBConfig, MLPConfig, TrackGNNConfig
from TrackSelectorDNN.models.netA import NetA, NetATransformer
from TrackSelectorDNN.models.netB import NetB, NetBTrackOnly
from TrackSelectorDNN.models.registry import get_activation, get_pooling

# -------------------------------------------------------------------------------------
def build_mlp(input_dim, cfg:MLPConfig):
    layers = []
    dim = input_dim

    act = get_activation(cfg.activation)
    layers.append(nn.Linear(dim, cfg.hidden_dim))

    if cfg.batchnorm:
        layers.append(nn.LayerNorm(cfg.hidden_dim))

    layers.append(act())

    for _ in range(cfg.hidden_layers-1):
        layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        if cfg.batchnorm:
            layers.append(nn.LayerNorm(cfg.hidden_dim))
        layers.append(act())

    return nn.Sequential(*layers)


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
            use_layernorm=cfg.use_layernorm,
            activation=activation,
            dropout_rate=cfg.dropout_rate
        )

    if cfg.kind == "transformer":
        return NetATransformer(
            input_dim=input_dim,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            latent_dim=latent_dim,
            dropout=cfg.dropout,
        )

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
            use_layernorm=cfg.use_layernorm,
            activation=activation,
            dropout_rate=cfg.dropout_rate
        )
    if cfg.kind == "track_only":
        return NetBTrackOnly(
            track_feat_dim=track_feat_dim,
            hidden_dim=cfg.hidden_dim,
            hidden_layers=cfg.hidden_layers,
            use_batchnorm=cfg.batchnorm,
            activation=activation,
        )
    raise ValueError(f"Unknown NetB kind: {cfg.kind}")

# -------------------------------------------------------------------------------------

def build_model(cfg: ModelConfig) -> nn.Module:
    """
    Build the correct torch.nn.Module from a validated ModelConfig.
    """
    from TrackSelectorDNN.models.track_classifier import PreselectorClassifier, TrackClassifier, TrackOnlyClassifier
    from TrackSelectorDNN.models.gnn_classifier import TrackGNN

    if cfg.type == "track_classifier":
        return TrackClassifier(cfg)
    if cfg.type == "track_only":
        return TrackOnlyClassifier(cfg)
    if cfg.type == "preselector":
        return PreselectorClassifier(cfg)
    if cfg.type == "gnn":
        return TrackGNN(cfg)

    raise ValueError(f"Unknown model type: {cfg.type}")

# ------------------------------------------------------------------------------------- 
