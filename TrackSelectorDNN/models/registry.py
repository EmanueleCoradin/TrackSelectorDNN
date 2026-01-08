"""
Module for registries of activations and pooling layers.
"""

import torch.nn as nn
from TrackSelectorDNN.models.pooling import (
    SoftmaxPooling, SumPooling, MeanPooling,
    SumPoolingInference, MeanPoolingInference
)

# ---- Activation Registry ----
ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
}

def get_activation(name: str):
    """Return activation class (not instance) given a string key."""
    if name is None:
        return nn.SiLU
    act = ACTIVATIONS.get(name.lower())
    if act is None:
        raise ValueError(f"Unknown activation: {name}. Valid: {list(ACTIVATIONS.keys())}")
    return act


# ---- Pooling Registry ----
POOLING_TYPES = {
    "softmax":   lambda latent_dim: SoftmaxPooling(latent_dim),
    "sum":       lambda latent_dim: SumPooling(),
    "mean":      lambda latent_dim: MeanPooling(),
    "sum-inference":       lambda latent_dim: SumPoolingInference(),
    "mean-inference":      lambda latent_dim: MeanPoolingInference(),
}

def get_pooling(name: str, latent_dim: int):
    """Return a pooling module instance given its name."""
    if name not in POOLING_TYPES:
        raise ValueError(f"Unknown pooling type: {name}. Valid: {list(POOLING_TYPES.keys())}")
    return POOLING_TYPES[name](latent_dim)
