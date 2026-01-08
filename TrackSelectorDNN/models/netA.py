"""
Module defining the NetA model, processing per-hit features.
"""
import torch
import torch.nn as nn

# -----------------------------
# Per-hit encoder: NetA
# -----------------------------

class NetA(nn.Module):
    """
    Per-hit DNN with variable number of hidden layers.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, hidden_layers,
                 use_batchnorm, activation):
        """
        Per-hit DNN with variable number of hidden layers.

        Args:
            input_dim (int):        Number of input features per hit.
            hidden_dim (int):       Width of each hidden layer.
            latent_dim (int):       Output size per hit.
            hidden_layers (int):    Number of hidden layers between input and output.
            use_batchnorm (bool):   Whether to use BatchNorm1d after each layer.
            activation (nn.Module): Activation function class (e.g. nn.ReLU, nn.SiLU).
        """
        super().__init__()

        layers = []
        act = activation()

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act)

        # Hidden layers 
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())

        # Output layer 
        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (N_hits_total, input_dim)
        return self.mlp(x)  # (N_hits_total, latent_dim)

# ------------------------------------------------------------------------------

#TODO: Implement the model
class NetATransformer(nn.Module):
    """
    Transformer implementation of the hit features processing.
    """
    def __init__(self,input_dim, d_model, n_heads, n_layers,
                 latent_dim, dropout):
        super().__init__()

    def forward(self, x):
        return x

# -------------------------------------------------------------------------------------
