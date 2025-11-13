import torch
import torch.nn as nn

class NetB(nn.Module):
    def __init__(self, latent_dim, track_feat_dim,
                 hidden_dim, hidden_layers,
                 use_batchnorm, activation):
        """
        Track-level NN.

        Args:
            latent_dim (int):       Dimension of pooled hit embedding from NetA.
            track_feat_dim (int):   Dimension of per-track features.
            hidden_dim (int):       Width of each hidden layer.
            hidden_layers (int):    Number of hidden layers before the output.
            use_batchnorm (bool):   Whether to include BatchNorm1d.
            activation (nn.Module): Activation function class (e.g., nn.ReLU, nn.SiLU).
        """
        super().__init__()

        layers = []
        act = activation()

        input_dim = latent_dim + track_feat_dim

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act)

        # Hidden layers (if hidden_layers > 1)
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())

        # Output layer (1 neuron for binary classification)
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, pooled_hit, track_features):
        """
        pooled_hit: (N_tracks, latent_dim)
        track_features: (N_tracks, track_feat_dim)
        """
        x = torch.cat([pooled_hit, track_features], dim=-1)
        logits = self.mlp(x)
        return torch.sigmoid(logits).squeeze(-1)
