import torch
import torch.nn as nn

class NetB(nn.Module):
    def __init__(self, latent_dim, track_feat_dim,
                 hidden_dim, hidden_layers,
                 use_batchnorm, activation):
        """
        Track-level NN with latent input from hit features.

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

        input_dim = latent_dim + track_feat_dim

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation())

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
        return logits.squeeze(-1)

class NetBTrackOnly(nn.Module):
    def __init__(self,
                 track_feat_dim,
                 hidden_dim,
                 hidden_layers,
                 use_batchnorm,
                 activation):
        """
        Track-level NN with input features just from tracks.
        If hidden_layers == 0 a pure linear model is returned.
        Args:
            track_feat_dim (int):   Dimension of per-track features.
            hidden_dim (int):       Width of each hidden layer.
            hidden_layers (int):    Number of hidden layers before the output.
            use_batchnorm (bool):   Whether to include BatchNorm1d.
            activation (nn.Module): Activation function class (e.g., nn.ReLU, nn.SiLU).
        """    
        super().__init__()

        layers = []
        
        in_dim = track_feat_dim

        if hidden_layers == 0:
            self.mlp = nn.Linear(input_dim, 1)
            return


        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation())
        
        for _ in range(hidden_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())

        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, track_features):
        return self.mlp(track_features).squeeze(-1)
