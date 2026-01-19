"""
This module implements a Graph Neural Network (GNN) classifier for track selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from TrackSelectorDNN.data_manager.dataset import FeatureBundle
from TrackSelectorDNN.configs.schema import TrackGNNConfig
from TrackSelectorDNN.models.factory import build_mlp

class TrackGNN(nn.Module):
    """
    Graph Neural Network (GNN) for track classification.
    The model assumes a 1D chain graph where edge i connects node i and i+1.
    """
    def __init__(
        self,
        cfg: TrackGNNConfig,
    ):

        super().__init__()

        self.node_update = build_mlp(
            input_dim=cfg.node_dim + 2 * cfg.edge_dim + cfg.global_dim,
            cfg=cfg.node_mlp,
        )

        self.edge_update = build_mlp(
            input_dim=cfg.edge_dim + 2 * cfg.node_mlp.hidden_dim + cfg.global_dim,
            cfg=cfg.edge_mlp,
        )

        self.edge_aggregation = BatchedSoftEdgeAggregator (
            input_dim=cfg.edge_mlp.hidden_dim,
            temperature=cfg.temperature,
        )

        self.global_update = build_mlp(
            input_dim=cfg.edge_mlp.hidden_dim,
            cfg=cfg.global_mlp,
        )

        self.global_classifier = nn.Sequential(
            build_mlp(
                input_dim=cfg.global_mlp.hidden_dim,
                cfg=cfg.global_classifier
            ),
            nn.Linear(cfg.global_classifier.hidden_dim, 1)  # single output
        )

    def forward(
        self,
        global_features,  # (B, G)
        node_features,    # (B, N, node_dim)
        edge_features,    # (B, N-1, edge_dim)
        node_mask,        # (B, N)
    ):
        B, N, _ = node_features.shape
        edge_mask = node_mask[:, :-1] & node_mask[:, 1:]

        # ----- Build forward/backward edges -----
        edge_forward = torch.zeros_like(node_features[..., :edge_features.size(-1)])
        edge_backward = torch.zeros_like(edge_forward)

        edge_forward[:, :-1] = edge_features
        edge_backward[:, 1:] = edge_features
        
        # Mask edges connected to invalid nodes
        edge_forward[:, :-1] *= edge_mask.unsqueeze(-1)
        edge_backward[:, 1:] *= edge_mask.unsqueeze(-1)

        # ----- Node update -----
        global_node = global_features.unsqueeze(1).repeat(1, N, 1)

        node_input = torch.cat(
            [node_features, edge_backward, edge_forward, global_node],
            dim=-1,
        )

        node_features_prime = self.node_update(node_input)

        # ----- Edge update -----
        global_edge = global_features.unsqueeze(1).repeat(1, N - 1, 1)

        edge_input = torch.cat(
            [
                edge_features,
                node_features_prime[:, :-1],
                node_features_prime[:, 1:],
                global_edge,
            ],
            dim=-1,
        )

        edge_features_prime = self.edge_update(edge_input)

        # ----- Global aggregation -----
        edge_aggregated = self.edge_aggregation(edge_features_prime, edge_mask)

        # ----- Global update -----
        global_input = torch.cat([edge_aggregated], dim=-1)
        global_features_prime = self.global_update(global_input)

        # ----- Classification -----
        y = self.global_classifier(global_features_prime)

        return y.squeeze(-1)
    
    def forward_bundle(self, features: FeatureBundle):
        """
        Forward method accepting a FeatureBundle.

        Args:
            features (FeatureBundle): Input features bundle.
        """
        return self.forward(
            global_features=features.global_features,
            node_features=features.hit_features,
            edge_features=features.edge_features,
            node_mask=features.mask,
        )

class BatchedSoftEdgeAggregator(nn.Module):
    def __init__(self, input_dim, temperature):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)
        self.temperature = temperature

    def forward(self, edge_features, edge_mask):
        scores = self.score(edge_features).squeeze(-1)
        mask = edge_mask.to(scores.dtype)
        scores = scores * mask + (1.0 - mask) * (-1e9)

        weights = torch.softmax(scores / self.temperature, dim=1).unsqueeze(-1)
        return (edge_features * weights).sum(dim=1)

