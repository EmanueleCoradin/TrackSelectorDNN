from pydantic import BaseModel, Field, validator
from typing import Literal
import yaml

class Config(BaseModel):
    # --- Model architecture ---
    hit_input_dim: int
    track_feat_dim: int
    latent_dim: int
    pooling_type: Literal["sum", "mean", "softmax"]
    
    netA_hidden_dim: int
    netA_hidden_layers: int
    netA_batchnorm: bool
    netA_activation: Literal["relu", "silu", "gelu", "tanh", "leakyrelu"]
    
    netB_hidden_dim: int
    netB_hidden_layers: int
    netB_batchnorm: bool
    netB_activation: Literal["relu", "silu", "gelu", "tanh", "leakyrelu"]

    # --- Training setup ---
    lr: float = Field(..., gt=0)
    epochs: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)

    # --- Data ---
    dataset_type: Literal["dummy", "production"]
    train_path: str
    n_tracks: int               # used only for dummy
    max_hits: int
    val_fraction: float = Field(..., ge=0.0, le=1.0)

    @validator("latent_dim")
    def latent_positive(cls, v):
        if v <= 0:
            raise ValueError("latent_dim must be > 0")
        return v


def load_config(path: str) -> Config:
    """Load and validate YAML config file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
