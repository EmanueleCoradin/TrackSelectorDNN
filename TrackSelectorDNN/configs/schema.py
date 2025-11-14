from importlib import resources
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml


class Config(BaseModel):
    """
    Configuration schema for the TrackSelectorDNN model and training pipeline.

    This class uses Pydantic to define and validate configuration parameters
    for model architecture, training setup, and dataset handling. It can
    be instantiated from a YAML configuration file using the `load_config` function.

    --- Attributes ---

    Model Architecture:
        hit_input_dim (int): Dimensionality of the input features for hits.
        track_feat_dim (int): Dimensionality of the track features.
        latent_dim (int): Size of the latent embedding vector (must be > 0).
        pooling_type (Literal["sum", "mean", "softmax"]): Pooling method to aggregate the results of NetAs.

        netA_hidden_dim (int): Hidden layer size for network A.
        netA_hidden_layers (int): Number of hidden layers in network A.
        netA_batchnorm (bool): Whether to use batch normalization in network A.
        netA_activation (Literal["relu", "silu", "gelu", "tanh", "leakyrelu"]): Activation function for network A.

        netB_hidden_dim (int): Hidden layer size for network B.
        netB_hidden_layers (int): Number of hidden layers in network B.
        netB_batchnorm (bool): Whether to use batch normalization in network B.
        netB_activation (Literal["relu", "silu", "gelu", "tanh", "leakyrelu"]): Activation function for network B.

    Training Setup:
        lr (float): Learning rate (must be > 0).
        epochs (int): Number of training epochs (must be > 0).
        batch_size (int): Training batch size (must be > 0).
        patience (int): Early stopping patience.
        delta (float): Minimum improvement for early stopping.
        base_checkpoint_directory (str): Directory to store model checkpoints.

    Data:
        dataset_type (Literal["dummy", "production"]): Type of dataset to use.
        dummy_load_path (Optional[str]): Path to dummy dataset (required if dataset_type is "dummy").
        train_path (Optional[str]): Path to training dataset (required if dataset_type is "production").
        val_path (Optional[str]): Path to validation dataset (required if dataset_type is "production").
        test_path (Optional[str]): Path to test dataset (required if dataset_type is "production").
        max_hits (int): Maximum number of hits per track.

    Validation:
        - `latent_dim` must be positive.
        - Depending on `dataset_type`, certain dataset paths are required:
            * "dummy": `dummy_load_path` is required.
            * "production": `train_path`, `val_path`, and `test_path` are required.

    Usage:
        config = load_config("my_config.yaml")
        print(config.lr, config.netA_activation)
    """
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
    patience: int
    delta: float
    base_checkpoint_directory: str

    # --- Data ---
    dataset_type: Literal["dummy", "production"]
    dummy_load_path: Optional[str] = None
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    max_hits: int

    @field_validator("latent_dim")
    def latent_positive(cls, v):
        if v <= 0:
            raise ValueError("latent_dim must be > 0")
        return v

    @model_validator(mode="after")
    def check_paths(self):
        if self.dataset_type == "dummy":
            if not self.dummy_load_path:
                raise ValueError("dummy_load_path is required when dataset_type='dummy'")

        if self.dataset_type == "production":
            missing = [
                name for name in ("train_path", "val_path", "test_path")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"Missing required fields for production dataset: {', '.join(missing)}"
                )

        return self
        
def load_config(filename: str):
    with resources.open_text("TrackSelectorDNN.configs", filename) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
