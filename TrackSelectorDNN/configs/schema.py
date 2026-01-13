"""
Configuration schema for the TrackSelectorDNN model and training pipeline.
"""

from importlib import resources
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

#-----------------------------------------------------------------------------

# ---------------------
# Model Related Config
# ---------------------

# ----------------------------------------------------------------------------

class MLPConfig(BaseModel):
    """
    Configuration schema for a MLP block.
    
    --- Attributes ---    
    hidden_dim (int): Hidden layer size.
    hidden_layers (int): Number of hidden layers.
    batchnorm (bool): Whether to use batch normalization after each layer.
    activation (Literal["relu", "silu", "gelu", "tanh", "leakyrelu"]): Activation function for each neuron.
    """

    hidden_dim: int = Field(gt=0)
    hidden_layers: int = Field(gt=0)
    batchnorm: bool
    activation: Literal["relu", "silu", "gelu", "tanh", "leakyrelu"]

# -------------------------------------------------------------------------------------

class NetAMLPConfig(BaseModel):
    """
    Configuration schema for a NetA based on a MLP block.
    
    --- Attributes ---    
    kind (Literal["mlp"]): Kind of architecture implemented. 
    hidden_dim (int): Hidden layer size.
    hidden_layers (int): Number of hidden layers.
    batchnorm (bool): Whether to use batch normalization after each layer.
    activation (Literal["relu", "silu", "gelu", "tanh", "leakyrelu"]): Activation function for each neuron.
    """
    kind: Literal["mlp"]

    hidden_dim: int = Field(gt=0)
    hidden_layers: int = Field(gt=0)
    batchnorm: bool
    activation: Literal["relu", "silu", "gelu", "tanh", "leakyrelu"]

#TODO: to be implemented in the library, left as a reference
class NetATransformerConfig(BaseModel):
    """
    Configuration schema for a NetA based on a transformer block.
    
    --- Attributes ---    
    kind (Literal["transformer"]): Kind of architecture implemented. 
    d_model (int): Dimensionality of the per-hit embedding space used internally by the transformer.
    
    n_heads (int): 
        Number of attention heads in the multi-head self-attention layers.
        Must evenly divide `d_model`. More heads allow the model to attend
        to multiple feature subspaces in parallel.
    
    n_layers (bool): Number of stacked transformer encoder layers.
    dropout (float): Dropout probability applied within the transformer encoder layers. Must be in the range [0.0, 1.0].
    """
    kind: Literal["transformer"]

    d_model: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    n_layers: int = Field(gt=0)
    dropout: float = Field(0.0, ge=0.0, le=1.0)

NetAConfig = Annotated[
    Union[
        NetAMLPConfig,
        NetATransformerConfig,
    ],
    Field(discriminator="kind"),
]

# -------------------------------------------------------------------------------------

class NetBMLPConfig(BaseModel):
    """
    Configuration schema for a track-level MLP that combines pooled hit
    embeddings from NetA with per-track features.
    """

    kind: Literal["mlp"]

    hidden_dim: int = Field(gt=0)
    hidden_layers: int = Field(ge=0)
    batchnorm: bool
    activation: Literal["relu", "silu", "gelu", "tanh", "leakyrelu"]

class NetBTrackOnlyConfig(BaseModel):
    """
    Configuration schema for a track-only classifier operating exclusively
    on per-track features, without any dependence on hit-level information.
    """

    kind: Literal["track_only"]

    hidden_dim: int = Field(gt=0)
    hidden_layers: int = Field(ge=0)
    batchnorm: bool
    activation: Literal["relu", "silu", "gelu", "tanh", "leakyrelu"]

NetBConfig = Annotated[
    Union[
        NetBMLPConfig,
        NetBTrackOnlyConfig,
    ],
    Field(discriminator="kind"),
]

# -------------------------------------------------------------------------------------

class TrackClassifierConfig(BaseModel):
    """
    Configuration schema for a classifier operating 
    on per-track features and hit-level information.
    """
    type: Literal["track_classifier"]

    hit_input_dim: int = Field(gt=0)
    track_feat_dim: int = Field(gt=0)
    latent_dim: int = Field(gt=0)
    pooling_type: Literal["sum", "mean", "softmax"]

    netA: NetAConfig
    netB: NetBConfig

class TrackOnlyClassifierConfig(BaseModel):
    """
    Configuration schema for a track-only classifier operating exclusively
    on per-track features, without any dependence on hit-level information.
    """
    type: Literal["track_only"]

    track_feat_dim: int = Field(gt=0)
    netB: NetBConfig

class PreselectorClassifierConfig(BaseModel):
    """
    Configuration schema for a preselector classifier operating 
    on per-track features.
    """
    type: Literal["preselector"]

    preselector_feat_dim: int = Field(gt=0)
    netB: NetBConfig

ModelConfig = Annotated[
    Union[
        TrackClassifierConfig,
        TrackOnlyClassifierConfig,
        PreselectorClassifierConfig,
    ],
    Field(discriminator="type"),
]
# -------------------------------------------------------------------------------------

# ------------------------
# Training Related Config
# ------------------------

# -------------------------------------------------------------------------------------

class SymmetryConfig(BaseModel):
    """
    Configuration schema for optional symmetry regularization during training.

    --- Attributes ---
        idxSymRecHitFeatures (Optional[List[int]]): Indices of hit features to symmetrize.
        idxSymRecoPixelTrackFeatures (Optional[List[int]]): Indices of pixel track features  to symmetrize.
        idxSymPreselectFeatures (Optional[List[int]]): Indices of preselection features to symmetrize.
        lambda_sym (Optional[float]): Weight for symmetric regularization term.
    """
    idxSymRecHitFeatures: Optional[List[int]] = None
    idxSymRecoPixelTrackFeatures: Optional[List[int]] = None
    idxSymPreselectFeatures: Optional[List[int]] = None
    lambda_sym: Optional[float] = None

class WeightsConfig(BaseModel):
    """
    Configuration schema for optional class weighting during training.

    --- Attributes ---
        w_true (Optional[float]): Weight for true tracks.
        w_fake (Optional[float]): Weight for fake tracks.
    """
    w_true: Optional[float] = None
    w_fake: Optional[float] = None

# -------------------------------------------------------------------------------------

class OptimizerConfig(BaseModel):
    """
    Configuration schema for optimizer to be used during training.

     --- Attributes ---
        name (Literal["adam", "adamw"]):
            Identifier of the optimizer to use. Supported options are
            currently Adam and AdamW.
        lr (float): Base learning rate for the optimizer. Must be strictly positive.
        weight_decay (float): 
            L2 regularization coefficient applied by the optimizer.
            Defaults to zero if not specified.
    """
    name: Literal["adam", "adamw"]
    lr: float = Field(..., gt=0)
    weight_decay: float = 0.0

class SchedulerConfig(BaseModel):
    """
    Learning-rate scheduler configuration for the training pipeline.

    This schema defines all supported LR scheduling strategies and their
    associated hyperparameters. It provides a unified interface for
    epoch-level schedulers (e.g. CosineAnnealingLR, ReduceLROnPlateau)
    and batch-level schedulers (e.g. OneCycleLR).

    --- Attributes ---
        name (Literal["none", "plateau", "cosine", "onecycle"]):
            - "none": No scheduler is applied.
            - "plateau": Uses ReduceLROnPlateau, adjusting LR based on validation loss.
            - "cosine": Uses CosineAnnealingLR with cosine decay.
            - "onecycle": Uses OneCycleLR; LR changes every batch.

        # --- Common optional fields ---
        min_lr (Optional[float]):
            Minimum LR floor used by schedulers such as ReduceLROnPlateau
            or CosineAnnealingLR. If not provided, the underlying PyTorch
            default is used.

        # --- ReduceLROnPlateau parameters ---
        factor (Optional[float]):
            Multiplicative factor for LR reduction.
        patience (Optional[int]):
            Number of epochs without improvement before reducing LR.

        # --- CosineAnnealingLR parameters ---
        T_max (Optional[int]):
            Number of epochs for a full cosine cycle.
        eta_min (Optional[float]):
            Minimum learning rate for cosine annealing.

        # --- OneCycleLR parameters ---
        max_lr (Optional[float]):
            Peak LR for OneCycleLR.
        pct_start (Optional[float]):
            Fraction of the cycle used for the LR increase phase.
        anneal_strategy (Optional[Literal["cos", "linear"]]):
            Schedule shape for LR decrease.
        div_factor (Optional[float]):
            Determines initial_lr = max_lr / div_factor.
        final_div_factor (Optional[float]):
            Determines minimum LR via final_lr = initial_lr / final_div_factor.
        three_phase (Optional[bool]):
            Whether to use the original 3-phase OneCycle policy.
    """

    name: Literal["none", "plateau", "cosine", "onecycle"] = "none"

    # --- Common optional fields ---
    min_lr: Optional[float] = None

    # --- Plateau parameters ---
    factor: Optional[float] = None
    patience: Optional[int] = None

    # --- Cosine parameters ---
    T_max: Optional[int] = None
    eta_min: Optional[float] = None

    # --- OneCycleLR parameters ---
    max_lr: Optional[float] = None
    pct_start: Optional[float] = None
    anneal_strategy: Optional[Literal["cos", "linear"]] = None
    div_factor: Optional[float] = None
    final_div_factor: Optional[float] = None
    three_phase: Optional[bool] = None

    @model_validator(mode="after")
    def check_scheduler_params(self):
        """
        Ensure that required parameters are provided and valid depending on the scheduler type.
        """
        if self.name == "plateau":
            if self.factor is None or not (0 < self.factor < 1):
                raise ValueError("plateau scheduler requires 'factor' in (0,1)")
            if self.patience is None or self.patience <= 0:
                raise ValueError("plateau scheduler requires 'patience' > 0")

        elif self.name == "cosine":
            if self.T_max is None or self.T_max <= 0:
                raise ValueError("cosine scheduler requires 'T_max' > 0")
            if self.eta_min is not None and self.eta_min < 0:
                raise ValueError("'eta_min' must be >= 0")

        elif self.name == "onecycle":
            if self.max_lr is None or self.max_lr <= 0:
                raise ValueError("onecycle scheduler requires 'max_lr' > 0")
            if self.pct_start is not None and not (0 < self.pct_start < 1):
                raise ValueError("'pct_start' must be in (0,1)")
            if self.div_factor is not None and self.div_factor <= 0:
                raise ValueError("'div_factor' must be > 0")
            if self.final_div_factor is not None and self.final_div_factor <= 0:
                raise ValueError("'final_div_factor' must be > 0")
            if self.anneal_strategy is not None and self.anneal_strategy not in ["cos", "linear"]:
                raise ValueError("'anneal_strategy' must be 'cos' or 'linear'")

        elif self.name == "none":
            # No parameters required
            pass

        else:
            raise ValueError(f"Unknown scheduler name: {self.name}")

        return self

# -------------------------------------------------------------------------------------

class TrainingConfig(BaseModel):
    """
    Configuration schema for the training pipeline.

    This class uses Pydantic to define and validate configuration parameters
    for training setup
    
    --- Attributes ---
        epochs (int): Number of training epochs (must be > 0).
        batch_size (int): Training batch size (must be > 0).
        patience (int): Early stopping patience.
        delta (float): Minimum improvement for early stopping.
        base_checkpoint_directory (str): Directory to store model checkpoints.
        symmetry (SymmetryConfig): 
            Nested configuration block controlling symmetric feature regularization options.
        weights (WeightsConfig): Nested configuration block defining class-wise sample weights
    """
    epochs: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)
    patience: int  = Field(..., gt=0)
    delta: float
    base_checkpoint_directory: str
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    symmetry: SymmetryConfig
    weights: WeightsConfig

# -------------------------------------------------------------------------------------

# --------------------
# Data Related Config
# --------------------

class DataConfig(BaseModel):
    """
    Configuration schema for the dataset handling.

    --- Attributes ---
        dataset_type (Literal["dummy", "production", "preselector"]): Type of dataset to use.
        dummy_load_path (Optional[str]): Path to dummy dataset (required if dataset_type is "dummy").
        train_path (Optional[str]): Path to training dataset (required if dataset_type is "production", "preselector").
        val_path (Optional[str]): Path to validation dataset (required if dataset_type is "production", "preselector").
        test_path (Optional[str]): Path to test dataset.
        max_hits (int): Maximum number of hits per track.
    """

    dataset_type: Literal["dummy", "production", "preselector"]
    dummy_load_path: Optional[str] = None
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    max_hits: int

    @model_validator(mode="after")
    def check_paths(self):
        """
        Ensure that required parameters are provided and valid depending on the scheduler type.
        """
        if self.dataset_type == "dummy":
            if not self.dummy_load_path:
                raise ValueError("dummy_load_path is required when dataset_type='dummy'")

        if self.dataset_type == "production" or self.dataset_type == "preselector":
            missing = [
                name for name in ("train_path", "val_path")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"Missing required fields for production dataset: {', '.join(missing)}"
                )

        return self

# -------------------------------------------------------------------------------------

# --------------------
# General Config
# --------------------

# -------------------------------------------------------------------------------------

class Config(BaseModel):
    """
    Configuration schema for the TrackSelectorDNN model and training pipeline.

    This class uses Pydantic to define and validate configuration parameters
    for model architecture, training setup, and dataset handling. It can
    be instantiated from a YAML configuration file using the `load_config` function.
    """
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

def load_config(filename: str | Path) -> Config:
    """
    Load a YAML configuration file from either:
    - a filesystem path
    - or the TrackSelectorDNN.configs package
    """
    if isinstance(filename, (str, Path)) and Path(filename).exists():
        with Path(filename).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    else:
        with resources.open_text("TrackSelectorDNN.configs", filename) as f:
            raw = yaml.safe_load(f)

    return Config(**raw)

# -------------------------------------------------------------------------------------
