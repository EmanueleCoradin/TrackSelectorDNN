# models/__init__.py
from .netA import NetA
from .netB import NetB
from .pooling import AttentionPooling, SumPooling, MeanPooling
from .track_classifier import TrackClassifier
from .registry import Registry

MODEL_REGISTRY = {
    "TrackClassifier": TrackClassifier,
    "NetA": NetA,
    "NetB": NetB,
}

__all__ = list(MODEL_REGISTRY.keys())
