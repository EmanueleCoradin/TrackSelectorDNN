# models/__init__.py
from .netA import NetA
from .netB import NetB
from .pooling import SoftmaxPooling#, SumPooling, MeanPooling
from .track_classifier import TrackClassifier

MODEL_REGISTRY = {
    "TrackClassifier": TrackClassifier,
    "NetA": NetA,
    "NetB": NetB,
}

__all__ = list(MODEL_REGISTRY.keys())
