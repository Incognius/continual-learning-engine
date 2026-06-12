"""Continual learning MLOps platform.

Public foundation re-exports so other modules can do:
    from continual_ml import get_settings, Record, Prediction
"""

from continual_ml.config import Settings, get_settings
from continual_ml.schemas import (
    DriftEvent,
    DriftType,
    FeatureVector,
    Prediction,
    Record,
    SourceSchema,
)

__all__ = [
    "Settings",
    "get_settings",
    "Record",
    "FeatureVector",
    "Prediction",
    "DriftEvent",
    "DriftType",
    "SourceSchema",
]

__version__ = "0.1.0"
