"""Data contracts shared across every layer of the platform.

These Pydantic models are the *lingua franca* between components: data sources
emit ``Record``s, the feature pipeline produces ``FeatureVector``s, the model
returns ``Prediction``s, and the drift detector raises ``DriftEvent``s. Because
every layer speaks these types instead of raw dicts, layers stay decoupled and a
new dataset only has to produce ``Record``s in this shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

# A raw feature value may be numeric or categorical (e.g. a taxi zone id or
# rate code). The feature pipeline is responsible for encoding categoricals
# into the numeric ``FeatureVector`` the model consumes.
FeatureValue = Union[float, int, str]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Record(BaseModel):
    """One observation arriving from the stream.

    ``target`` is optional because in a real stream the label often arrives
    *after* the prediction (delayed feedback). The engine predicts when a record
    arrives and learns once a target is known.
    """

    record_id: str
    timestamp: datetime = Field(default_factory=_utcnow)
    features: dict[str, FeatureValue]
    target: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceSchema(BaseModel):
    """Self-description of a data source.

    Lets downstream components (feature pipeline, model, drift) configure
    themselves from the source instead of hard-coding column names — so a new
    dataset is wired in by implementing a source, not by editing those layers.
    """

    name: str
    task: Literal["regression", "classification"]
    target_name: str
    feature_names: list[str]
    categorical_features: list[str] = Field(default_factory=list)


class FeatureVector(BaseModel):
    """Model-ready features after the online feature pipeline has run."""

    record_id: str
    values: dict[str, float]


class Prediction(BaseModel):
    """A single inference result with the provenance needed for monitoring."""

    record_id: str
    prediction: float
    model_version: str
    latency_ms: float
    timestamp: datetime = Field(default_factory=_utcnow)


class DriftType(str, Enum):
    CONCEPT = "concept"   # the input→target relationship changed
    FEATURE = "feature"   # the input distribution changed


class DriftEvent(BaseModel):
    """Emitted when a detector flags drift; consumed by the engine to react."""

    drift_type: DriftType
    detected_at: datetime = Field(default_factory=_utcnow)
    sample_index: int
    detector: str
    score: Optional[float] = None
    details: dict[str, Any] = Field(default_factory=dict)
    action_taken: Optional[str] = None
