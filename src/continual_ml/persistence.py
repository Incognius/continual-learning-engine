"""Model-bundle persistence.

The deployable unit is **not** just the model — the online target encoders
("zone-pair memory") live in the ``FeaturePipeline``. A bundle therefore pickles
the model *and* the pipeline (plus schema/metadata) so a reloaded champion
reproduces predictions exactly, including its learned zone-pair statistics.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from continual_ml.features.feature_pipeline import FeaturePipeline
from continual_ml.models.online_model import OnlineModel
from continual_ml.schemas import SourceSchema


@dataclass
class ModelBundle:
    model: OnlineModel
    features: FeaturePipeline
    schema: SourceSchema
    meta: dict[str, Any] = field(default_factory=dict)

    def predict(self, raw_features: dict) -> float:
        """Reproduce a prediction from raw record features (frozen encoders)."""
        feats = self.features.transform(raw_features_to_record(raw_features),
                                        update_stats=False)
        return self.model.predict_one(feats)


def save_bundle(bundle: ModelBundle, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(bundle, fh)
    return path.stat().st_size


def load_bundle(path: str | Path) -> ModelBundle:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


# Local import kept here to avoid a module-level cycle.
def raw_features_to_record(raw_features: dict):
    from continual_ml.schemas import Record
    return Record(record_id="bundle", features=raw_features)
