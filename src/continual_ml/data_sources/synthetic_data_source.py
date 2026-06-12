"""Synthetic regression stream with injectable concept drift.

Deterministic (seeded) so it is ideal for tests and CI. The target is a linear
function of the features plus noise; at ``drift.at_sample`` the underlying
weights shift, which is a genuine *concept* drift (the input→target relationship
changes) that the drift detector and retraining logic must catch.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

from continual_ml.config import SyntheticConfig
from continual_ml.data_sources.base_data_source import BaseDataSource
from continual_ml.schemas import Record, SourceSchema

_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)


class SyntheticDataSource(BaseDataSource):
    """Generates an (optionally infinite) stream of linear-regression samples."""

    def __init__(self, config: SyntheticConfig, max_samples: Optional[int] = None):
        self._cfg = config
        self._max_samples = max_samples
        self._feature_names = [f"x{i}" for i in range(config.n_features)]

    def schema(self) -> SourceSchema:
        return SourceSchema(
            name="synthetic",
            task="regression",
            target_name="y",
            feature_names=list(self._feature_names),
            categorical_features=[],
        )

    def _weights(self, rng: random.Random) -> list[float]:
        return [rng.uniform(-2.0, 2.0) for _ in range(self._cfg.n_features)]

    def stream(self) -> Iterator[Record]:
        rng = random.Random(self._cfg.seed)
        weights = self._weights(rng)
        bias = rng.uniform(-1.0, 1.0)
        drift = self._cfg.drift

        i = 0
        while self._max_samples is None or i < self._max_samples:
            # Apply concept drift: shift the weights once we pass the trigger.
            if drift.enabled and i == drift.at_sample:
                weights = [w * drift.magnitude + rng.uniform(-1, 1) for w in weights]
                bias += drift.magnitude

            features = {name: rng.gauss(0.0, 1.0) for name in self._feature_names}
            target = bias + sum(
                w * features[name] for w, name in zip(weights, self._feature_names)
            )
            target += rng.gauss(0.0, self._cfg.noise_std)

            yield Record(
                record_id=f"syn-{i}",
                timestamp=_EPOCH + timedelta(seconds=i),
                features=features,
                target=target,
                metadata={"drifted": drift.enabled and i >= drift.at_sample},
            )
            i += 1
