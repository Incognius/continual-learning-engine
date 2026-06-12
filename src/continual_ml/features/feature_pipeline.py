"""Online feature pipeline.

Turns a raw ``Record`` into the engineered feature dict the model consumes. It
is *schema-driven*: it reads the source's ``SourceSchema`` to know which columns
are numeric vs categorical, so the same pipeline works for synthetic data, taxi
data, or any future dataset without edits.

Responsibilities:
  * Pass through known features; split numeric vs categorical.
  * Add light, leakage-free derived features (cyclical time encodings when the
    relevant columns exist).
  * Maintain running statistics (Welford) on numeric inputs — these are the
    reference the feature-drift detector compares against and are exposed for
    monitoring.

Encoding/scaling of categoricals is intentionally left to the model's River
pipeline so there is exactly one place that owns numeric vectorization.
"""

from __future__ import annotations

import math
from typing import Optional

from continual_ml.config import FeaturesConfig
from continual_ml.schemas import FeatureValue, Record, SourceSchema

_CYCLICAL = {"pickup_hour": 24.0, "pickup_dayofweek": 7.0, "pickup_month": 12.0}


class _Welford:
    """Numerically stable online mean/variance."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


class _RunningMean:
    """Online count + mean for a single bucket (global baseline)."""

    __slots__ = ("n", "mean")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0

    def update(self, y: float) -> None:
        self.n += 1
        self.mean += (y - self.mean) / self.n


class _MultiHorizonMean:
    """Per-bucket memory at several horizons.

    ``long`` (alpha 0) is a cumulative mean — stable geographic structure that
    never fades. Fixed-alpha EWMAs (``med``, ``short``) decay older observations
    so recent traffic conditions dominate. The short↔long gap is itself a signal
    of current deviation from the corridor's baseline.
    """

    __slots__ = ("n", "_alphas", "_vals")

    def __init__(self, alphas: dict[str, float]) -> None:
        self.n = 0
        self._alphas = alphas
        self._vals = {name: 0.0 for name in alphas}

    def update(self, y: float) -> None:
        self.n += 1
        for name, alpha in self._alphas.items():
            if self.n == 1:
                self._vals[name] = y
            elif alpha <= 0.0:                       # cumulative (long) memory
                self._vals[name] += (y - self._vals[name]) / self.n
            else:                                     # EWMA (shorter horizons)
                self._vals[name] += alpha * (y - self._vals[name])

    def values(self) -> dict[str, float]:
        return self._vals


class FeaturePipeline:
    def __init__(self, schema: SourceSchema, config: Optional[FeaturesConfig] = None):
        self._schema = schema
        self._categorical = set(schema.categorical_features)
        self._numeric = [f for f in schema.feature_names if f not in self._categorical]
        self._stats: dict[str, _Welford] = {f: _Welford() for f in self._numeric}
        self._count = 0

        # --- online multi-horizon target encoders ("zone-pair memory") ------
        config = config or FeaturesConfig()
        self._encoders: list[tuple[str, ...]] = [tuple(e) for e in config.target_encoders]
        self._te_smoothing = config.te_smoothing
        self._horizons = dict(config.te_horizons) or {"long": 0.0}
        # One feature per (encoder × horizon).
        self._te_names = [
            "te_" + "_".join(e) + "_" + h
            for e in self._encoders for h in self._horizons
        ]
        self._te_stats: dict[tuple, dict[str, _MultiHorizonMean]] = {e: {} for e in self._encoders}
        self._global = _RunningMean()

    @property
    def numeric_features(self) -> list[str]:
        return list(self._numeric) + list(self._te_names)

    def _encoder_key(self, enc: tuple[str, ...], features: dict) -> Optional[str]:
        if not all(name in features for name in enc):
            return None
        return "|".join(str(features[name]) for name in enc)

    @property
    def categorical_features(self) -> list[str]:
        return list(self._categorical)

    def running_stats(self) -> dict[str, dict[str, float]]:
        return {
            f: {"mean": w.mean, "std": w.std, "n": w.n} for f, w in self._stats.items()
        }

    def transform(self, record: Record, update_stats: bool = True) -> dict[str, FeatureValue]:
        """Produce the engineered feature dict for one record.

        ``update_stats=False`` (used by ad-hoc /predict) computes features
        without mutating the running statistics that feed drift detection.
        """
        if update_stats:
            self._count += 1
        out: dict[str, FeatureValue] = {}

        for name, value in record.features.items():
            if name in self._categorical:
                out[name] = str(value)
                continue

            fval = float(value)
            out[name] = fval
            if update_stats and name in self._stats:
                self._stats[name].update(fval)

            # Cyclical encoding for periodic time columns (leakage-free).
            period = _CYCLICAL.get(name)
            if period:
                angle = 2.0 * math.pi * (fval % period) / period
                out[f"{name}_sin"] = math.sin(angle)
                out[f"{name}_cos"] = math.cos(angle)

        # Online target-encoding features: emit the CURRENT estimate at each
        # horizon (the model then predicts; encoders are updated later via
        # update_target). Unseen/low-count buckets shrink toward the global mean.
        g = self._global.mean
        k = self._te_smoothing
        for enc in self._encoders:
            key = self._encoder_key(enc, out)
            if key is None:
                continue
            prefix = "te_" + "_".join(enc) + "_"
            bucket = self._te_stats[enc].get(key)
            if bucket is None:
                for h in self._horizons:
                    out[prefix + h] = g
            else:
                vals = bucket.values()
                for h in self._horizons:
                    out[prefix + h] = (bucket.n * vals[h] + k * g) / (bucket.n + k)

        return out

    def update_target(self, features: dict, y: float) -> None:
        """Update the online target encoders once the true label is known.

        Called by the engine AFTER prediction + model update, so the encoder
        value a record sees is never derived from its own label.
        """
        self._global.update(y)
        for enc in self._encoders:
            key = self._encoder_key(enc, features)
            if key is None:
                continue
            bucket = self._te_stats[enc].get(key)
            if bucket is None:
                bucket = _MultiHorizonMean(self._horizons)
                self._te_stats[enc][key] = bucket
            bucket.update(y)
