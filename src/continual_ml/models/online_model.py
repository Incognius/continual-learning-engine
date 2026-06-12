"""Online (incremental) regression model built on River.

Wraps a River pipeline behind a tiny ``predict_one`` / ``learn_one`` interface.
Two design choices worth noting:

1. **Schema-driven pipeline.** Categorical features are one-hot encoded and
   numeric features are standardized, in separate branches, based on the
   ``SourceSchema``. No column names are hard-coded.

2. **Online target scaling.** Trip durations span seconds→thousands; feeding
   raw targets to an SGD learner is numerically nasty. We standardize the target
   with a running mean/std (updated *before* learning, so there is no leakage of
   the current label into its own prediction) and invert on the way out. For the
   synthetic source this is a near-identity transform, so the model code stays
   dataset-agnostic.
"""

from __future__ import annotations

import math
from typing import Any

from river import compose, forest, linear_model, optim, preprocessing, tree

from continual_ml.config import ModelConfig
from continual_ml.schemas import FeatureValue, SourceSchema


def _build_regressor(cfg: ModelConfig):
    if cfg.type == "linear":
        # Adam (not plain SGD): with high-cardinality one-hot zone features and a
        # standardized target, plain SGD at lr=0.01 diverges; Adam's per-parameter
        # adaptive steps stay stable and converge to a lower error. L2 weight decay
        # bounds the weights so the model does not overfit the most-recent window
        # and extrapolate wildly on older data.
        return linear_model.LinearRegression(
            optimizer=optim.Adam(cfg.learning_rate), l2=cfg.l2
        )
    # max_size (MB) caps tree memory so the saved artifact can't balloon in
    # production; grace_period trades a little accuracy for far fewer splits.
    if cfg.type == "hoeffding_tree":
        return tree.HoeffdingTreeRegressor(grace_period=200, max_size=30)
    if cfg.type == "adaptive_tree":
        return tree.HoeffdingAdaptiveTreeRegressor(seed=42, grace_period=200, max_size=30)
    if cfg.type == "adaptive_forest":
        return forest.ARFRegressor(seed=42)
    raise ValueError(f"Unknown model.type '{cfg.type}'")


def _build_pipeline(schema: SourceSchema, cfg: ModelConfig):
    ignore = set(cfg.ignore_features)
    categorical = [c for c in schema.categorical_features if c not in ignore]
    regressor = _build_regressor(cfg)

    if categorical:
        numeric_branch = compose.Discard(*categorical) | preprocessing.StandardScaler()
        categorical_branch = compose.Select(*categorical) | preprocessing.OneHotEncoder()
        features = numeric_branch + categorical_branch
    else:
        features = preprocessing.StandardScaler()

    return features | regressor


class OnlineModel:
    def __init__(self, schema: SourceSchema, config: ModelConfig):
        self._schema = schema
        self._config = config
        self._warmup = max(config.warmup, 2)
        self._ignore = set(config.ignore_features)
        self.model = _build_pipeline(schema, config)
        # Running target statistics (Welford) for online standardization.
        self._t_n = 0
        self._t_mean = 0.0
        self._t_m2 = 0.0
        self.samples_seen = 0

    # --- target scaling helpers ---------------------------------------------
    @property
    def _t_std(self) -> float:
        if self._t_n < 2:
            return 1.0
        std = math.sqrt(self._t_m2 / self._t_n)
        return std if std > 1e-9 else 1.0

    def _update_target_stats(self, y: float) -> None:
        self._t_n += 1
        delta = y - self._t_mean
        self._t_mean += delta / self._t_n
        self._t_m2 += delta * (y - self._t_mean)

    def _model_input(self, features: dict[str, FeatureValue]) -> dict[str, FeatureValue]:
        if not self._ignore:
            return features
        return {k: v for k, v in features.items() if k not in self._ignore}

    # --- inference / learning ------------------------------------------------
    def predict_one(self, features: dict[str, FeatureValue]) -> float:
        scaled = self.model.predict_one(self._model_input(features))
        if scaled is None:
            scaled = 0.0
        return scaled * self._t_std + self._t_mean

    def learn_one(self, features: dict[str, FeatureValue], y: float) -> None:
        self._update_target_stats(y)
        # Warm-up: collect enough targets for a stable mean/std before training,
        # so the SGD learner only ever sees O(1) standardized targets and cannot
        # diverge on the first (effectively unscaled) samples.
        if self._t_n <= self._warmup:
            return
        y_scaled = (y - self._t_mean) / self._t_std
        self.model.learn_one(self._model_input(features), y_scaled)
        self.samples_seen += 1

    # --- introspection -------------------------------------------------------
    def params(self) -> dict[str, Any]:
        return {
            "model_type": self._config.type,
            "learning_rate": self._config.learning_rate,
            "task": self._schema.task,
            "n_categorical": len(self._schema.categorical_features),
            "target_name": self._schema.target_name,
        }
