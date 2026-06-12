"""Typed, validated configuration loader.

Loads ``configs/config.yaml`` and layers environment-variable overrides on top
(prefix ``CML_``, nesting separator ``__``). Access settings anywhere via the
cached ``get_settings()`` — this is the single source of truth for runtime
behavior, so swapping a dataset or a model is a config change, not a code change.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


# --- Nested config sections (mirror configs/config.yaml) ---------------------

class AppConfig(BaseModel):
    name: str = "continual-ml-platform"
    environment: Literal["development", "production"] = "development"
    log_level: str = "INFO"


class SyntheticDrift(BaseModel):
    enabled: bool = True
    at_sample: int = 5000
    magnitude: float = 2.0


class SyntheticConfig(BaseModel):
    n_features: int = 6
    seed: int = 42
    noise_std: float = 0.5
    drift: SyntheticDrift = Field(default_factory=SyntheticDrift)


class NYCTaxiConfig(BaseModel):
    # Path to the cleaned parquet produced by scripts/prepare_nyc_taxi.py.
    # The source only *reads* this; downloading/cleaning happens offline.
    processed_path: str = "data/processed/nyc_taxi.parquet"
    # Cached zone centroids (scripts/prepare_zones.py) for geographic features.
    zone_centroids_path: str = "data/zone_centroids.csv"
    # Rush-hour windows (local hour ranges) used to derive the is_rush_hour flag.
    rush_hours: list[int] = Field(default_factory=lambda: [7, 8, 9, 16, 17, 18, 19])
    # Expose `route_distance_km` (from trip_distance) as a feature. In this
    # backtest it stands in for the PLANNED route distance a production router
    # (Google Maps / Uber) provides at request time — a valid, highly predictive
    # feature there, simulated here. Set false for the strict no-route backtest.
    use_route_distance: bool = True


class DataSourceConfig(BaseModel):
    type: str = "synthetic"
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)
    nyc_taxi: NYCTaxiConfig = Field(default_factory=NYCTaxiConfig)


class StreamConfig(BaseModel):
    rate_per_sec: float = 10.0
    jitter: float = 0.0
    max_samples: Optional[int] = None
    mode: Literal["fixed", "time_warp"] = "fixed"  # time_warp uses record timestamps
    time_warp_factor: float = 600.0                 # real seconds compressed per wall second
    autostart: bool = True                          # engine starts streaming on boot


class FeaturesConfig(BaseModel):
    rolling_window: int = 50
    lags: list[int] = Field(default_factory=lambda: [1, 2, 3])
    # Online target encoders: each inner list is a feature combination whose
    # running mean target becomes a feature (e.g. mean duration per zone-pair).
    # Updated only AFTER the label is known, so there is no leakage.
    target_encoders: list[list[str]] = Field(default_factory=list)
    te_smoothing: float = 20.0   # shrink low-count combos toward the global mean
    # Multi-horizon memory per encoder: each entry name->EWMA alpha (0.0 = a
    # cumulative mean, i.e. infinite/long memory). Lets stable geographic
    # structure persist (long) while recent traffic adapts (short). Each horizon
    # emits its own feature, e.g. te_pu_zone_do_zone_short.
    te_horizons: dict[str, float] = Field(
        default_factory=lambda: {"long": 0.0, "med": 0.05, "short": 0.3}
    )


class ModelConfig(BaseModel):
    task: Literal["regression", "classification"] = "regression"
    # linear | hoeffding_tree | adaptive_tree | adaptive_forest
    type: str = "hoeffding_tree"
    learning_rate: float = 0.01
    l2: float = 2.0             # L2 weight decay for the linear model
    warmup: int = 100           # collect target stats before the learner starts
    # Features the model should NOT consume directly (still used upstream for geo
    # + target encoding). Dropping the high-cardinality raw zones keeps the tree
    # small since te_pu_zone_do_zone already summarizes them.
    ignore_features: list[str] = Field(default_factory=list)
    rolling_window: int = 1000   # window for rolling performance metrics
    params: dict[str, Any] = Field(default_factory=dict)
    metric: str = "mae"


class ConceptDriftConfig(BaseModel):
    detector: Literal["adwin", "ddm", "page_hinkley"] = "adwin"
    delta: float = 0.002
    # On each ADWIN trigger, compare mean error over this many samples before vs
    # after; only a material shift is a *validated* drift (others are anomalies).
    validation_window: int = 1500
    validation_min_rel_shift: float = 0.10   # >10% mean change
    validation_min_effect: float = 0.20      # Cohen's d > 0.2


class FeatureDriftConfig(BaseModel):
    reference_size: int = 5000   # broad reference so diurnal cycles aren't false positives
    window_size: int = 2000
    check_every: int = 1000
    psi_threshold: float = 0.2   # PSI > 0.2 => significant population shift


class DriftConfig(BaseModel):
    concept: ConceptDriftConfig = Field(default_factory=ConceptDriftConfig)
    feature: FeatureDriftConfig = Field(default_factory=FeatureDriftConfig)


class EngineConfig(BaseModel):
    log_every: int = 500            # push metrics to MLflow every N samples
    recent_buffer: int = 200        # points kept for the live frontend charts
    # Versioning is NOT triggered by raw drift events. A *validated* concept
    # drift in the degradation direction sets a "retrain recommended" alert;
    # actual new versions are created only by the shadow-validate promote flow
    # (scripts/retrain.py). This cooldown rate-limits repeat alerts.
    retrain_alert_cooldown: int = 20000
    # If set and the bundle's schema matches the active source, the engine
    # warm-starts from this champion instead of cold-starting.
    warm_start_path: Optional[str] = "artifacts/online_model.pkl"


class MLflowConfig(BaseModel):
    tracking_uri: str = "http://mlflow:5000"
    experiment_name: str = "continual-learning"
    registered_model_name: str = "continual-online-model"
    register_on_drift: bool = True


class MonitoringConfig(BaseModel):
    metrics_path: str = "/metrics"


# --- Top-level settings ------------------------------------------------------

class Settings(BaseSettings):
    """Root settings object. Built from YAML, then env overrides applied."""

    model_config = SettingsConfigDict(
        env_prefix="CML_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: AppConfig = Field(default_factory=AppConfig)
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Define source precedence (earlier sources win).

        env vars > .env file > YAML file > field defaults. The YAML path is
        resolved from ``CML_CONFIG_PATH`` so overriding it stays env-driven.
        """
        config_path = Path(os.getenv("CML_CONFIG_PATH", "configs/config.yaml"))
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=config_path)
        return (init_settings, env_settings, dotenv_settings, yaml_source, file_secret_settings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the validated settings singleton.

    Precedence (lowest → highest): field defaults < YAML file < env vars.
    The YAML path comes from ``CML_CONFIG_PATH`` (default ``configs/config.yaml``).
    """
    return Settings()
