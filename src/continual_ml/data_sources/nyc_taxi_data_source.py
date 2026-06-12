"""NYC TLC yellow-taxi stream — a real-world replacement for the synthetic source.

Replays cleaned trips in pickup-time order as an online regression stream whose
target is **trip duration in seconds** (ETA prediction). Crucially, only
features knowable *at pickup* are exposed — origin/destination zone, time-of-day,
passenger count, rate code. ``trip_distance`` is intentionally excluded because
it is the metered distance of the *completed* trip and would leak the target.

The duration label models delayed feedback: it is "known" only at dropoff, which
is exactly the test-then-train pattern the engine implements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from continual_ml.config import NYCTaxiConfig
from continual_ml.data_sources.base_data_source import BaseDataSource
from continual_ml.geo import ZoneGeo
from continual_ml.schemas import Record, SourceSchema

# Numeric vs categorical split is declared here so the feature pipeline can
# encode the right columns without inspecting the data. The geo.* features come
# from the zone centroids and are knowable at pickup (no leakage).
_NUMERIC_FEATURES = [
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
    "passenger_count",
    "gc_distance_km",   # straight-line distance between zone centroids
    "bearing_sin",      # direction of travel (circular -> sin/cos)
    "bearing_cos",
    "same_borough",
    "pu_airport",
    "do_airport",
]
_CATEGORICAL_FEATURES = ["pu_zone", "do_zone", "ratecode", "pu_borough", "do_borough"]
_TARGET = "target_duration_s"


class NYCTaxiDataSource(BaseDataSource):
    """Streams cleaned NYC taxi trips from a prepared parquet file."""

    def __init__(self, config: NYCTaxiConfig):
        self._cfg = config
        self._path = Path(config.processed_path)
        self._geo = ZoneGeo(config.zone_centroids_path)
        self._numeric = list(_NUMERIC_FEATURES)
        if config.use_route_distance:
            self._numeric.append("route_distance_km")
        self._feature_names = self._numeric + _CATEGORICAL_FEATURES

    @property
    def geo(self) -> ZoneGeo:
        return self._geo

    def schema(self) -> SourceSchema:
        return SourceSchema(
            name="nyc_taxi",
            task="regression",
            target_name="trip_duration_s",
            feature_names=list(self._feature_names),
            categorical_features=list(_CATEGORICAL_FEATURES),
        )

    def _load(self) -> pd.DataFrame:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Processed taxi data not found at '{self._path}'. "
                f"Run: python scripts/prepare_nyc_taxi.py first."
            )
        df = pd.read_parquet(self._path)
        # Defensive: guarantee chronological replay regardless of how it was saved.
        return df.sort_values("pickup_datetime").reset_index(drop=True)

    def stream(self) -> Iterator[Record]:
        df = self._load()
        for row in df.itertuples(index=False):
            pu_zone, do_zone = str(row.pu_zone), str(row.do_zone)
            geo = self._geo.features(pu_zone, do_zone)
            features = {
                "pickup_hour": float(row.pickup_hour),
                "pickup_dayofweek": float(row.pickup_dayofweek),
                "pickup_month": float(row.pickup_month),
                "is_weekend": float(row.is_weekend),
                "is_rush_hour": float(row.is_rush_hour),
                "passenger_count": float(row.passenger_count),
                # Geographic features (leakage-free; from zone centroids).
                "gc_distance_km": geo["gc_distance_km"],
                "bearing_sin": geo["bearing_sin"],
                "bearing_cos": geo["bearing_cos"],
                "same_borough": geo["same_borough"],
                "pu_airport": geo["pu_airport"],
                "do_airport": geo["do_airport"],
                # Categoricals as strings so the pipeline can one-hot them.
                "pu_zone": pu_zone,
                "do_zone": do_zone,
                "ratecode": str(row.ratecode),
                "pu_borough": geo["pu_borough"],
                "do_borough": geo["do_borough"],
            }
            if self._cfg.use_route_distance:
                features["route_distance_km"] = float(getattr(row, "route_distance_km", 0.0))
            yield Record(
                record_id=str(row.record_id),
                timestamp=row.pickup_datetime.to_pydatetime(),
                features=features,
                target=float(row.target_duration_s),
            )
