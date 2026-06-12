"""Factory that builds the configured data source.

The engine calls ``build_data_source(settings)`` and never imports a concrete
source directly — adding a new dataset means registering it here, with no
changes to the consumer code.
"""

from __future__ import annotations

from continual_ml.config import Settings
from continual_ml.data_sources.base_data_source import BaseDataSource
from continual_ml.data_sources.nyc_taxi_data_source import NYCTaxiDataSource
from continual_ml.data_sources.synthetic_data_source import SyntheticDataSource


def build_data_source(settings: Settings) -> BaseDataSource:
    ds = settings.data_source
    source_type = ds.type.lower()

    if source_type == "synthetic":
        return SyntheticDataSource(
            ds.synthetic, max_samples=settings.stream.max_samples
        )
    if source_type == "nyc_taxi":
        return NYCTaxiDataSource(ds.nyc_taxi)

    raise ValueError(
        f"Unknown data_source.type '{ds.type}'. "
        f"Supported: 'synthetic', 'nyc_taxi'."
    )
