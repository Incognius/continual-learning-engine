"""Data source layer — the plug point for new datasets."""

from continual_ml.data_sources.base_data_source import BaseDataSource
from continual_ml.data_sources.factory import build_data_source
from continual_ml.data_sources.nyc_taxi_data_source import NYCTaxiDataSource
from continual_ml.data_sources.synthetic_data_source import SyntheticDataSource

__all__ = [
    "BaseDataSource",
    "build_data_source",
    "SyntheticDataSource",
    "NYCTaxiDataSource",
]
