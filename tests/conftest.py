"""Test configuration: keep MLflow local/offline and the stream slow."""

import os

os.environ.setdefault("CML_MLFLOW__TRACKING_URI", "sqlite:///test_mlflow.db")
os.environ.setdefault("CML_DATA_SOURCE__TYPE", "synthetic")
os.environ.setdefault("CML_STREAM__RATE_PER_SEC", "500")
os.environ.setdefault("CML_ENGINE__WARM_START_PATH", "")  # cold start in tests
