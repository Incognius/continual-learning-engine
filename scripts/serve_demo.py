"""Local demo launcher — runs the API against the NYC taxi data with a local
SQLite MLflow backend (no Docker required).

    python scripts/serve_demo.py

Then open http://127.0.0.1:8000/. Environment variables still win, so you can
override any of these defaults from the shell.
"""

import os
import sys

os.environ.setdefault("CML_DATA_SOURCE__TYPE", "nyc_taxi")
os.environ.setdefault("CML_MLFLOW__TRACKING_URI", "sqlite:///mlflow.db")
os.environ.setdefault("CML_STREAM__RATE_PER_SEC", "400")

sys.path.insert(0, "src")

import uvicorn  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "continual_ml.api.prediction_service:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
