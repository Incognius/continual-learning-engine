"""MLflow bridge — experiment tracking + model lifecycle.

Logs params/metrics/artifacts to MLflow and snapshots a new *model version* on
drift (or on demand). Every MLflow interaction is wrapped so that a tracking
outage degrades gracefully (``enabled = False``) instead of taking down the
online learning loop — the model keeps learning regardless.

Backend: in v1 this points at an MLflow server (Docker) or a local SQLite store.
Both support the model registry; a bare file store does not, so version
registration is best-effort and guarded.
"""

from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

from continual_ml.config import MLflowConfig

logger = logging.getLogger("continual_ml.registry")


class ModelRegistry:
    def __init__(self, config: MLflowConfig):
        self._cfg = config
        self.enabled = False
        self.version = 0
        self._run_id: Optional[str] = None
        self._client: Optional[MlflowClient] = None

        try:
            mlflow.set_tracking_uri(config.tracking_uri)
            mlflow.set_experiment(config.experiment_name)
            self._client = MlflowClient()
            self.enabled = True
            logger.info("MLflow tracking at %s", config.tracking_uri)
        except Exception as exc:  # noqa: BLE001 - tracking must never be fatal
            logger.warning("MLflow disabled (%s): %s", config.tracking_uri, exc)

    def start_run(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            run = mlflow.start_run(run_name="continual-learning-session")
            self._run_id = run.info.run_id
            mlflow.log_params(params)
            self._ensure_registered_model()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow start_run failed: %s", exc)
            self.enabled = False

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not self.enabled:
            return
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as exc:  # noqa: BLE001
            logger.debug("MLflow log_metrics failed: %s", exc)

    def snapshot_model(self, model_obj: Any, reason: str, step: int) -> int:
        """Pickle the model, log it as an artifact, and register a new version."""
        self.version += 1
        if not self.enabled:
            return self.version
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "online_model.pkl"
                with path.open("wb") as fh:
                    pickle.dump(model_obj, fh)
                artifact_path = f"model_v{self.version}"
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
                mlflow.set_tag(f"snapshot_{self.version}_reason", reason)
                self._register_version(artifact_path, step, reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow snapshot failed: %s", exc)
        return self.version

    def end_run(self) -> None:
        if self.enabled:
            try:
                mlflow.end_run()
            except Exception:  # noqa: BLE001
                pass

    # --- internals -----------------------------------------------------------
    def _ensure_registered_model(self) -> None:
        try:
            self._client.create_registered_model(self._cfg.registered_model_name)
        except Exception:  # noqa: BLE001 - already exists is fine
            pass

    def _register_version(self, artifact_path: str, step: int, reason: str) -> None:
        try:
            source = mlflow.get_artifact_uri(artifact_path)
            mv = self._client.create_model_version(
                name=self._cfg.registered_model_name,
                source=source,
                run_id=self._run_id,
                tags={"reason": reason, "step": str(step)},
            )
            logger.info("Registered %s v%s", self._cfg.registered_model_name, mv.version)
        except Exception as exc:  # noqa: BLE001 - file store doesn't support registry
            logger.debug("Model version registration skipped: %s", exc)
