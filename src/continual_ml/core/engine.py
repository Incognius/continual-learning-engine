"""The continual-learning engine — the orchestrator and heart of the platform.

Implements the prequential (test-then-train) loop and owns every component:
feature pipeline, online model, metrics, drift detectors, and the MLflow
registry. The API layer is intentionally thin: it pushes records in and reads
stats out; all the ML logic lives here.

Per labeled record the loop is:
    features  ->  predict  ->  evaluate (BEFORE learning)  ->  drift checks
              ->  learn (online update)  ->  publish metrics
On concept drift, a new model version is snapshotted to MLflow.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Optional

from continual_ml.config import Settings
from continual_ml.core.evaluation import RegressionMetrics
from continual_ml.features.feature_pipeline import FeaturePipeline
from continual_ml.models.model_registry import ModelRegistry
from continual_ml.models.online_model import OnlineModel
from continual_ml.monitoring.drift_detector import (
    ConceptDriftDetector,
    FeatureDriftDetector,
)
from continual_ml.monitoring.metrics_collector import MetricsCollector
from continual_ml.schemas import DriftEvent, DriftType, Prediction, Record, SourceSchema

logger = logging.getLogger("continual_ml.engine")


class ContinualLearningEngine:
    def __init__(self, settings: Settings, schema: SourceSchema):
        self._settings = settings
        self._schema = schema

        self.features = FeaturePipeline(schema, settings.features)
        self.model = OnlineModel(schema, settings.model)
        self.warm_started = self._maybe_warm_start(settings.engine.warm_start_path, schema)
        self.metrics = MetricsCollector()
        self.evaluation = RegressionMetrics(window=settings.model.rolling_window)
        self.registry = ModelRegistry(settings.mlflow)

        self.concept_drift = ConceptDriftDetector(settings.drift.concept)
        self.feature_drift = FeatureDriftDetector(
            settings.drift.feature,
            numeric_features=self.features.numeric_features,
            categorical_features=self.features.categorical_features,
        )

        # State + live buffers for the frontend.
        self._index = 0
        self._last_alert = -(10**9)
        self.concept_drift_count = 0      # validated concept drifts
        self.spurious_drift_count = 0     # ADWIN triggers that failed validation
        self.feature_drift_count = 0
        self.retrain_recommended = False  # set by a validated degradation
        self.model_version = 1 if self.warm_started else 0
        self.last_prediction = 0.0
        self.last_target = 0.0
        self._err_buffer: deque[float] = deque(
            maxlen=2 * settings.drift.concept.validation_window
        )

        buf = settings.engine.recent_buffer
        self.recent: deque[dict] = deque(maxlen=buf)
        self.history: deque[dict] = deque(maxlen=400)
        self.drift_events: deque[dict] = deque(maxlen=50)
        self.stream_status: dict[str, Any] = {}

        # Per-segment rolling error (catches localized degradation, e.g. a model
        # that looks fine overall but is bad on airport runs or at 5pm).
        self._seg_borough: dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))
        self._seg_hour: dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))

    def _maybe_warm_start(self, path: Optional[str], schema: SourceSchema) -> bool:
        """Load a champion bundle if present and schema-compatible."""
        if not path:
            return False
        from pathlib import Path

        if not Path(path).exists():
            return False
        try:
            from continual_ml.persistence import load_bundle

            bundle = load_bundle(path)
            if bundle.schema.name != schema.name:
                logger.info("Ignoring warm-start %s (schema mismatch)", path)
                return False
            self.model = bundle.model
            self.features = bundle.features
            logger.info("Warm-started from champion %s", path)
            return True
        except Exception as exc:  # noqa: BLE001 - never block startup on this
            logger.warning("Warm-start failed (%s): %s", path, exc)
            return False

    # --- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self.registry.start_run(self.model.params())
        self.metrics.set_model_version(self.model_version)
        logger.info("Engine started for source '%s'", self._schema.name)

    def stop(self) -> None:
        self.registry.end_run()

    # --- ad-hoc inference (no learning) -------------------------------------
    def predict(self, raw_features: dict[str, Any]) -> Prediction:
        record = Record(record_id=f"adhoc-{self._index}", features=raw_features)
        t0 = time.perf_counter()
        feats = self.features.transform(record, update_stats=False)
        value = self.model.predict_one(feats)
        latency = time.perf_counter() - t0
        self.metrics.observe_prediction(latency)
        return Prediction(
            record_id=record.record_id,
            prediction=value,
            model_version=str(self.model_version),
            latency_ms=latency * 1000.0,
        )

    # --- the prequential loop ------------------------------------------------
    def process(self, record: Record) -> Prediction:
        t0 = time.perf_counter()
        feats = self.features.transform(record, update_stats=True)
        value = self.model.predict_one(feats)
        latency = time.perf_counter() - t0

        self.metrics.record_sample()
        self.metrics.observe_prediction(latency)
        self.predictions_seen = self._index
        self.last_prediction = value

        if record.target is not None:
            y = float(record.target)
            self.last_target = y
            self.evaluation.update(y, value)
            self._update_segments(feats, abs(y - value))
            self._check_concept_drift(y, value)
            self._check_feature_drift(feats)
            self.model.learn_one(feats, y)
            # Update online target encoders AFTER learning (leakage-free).
            self.features.update_target(feats, y)
            self._publish(y, value)

        self._record_point(record, value)
        self._index += 1
        return Prediction(
            record_id=record.record_id,
            prediction=value,
            model_version=str(self.model_version),
            latency_ms=latency * 1000.0,
        )

    # --- drift handling ------------------------------------------------------
    def _check_concept_drift(self, y_true: float, y_pred: float) -> None:
        error = abs(y_true - y_pred)
        self._err_buffer.append(error)
        if not self.concept_drift.update(error):
            return

        # Validate the ADWIN trigger: compare mean error in the window before vs
        # after the event. Only a material, sized shift is a *real* drift; the
        # rest are anomalies (a transient spike, not a distribution change).
        validated, rel_shift, direction = self._validate_drift()
        if not validated:
            self.spurious_drift_count += 1
            self.metrics.record_spurious_drift()
            self._log_event(DriftEvent(
                drift_type=DriftType.CONCEPT, sample_index=self._index,
                detector=self._settings.drift.concept.detector,
                details={"rel_shift": round(rel_shift, 3)},
                action_taken="not validated (anomaly)",
            ))
            return

        self.concept_drift_count += 1
        self.metrics.record_concept_drift()
        action = f"validated {direction} {abs(rel_shift)*100:.0f}%"
        # A validated *degradation* recommends a retrain (decision left to the
        # promote flow). We never auto-version here — a drift is not a better model.
        if direction == "worse":
            cooldown = self._settings.engine.retrain_alert_cooldown
            if self._index - self._last_alert >= cooldown:
                self.retrain_recommended = True
                self._last_alert = self._index
                self.metrics.set_retrain_recommended(True)
                action += " -> RETRAIN RECOMMENDED"
        self._log_event(DriftEvent(
            drift_type=DriftType.CONCEPT, sample_index=self._index,
            detector=self._settings.drift.concept.detector,
            score=rel_shift, details={"rel_shift": round(rel_shift, 3)},
            action_taken=action,
        ))

    def _validate_drift(self) -> tuple[bool, float, str]:
        """Before/after window test on the recent error buffer."""
        w = self._settings.drift.concept.validation_window
        if len(self._err_buffer) < 2 * w:
            return False, 0.0, "insufficient"
        buf = list(self._err_buffer)
        before, after = buf[-2 * w:-w], buf[-w:]
        mb = sum(before) / w
        ma = sum(after) / w
        var_b = sum((x - mb) ** 2 for x in before) / w
        var_a = sum((x - ma) ** 2 for x in after) / w
        pooled = ((var_b + var_a) / 2) ** 0.5 + 1e-9
        rel = (ma - mb) / mb if mb > 1e-9 else 0.0
        effect = abs(ma - mb) / pooled
        cfg = self._settings.drift.concept
        valid = abs(rel) >= cfg.validation_min_rel_shift and effect >= cfg.validation_min_effect
        return valid, rel, ("worse" if ma > mb else "better")

    def _check_feature_drift(self, feats: dict) -> None:
        psi = self.feature_drift.update(feats)
        if psi is None:
            return
        self.metrics.set_feature_drift(psi)
        if self.feature_drift.is_drift(psi):
            self.feature_drift_count += 1
            self.metrics.record_feature_drift()
            worst = max(psi, key=psi.get)
            event = DriftEvent(
                drift_type=DriftType.FEATURE,
                sample_index=self._index,
                detector="psi",
                score=psi[worst],
                details={"feature": worst, "psi": round(psi[worst], 3)},
            )
            self._log_event(event)

    def _update_segments(self, feats: dict, error: float) -> None:
        borough = str(feats.get("pu_borough", "all"))
        self._seg_borough[borough].append(error)
        if "pickup_hour" in feats:
            self._seg_hour[str(int(feats["pickup_hour"]))].append(error)

    def _segment_maes(self, segments: dict[str, deque]) -> dict[str, float]:
        return {k: round(sum(v) / len(v), 1) for k, v in segments.items() if v}

    def _log_event(self, event: DriftEvent) -> None:
        self.drift_events.appendleft(
            {
                "type": event.drift_type.value,
                "index": event.sample_index,
                "detector": event.detector,
                "score": event.score,
                "details": event.details,
                "action": event.action_taken,
                "at": event.detected_at.isoformat(),
            }
        )
        logger.info("DRIFT %s @ %s %s", event.drift_type.value, event.sample_index, event.details)

    # --- publishing ----------------------------------------------------------
    def _publish(self, y_true: float, y_pred: float) -> None:
        snap = self.evaluation.snapshot()
        self.metrics.set_performance(snap)
        self.metrics.set_target(y_true, y_pred)
        if self._index % self._settings.engine.log_every == 0:
            self.registry.log_metrics(
                {
                    "mae": snap["mae"],
                    "rmse": snap["rmse"],
                    "r2": snap["r2"],
                    "rolling_mae": snap["rolling_mae"],
                    "concept_drift_events": float(self.concept_drift_count),
                    "feature_drift_events": float(self.feature_drift_count),
                },
                step=self._index,
            )
        if self._index % 100 == 0:
            self.history.append(
                {
                    "index": self._index,
                    "rolling_mae": round(snap["rolling_mae"], 3),
                    "rolling_rmse": round(snap["rolling_rmse"], 3),
                    "rolling_r2": round(snap["rolling_r2"], 3),
                }
            )
            for borough, mae in self._segment_maes(self._seg_borough).items():
                self.metrics.set_segment_mae("borough", borough, mae)
            for hour, mae in self._segment_maes(self._seg_hour).items():
                self.metrics.set_segment_mae("hour", hour, mae)

    def _record_point(self, record: Record, value: float) -> None:
        self.recent.append(
            {
                "index": self._index,
                "ts": record.timestamp.isoformat(),
                "actual": None if record.target is None else round(float(record.target), 2),
                "predicted": round(value, 2),
            }
        )

    # --- snapshot for the API/frontend --------------------------------------
    def stats(self) -> dict[str, Any]:
        snap = self.evaluation.snapshot()
        return {
            "source": self._schema.name,
            "task": self._schema.task,
            "target_name": self._schema.target_name,
            "model_type": self._settings.model.type,
            "schema": {
                "numeric": self.features.numeric_features,
                "categorical": self.features.categorical_features,
            },
            "samples_processed": self._index,
            "labeled_samples": int(snap["n"]),
            "performance": {k: round(v, 4) for k, v in snap.items()},
            "model_version": self.model_version,
            "mlflow_enabled": self.registry.enabled,
            "drift": {
                "concept_events": self.concept_drift_count,       # validated only
                "spurious_events": self.spurious_drift_count,      # failed validation
                "feature_events": self.feature_drift_count,
                "retrain_recommended": self.retrain_recommended,
                "feature_psi": {k: round(v, 4) for k, v in self.feature_drift.last_psi.items()},
            },
            "last": {"actual": round(self.last_target, 2), "predicted": round(self.last_prediction, 2)},
            "segments": {
                "borough": self._segment_maes(self._seg_borough),
                "hour": self._segment_maes(self._seg_hour),
            },
            "stream": self.stream_status,
            "recent": list(self.recent),
            "history": list(self.history),
            "drift_events": list(self.drift_events),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
