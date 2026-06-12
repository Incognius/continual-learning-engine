"""Prometheus instrumentation.

All instruments live in a private ``CollectorRegistry`` so the ``/metrics``
endpoint exposes exactly our platform metrics (no duplicate-registration issues
on reload). Prometheus scrapes ``/metrics``; Grafana visualizes.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class MetricsCollector:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()

        self.samples_total = Counter(
            "cml_samples_processed_total",
            "Total records processed by the engine.",
            registry=self.registry,
        )
        self.predictions_total = Counter(
            "cml_predictions_total",
            "Total predictions served.",
            registry=self.registry,
        )
        self.prediction_latency = Histogram(
            "cml_prediction_latency_seconds",
            "Prediction latency in seconds.",
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
            registry=self.registry,
        )
        self.performance = Gauge(
            "cml_model_performance",
            "Rolling model performance.",
            ["metric"],
            registry=self.registry,
        )
        self.feature_drift = Gauge(
            "cml_feature_drift_psi",
            "Population Stability Index per feature.",
            ["feature"],
            registry=self.registry,
        )
        self.concept_drift_total = Counter(
            "cml_concept_drift_events_total",
            "Number of concept-drift events detected.",
            registry=self.registry,
        )
        self.feature_drift_total = Counter(
            "cml_feature_drift_events_total",
            "Number of feature-drift events detected.",
            registry=self.registry,
        )
        self.model_version = Gauge(
            "cml_model_version",
            "Deployed model version (changed only by the promote flow).",
            registry=self.registry,
        )
        self.retrain_recommended = Gauge(
            "cml_retrain_recommended",
            "1 if a validated degradation has recommended a retrain.",
            registry=self.registry,
        )
        self.spurious_drift_total = Counter(
            "cml_spurious_drift_events_total",
            "ADWIN triggers that failed before/after validation (anomalies).",
            registry=self.registry,
        )
        self.target_value = Gauge(
            "cml_target_value",
            "Most recent observed target / prediction.",
            ["kind"],
            registry=self.registry,
        )
        self.segment_mae = Gauge(
            "cml_segment_rolling_mae",
            "Rolling MAE per segment (e.g. borough, hour-of-day).",
            ["kind", "segment"],
            registry=self.registry,
        )

    # --- recording helpers ---------------------------------------------------
    def observe_prediction(self, latency_s: float) -> None:
        self.predictions_total.inc()
        self.prediction_latency.observe(latency_s)

    def record_sample(self) -> None:
        self.samples_total.inc()

    def set_performance(self, snapshot: dict[str, float]) -> None:
        for name in ("mae", "rmse", "r2", "rolling_mae", "rolling_rmse", "rolling_r2"):
            if name in snapshot:
                self.performance.labels(metric=name).set(snapshot[name])

    def set_feature_drift(self, psi: dict[str, float]) -> None:
        for feature, value in psi.items():
            self.feature_drift.labels(feature=feature).set(value)

    def record_concept_drift(self) -> None:
        self.concept_drift_total.inc()

    def record_feature_drift(self) -> None:
        self.feature_drift_total.inc()

    def set_model_version(self, version: int) -> None:
        self.model_version.set(version)

    def set_retrain_recommended(self, on: bool) -> None:
        self.retrain_recommended.set(1.0 if on else 0.0)

    def record_spurious_drift(self) -> None:
        self.spurious_drift_total.inc()

    def set_target(self, observed: float, predicted: float) -> None:
        self.target_value.labels(kind="observed").set(observed)
        self.target_value.labels(kind="predicted").set(predicted)

    def set_segment_mae(self, kind: str, segment: str, mae: float) -> None:
        self.segment_mae.labels(kind=kind, segment=segment).set(mae)

    def expose(self) -> bytes:
        return generate_latest(self.registry)
