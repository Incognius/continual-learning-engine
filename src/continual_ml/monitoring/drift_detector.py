"""Drift detection: concept drift (online) + feature drift (windowed PSI).

* **Concept drift** — a River detector (ADWIN / Page-Hinkley) watching the live
  prediction-error stream. When the error distribution shifts, the input→target
  relationship has changed. This is genuinely online: one number in per sample.

* **Feature drift** — Population Stability Index (PSI) between a fixed reference
  window (the first ``reference_size`` samples) and a sliding current window.
  PSI is the same idea Evidently reports; implementing it directly keeps the core
  dependency-light and version-stable. Evidently can be dropped in as an
  alternative report generator without touching the engine.

PSI interpretation: < 0.1 no shift, 0.1–0.2 moderate, > 0.2 significant.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Optional

from river import drift

from continual_ml.config import ConceptDriftConfig, FeatureDriftConfig

_PSI_EPS = 1e-6


class ConceptDriftDetector:
    def __init__(self, config: ConceptDriftConfig):
        if config.detector == "adwin":
            self._detector = drift.ADWIN(delta=config.delta)
        elif config.detector == "page_hinkley":
            self._detector = drift.PageHinkley()
        elif config.detector == "ddm":
            self._detector = drift.binary.DDM()
        else:
            raise ValueError(f"Unknown concept detector '{config.detector}'")

    def update(self, error: float) -> bool:
        """Feed one prediction error; return True if drift was detected."""
        self._detector.update(error)
        return bool(self._detector.drift_detected)


class _NumericPSI:
    """PSI for a single numeric feature using fixed reference quantile bins."""

    def __init__(self, n_bins: int = 10):
        self._n_bins = n_bins
        self._edges: Optional[list[float]] = None
        self._ref_dist: Optional[list[float]] = None

    def fit_reference(self, values: list[float]) -> None:
        s = sorted(values)
        if not s:
            return
        # Quantile bin edges from the reference sample.
        edges = [s[min(int(q * len(s)), len(s) - 1)] for q in
                 [i / self._n_bins for i in range(1, self._n_bins)]]
        self._edges = edges
        self._ref_dist = self._distribution(values)

    def _bin(self, x: float) -> int:
        assert self._edges is not None
        for i, edge in enumerate(self._edges):
            if x <= edge:
                return i
        return len(self._edges)

    def _distribution(self, values: list[float]) -> list[float]:
        counts = [0] * (self._n_bins)
        for v in values:
            counts[self._bin(v)] += 1
        total = sum(counts) or 1
        return [c / total for c in counts]

    def psi(self, current: list[float]) -> float:
        if self._ref_dist is None or not current:
            return 0.0
        cur = self._distribution(current)
        total = 0.0
        for r, c in zip(self._ref_dist, cur):
            r = max(r, _PSI_EPS)
            c = max(c, _PSI_EPS)
            total += (c - r) * math.log(c / r)
        return total


class _CategoricalPSI:
    """PSI for a categorical feature using category frequencies as bins."""

    def __init__(self) -> None:
        self._ref: Optional[dict[str, float]] = None

    def fit_reference(self, values: list[str]) -> None:
        counts = Counter(values)
        total = sum(counts.values()) or 1
        self._ref = {k: v / total for k, v in counts.items()}

    def psi(self, current: list[str]) -> float:
        if not self._ref or not current:
            return 0.0
        counts = Counter(current)
        total = sum(counts.values()) or 1
        keys = set(self._ref) | set(counts)
        out = 0.0
        for k in keys:
            r = max(self._ref.get(k, 0.0), _PSI_EPS)
            c = max(counts.get(k, 0) / total, _PSI_EPS)
            out += (c - r) * math.log(c / r)
        return out


class FeatureDriftDetector:
    """Tracks PSI per feature; flags drift when any feature exceeds the threshold."""

    def __init__(
        self,
        config: FeatureDriftConfig,
        numeric_features: list[str],
        categorical_features: list[str],
    ):
        self._cfg = config
        self._numeric = numeric_features
        self._categorical = categorical_features
        self._ref_buffer: list[dict] = []
        self._reference_ready = False
        self._windows: dict[str, deque] = {
            f: deque(maxlen=config.window_size)
            for f in numeric_features + categorical_features
        }
        self._num_psi = {f: _NumericPSI() for f in numeric_features}
        self._cat_psi = {f: _CategoricalPSI() for f in categorical_features}
        self._count = 0
        self.last_psi: dict[str, float] = {}

    def update(self, features: dict) -> Optional[dict[str, float]]:
        """Add a sample; periodically returns a {feature: psi} map, else None."""
        self._count += 1

        if not self._reference_ready:
            self._ref_buffer.append(features)
            if len(self._ref_buffer) >= self._cfg.reference_size:
                self._fit_reference()
            return None

        for f in self._windows:
            if f in features:
                self._windows[f].append(features[f])

        if self._count % self._cfg.check_every != 0:
            return None
        return self._compute_psi()

    def _fit_reference(self) -> None:
        for f, p in self._num_psi.items():
            p.fit_reference([float(r[f]) for r in self._ref_buffer if f in r])
        for f, p in self._cat_psi.items():
            p.fit_reference([str(r[f]) for r in self._ref_buffer if f in r])
        self._reference_ready = True
        self._ref_buffer = []

    def _compute_psi(self) -> dict[str, float]:
        psi: dict[str, float] = {}
        for f, p in self._num_psi.items():
            psi[f] = p.psi([float(v) for v in self._windows[f]])
        for f, p in self._cat_psi.items():
            psi[f] = p.psi([str(v) for v in self._windows[f]])
        self.last_psi = psi
        return psi

    def is_drift(self, psi: dict[str, float]) -> bool:
        return any(v > self._cfg.psi_threshold for v in psi.values())
