import random

from continual_ml.config import ConceptDriftConfig, FeatureDriftConfig
from continual_ml.monitoring.drift_detector import (
    ConceptDriftDetector,
    FeatureDriftDetector,
)


def test_concept_drift_detects_error_shift():
    det = ConceptDriftDetector(ConceptDriftConfig(detector="adwin", delta=0.002))
    rng = random.Random(0)
    detected = False
    # Stable low error, then a sustained jump.
    for _ in range(1000):
        detected |= det.update(abs(rng.gauss(0, 0.1)))
    for _ in range(1000):
        detected |= det.update(abs(rng.gauss(5, 0.1)))
    assert detected


def test_feature_drift_psi_flags_distribution_shift():
    cfg = FeatureDriftConfig(reference_size=300, window_size=300, check_every=50, psi_threshold=0.2)
    det = FeatureDriftDetector(cfg, numeric_features=["f"], categorical_features=[])
    rng = random.Random(1)

    # Reference distribution.
    for _ in range(300):
        det.update({"f": rng.gauss(0.0, 1.0)})
    # Shifted distribution should drive PSI up.
    last = {}
    for _ in range(600):
        out = det.update({"f": rng.gauss(6.0, 1.0)})
        if out:
            last = out
    assert last
    assert last["f"] > 0.2
    assert det.is_drift(last)


def test_feature_drift_quiet_when_stable():
    cfg = FeatureDriftConfig(reference_size=300, window_size=300, check_every=50, psi_threshold=0.2)
    det = FeatureDriftDetector(cfg, numeric_features=["f"], categorical_features=[])
    rng = random.Random(2)
    out = {}
    for _ in range(900):
        res = det.update({"f": rng.gauss(0.0, 1.0)})
        if res:
            out = res
    # Same distribution => low PSI, no drift.
    assert out
    assert not det.is_drift(out)
