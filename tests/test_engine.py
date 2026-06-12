import itertools

from continual_ml.config import get_settings
from continual_ml.core.engine import ContinualLearningEngine
from continual_ml.data_sources import build_data_source


def _engine_and_source():
    settings = get_settings()
    settings.data_source.type = "synthetic"
    settings.data_source.synthetic.n_features = 5
    # Offline registry for tests.
    settings.mlflow.tracking_uri = "sqlite:///test_mlflow.db"
    source = build_data_source(settings)
    engine = ContinualLearningEngine(settings, source.schema())
    return engine, source


def test_engine_learns_and_improves():
    engine, source = _engine_and_source()
    engine.start()

    errors_early, errors_late = [], []
    for i, rec in enumerate(itertools.islice(source.stream(), 4000)):
        engine.process(rec)
        if 200 <= i < 400:
            errors_early.append(engine.evaluation.rolling_mae)
        if 3800 <= i < 4000:
            errors_late.append(engine.evaluation.rolling_mae)
    engine.stop()

    stats = engine.stats()
    assert stats["labeled_samples"] == 4000
    assert stats["performance"]["mae"] > 0
    # The online model should reduce error over the stream.
    assert sum(errors_late) / len(errors_late) < sum(errors_early) / len(errors_early)


def test_engine_predict_is_inference_only():
    engine, _ = _engine_and_source()
    engine.start()
    before = engine._index
    pred = engine.predict({f"x{i}": 0.5 for i in range(5)})
    engine.stop()
    assert isinstance(pred.prediction, float)
    assert pred.latency_ms >= 0
    # Ad-hoc prediction must not advance the learning index.
    assert engine._index == before
