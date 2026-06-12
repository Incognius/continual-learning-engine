import itertools

from continual_ml.config import SyntheticConfig
from continual_ml.data_sources.synthetic_data_source import SyntheticDataSource
from continual_ml.schemas import Record


def test_synthetic_schema_and_records():
    src = SyntheticDataSource(SyntheticConfig(n_features=4, seed=1), max_samples=100)
    schema = src.schema()
    assert schema.task == "regression"
    assert schema.feature_names == ["x0", "x1", "x2", "x3"]
    assert schema.categorical_features == []

    records = list(src.stream())
    assert len(records) == 100
    assert all(isinstance(r, Record) for r in records)
    assert all(r.target is not None for r in records)
    assert all(set(r.features) == set(schema.feature_names) for r in records)


def test_synthetic_is_deterministic():
    cfg = SyntheticConfig(n_features=3, seed=7)
    a = list(itertools.islice(SyntheticDataSource(cfg, max_samples=20).stream(), 20))
    b = list(itertools.islice(SyntheticDataSource(cfg, max_samples=20).stream(), 20))
    assert [r.target for r in a] == [r.target for r in b]


def test_synthetic_injects_concept_drift():
    cfg = SyntheticConfig(n_features=3, seed=3)
    cfg.drift.enabled = True
    cfg.drift.at_sample = 50
    records = list(itertools.islice(SyntheticDataSource(cfg, max_samples=60).stream(), 60))
    assert records[49].metadata["drifted"] is False
    assert records[55].metadata["drifted"] is True
