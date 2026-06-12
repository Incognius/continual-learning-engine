import math

from continual_ml.config import FeaturesConfig
from continual_ml.features.feature_pipeline import FeaturePipeline
from continual_ml.schemas import Record, SourceSchema


def _schema():
    return SourceSchema(
        name="t", task="regression", target_name="y",
        feature_names=["a", "cat"], categorical_features=["cat"],
    )


def test_online_target_encoding_is_leakage_free_and_learns():
    cfg = FeaturesConfig(target_encoders=[["cat"]], te_smoothing=1.0,
                         te_horizons={"long": 0.0, "short": 0.5})
    fp = FeaturePipeline(_schema(), cfg)

    # First time we see cat='x' the encoder is empty -> falls back to global mean.
    feats = fp.transform(Record(record_id="1", features={"a": 1.0, "cat": "x"}))
    assert feats["te_cat_long"] == 0.0   # global mean starts at 0, no leakage of y
    assert feats["te_cat_short"] == 0.0

    # After observing labels for 'x', both horizons move toward their mean.
    for y in (100.0, 100.0, 100.0, 100.0):
        f = fp.transform(Record(record_id="i", features={"a": 1.0, "cat": "x"}))
        fp.update_target(f, y)
    est = fp.transform(Record(record_id="2", features={"a": 1.0, "cat": "x"}))
    assert 50.0 < est["te_cat_long"] <= 100.0   # learned, shrunk toward global
    assert 50.0 < est["te_cat_short"] <= 100.0


def test_multi_horizon_short_reacts_faster_than_long():
    cfg = FeaturesConfig(target_encoders=[["cat"]], te_smoothing=0.0,
                         te_horizons={"long": 0.0, "short": 0.5})
    fp = FeaturePipeline(_schema(), cfg)
    # Warm both horizons at 100, then a sudden regime change to 300.
    for _ in range(50):
        f = fp.transform(Record(record_id="i", features={"a": 1.0, "cat": "x"}))
        fp.update_target(f, 100.0)
    for _ in range(5):
        f = fp.transform(Record(record_id="i", features={"a": 1.0, "cat": "x"}))
        fp.update_target(f, 300.0)
    out = fp.transform(Record(record_id="2", features={"a": 1.0, "cat": "x"}))
    # Short memory should have moved toward 300 far more than long memory.
    assert out["te_cat_short"] > out["te_cat_long"]


def test_target_encoder_skipped_when_columns_absent():
    cfg = FeaturesConfig(target_encoders=[["pu_zone", "do_zone"]])
    fp = FeaturePipeline(_schema(), cfg)
    feats = fp.transform(Record(record_id="1", features={"a": 1.0, "cat": "x"}))
    assert not any(k.startswith("te_pu_zone_do_zone") for k in feats)  # skipped


def test_zone_geo_features():
    from continual_ml.geo import ZoneGeo
    from pathlib import Path
    import pytest

    path = "data/zone_centroids.csv"
    if not Path(path).exists():
        pytest.skip("zone centroids not prepared")
    geo = ZoneGeo(path)
    assert geo.loaded_zones > 200
    f = geo.features("132", "138")  # JFK -> LaGuardia
    assert f["gc_distance_km"] > 10  # the two airports are far apart
    assert f["pu_airport"] == 1.0 and f["do_airport"] == 1.0
    assert abs(f["bearing_sin"] ** 2 + f["bearing_cos"] ** 2 - 1.0) < 1e-3  # 4dp rounding
