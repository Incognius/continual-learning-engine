"""Baselines, batch ceiling, and the cost of continual learning.

All setups are scored on the SAME chronological holdout (last 20% of the data):

  floor:    global-mean predictor; pure zone-pair memory predictor
  ablation: online model WITHOUT vs WITH the route-distance feature
  ceiling:  batch (multi-epoch over shuffled train) — the static upper bound
  ours:     online single-pass (continual)
  realistic: batch-warmup on the first 40%, then continual on the rest

The batch − online gap is the price we pay for continual adaptation.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np

os.environ.setdefault("CML_DATA_SOURCE__TYPE", "nyc_taxi")
sys.path.insert(0, "src")

from continual_ml.config import get_settings  # noqa: E402
from continual_ml.data_sources import build_data_source  # noqa: E402
from continual_ml.features.feature_pipeline import FeaturePipeline  # noqa: E402
from continual_ml.models.online_model import OnlineModel  # noqa: E402

BASE_IGNORE = ["pu_zone", "do_zone"]


def metrics(model, fp, records, drop_route):
    errs, ys = [], []
    for r in records:
        f = fp.transform(r, update_stats=False)
        if drop_route:
            f.pop("route_distance_km", None)
        p = model.predict_one(f)
        errs.append(abs(p - r.target))
        ys.append(r.target)
    errs, ys = np.array(errs), np.array(ys)
    sst = ((ys - ys.mean()) ** 2).sum()
    return errs.mean(), 1 - (errs ** 2).sum() / sst


def make_model(settings, schema, use_route):
    cfg = settings.model.model_copy(deep=True)
    cfg.ignore_features = BASE_IGNORE + ([] if use_route else ["route_distance_km"])
    return OnlineModel(schema, cfg)


def train_online(model, fp, records, drop_route):
    for r in records:
        f = fp.transform(r)
        if drop_route:
            f.pop("route_distance_km", None)
        model.predict_one(f)
        model.learn_one(f, r.target)
        fp.update_target(f, r.target)


def train_batch(model, fp, records, drop_route, epochs):
    idx = list(range(len(records)))
    for _ in range(epochs):
        random.shuffle(idx)
        for i in idx:
            r = records[i]
            f = fp.transform(r)
            if drop_route:
                f.pop("route_distance_km", None)
            model.predict_one(f)
            model.learn_one(f, r.target)
            fp.update_target(f, r.target)


def main() -> int:
    random.seed(0)
    settings = get_settings()
    src = build_data_source(settings)
    schema = src.schema()
    recs = list(src.stream())
    n = len(recs)
    split = int(n * 0.8)
    train, holdout = recs[:split], recs[split:]
    print(f"data n={n:,}  train={len(train):,}  holdout={len(holdout):,}\n")

    rows = []

    # --- floor 1: global mean ----------------------------------------------
    mean = np.mean([r.target for r in train])
    err = np.mean([abs(mean - r.target) for r in holdout])
    rows.append(("global-mean (floor)", err, 0.0))

    # --- floor 2: pure zone-pair memory ------------------------------------
    fp = FeaturePipeline(schema, settings.features)
    for r in train:
        fp.update_target(fp.transform(r), r.target)
    errs, ys = [], []
    for r in holdout:
        f = fp.transform(r, update_stats=False)
        p = f.get("te_pu_zone_do_zone_long", mean)
        errs.append(abs(p - r.target)); ys.append(r.target)
    errs, ys = np.array(errs), np.array(ys)
    rows.append(("zone-pair memory only", errs.mean(),
                 1 - (errs ** 2).sum() / ((ys - ys.mean()) ** 2).sum()))

    # --- online, no route distance -----------------------------------------
    fp = FeaturePipeline(schema, settings.features)
    m = make_model(settings, schema, use_route=False)
    train_online(m, fp, train, drop_route=True)
    rows.append(("online, NO route dist", *metrics(m, fp, holdout, drop_route=True)))

    # --- online, with route distance (continual = ours) --------------------
    fp = FeaturePipeline(schema, settings.features)
    m = make_model(settings, schema, use_route=True)
    train_online(m, fp, train, drop_route=False)
    rows.append(("online +route (OURS)", *metrics(m, fp, holdout, drop_route=False)))

    # --- batch ceiling (2 epochs, shuffled) --------------------------------
    fp = FeaturePipeline(schema, settings.features)
    m = make_model(settings, schema, use_route=True)
    train_batch(m, fp, train, drop_route=False, epochs=2)
    rows.append(("batch 2-epoch +route (ceiling)", *metrics(m, fp, holdout, drop_route=False)))

    # --- realistic: batch-warmup 40% then continual ------------------------
    warm = int(len(train) * 0.5)  # 40% of all data = 50% of train
    fp = FeaturePipeline(schema, settings.features)
    m = make_model(settings, schema, use_route=True)
    train_batch(m, fp, train[:warm], drop_route=False, epochs=2)   # batch warm-up
    train_online(m, fp, train[warm:], drop_route=False)            # then continual
    rows.append(("batch-warmup + continual +route", *metrics(m, fp, holdout, drop_route=False)))

    print(f"{'setup':34s} {'MAE(s)':>9} {'R2':>7}")
    print("-" * 54)
    for name, mae, r2 in rows:
        print(f"{name:34s} {mae:9.1f} {r2:7.3f}")

    online_r2 = rows[3][2]
    batch_r2 = rows[4][2]
    print(f"\ncontinual-learning cost (batch R2 - online R2) = {batch_r2 - online_r2:+.3f}")
    print(f"route-distance lift (R2) = {rows[3][2] - rows[2][2]:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
