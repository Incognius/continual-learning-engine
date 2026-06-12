"""Train the deployable champion — realistically.

Instead of cold-starting an online model from zero (unrealistic for a real
deployment), this does what a real ETA team would: **batch-train on a chunk of
history, then switch to continual (online) learning** for the rest. It then
evaluates on a held-out *recent* slice (not prequential-only), saves the
deployable bundle (model + zone-pair memory), writes metrics, and registers an
MLflow version with the `champion` alias.

    python scripts/train.py                       # batch-warmup 40% + continual
    python scripts/train.py --batch-warmup-frac 0 # pure online (for comparison)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("CML_DATA_SOURCE__TYPE", "nyc_taxi")
os.environ.setdefault("CML_MLFLOW__TRACKING_URI", "sqlite:///mlflow.db")
sys.path.insert(0, "src")

from continual_ml.config import get_settings  # noqa: E402
from continual_ml.data_sources import build_data_source  # noqa: E402
from continual_ml.features.feature_pipeline import FeaturePipeline  # noqa: E402
from continual_ml.models.online_model import OnlineModel  # noqa: E402
from continual_ml.persistence import ModelBundle, load_bundle, save_bundle  # noqa: E402


def train_online(model, fp, records):
    for r in records:
        f = fp.transform(r)
        model.predict_one(f)
        model.learn_one(f, r.target)
        fp.update_target(f, r.target)


def train_batch(model, fp, records, epochs):
    idx = list(range(len(records)))
    for _ in range(epochs):
        random.shuffle(idx)
        for i in idx:
            r = records[i]
            f = fp.transform(r)
            model.predict_one(f)
            model.learn_one(f, r.target)
            fp.update_target(f, r.target)


def evaluate(model, fp, records):
    errs, ys = [], []
    seg: dict[str, list] = defaultdict(list)
    for r in records:
        f = fp.transform(r, update_stats=False)
        p = model.predict_one(f)
        e = abs(p - r.target)
        errs.append(e); ys.append(r.target)
        seg[str(f.get("pu_borough", "all"))].append(e)
    errs, ys = np.array(errs), np.array(ys)
    sst = ((ys - ys.mean()) ** 2).sum() or 1.0
    return {
        "mae": float(errs.mean()),
        "rmse": float(np.sqrt((errs ** 2).mean())),
        "r2": float(1 - (errs ** 2).sum() / sst),
        "by_borough": {b: round(float(np.mean(v)), 1) for b, v in seg.items()},
    }


def register_mlflow(settings, path, metrics):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        mlflow.set_experiment(settings.mlflow.experiment_name)
        name = settings.mlflow.registered_model_name
        with mlflow.start_run(run_name="train-champion"):
            mlflow.log_params(settings.model.model_dump())
            mlflow.log_metrics({f"holdout_{k}": v for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            mlflow.log_artifact(str(path), artifact_path="champion")
            c = MlflowClient()
            try:
                c.create_registered_model(name)
            except Exception:  # noqa: BLE001
                pass
            mv = c.create_model_version(name, mlflow.get_artifact_uri("champion"),
                                        run_id=mlflow.active_run().info.run_id)
            c.set_registered_model_alias(name, "champion", mv.version)
            return int(mv.version)
    except Exception as exc:  # noqa: BLE001
        print(f"[mlflow] skipped: {exc}")
        return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-warmup-frac", type=float, default=0.4,
                   help="fraction of all data to batch-train before going continual")
    p.add_argument("--epochs", type=int, default=2, help="epochs for the batch warm-up")
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--source", default=None)
    args = p.parse_args(argv)
    random.seed(0)

    settings = get_settings()
    if args.source:
        settings.data_source.type = args.source
    src = build_data_source(settings)
    schema = src.schema()
    recs = list(src.stream())
    n = len(recs)
    split = int(n * (1 - args.holdout_frac))
    train, holdout = recs[:split], recs[split:]
    warm = int(n * args.batch_warmup_frac)

    print(f"[train] source={schema.name} model={settings.model.type} "
          f"route_dist={'route_distance_km' in schema.feature_names}")
    print(f"[train] n={n:,}  batch-warmup={warm:,} ({args.epochs} epochs)  "
          f"continual={max(split-warm,0):,}  holdout={len(holdout):,}")

    fp = FeaturePipeline(schema, settings.features)
    model = OnlineModel(schema, settings.model)
    t0 = time.perf_counter()
    if warm > 0:
        train_batch(model, fp, train[:warm], epochs=args.epochs)
    train_online(model, fp, train[warm:])
    dt = time.perf_counter() - t0

    hold = evaluate(model, fp, holdout)
    print(f"[holdout] MAE={hold['mae']:.1f}s  RMSE={hold['rmse']:.1f}s  R2={hold['r2']:.3f}  "
          f"({dt:.0f}s)")

    artifacts = Path("artifacts"); artifacts.mkdir(exist_ok=True)
    path = artifacts / "online_model.pkl"
    bundle = ModelBundle(model=model, features=fp, schema=schema, meta={
        "trained_samples": split, "batch_warmup": warm, "epochs": args.epochs,
        "source": schema.name, "model_type": settings.model.type,
    })
    size_kb = save_bundle(bundle, path) / 1024
    version = register_mlflow(settings, path, hold)

    json.dump({
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "regime": f"batch-warmup({args.batch_warmup_frac:.0%},{args.epochs}ep)+continual",
        "source": schema.name, "model_type": settings.model.type,
        "route_distance": "route_distance_km" in schema.feature_names,
        "n_train": split, "batch_warmup": warm, "holdout": len(holdout),
        "holdout_metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in hold.items()},
        "champion_version": version, "weights_kb": round(size_kb, 1),
    }, open(artifacts / "metrics.json", "w"), indent=2)

    reloaded = load_bundle(path)
    sample = {f: (0.0 if f not in schema.categorical_features else "1")
              for f in schema.feature_names}
    print("-" * 60)
    print(f"[save ] bundle -> {path} ({size_kb:.0f} KB) · champion v{version}")
    print(f"[check] reload predict = {reloaded.predict(sample):.0f}s  (load OK)")
    print(f"[borough MAE] {hold['by_borough']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
