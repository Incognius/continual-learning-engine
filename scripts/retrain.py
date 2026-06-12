"""Scheduled rolling-window retrain with shadow-validate-before-promote.

This is the *weekly* tier of the deployment cadence (see REPORT.md §Operations):
continuous online learning keeps the live model fresh per-trip, while this job
periodically trains a fresh candidate on a rolling recent window and promotes it
to champion ONLY if it beats the incumbent on a held-out recent slice. That
bounds drift accumulation and pulls in structural changes (new zones, holidays,
construction) without ever shipping a regression.

    python scripts/retrain.py --window-frac 0.8 --val-frac 0.1 --margin 0.0

Flow:
  records (chronological)
    └─ holdout = most recent val_frac           (validation, unseen)
    └─ window  = last window_frac before holdout (candidate training data)
  train candidate on window  ->  evaluate on holdout
  load champion              ->  evaluate on holdout
  promote candidate iff  cand_MAE < champ_MAE * (1 - margin)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CML_DATA_SOURCE__TYPE", "nyc_taxi")
os.environ.setdefault("CML_MLFLOW__TRACKING_URI", "sqlite:///mlflow.db")
sys.path.insert(0, "src")

from continual_ml.config import get_settings  # noqa: E402
from continual_ml.data_sources import build_data_source  # noqa: E402
from continual_ml.features.feature_pipeline import FeaturePipeline  # noqa: E402
from continual_ml.models.online_model import OnlineModel  # noqa: E402
from continual_ml.persistence import ModelBundle, load_bundle, save_bundle  # noqa: E402

CHAMPION = Path("artifacts/online_model.pkl")


def _evaluate(model: OnlineModel, features: FeaturePipeline, records) -> dict:
    """Frozen evaluation (no learning) on a holdout, with per-borough MAE."""
    errs, ys = [], []
    seg: dict[str, list] = defaultdict(list)
    for r in records:
        f = features.transform(r, update_stats=False)
        p = model.predict_one(f)
        e = abs(p - r.target)
        errs.append(e)
        ys.append(r.target)
        seg[str(f.get("pu_borough", "all"))].append(e)
    n = len(errs)
    mean = sum(ys) / n
    sst = sum((y - mean) ** 2 for y in ys) or 1.0
    sse = sum(e * e for e in errs)
    return {
        "mae": sum(errs) / n,
        "rmse": math.sqrt(sse / n),
        "r2": 1 - sse / sst,
        "n": n,
        "by_borough": {b: round(sum(v) / len(v), 1) for b, v in seg.items()},
    }


def _train_candidate(settings, schema, records) -> tuple[OnlineModel, FeaturePipeline]:
    features = FeaturePipeline(schema, settings.features)
    model = OnlineModel(schema, settings.model)
    for r in records:
        f = features.transform(r)
        model.predict_one(f)
        model.learn_one(f, r.target)
        features.update_target(f, r.target)
    return model, features


def _register_champion(settings, bundle_path: Path, metrics: dict) -> None:
    """Best-effort MLflow registration + 'champion' alias for the promoted model."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        mlflow.set_experiment(settings.mlflow.experiment_name)
        name = settings.mlflow.registered_model_name
        with mlflow.start_run(run_name="rolling-retrain-promote"):
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            mlflow.log_artifact(str(bundle_path), artifact_path="champion")
            client = MlflowClient()
            try:
                client.create_registered_model(name)
            except Exception:  # noqa: BLE001
                pass
            source = mlflow.get_artifact_uri("champion")
            mv = client.create_model_version(name=name, source=source,
                                             run_id=mlflow.active_run().info.run_id)
            client.set_registered_model_alias(name, "champion", mv.version)
            print(f"[mlflow] registered {name} v{mv.version} alias=champion")
    except Exception as exc:  # noqa: BLE001
        print(f"[mlflow] registration skipped: {exc}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window-frac", type=float, default=0.8,
                   help="rolling training window as a fraction of pre-holdout data")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="most-recent fraction held out for validation")
    p.add_argument("--margin", type=float, default=0.0,
                   help="candidate must beat champion MAE by this fraction to promote")
    args = p.parse_args(argv)

    settings = get_settings()
    settings.engine.warm_start_path = None
    source = build_data_source(settings)
    schema = source.schema()

    records = list(source.stream())
    n = len(records)
    val_start = int(n * (1 - args.val_frac))
    holdout = records[val_start:]
    pre = records[:val_start]
    window_start = int(len(pre) * (1 - args.window_frac))
    window = pre[window_start:]
    print(f"[retrain] total={n:,}  window={len(window):,}  holdout={len(holdout):,}")

    t0 = time.perf_counter()
    cand_model, cand_features = _train_candidate(settings, schema, window)
    cand = _evaluate(cand_model, cand_features, holdout)
    print(f"[candidate] MAE={cand['mae']:.1f}s  R2={cand['r2']:.3f}  ({time.perf_counter()-t0:.0f}s)")

    if CHAMPION.exists():
        champ_bundle = load_bundle(CHAMPION)
        champ = _evaluate(champ_bundle.model, champ_bundle.features, holdout)
        print(f"[champion ] MAE={champ['mae']:.1f}s  R2={champ['r2']:.3f}")
    else:
        champ = None
        print("[champion ] none on disk")

    promote = champ is None or cand["mae"] < champ["mae"] * (1 - args.margin)
    decision = "PROMOTE" if promote else "KEEP CHAMPION"
    print(f"[decision ] {decision}")

    if promote:
        bundle = ModelBundle(
            model=cand_model, features=cand_features, schema=schema,
            meta={"trained_samples": len(window), "source": schema.name,
                  "promoted_at": datetime.now(timezone.utc).isoformat()},
        )
        size_kb = save_bundle(bundle, CHAMPION) / 1024
        print(f"[promote  ] champion -> {CHAMPION} ({size_kb:.0f} KB)")
        _register_champion(settings, CHAMPION, cand)

    report = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "window_samples": len(window),
        "holdout_samples": len(holdout),
        "candidate": cand,
        "champion": champ,
        "margin": args.margin,
        "decision": decision,
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/retrain_report.json").write_text(json.dumps(report, indent=2))
    print("[report  ] artifacts/retrain_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
