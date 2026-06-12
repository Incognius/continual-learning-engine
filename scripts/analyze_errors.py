"""Diagnostic: is there exploitable structure left in the residuals?

Evaluates the champion model over the dataset and looks for:
  * diurnal / weekly periodicity in error (systematic bias by hour / weekday)
  * temporal autocorrelation of residuals (traffic "regimes" the model misses)
  * the error tail (how much is irreducible outlier noise)

These findings drive the modeling discussion (do we need boosting / a recent-
residual feature, or is the remaining error mostly irreducible?).
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

os.environ.setdefault("CML_DATA_SOURCE__TYPE", "nyc_taxi")
sys.path.insert(0, "src")

from continual_ml.config import get_settings  # noqa: E402
from continual_ml.data_sources import build_data_source  # noqa: E402
from continual_ml.persistence import load_bundle  # noqa: E402


def main() -> int:
    b = load_bundle("artifacts/online_model.pkl")
    src = build_data_source(get_settings())
    recs = list(src.stream())

    resid, abserr = [], []
    by_hour: dict[int, list] = defaultdict(list)
    by_dow: dict[int, list] = defaultdict(list)
    for r in recs:
        f = b.features.transform(r, update_stats=False)
        p = b.model.predict_one(f)
        e = p - r.target
        resid.append(e)
        abserr.append(abs(e))
        by_hour[int(r.features["pickup_hour"])].append(e)
        by_dow[int(r.features["pickup_dayofweek"])].append(e)

    resid = np.array(resid)
    abserr = np.array(abserr)

    print(f"n={len(resid):,}  MAE={abserr.mean():.1f}s  RMSE={np.sqrt((resid**2).mean()):.1f}s")
    print(f"overall mean residual (bias) = {resid.mean():+.1f}s")
    print()

    print("--- diurnal: mean signed residual (bias) and MAE per pickup hour ---")
    for h in range(24):
        if by_hour[h]:
            arr = np.array(by_hour[h])
            print(f"  {h:02d}h  bias={arr.mean():+6.1f}s  MAE={np.abs(arr).mean():6.1f}s  n={len(arr):>6}")

    print("--- weekly: mean signed residual and MAE per weekday (0=Mon) ---")
    for d in range(7):
        if by_dow[d]:
            arr = np.array(by_dow[d])
            print(f"  dow{d}  bias={arr.mean():+6.1f}s  MAE={np.abs(arr).mean():6.1f}s  n={len(arr):>6}")

    print("--- residual autocorrelation (time-ordered trips) ---")
    r0 = resid - resid.mean()
    denom = (r0 * r0).sum()
    for lag in (1, 10, 50, 200, 1000):
        ac = (r0[lag:] * r0[:-lag]).sum() / denom
        print(f"  lag {lag:>4}: autocorr = {ac:+.3f}")

    print("--- error tail (|error| percentiles) ---")
    for q in (50, 75, 90, 95, 99, 99.9):
        print(f"  p{q}: {np.percentile(abserr, q):7.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
