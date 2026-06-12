"""Are ADWIN concept-drift events real distribution shifts, or noise?

For each ADWIN event we compare a window of errors BEFORE vs AFTER the event and
declare it a *valid* shift only if the mean moves materially (Cohen's d > 0.2 and
>10% relative). Two controls:
  * shuffle test  — ADWIN on a time-shuffled error stream has NO real drift, so
                    any events it fires are pure false positives.
  * outlier clustering — do >p99 errors cluster in time (regime) or are they
                    temporally random (true outliers)?
"""

from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("CML_DATA_SOURCE__TYPE", "nyc_taxi")
sys.path.insert(0, "src")

from river import drift  # noqa: E402

from continual_ml.config import get_settings  # noqa: E402
from continual_ml.data_sources import build_data_source  # noqa: E402
from continual_ml.features.feature_pipeline import FeaturePipeline  # noqa: E402
from continual_ml.models.online_model import OnlineModel  # noqa: E402

W = 2000  # before/after comparison window


def adwin_events(errs, delta) -> list[int]:
    d = drift.ADWIN(delta=delta)
    ev = []
    for i, e in enumerate(errs):
        d.update(float(e))
        if d.drift_detected:
            ev.append(i)
    return ev


def main() -> int:
    s = get_settings()
    src = build_data_source(s)
    fp = FeaturePipeline(src.schema(), s.features)
    m = OnlineModel(src.schema(), s.model)

    errs = []
    for r in src.stream():
        f = fp.transform(r)
        p = m.predict_one(f)
        errs.append(abs(p - r.target))
        m.learn_one(f, r.target)
        fp.update_target(f, r.target)
    errs = np.array(errs)

    delta = s.drift.concept.delta
    events = adwin_events(errs, delta)
    print(f"errors n={len(errs):,}  ADWIN delta={delta}  events={len(events)}")

    checked = [i for i in events if i - W >= 0 and i + W <= len(errs)]
    valid = degr = impr = 0
    rels = []
    for i in checked:
        before, after = errs[i - W:i], errs[i:i + W]
        mb, ma = before.mean(), after.mean()
        pooled = np.sqrt((before.var() + after.var()) / 2) + 1e-9
        d = (ma - mb) / pooled            # Cohen's d
        rel = (ma - mb) / mb
        rels.append(rel)
        if abs(d) > 0.2 and abs(rel) > 0.10:
            valid += 1
            degr += ma > mb
            impr += ma <= mb
    n = max(len(checked), 1)
    print(f"validated {len(checked)} events  ->  REAL shifts={valid} ({100*valid/n:.0f}%)  "
          f"spurious={len(checked)-valid} ({100*(len(checked)-valid)/n:.0f}%)")
    print(f"  of real shifts: degradations={degr}, improvements={impr} "
          f"(ADWIN-on-error also fires on the model getting *better*)")
    print(f"  median |relative mean shift| at events = {np.median(np.abs(rels))*100:.1f}%")

    # control 1: shuffle removes all temporal structure
    sh = errs.copy()
    np.random.seed(0)
    np.random.shuffle(sh)
    sh_ev = adwin_events(sh, delta)
    print(f"shuffle control: ADWIN on time-shuffled errors -> {len(sh_ev)} events "
          f"(pure false positives; ideal = 0)")

    # control 2: do extreme errors cluster in time?
    thr = np.percentile(errs, 99)
    big = (errs > thr).astype(float)
    b0 = big - big.mean()
    ac = float((b0[1:] * b0[:-1]).sum() / (b0 * b0).sum())
    print(f"outlier clustering: >p99 ({thr:.0f}s) indicator lag-1 autocorr = {ac:+.3f}  "
          f"(~0 => temporally random outliers, not a regime shift)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
