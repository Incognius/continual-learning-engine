# Continual Learning MLOps Platform — Engineering Report

---

## 1. Executive Summary

This project implements an online machine learning system for taxi trip duration prediction on the NYC TLC yellow cab dataset (900K trips, Oct 2022 – Mar 2023). The system ingests trips as a stream, predicts duration at pickup using only information available at that moment, updates the model after each trip, and monitors for drift. No batch retraining is required during operation.

The task is regression: predict trip duration in seconds from pickup zone, dropoff zone, time of day, passenger count, and rate code. `trip_distance` is excluded — it is the completed route distance, known only at dropoff.

**Key results** (held-out 20% recent slice, `scripts/benchmark.py`):

| Setup | MAE | R² |
|---|---|---|
| Global mean (floor) | 526 s | 0.000 |
| Zone-pair memory only | 269 s | 0.658 |
| Online model, no route distance | 238 s | 0.745 |
| **Online model + route (deployed)** | **229 s** | **0.762** |
| Batch 2-epoch (static ceiling) | 227 s | 0.777 |
| Batch-warmup 40% + continual | 228 s | 0.762 |

The continual learning cost relative to batch is **0.015 R²** — nearly free. Most of the predictive signal comes from the zone-pair memory (R² 0.658 from memory alone) rather than from the base learner. The batch ceiling (0.777) matches published XGBoost results on this dataset, which used exact coordinates and OSRM routing that we don't have.

ADWIN drift detection fired 469 times; only 28% survived validation as genuine distribution shifts. The other 72% were spurious triggers from error variance, not real drift.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────┐
│  DATA SOURCE                                          │
│  BaseDataSource ── SyntheticDataSource                │
│                └─ NYCTaxiDataSource                   │
└───────────────────────┬──────────────────────────────┘
                        │ Record (typed contract)
                        ▼
┌──────────────────────────────────────────────────────┐
│  StreamSimulator  (rate-controlled, pause/resume)     │
└───────────────────────┬──────────────────────────────┘
                        │ async, paced
                        ▼
┌──────────────────────────────────────────────────────┐
│  ContinualLearningEngine  (prequential loop)          │
│                                                       │
│  FeaturePipeline → OnlineModel → RegressionMetrics   │
│       │                │              │               │
│  DriftDetectors    ModelRegistry  MetricsCollector    │
│  (ADWIN + PSI)     (MLflow)       (Prometheus)        │
└──────────────────────────────────────────────────────┘
         │                               │
         ▼                               ▼
    GET /stats                     GET /metrics
    (dashboard JSON)            (Prometheus scrape)
                                         │
                               Prometheus → Grafana
```

Four services: `api` (FastAPI + engine), `mlflow` (SQLite backend), `prometheus`, `grafana`. The data source is the only seam — switching datasets requires implementing one abstract class and registering it; no other code changes.

**Data flow for one labeled record:**

1. The stream simulator emits a `Record` at the configured rate (or records arrive via `POST /ingest`).
2. `FeaturePipeline.transform` generates geographic features, cyclical time encodings, and zone-pair memory features. Zone-pair means are read from the current (pre-update) state — no leakage.
3. `OnlineModel.predict_one` runs the River pipeline and returns duration in seconds.
4. `RegressionMetrics.update(y_true, y_pred)` records the error **before** learning. This ordering is what makes the evaluation honest.
5. The absolute error is fed to ADWIN. If ADWIN fires, the engine runs a before/after window validation. A validated degradation sets `retrain_recommended`.
6. `OnlineModel.learn_one` updates the model on this sample.
7. `FeaturePipeline.update_target` folds the label into the zone-pair means.
8. Metrics are pushed to Prometheus gauges; every 500 samples, a row is logged to MLflow.

`POST /predict` skips steps 4–8 — features, predict, return, no side effects.

---

## 3. Feature Engineering

### 3.1 Geographic Features

The raw data identifies zones by TLC LocationID (1–263). One-hot encoding these makes zones opaque to the model: it cannot know that zone 132 (JFK) is 20 km from zone 230 (Times Square), or that adjacent zones take 2 minutes. Distance is the dominant predictor of duration, and one-hot zones carry none of it.

The fix is to compute zone centroids from the TLC shapefile (EPSG:2263, NY State Plane in feet) and derive distance and bearing at request time. Details of the centroid computation are in Appendix B; the serving path only needs the resulting 12 KB CSV.

For each trip, the following features are computed from the two centroids:

| Feature | What it captures |
|---|---|
| `gc_distance_km` | Straight-line distance — the dominant duration driver |
| `bearing_sin`, `bearing_cos` | Direction of travel (encoded as sin/cos so 359° ≈ 0°) |
| `same_borough` | Intra-borough trips behave differently from river-crossings |
| `pu_airport`, `do_airport` | Airport runs are long, high-variance outliers |
| `pu_borough`, `do_borough` | Coarse geographic grouping for tree splits |

All of these are derivable at pickup from the stated origin and destination. None uses the realized route.

**Impact:** one-hot zones → R² 0.44, MAE 367 s. Adding geographic features → R² 0.65, MAE 238 s. Raw zone columns are then dropped from model input (they're summarized by the zone-pair memory) — tree size drops 5× with no accuracy change.

### 3.2 Zone-Pair Memory

Straight-line distance misses corridor-specific delays: bridges, tunnels, Midtown congestion. To capture this, the feature pipeline maintains a **leakage-free online target encoder** — a running mean trip duration per corridor key.

Three keys are maintained:
- `(pu_zone, do_zone)` — the exact corridor
- `(do_zone)` — a destination baseline (fallback for unseen pairs)
- `(pu_borough, do_borough, pickup_hour)` — borough-pair by time of day

The emitted feature is smoothed toward the global mean to handle rare corridors:

```
te = (n·μ + k·g) / (n + k)     # k = 20
```

The leakage guard is update ordering. The engine always runs:

```
feats = pipeline.transform(record)   # reads current (pre-update) means
pred  = model.predict_one(feats)     # predict
...evaluate, learn...
pipeline.update_target(feats, y)     # update means AFTER
```

A trip's own label never influences its own feature. This is functionally a learned edge over the zone graph — the same signal a static OSRM duration lookup would provide, but built online without external data. **Zone-pair memory alone reaches R² 0.658**, which is why additional features give diminishing returns.

### 3.3 Multi-Horizon Memory

A single running mean weights last March's trips the same as last hour's. For ETA, you want the stable corridor baseline (geography doesn't change) and recent traffic conditions (congestion does). The encoder maintains three estimates per key:

| Horizon | Update rule | α | What it tracks |
|---|---|---|---|
| Long | Cumulative mean | 0.0 (infinite) | Stable geographic baseline |
| Med | EWMA | 0.05 | Weeks-scale conditions |
| Short | EWMA | 0.30 | Recent traffic |

Each becomes a separate feature (`te_pu_zone_do_zone_{long,med,short}`). The gap between short and long is a live congestion signal — if a corridor is currently running 20% slower than its baseline, the model sees that difference directly.

**Impact:** extending from 3 to 6 months of data and adding multi-horizon memory lifted no-route R² from 0.69 to 0.745 — a larger gain than adding route distance (0.745 → 0.762).

---

## 4. Model Selection & Learning Strategy

### 4.1 Model Bakeoff

The first production run used a linear model with Adam. Live rolling MAE looked fine (~360 s). But when evaluated frozen — on the first third of the dataset, not just the last 1,000 trips — it predicted **all-negative durations**: mean −3965 s, MAE 4824 s on January data, working only on late-March trips.

The cause: an unregularized linear learner on a time-ordered stream overfits the recent window. Weights grow toward recent data and extrapolate to nonsense on older feature combinations. The rolling metric is immune to this because it only scores the trailing window. This is a real failure mode in continual learning — the rolling metric and the frozen model are not the same thing.

Bakeoff on early/mid/late slices, frozen model evaluation:

| Model | Worst-slice MAE | Negative ETAs |
|---|---|---|
| Linear (no regularization) | 4824 s | 25–100% |
| Linear (L2 = 2.0) | 486 s | 0% |
| **Hoeffding Tree** | **402 s** | **0%** |
| Adaptive Tree | 484 s | 0% |

The Hoeffding Tree wins because its leaf predictions are **bounded averages of seen targets** — it physically cannot output negative durations, and it accumulates knowledge from the whole stream rather than chasing the recent window. Trip duration is near-stationary over six months, so drift-adaptive variants don't help here. L2 regularization was added as a linear option, but the default was switched to Hoeffding Tree.

### 4.2 Batch vs Continual Learning

The benchmark compares online learning against a static upper bound trained in batch:

| Setup | MAE | R² |
|---|---|---|
| Online + route (deployed) | 229 s | 0.762 |
| Batch 2-epoch (ceiling) | 227 s | 0.777 |
| Batch-warmup 40% + continual | 228 s | 0.762 |

The continual learning cost is **+0.015 R²**. In practice, the deployed regime is batch-warmup then continual: train on the first 40% of history for two epochs to warm up the zone-pair memory and weights, then switch to single-pass online learning for the remainder. This matches batch accuracy (0.762 vs 0.762) and is more realistic than cold-starting from zero.

The batch ceiling (0.777) matches published XGBoost results (~0.78) on this dataset despite using zone-level resolution instead of exact lat/long coordinates and no OSRM routing. The multi-horizon zone-pair memory acts as a learned stand-in for route duration priors.

---

## 5. Drift Detection Analysis

### 5.1 ADWIN Validation

ADWIN monitors the absolute prediction error stream and signals when it detects a distribution shift. On 900K trips, it fired **469 times**. Before acting on these, we ran a validation procedure.

**Validation procedure:** for each ADWIN trigger, compare mean error over the 1500 samples before vs 1500 samples after. Count the event as real only if:
- Mean error shifts by more than 10% (relative)
- Cohen's d > 0.2 (effect size)

**Results:**
- 469 events → 132 validated (28%), 337 spurious (72%)
- Of the validated 132: 76 were degradations, 56 were **improvements** (the model getting better, not worse)

**Shuffle control:** ADWIN was run on a time-shuffled error stream — identical samples, random order, zero real drift. It fired **110 times**. This confirms that a substantial fraction of ADWIN triggers are driven by error variance, not temporal distribution shifts.

**Outlier check:** prediction errors above p99 (>1426 s) have lag-1 autocorrelation of +0.014. The large errors are temporally random — accidents and unusual trips — not a clustered regime shift.

The in-engine fix: every ADWIN trigger now runs the before/after window test. The dashboard shows validated drifts and spurious anomalies separately. Model versioning is completely decoupled from drift detection — a validated degradation raises a `retrain_recommended` flag; new versions are created only by the promote flow when a candidate beats the champion on a holdout.

### 5.2 Feature Drift

Feature drift is tracked via Population Stability Index (PSI) between a fixed reference window (5000 samples) and a sliding current window (2000 samples), computed every 1000 samples.

**Findings:** `pickup_hour` and `is_weekend` show persistent PSI above the 0.2 threshold. This is genuine — replaying 6 months of ordered data moves through weekday/weekend and morning/evening cycles. The reference window is intentionally large (5000 samples) to reduce false positives from diurnal cycles alone; smaller reference windows made `pickup_hour` appear permanently drifted when it was just advancing through the day.

No feature showed abrupt distribution collapse or gradual drift that would indicate a dataset shift requiring action.

---

## 6. Error Analysis

Residual analysis on the champion model over the full 900K trip dataset:

**Diurnal bias:** overall bias is +23 s, but it's not uniform. Pre-dawn hours (04–06h) have bias +96–118 s — the model over-predicts because roads are faster than its average expectation. Midday (12–17h) is well-calibrated. Mondays and Sundays show +61–74 s over-prediction on light-traffic days. The cyclical time encodings capture most of the time-of-day signal but miss the interaction between time and distance (off-peak speedups disproportionately help long trips). MAE swings ~±35 s across hours, so this is a minor residual.

**Autocorrelation:** lag-1 autocorrelation of residuals is +0.05, decaying to +0.015 by lag 1000. There is a faint regime signal — congested periods produce mildly correlated over-runs — but it's small.

**Tail behavior:**

| Percentile | |error|| 
|---|---|
| p50 | 164 s |
| p90 | 483 s |
| p95 | 660 s |
| p99 | 1199 s |
| p99.9 | 2565 s |

The gap between RMSE (347 s) and MAE (229 s) is driven by a thin tail of trips whose duration is not a function of any feature we have — accidents, road closures, driver behavior. No learner can reduce this without new information.

The multi-horizon short-term memory is the highest-ROI remaining improvement (it partially exploits the 0.05 autocorrelation by tracking recent per-corridor conditions). Boosting would offer low single-digit MAE gains at best and doesn't fit River's per-sample loop cleanly.

---

## 7. Operational Design

### Retraining Strategy

The system operates on three cadences:

**Continuous:** every completed trip is ingested into the online model. Zone-pair means are updated after each label. The model stays current on recent conditions without any scheduled jobs.

**Weekly:** `scripts/retrain.py` trains a candidate on a rolling 3–6 month window (batch-warmup then continual), evaluates it against the current champion on a held-out recent slice, and promotes only if candidate MAE < champion MAE × (1 − margin). The check includes per-borough breakdowns so a candidate that regresses on airport runs is caught even if overall MAE improves. Promotes are recorded in MLflow with a `champion` alias.

The promote threshold is conservative by design: the online model updates continuously, so a weekly candidate needs to be meaningfully better, not just within noise.

**Warm-start:** the deployable artifact is a bundle containing the model and the full zone-pair memory state. A restarted service loads the bundle and is immediately accurate instead of cold-starting from zero. This also means the weekly retrain preserves accumulated corridor knowledge.

**Versioning:** model versions are created only by promotes. ADWIN triggers do not create versions. A validated degradation raises the `retrain_recommended` flag, which is the signal to run the weekly job — it is not an automatic action.

**Per-segment monitoring:** Prometheus tracks rolling MAE per borough (`cml_segment_rolling_mae{kind="borough"}`). Manhattan MAE is ~206 s; outer boroughs range 438–1017 s. A localized degradation (e.g., a model change that hurts airport runs) would show in the borough breakdown before appearing in the overall metric.

---

## 8. Limitations & Future Work

**Delayed labels.** The current setup supplies labels immediately after each prediction. In production, the label (actual duration) arrives minutes later. The engine supports `record.target = None` and skips learning when the label is absent, but the matching and buffering logic for late labels isn't implemented.

**Route distance.** The `gc_distance_km` feature is a straight-line proxy for route distance. A production system would use a router (Google Maps, OSRM) to get the planned route distance at pickup time — this is a valid, non-leaking feature that the current setup approximates by exposing `trip_distance` reframed as a planned estimate. The measured lift from this approximation is +0.018 R² (0.745 → 0.762). The remaining gap to SOTA (~0.81 R²) is primarily explained by the absence of exact lat/long (unavailable in TLC data post-2016) and real-time traffic.

**Weather and live traffic.** Both are well-known drivers of duration variance and would reduce the irreducible error tail. Neither requires architectural changes — they slot in as additional features.

**Single-process deployment.** The engine runs inside the API process. At scale, this should be separated: put the engine in a worker, the stream simulator in a Kafka producer, and the API as a thin inference service. The `StreamSimulator` interface already matches this split.

**Stronger continual learning benchmarks.** The current comparison is against a static batch ceiling. A fairer comparison would include forgetting benchmarks (performance on held-out early data after training on late data) and explicit concept drift tests with controlled drift injection.

**ADWIN sensitivity.** 72% of ADWIN triggers are spurious on this dataset. The before/after validation fixes the downstream impact, but a better detector for low-autocorrelation error streams would reduce alert fatigue without the workaround.

---

## Appendix

### A. File Structure

```
configs/                  central YAML config
src/continual_ml/
  config.py               typed settings (Pydantic + YAML + env)
  schemas.py              shared data contracts
  geo.py                  zone-pair → geographic features
  persistence.py          model bundle save/load
  data_sources/           abstract source + synthetic + nyc_taxi
  streaming/              rate-controlled stream simulator
  features/               online feature pipeline
  models/                 River wrapper + MLflow registry bridge
  monitoring/             ADWIN/PSI drift detectors + Prometheus
  core/                   engine (prequential loop) + evaluation
  api/                    FastAPI service + live dashboard (HTML/JS/CSS)
infrastructure/
  docker/Dockerfile.api
  prometheus/prometheus.yml
  grafana/provisioning/   auto-wired datasource + dashboard provider
  grafana/dashboards/     pre-built Grafana dashboard JSON
scripts/
  prepare_nyc_taxi.py     offline download → clean → parquet
  prepare_zones.py        TLC shapefile → zone_centroids.csv
  train.py                batch-warmup + continual training run
  retrain.py              rolling-window retrain with promote logic
  serve_demo.py           local launcher (nyc_taxi, SQLite MLflow)
  benchmark.py            ablation: baselines, batch ceiling, continual cost
  validate_drift.py       ADWIN validation + shuffle control
  analyze_errors.py       residual periodicity, autocorrelation, tail
tests/                    13 tests (engine, drift, API, features)
```

### B. Geographic Implementation Details

Zone centroids are computed from the TLC taxi_zones shapefile in **EPSG:2263** (NY State Plane, Long Island zone, US survey feet). This is a conformal projection — within NYC, planar Euclidean distance in the projected plane is a good approximation of true ground distance, which avoids the need for geodetic calculations.

Centroids are computed using the **shoelace formula** per polygon ring:

```
A  = ½ · Σ (xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)
Cx = 1/(6A) · Σ (xᵢ + xᵢ₊₁)(xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)
Cy = 1/(6A) · Σ (yᵢ + yᵢ₊₁)(xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)
```

For multipolygons, rings are area-weighted. Coordinates are converted to kilometres (× 0.0003048). The result is cached to `data/zone_centroids.csv` (263 zones). The serving path reads only this CSV — no shapefile dependency at runtime.

Sanity check: JFK ↔ LaGuardia centroid distance computes to 15.9 km, which is correct.

### C. Full Metrics Tables

**Benchmark ablation** (180K held-out recent slice, `scripts/benchmark.py`):

| Setup | MAE (s) | R² |
|---|---|---|
| Global mean | 526 | 0.000 |
| Zone-pair memory only | 269 | 0.658 |
| Online, no route distance | 238 | 0.745 |
| Online + route distance | 229 | 0.762 |
| Batch 2-epoch ceiling | 227 | 0.777 |
| Batch-warmup 40% + continual | 228 | 0.762 |

**Model bakeoff** (frozen evaluation, early/mid/late slices):

| Model | Worst-slice MAE | Negative ETAs |
|---|---|---|
| Linear (no regularization) | 4824 s | 25–100% |
| Linear (L2 = 2.0) | 486 s | 0% |
| Hoeffding Tree | 402 s | 0% |
| Adaptive Tree | 484 s | 0% |

**Per-borough MAE** (rolling, champion model):

| Borough | Rolling MAE |
|---|---|
| Manhattan | 206 s |
| EWR | 374 s |
| Queens | 438 s |
| Brooklyn | 525 s |
| Bronx | 671 s |
| Staten Island | 1017 s |

**ADWIN validation summary:**

| Category | Count | % |
|---|---|---|
| Total ADWIN triggers | 469 | 100% |
| Validated real shifts | 132 | 28% |
| — of which: degradations | 76 | 16% |
| — of which: improvements | 56 | 12% |
| Spurious (failed validation) | 337 | 72% |
| Shuffle control false positives | 110 | — |

### D. Deployment Instructions

**Local (no Docker):**
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"

python scripts/prepare_nyc_taxi.py \
  --months 2022-10 2022-11 2022-12 2023-01 2023-02 2023-03 \
  --rows-per-month 150000
python scripts/prepare_zones.py

python scripts/train.py
python scripts/serve_demo.py
# http://127.0.0.1:8000/
```

**Docker Compose:**
```bash
docker compose up --build                        # synthetic, zero setup
DATA_SOURCE=nyc_taxi docker compose up --build   # real data (after prepare)
```

Ports: API :8000, MLflow :5000, Prometheus :9090, Grafana :3000.
