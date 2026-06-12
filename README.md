# Continual Learning MLOps Platform

A continual-learning platform for streaming machine learning systems.

The project simulates a data stream, generates features, serves predictions,
evaluates model performance, updates the model online, and monitors for drift.
The architecture is dataset-agnostic: adding a new data source only requires
implementing a data-source adapter.

## Stack (v1)

| Concern | Tool |
|---|---|
| Online learning | [River](https://riverml.xyz) |
| Serving | FastAPI + Uvicorn |
| Experiments / model lifecycle | MLflow (SQLite backend + local artifacts) |
| Drift | River detectors (concept) + Evidently (feature/data) |
| Metrics | Prometheus |
| Dashboards | Grafana |
| Packaging / run | Docker + Docker Compose |

_Deferred by design (added later behind existing interfaces): Kafka, PostgreSQL, Airflow._

## Layout

```
configs/                  central YAML config (single source of truth)
src/continual_ml/
  config.py               typed settings loader (YAML + env overrides)
  schemas.py              shared data contracts (Record/Prediction/DriftEvent)
  data_sources/           abstract source + synthetic impl (the plug point)
  streaming/              rate-controlled stream simulator (Kafka stand-in)
  features/               online feature pipeline
  models/                 River model wrapper + MLflow registry bridge
  monitoring/             drift detector + Prometheus metrics
  core/                   continual-learning engine (the loop)
  api/                    FastAPI service
infrastructure/           Dockerfile, Prometheus + Grafana config
tests/
```

## Configuration

All behavior is driven by [`configs/config.yaml`](configs/config.yaml). Override any
value with an env var: prefix `CML_`, nest with `__`. Example:

```
CML_STREAM__RATE_PER_SEC=50
CML_DATA_SOURCE__TYPE=synthetic
```

Copy `.env.example` → `.env` for local overrides.

## Implemented Components

- Configuration system
- Data-source abstraction
- Stream simulator
- Online feature pipeline
- River-based continual learner
- Drift detection
- Monitoring and metrics
- FastAPI serving layer
- Dockerized deployment


## Performance (900k trips Oct 2022–Mar 2023, batch-warmup 40% + continual)

- **R²:** 0.761
- **MAE:** 232 s (~4 min)
- **Latency:** <1 ms
- **Model bundle:** 13.9 MB
- **Batch ceiling:** 0.777 (continual model within 0.016 R²)

The largest gains came from geographic features (distance, bearing, borough metadata), multi-horizon zone-pair memory, and route-distance experiments. These improvements increased performance from **R² 0.44 → 0.762** and reduced **MAE from 367 s → 232 s**.

ADWIN drift events were validated against controls and manual analysis; most detections were transient error spikes rather than genuine distribution shifts. Borough-level MAE ranged from **206 s in Manhattan** to **438–1017 s in outer boroughs**.

> **Using the real taxi data:** the API defaults to `synthetic` for zero-setup
> CI. For the real model run `python scripts/serve_demo.py` (or set
> `CML_DATA_SOURCE__TYPE=nyc_taxi`) — see prep steps below.

## Real dataset: NYC TLC yellow taxi

The platform ships with a real streaming source alongside the synthetic one.
Predict **trip duration (ETA)** in seconds using only pickup-time features
(origin/destination zone, time-of-day, passenger count, rate code).
`trip_distance` is deliberately excluded as target leakage.

Prepare the dataset once (downloads, cleans, caches to `data/processed/`):

```bash
python scripts/prepare_nyc_taxi.py --months 2022-10 2022-11 2022-12 2023-01 2023-02 2023-03 --rows-per-month 150000
python scripts/prepare_zones.py
```

Then switch the active source in `configs/config.yaml`:

```yaml
data_source:
  type: nyc_taxi          # was: synthetic
```

No other changes are required.

## Prerequisites

- Python 3.11+
- Docker Desktop (for the full stack) — **not yet installed on this machine**

## Quick start

### Local (no Docker) — fastest path
```bash
python -m venv .venv && .venv\Scripts\activate   # Windows PowerShell
pip install -e ".[dev]"

# download and prepare taxi data
python scripts/prepare_nyc_taxi.py --months 2022-10 2022-11 2022-12 2023-01 2023-02 2023-03 --rows-per-month 150000
python scripts/prepare_zones.py

# train and save model bundle
python scripts/train.py

# retrain model
python scripts/retrain.py

# start API and dashboard
python scripts/serve_demo.py
```
Open **http://127.0.0.1:8000/** for the dashboard.

### Full stack (Docker Compose)
```bash
docker compose up --build                       # synthetic data, zero setup
DATA_SOURCE=nyc_taxi docker compose up --build  # real taxi data (after prepare)
```
Dashboard `:8000` · MLflow `:5000` · Prometheus `:9090` · Grafana `:3000`.

## Endpoints

**GET**
- `/` — Dashboard
- `/health` — Health check
- `/metrics` — Prometheus metrics
- `/stats` — Runtime statistics (JSON)
- `/zones` — Available taxi zones

**POST**
- `/predict_trip` — Predict ETA from a zone pair (computes geo features)
- `/predict` — Predict ETA from raw features
- `/ingest` — Ingest a new record
- `/stream/pause` — Pause stream
- `/stream/resume` — Resume stream
- `/stream/rate` — Update stream rate

### Tests
```bash
pytest -q     # 13 passing
```
