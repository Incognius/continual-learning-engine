"""FastAPI service — the platform's outward face.

Runs the continual-learning engine as a background task fed by the stream
simulator, and exposes:

    GET  /             -> live dashboard (frontend)
    GET  /health       -> liveness/readiness
    GET  /metrics      -> Prometheus exposition
    GET  /stats        -> JSON snapshot for the dashboard
    GET  /zones        -> zone catalog for the trip-predictor UI
    POST /predict_trip -> trip-level ETA (server computes geo features)
    POST /predict      -> ad-hoc inference on raw features (no learning)
    POST /ingest       -> push one record into the learning loop
    POST /stream/...   -> pause | resume | rate (live stream control)

The engine is single-threaded and its ``process`` has no ``await`` inside, so
the background stream and HTTP handlers can share it without races on the
asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel

from continual_ml.config import get_settings
from continual_ml.core.engine import ContinualLearningEngine
from continual_ml.data_sources import build_data_source
from continual_ml.schemas import Record
from continual_ml.streaming import StreamSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("continual_ml.api")

_STATIC_DIR = Path(__file__).parent / "static"


class AppState:
    engine: Optional[ContinualLearningEngine] = None
    simulator: Optional[StreamSimulator] = None
    task: Optional[asyncio.Task] = None
    geo: Optional[object] = None          # ZoneGeo, if the source is geographic
    rush_hours: set = frozenset()


state = AppState()


async def _run_stream() -> None:
    """Background loop: pull paced records from the simulator into the engine."""
    assert state.simulator and state.engine
    try:
        async for record in state.simulator.stream():
            state.engine.process(record)
            state.engine.stream_status = state.simulator.status
    except asyncio.CancelledError:  # graceful shutdown
        raise
    except Exception:  # noqa: BLE001
        logger.exception("Stream loop crashed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    source = build_data_source(settings)
    logger.info("Data source: %s", source.name)

    state.engine = ContinualLearningEngine(settings, source.schema())
    state.engine.start()
    state.simulator = StreamSimulator(source, settings.stream)
    state.task = asyncio.create_task(_run_stream())
    # Geographic source? expose its zone catalog for the trip-predictor UI.
    state.geo = getattr(source, "geo", None)
    state.rush_hours = set(settings.data_source.nyc_taxi.rush_hours)

    try:
        yield
    finally:
        if state.simulator:
            state.simulator.stop()
        if state.task:
            state.task.cancel()
            try:
                await state.task
            except asyncio.CancelledError:
                pass
        if state.engine:
            state.engine.stop()


app = FastAPI(title="Continual Learning MLOps Platform", version="0.1.0", lifespan=lifespan)


# --- request models ----------------------------------------------------------
class PredictRequest(BaseModel):
    features: dict[str, Any]


class IngestRequest(BaseModel):
    features: dict[str, Any]
    target: Optional[float] = None
    record_id: Optional[str] = None


class TripRequest(BaseModel):
    """High-level trip inputs; the server derives all geographic features."""
    pu_zone: str
    do_zone: str
    pickup_hour: int = 18
    pickup_dayofweek: int = 2
    pickup_month: int = 2
    passenger_count: int = 1
    ratecode: str = "1"


# --- endpoints ---------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    eng = state.engine
    return {
        "status": "ok" if eng else "starting",
        "source": eng._schema.name if eng else None,
        "samples_processed": eng._index if eng else 0,
    }


@app.get("/metrics")
def metrics() -> Response:
    if not state.engine:
        return Response("", media_type=CONTENT_TYPE_LATEST)
    return Response(state.engine.metrics.expose(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stats")
def stats() -> JSONResponse:
    if not state.engine:
        return JSONResponse({"status": "starting"})
    if state.simulator:
        state.engine.stream_status = state.simulator.status
    return JSONResponse(state.engine.stats())


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if not state.engine:
        return {"error": "engine not ready"}
    return state.engine.predict(req.features).model_dump(mode="json")


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    if not state.engine:
        return {"error": "engine not ready"}
    record = Record(
        record_id=req.record_id or f"ingest-{state.engine._index}",
        features=req.features,
        target=req.target,
    )
    return state.engine.process(record).model_dump(mode="json")


@app.get("/zones")
def zones() -> dict:
    """Zone catalog for the trip-predictor dropdowns (geographic sources only)."""
    if state.geo is None:
        return {"available": False, "zones": []}
    return {"available": True, "zones": state.geo.catalog()}


@app.post("/predict_trip")
def predict_trip(req: TripRequest) -> dict:
    """Trip-level ETA: the server computes geo features from the zone pair."""
    if not state.engine:
        return {"error": "engine not ready"}
    if state.geo is None:
        return {"error": "geographic features unavailable for this data source"}

    geo = state.geo.features(req.pu_zone, req.do_zone)
    features = {
        "pickup_hour": float(req.pickup_hour),
        "pickup_dayofweek": float(req.pickup_dayofweek),
        "pickup_month": float(req.pickup_month),
        "is_weekend": 1.0 if req.pickup_dayofweek >= 5 else 0.0,
        "is_rush_hour": 1.0 if req.pickup_hour in state.rush_hours else 0.0,
        "passenger_count": float(req.passenger_count),
        "gc_distance_km": geo["gc_distance_km"],
        "bearing_sin": geo["bearing_sin"],
        "bearing_cos": geo["bearing_cos"],
        "same_borough": geo["same_borough"],
        "pu_airport": geo["pu_airport"],
        "do_airport": geo["do_airport"],
        "pu_zone": req.pu_zone,
        "do_zone": req.do_zone,
        "ratecode": req.ratecode,
        "pu_borough": geo["pu_borough"],
        "do_borough": geo["do_borough"],
    }
    pred = state.engine.predict(features)
    return {
        "prediction_s": round(pred.prediction, 1),
        "prediction_min": round(pred.prediction / 60.0, 1),
        "distance_km": geo["gc_distance_km"],
        "pu_borough": geo["pu_borough"],
        "do_borough": geo["do_borough"],
        "is_rush_hour": bool(features["is_rush_hour"]),
        "latency_ms": pred.latency_ms,
    }


@app.post("/stream/pause")
def stream_pause() -> dict:
    if state.simulator:
        state.simulator.pause()
    return state.simulator.status if state.simulator else {}


@app.post("/stream/resume")
def stream_resume() -> dict:
    if state.simulator:
        state.simulator.resume()
    return state.simulator.status if state.simulator else {}


@app.post("/stream/rate")
def stream_rate(rate_per_sec: float) -> dict:
    if state.simulator:
        state.simulator.set_rate(rate_per_sec)
    return state.simulator.status if state.simulator else {}


# --- frontend ----------------------------------------------------------------
@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
