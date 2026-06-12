from fastapi.testclient import TestClient

from continual_ml.api.prediction_service import app


def test_api_endpoints():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        # Ad-hoc prediction (synthetic source -> numeric features).
        resp = client.post("/predict", json={"features": {f"x{i}": 0.1 for i in range(6)}})
        assert resp.status_code == 200
        assert "prediction" in resp.json()

        # Ingest a labeled record into the learning loop.
        ing = client.post(
            "/ingest",
            json={"features": {f"x{i}": 0.2 for i in range(6)}, "target": 1.0},
        )
        assert ing.status_code == 200
        assert "prediction" in ing.json()

        # Prometheus exposition.
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert b"cml_samples_processed_total" in metrics.content

        # Stream controls.
        assert client.post("/stream/pause").json()["running"] is False
        assert client.post("/stream/resume").json()["running"] is True

        # Synthetic source has no geography -> zone catalog unavailable,
        # and the trip endpoint reports geo unavailable rather than crashing.
        assert client.get("/zones").json()["available"] is False
        trip = client.post("/predict_trip", json={"pu_zone": "1", "do_zone": "2"})
        assert "error" in trip.json()
