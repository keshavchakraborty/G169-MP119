"""Tests for the FastAPI prediction service, schema validation, and logging."""

import json
import logging
from fastapi.testclient import TestClient
from app.main import app, JsonFormatter


client = TestClient(app)

VALID_PAYLOAD = {
    "loom_age_yr": 12.0,
    "thread_count": 120.0,
    "power_cuts_min": 45.0,
    "weaver_exp_yr": 15.0,
    "humidity_pct": 60.0,
    "shift_hours": 8.0,
    "fabric": "cotton",
    "shed": "Shed 1",
}


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_predict_success() -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    # Output in metres per shift should be a reasonable positive number
    assert data["prediction"] > 0


def test_predict_rejects_power_cuts_above_250() -> None:
    # Rule 1: power_cuts_min above 250 is impossible for us. Reject it.
    invalid_payload = {**VALID_PAYLOAD, "power_cuts_min": 250.1}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_accepts_power_cuts_at_boundary_250() -> None:
    # Boundary check for power_cuts_min <= 250
    boundary_payload = {**VALID_PAYLOAD, "power_cuts_min": 250.0}
    response = client.post("/predict", json=boundary_payload)
    assert response.status_code == 200


def test_predict_rejects_shift_hours_below_3() -> None:
    # Rule 2: shift_hours below 3 is always a typing mistake. Reject it.
    invalid_payload = {**VALID_PAYLOAD, "shift_hours": 2.9}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_accepts_shift_hours_at_boundary_3() -> None:
    # Boundary check for shift_hours >= 3
    boundary_payload = {**VALID_PAYLOAD, "shift_hours": 3.0}
    response = client.post("/predict", json=boundary_payload)
    assert response.status_code == 200


def test_predict_rejects_missing_field() -> None:
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "shed"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_json_formatter_outputs_valid_json() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Sample log message",
        args=(),
        exc_info=None,
    )
    record.props = {"inputs": {"power_cuts_min": 20}, "prediction": 45.2}
    formatted = formatter.format(record)
    parsed = json.loads(formatted)
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Sample log message"
    assert parsed["prediction"] == 45.2
    assert "timestamp" in parsed
