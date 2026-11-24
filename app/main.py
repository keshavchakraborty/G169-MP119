"""FastAPI prediction service with structured JSON logging and schema validation."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class JsonFormatter(logging.Formatter):
    """Format logs as single-line JSON records."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "props") and isinstance(record.props, dict):
            log_payload.update(record.props)
        return json.dumps(log_payload)


# Set up JSON logging to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())

logger = logging.getLogger("mp119.service")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(handler)
logger.propagate = False


class PredictionRequest(BaseModel):
    loom_age_yr: float
    thread_count: float
    power_cuts_min: float = Field(
        ...,
        le=250,
        description="Power cuts during the shift in minutes; rejected if > 250",
    )
    weaver_exp_yr: float
    humidity_pct: float
    shift_hours: float = Field(
        ...,
        ge=3,
        description="Shift length in hours; rejected if < 3",
    )
    fabric: str
    shed: str


class PredictionResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: str


app = FastAPI(
    title="MP119 Power Loom Output Service",
    description="Inference API for Bhopal Weavers' Cluster power-loom output prediction.",
    version="1.0",
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
_model: Optional[Any] = None


def get_model() -> Any:
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise RuntimeError(f"Champion model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, str]:
    """Health check endpoint required by the monitoring and orchestration probes."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> dict[str, float]:
    """Generate cloth output prediction in metres for a loom shift."""
    try:
        model = get_model()
    except Exception as exc:
        logger.error(
            "Model loading failed",
            extra={"props": {"error": str(exc)}},
        )
        raise HTTPException(status_code=503, detail="Model unavailable") from exc

    features = [
        "loom_age_yr",
        "thread_count",
        "power_cuts_min",
        "weaver_exp_yr",
        "humidity_pct",
        "shift_hours",
        "fabric",
        "shed",
    ]
    data = pd.DataFrame([{col: getattr(payload, col) for col in features}])
    prediction = float(model.predict(data)[0])

    logger.info(
        "Prediction made",
        extra={
            "props": {
                "inputs": {col: getattr(payload, col) for col in features},
                "prediction": round(prediction, 4),
            }
        },
    )

    return {"prediction": prediction}
