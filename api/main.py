"""FastAPI service exposing the PhishGuard phishing classifier."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("phishguard.api")
APP = FastAPI(title="PhishGuard API", version="1.0.0")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "ml" / "phishguard_model.joblib"
DASHBOARD_PATH = PROJECT_ROOT / "dashboard"
LABEL_MAPPING: Dict[int, str] = {0: "legitimate", 1: "phishing"}


class PredictionRequest(BaseModel):
    """Model for a single email prediction request."""

    text: str = Field(..., description="Plain-text body of the email to evaluate.")


class PredictionResponse(BaseModel):
    """Response payload returned by the prediction endpoint."""

    label: str
    is_phishing: bool
    confidence: float


def load_model(path: Path) -> object:
    """Load a serialized model pipeline from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python ml/train_model.py` first."
        )

    LOGGER.info("Loading model from %s", path)
    model = joblib.load(path)
    return model


@APP.on_event("startup")
def startup_event() -> None:
    """Load the model when the API starts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    try:
        APP.state.model = load_model(MODEL_PATH)
        LOGGER.info("Model loaded successfully.")
    except FileNotFoundError as err:
        LOGGER.warning("%s", err)
        APP.state.model = None


@APP.get("/health")
def healthcheck() -> Dict[str, str]:
    """Simple health-check endpoint."""
    status = "ready" if getattr(APP.state, "model", None) else "model_missing"
    return {"status": status}


def _ensure_model_loaded() -> object:
    """Ensure the model is available before handling predictions."""
    model = getattr(APP.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train the model first with `python ml/train_model.py`.",
        )
    return model


def _format_prediction(raw_prediction: int, probability: float) -> PredictionResponse:
    """Convert raw model output to API response."""
    label = LABEL_MAPPING.get(raw_prediction, "unknown")
    return PredictionResponse(
        label=label,
        is_phishing=bool(raw_prediction),
        confidence=round(probability, 4),
    )


@APP.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Return phishing prediction for the provided email text."""
    model = _ensure_model_loaded()

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Email text must not be empty.")

    prediction = model.predict([request.text])[0]
    probabilities = getattr(model, "predict_proba", None)
    confidence: Optional[float] = None
    if callable(probabilities):
        confidence = float(probabilities([request.text])[0][int(prediction)])
    else:
        LOGGER.warning("Model does not expose predict_proba; default confidence used.")
        confidence = 0.0

    return _format_prediction(int(prediction), confidence)


APP.mount(
    "/",
    StaticFiles(directory=str(DASHBOARD_PATH), html=True),
    name="dashboard",
)
