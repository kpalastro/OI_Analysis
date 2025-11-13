"""
FastAPI-based serving layer for the alpha model.

Provides `/predict` and `/health` endpoints. Loads the latest trained
model, scaler, feature list, and metadata from the configured model directory.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

DEFAULT_MODEL_DIR = Path("ml_system/models/alpha")
CLASS_LABELS = {-1: "Bearish", 0: "Neutral", 1: "Bullish"}


class PredictRequest(BaseModel):
    """Schema for prediction requests."""

    features: Dict[str, float] = Field(
        ..., description="Mapping of feature name to value. Must include all trained features."
    )

    @validator("features")
    def validate_features(cls, value: Dict[str, float]) -> Dict[str, float]:
        if not value:
            raise ValueError("At least one feature value must be provided.")
        return value


class PredictResponse(BaseModel):
    """Schema for prediction responses."""

    prediction: str
    prediction_code: int
    confidence: float
    probabilities: Dict[str, float]
    missing_features: Optional[List[str]] = None


class ModelArtifacts:
    """Container for loaded model artifacts."""

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.metadata: Dict[str, str] | None = None

    def load(self) -> None:
        """Load model, scaler, feature names, and metadata from disk."""
        model_path = self.model_dir / "alpha_model_xgboost.pkl"
        if not model_path.exists():
            # fallback to LightGBM naming convention
            model_path = self.model_dir / "alpha_model_lightgbm.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No alpha model artifact found in {self.model_dir}. "
                "Train the model before starting the server."
            )

        scaler_path = self.model_dir / "alpha_scaler_xgboost.pkl"
        if not scaler_path.exists():
            scaler_path = self.model_dir / "alpha_scaler_lightgbm.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler artifact missing in {self.model_dir}.")

        feature_path = self.model_dir / "alpha_feature_names.pkl"
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature names artifact missing in {self.model_dir}.")

        metadata_path = self.model_dir / "alpha_model_metadata.pkl"

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(feature_path)
        self.metadata = joblib.load(metadata_path) if metadata_path.exists() else None

        if not isinstance(self.feature_names, (list, tuple)):
            raise ValueError("Feature names artifact must be a list of strings.")

    def ensure_loaded(self) -> None:
        if self.model is None or self.scaler is None or not self.feature_names:
            self.load()


@lru_cache(maxsize=1)
def get_artifacts() -> ModelArtifacts:
    """Cached accessor for model artifacts."""
    model_dir = Path(os.environ.get("ALPHA_MODEL_DIR", DEFAULT_MODEL_DIR))
    artifacts = ModelArtifacts(model_dir)
    artifacts.ensure_loaded()
    return artifacts


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(title="Alpha Model Serving API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def healthcheck() -> Dict[str, str]:
        """Simple health check endpoint."""
        try:
            artifacts = get_artifacts()
            status = "ready" if artifacts.model is not None else "loading"
            metadata_info = artifacts.metadata or {}
        except Exception as exc:  # pragma: no cover - only for runtime diagnostics
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "status": status,
            "model_dir": str(artifacts.model_dir),
            "metadata": metadata_info,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        """Generate a prediction from the alpha model."""
        artifacts = get_artifacts()
        artifacts.ensure_loaded()
        missing = [feat for feat in artifacts.feature_names if feat not in payload.features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing feature values for: {', '.join(missing)}",
            )

        ordered_features = [payload.features[name] for name in artifacts.feature_names]
        feature_array = np.array(ordered_features, dtype=np.float32).reshape(1, -1)
        scaled_features = artifacts.scaler.transform(feature_array)

        probabilities = artifacts.model.predict_proba(scaled_features)[0]
        model_classes = getattr(artifacts.model, "classes_", sorted(CLASS_LABELS.keys()))
        prediction_code = int(artifacts.model.predict(scaled_features)[0])
        class_probabilities = {
            CLASS_LABELS.get(int(cls), str(cls)): float(prob)
            for cls, prob in zip(model_classes, probabilities)
        }

        return PredictResponse(
            prediction=CLASS_LABELS.get(prediction_code, str(prediction_code)),
            prediction_code=prediction_code,
            confidence=float(np.max(probabilities)),
            probabilities=class_probabilities,
            missing_features=None,
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ml_system.serving.model_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("ALPHA_MODEL_PORT", 8000)),
        reload=False,
    )

