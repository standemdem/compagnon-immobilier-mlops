import json
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI

from app.model_loader import get_model
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from app.security import verify_api_key


METADATA_PATH = Path(
    "data/models/"
    "prix_m2_pipeline_2020_2023.metadata.json"
)


app = FastAPI(
    title="Compagnon Immobilier API",
    description=(
        "API de prédiction du prix au m² "
        "des appartements en France métropolitaine."
    ),
    version="0.1.0",
)

@app.get("/")
def root() -> dict[str, str]:
    """
    Point d'entrée simple de l'API.
    """

    return {
        "name": "Compagnon Immobilier API",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }

@app.get("/health")
def health() -> dict[str, str]:
    """
    Vérifie que l'API répond.
    """

    return {
        "status": "ok",
    }


@app.get("/model/info")
def model_info() -> dict:
    """
    Retourne les métadonnées du modèle actuellement utilisé.
    """

    if not METADATA_PATH.exists():
        return {
            "status": "metadata_not_found",
        }

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    return metadata


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
    _: str = Depends(verify_api_key),
) -> PredictionResponse:
    """
    Prédit le prix au m² d'un appartement.
    """

    model = get_model()

    input_df = pd.DataFrame([request.model_dump()])

    prediction = model.predict(input_df)[0]

    return PredictionResponse(
        prix_m2=float(prediction),
    )