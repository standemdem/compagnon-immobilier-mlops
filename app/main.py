import time
import pandas as pd
from typing import Literal
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Response

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from app.model_loader import (
    get_model,
    get_model_info,
)

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from app.security import verify_api_key


PREDICTION_COUNTER = Counter(
    "compagnon_predictions_total",
    "Nombre total de prédictions effectuées.",
)

PREDICTION_ERRORS = Counter(
    "compagnon_prediction_errors_total",
    "Nombre total d'erreurs pendant les prédictions.",
)

PREDICTION_LATENCY = Histogram(
    "compagnon_prediction_duration_seconds",
    "Durée d'exécution des prédictions.",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Charge les champions MLflow au démarrage de l'API.
    """

    get_model("france")
    get_model("paris")

    yield

app = FastAPI(
    title="Compagnon Immobilier API",
    description=(
        "API de prédiction du prix au m² "
        "des appartements en France métropolitaine."
    ),
    version="0.1.0",
    lifespan=lifespan,
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
def model_info(
    scope: Literal["france", "paris"] = "france",
    _: str = Depends(verify_api_key)
) -> dict:
    """
    Retourne les informations du champion MLflow
    correspondant au scope demandé.
    """

    return get_model_info(scope)


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

    start_time = time.perf_counter()

    try:
        model = get_model(request.scope)

        input_df = pd.DataFrame([
            request.model_dump(
            exclude={"scope"}
            )
        ])

        prediction = model.predict(input_df)[0]

        PREDICTION_COUNTER.inc()

        return PredictionResponse(
            prix_m2=float(prediction),
        )

    except Exception:
        PREDICTION_ERRORS.inc()
        raise

    finally:
        duration = (
            time.perf_counter() - start_time
        )

        PREDICTION_LATENCY.observe(duration)

@app.get("/metrics")
def metrics() -> Response:
    """
    Expose les métriques Prometheus de l'API.
    """

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )