import os
from functools import lru_cache

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MLFLOW_REGISTERED_MODEL_NAME = (
    "compagnon-immobilier-prix-m2"
)

MLFLOW_MODEL_ALIAS = "champion"


@lru_cache(maxsize=1)
def get_model() -> Pipeline:
    """
    Charge depuis MLflow Model Registry
    la version du modèle portant l'alias champion.

    Le cache évite de recharger le modèle
    à chaque requête API.
    """

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    model_uri = (
        f"models:/"
        f"{MLFLOW_REGISTERED_MODEL_NAME}"
        f"@{MLFLOW_MODEL_ALIAS}"
    )

    model = mlflow.sklearn.load_model(
        model_uri
    )

    if not isinstance(model, Pipeline):
        raise TypeError(
            "Le modèle MLflow chargé "
            "n'est pas un Pipeline sklearn."
        )

    return model