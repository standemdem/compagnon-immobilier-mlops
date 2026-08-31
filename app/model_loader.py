import os
from functools import lru_cache

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MLFLOW_MODEL_ALIAS = "champion"

REGISTERED_MODELS = {
    "france": "compagnon-immobilier-prix-m2-france",
    "paris": "compagnon-immobilier-prix-m2-paris",
}


@lru_cache(maxsize=2)
def get_model(scope: str) -> Pipeline:
    """
    Charge depuis MLflow Model Registry
    le champion correspondant au scope demandé.

    Un modèle est mis en cache par scope.
    """

    if scope not in REGISTERED_MODELS:
        raise ValueError(
            f"Scope non supporté : {scope}"
        )

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    registered_model_name = (
        REGISTERED_MODELS[scope]
    )

    model_uri = (
        f"models:/"
        f"{registered_model_name}"
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

def get_model_info(scope: str) -> dict:
    """
    Retourne les informations du champion MLflow
    correspondant au scope demandé.
    """

    if scope not in REGISTERED_MODELS:
        raise ValueError(
            f"Scope non supporté : {scope}"
        )

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    registered_model_name = (
        REGISTERED_MODELS[scope]
    )

    client = MlflowClient()

    model_version = (
        client.get_model_version_by_alias(
            registered_model_name,
            MLFLOW_MODEL_ALIAS,
        )
    )

    return {
        "scope": scope,
        "registered_model": registered_model_name,
        "alias": MLFLOW_MODEL_ALIAS,
        "version": int(model_version.version),
        "run_id": model_version.run_id,
    }