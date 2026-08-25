from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


MODEL_PATH = Path(
    "data/models/"
    "prix_m2_pipeline_2020_2023.joblib"
)


@lru_cache(maxsize=1)
def get_model() -> Pipeline:
    """
    Charge le pipeline ML depuis le disque.

    Le cache évite de recharger le modèle
    à chaque requête API.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if not isinstance(model, Pipeline):
        raise TypeError(
            "Le modèle chargé n'est pas un Pipeline sklearn."
        )

    return model