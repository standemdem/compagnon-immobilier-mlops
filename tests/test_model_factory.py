import pytest
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from compagnon_immo.models.train import (
    build_model_pipeline,
)


def test_build_random_forest_pipeline():
    pipeline = build_model_pipeline(
        model_type="random_forest",
        model_params={
            "n_estimators": 10,
            "random_state": 42,
            "n_jobs": 1,
        },
        scope="france"
    )

    model = pipeline.named_steps["model"]

    assert isinstance(
        model,
        RandomForestRegressor,
    )

    assert model.n_estimators == 10


def test_build_hist_gradient_boosting_pipeline():
    pipeline = build_model_pipeline(
        model_type="hist_gradient_boosting",
        model_params={
            "max_iter": 10,
            "random_state": 42,
        },
        scope="france"
    )

    model = pipeline.named_steps["model"]

    assert isinstance(
        model,
        HistGradientBoostingRegressor,
    )

    assert model.max_iter == 10


def test_build_pipeline_with_unknown_model():
    with pytest.raises(
        ValueError,
        match="Type de modèle non supporté",
    ):
        build_model_pipeline(
            model_type="modele_inconnu",
            model_params={},
            scope="france"
        )

import pandas as pd

from compagnon_immo.models.train import build_model_pipeline


def test_random_forest_pipeline_paris():
    X = pd.DataFrame({
        "surface_reelle_bati": [30, 40, 50, 60],
        "nombre_pieces_principales": [1, 2, 2, 3],
        "latitude": [48.86, 48.87, 48.85, 48.84],
        "longitude": [2.34, 2.35, 2.36, 2.33],
        "has_dependance": [0, 0, 1, 0],
        "nom_commune": [
            "Paris 1er Arrondissement",
            "Paris 2e Arrondissement",
            "Paris 1er Arrondissement",
            "Paris 2e Arrondissement",
        ],
    })

    y = [10000, 9500, 10500, 9700]

    pipeline = build_model_pipeline(
        model_type="random_forest",
        model_params={
            "n_estimators": 2,
            "random_state": 42,
            "n_jobs": 1,
            "max_depth": 3,
            "min_samples_leaf": 1,
        },
        scope="paris",
    )

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)


def test_hist_gradient_boosting_pipeline_paris():
    X = pd.DataFrame({
        "surface_reelle_bati": [30, 40, 50, 60, 70, 80],
        "nombre_pieces_principales": [1, 2, 2, 3, 3, 4],
        "latitude": [48.86, 48.87, 48.85, 48.84, 48.88, 48.83],
        "longitude": [2.34, 2.35, 2.36, 2.33, 2.32, 2.37],
        "has_dependance": [0, 0, 1, 0, 1, 0],
        "nom_commune": [
            "Paris 1er Arrondissement",
            "Paris 2e Arrondissement",
            "Paris 3e Arrondissement",
            "Paris 1er Arrondissement",
            "Paris 2e Arrondissement",
            "Paris 3e Arrondissement",
        ],
    })

    y = [10000, 9500, 9000, 10500, 9700, 8800]

    pipeline = build_model_pipeline(
        model_type="hist_gradient_boosting",
        model_params={
            "max_iter": 5,
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
        scope="paris",
    )

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)