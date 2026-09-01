import pytest
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from compagnon_immo.models.train import (
    build_model_pipeline,
    compute_target_bounds,
    temporal_train_test_split,
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

def test_compute_target_bounds_rejects_empty_target():
    df = pd.DataFrame(
        {
            "prix_m2": [None, None],
        }
    )

    with pytest.raises(
        ValueError,
        match="aucune valeur exploitable",
    ):
        compute_target_bounds(df)

def test_temporal_train_test_split():
    df = pd.DataFrame(
        {
            "annee_mutation": [2020, 2021, 2023, 2024, 2024],
            "prix_m2": [1000, 1100, 1200, 1300, 1400],
        }
    )

    train, test = temporal_train_test_split(
        df,
        test_year=2024,
    )

    assert set(train["annee_mutation"]) == {
        2020,
        2021,
        2023,
    }

    assert set(test["annee_mutation"]) == {
        2024,
    }

    assert len(train) == 3
    assert len(test) == 2

def test_temporal_train_test_split_rejects_missing_test_year():
    df = pd.DataFrame(
        {
            "annee_mutation": [2020, 2021, 2022, 2023],
            "prix_m2": [1000, 1100, 1200, 1300],
        }
    )

    with pytest.raises(
        ValueError,
        match="Aucune observation trouvée pour 2024",
    ):
        temporal_train_test_split(
            df,
            test_year=2024,
        )

def test_random_forest_pipeline_france_fit_predict():
    X = pd.DataFrame(
        {
            "surface_reelle_bati": [40, 50, 60, 70],
            "nombre_pieces_principales": [2, 2, 3, 4],
            "latitude": [48.85, 48.86, 45.75, 43.30],
            "longitude": [2.35, 2.36, 4.85, 5.37],
            "has_dependance": [0, 1, 0, 1],
            "nom_commune": [
                "Paris",
                "Paris",
                "Lyon",
                "Marseille",
            ],
        }
    )

    y = [9000, 9500, 4500, 4000]

    pipeline = build_model_pipeline(
        model_type="random_forest",
        model_params={
            "n_estimators": 2,
            "random_state": 42,
            "n_jobs": 1,
        },
        scope="france",
    )

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)

def test_hist_gradient_boosting_pipeline_france_fit_predict():
    X = pd.DataFrame(
        {
            "surface_reelle_bati": [40, 50, 60, 70, 80, 90],
            "nombre_pieces_principales": [2, 2, 3, 4, 4, 5],
            "latitude": [48.85, 48.86, 45.75, 43.30, 44.84, 47.22],
            "longitude": [2.35, 2.36, 4.85, 5.37, -0.58, -1.55],
            "has_dependance": [0, 1, 0, 1, 0, 1],
            "nom_commune": [
                "Paris",
                "Paris",
                "Lyon",
                "Marseille",
                "Bordeaux",
                "Nantes",
            ],
        }
    )

    y = [9000, 9500, 4500, 4000, 5000, 4200]

    pipeline = build_model_pipeline(
        model_type="hist_gradient_boosting",
        model_params={
            "max_iter": 5,
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
        scope="france",
    )

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)

def test_build_pipeline_with_unknown_scope():
    with pytest.raises(
        ValueError,
        match="Scope non supporté : europe",
    ):
        build_model_pipeline(
            model_type="random_forest",
            model_params={
                "n_estimators": 2,
                "random_state": 42,
            },
            scope="europe",
        )
