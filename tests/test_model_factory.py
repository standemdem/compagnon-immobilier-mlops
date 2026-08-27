import pytest

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
        )