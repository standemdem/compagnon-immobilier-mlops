import pandas as pd
import pytest

from compagnon_immo.models.transformers import (
    CommuneSalesEncoder,
    FeatureSelector,
)

# Tests de CommuneSalesEncoder

def test_commune_sales_encoder_fit_transform():
    df = pd.DataFrame(
        {
            "nom_commune": [
                "Paris",
                "Paris",
                "Lyon",
                "Marseille",
                "Marseille",
                "Marseille",
            ]
        }
    )

    transformer = CommuneSalesEncoder()

    result = transformer.fit_transform(df)

    assert "nb_ventes_commune" in result.columns

    assert result.loc[0, "nb_ventes_commune"] == 2
    assert result.loc[1, "nb_ventes_commune"] == 2
    assert result.loc[2, "nb_ventes_commune"] == 1
    assert result.loc[3, "nb_ventes_commune"] == 3


def test_commune_sales_encoder_unknown_commune():
    train = pd.DataFrame(
        {
            "nom_commune": [
                "Paris",
                "Paris",
                "Lyon",
                "Marseille",
                "Marseille",
                "Marseille",
            ]
        }
    )

    test = pd.DataFrame(
        {
            "nom_commune": [
                "Paris",
                "Bordeaux",
            ]
        }
    )

    transformer = CommuneSalesEncoder()
    transformer.fit(train)

    result = transformer.transform(test)

    assert result.loc[0, "nb_ventes_commune"] == 2

    expected_fallback = transformer.median_

    assert result.loc[1, "nb_ventes_commune"] == expected_fallback

# Test de FeatureSelector

def test_feature_selector_selects_expected_columns():
    df = pd.DataFrame(
        {
            "surface_reelle_bati": [50, 70],
            "latitude": [48.85, 45.75],
            "longitude": [2.35, 4.85],
            "colonne_inutile": [100, 200],
        }
    )

    features = [
        "surface_reelle_bati",
        "latitude",
        "longitude",
    ]

    selector = FeatureSelector(features=features)

    result = selector.fit_transform(df)

    assert list(result.columns) == features
    assert "colonne_inutile" not in result.columns
    assert len(result) == 2


def test_feature_selector_missing_column():
    df = pd.DataFrame(
        {
            "surface_reelle_bati": [50],
            "latitude": [48.85],
        }
    )

    selector = FeatureSelector(
        features=[
            "surface_reelle_bati",
            "latitude",
            "longitude",
        ]
    )

    with pytest.raises(ValueError, match="longitude"):
        selector.fit(df)

def test_commune_sales_encoder_rejects_empty_commune():
    X = pd.DataFrame(
        {
            "nom_commune": [None, None],
        }
    )

    encoder = CommuneSalesEncoder()

    with pytest.raises(
        ValueError,
        match="aucune valeur exploitable",
    ):
        encoder.fit(X)
