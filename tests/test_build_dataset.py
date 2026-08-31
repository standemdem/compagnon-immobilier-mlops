import pandas as pd
import pytest
from compagnon_immo.data.build_dataset import (
    build_paris_model_dataset,
    filter_paris,
)


def test_filter_paris_keeps_only_department_75():
    df = pd.DataFrame(
        {
            "code_departement": [
                "75",
                "92",
                "93",
                "75",
            ],
            "valeur_fonciere": [
                100000,
                200000,
                300000,
                400000,
            ],
        }
    )

    result = filter_paris(df)

    assert len(result) == 2
    assert result["code_departement"].tolist() == ["75", "75"]

def test_filter_paris_raises_error_without_department_column():
    df = pd.DataFrame(
        {
            "valeur_fonciere": [
                100000,
                200000,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="code_departement",
    ):
        filter_paris(df)

def test_build_paris_model_dataset_keeps_only_paris_rows():
    df = pd.DataFrame(
        {
            "date_mutation": [
                "2024-01-10",
                "2024-02-10",
            ],
            "valeur_fonciere": [
                500000,
                300000,
            ],
            "surface_reelle_bati": [
                50,
                60,
            ],
            "nombre_pieces_principales": [
                2,
                3,
            ],
            "latitude": [
                48.8566,
                48.9000,
            ],
            "longitude": [
                2.3522,
                2.2500,
            ],
            "has_dependance": [
                False,
                False,
            ],
            "nom_commune": [
                "Paris",
                "Boulogne-Billancourt",
            ],
            "code_departement": [
                "75",
                "92",
            ],
        }
    )

    result = build_paris_model_dataset(df)

    assert len(result) == 1
    assert result.iloc[0]["nom_commune"] == "Paris"
    assert result.iloc[0]["prix_m2"] == 10000