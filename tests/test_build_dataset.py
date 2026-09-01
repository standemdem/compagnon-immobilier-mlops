import pandas as pd
import pytest
from compagnon_immo.data.build_dataset import (
    add_price_per_square_meter,
    add_mutation_year,
    build_paris_model_dataset,
    concatenate_years,
    filter_metropolitan_france,
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

def test_add_price_per_square_meter_filters_invalid_values():
    df = pd.DataFrame(
        {
            "valeur_fonciere": [
                200000,
                200000,
                0,
                -100000,
                None,
            ],
            "surface_reelle_bati": [
                50,
                0,
                50,
                50,
                50,
            ],
        }
    )

    result = add_price_per_square_meter(df)

    assert len(result) == 1
    assert result.iloc[0]["prix_m2"] == 4000

def test_add_mutation_year_rejects_invalid_date():
    df = pd.DataFrame(
        {
            "date_mutation": [
                "2024-01-10",
                "date-invalide",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="date_mutation sont invalides",
    ):
        add_mutation_year(df)

def test_filter_metropolitan_france_keeps_mainland_and_corsica():
    df = pd.DataFrame(
        {
            "code_departement": [
                "01",
                "75",
                "95",
                "2A",
                "2B",
                "971",
                "974",
            ],
        }
    )

    result = filter_metropolitan_france(df)

    assert result["code_departement"].tolist() == [
        "01",
        "75",
        "95",
        "2A",
        "2B",
    ]

def test_concatenate_years_rejects_different_schemas():
    df_2023 = pd.DataFrame(
        {
            "prix_m2": [3000],
            "annee_mutation": [2023],
        }
    )

    df_2024 = pd.DataFrame(
        {
            "annee_mutation": [2024],
            "prix_m2": [3200],
        }
    )

    with pytest.raises(
        ValueError,
        match="même schéma",
    ):
        concatenate_years([df_2023, df_2024])
