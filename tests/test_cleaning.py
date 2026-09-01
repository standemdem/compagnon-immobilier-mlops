import pandas as pd

from compagnon_immo.data.cleaning import (
    filter_sales,
    filter_mutations_with_apartment,
    filter_allowed_property_types,
    filter_single_apartment_mutations,
    clean_apartment_sales,
)


def test_filter_sales_keeps_only_sales():
    df = pd.DataFrame(
        {
            "id_mutation": ["m1", "m2", "m3"],
            "nature_mutation": ["Vente", "Echange", "Vente"],
        }
    )

    result = filter_sales(df)

    assert list(result["id_mutation"]) == ["m1", "m3"]
    assert (result["nature_mutation"] == "Vente").all()


def test_filter_mutations_with_apartment_keeps_whole_mutation():
    df = pd.DataFrame(
        {
            "id_mutation": ["m1", "m1", "m2", "m3"],
            "type_local": [
                "Appartement",
                "Dépendance",
                "Maison",
                "Appartement",
            ],
        }
    )

    result = filter_mutations_with_apartment(df)

    assert set(result["id_mutation"]) == {"m1", "m3"}

    # La dépendance de m1 doit être conservée avec l'appartement.
    assert len(result[result["id_mutation"] == "m1"]) == 2


def test_filter_allowed_property_types_excludes_mixed_mutations():
    df = pd.DataFrame(
        {
            "id_mutation": [
                "m1",
                "m1",
                "m2",
                "m2",
                "m3",
                "m3",
            ],
            "type_local": [
                "Appartement",
                "Dépendance",
                "Appartement",
                "Maison",
                "Appartement",
                None,
            ],
        }
    )

    result = filter_allowed_property_types(df)

    # m1 est valide : Appartement + Dépendance
    # m2 est exclue : présence d'une Maison
    # m3 est valide : les valeurs nulles sont tolérées
    assert set(result["id_mutation"]) == {"m1", "m3"}


def test_filter_single_apartment_mutations():
    df = pd.DataFrame(
        {
            "id_mutation": [
                "m1",
                "m1",
                "m2",
                "m2",
                "m3",
            ],
            "type_local": [
                "Appartement",
                "Dépendance",
                "Appartement",
                "Appartement",
                "Appartement",
            ],
        }
    )

    result = filter_single_apartment_mutations(df)

    assert set(result["id_mutation"]) == {"m1", "m3"}

def test_clean_apartment_sales_end_to_end():
    df = pd.DataFrame(
        {
            "id_mutation": [
                "m1", "m1",
                "m2",
                "m3", "m3",
                "m4", "m4",
            ],
            "nature_mutation": [
                "Vente", "Vente",
                "Echange",
                "Vente", "Vente",
                "Vente", "Vente",
            ],
            "type_local": [
                "Appartement", "Dépendance",
                "Appartement",
                "Appartement", "Maison",
                "Appartement", None,
            ],
            "surface_terrain": [
                10.0, 5.0,
                20.0,
                30.0, 10.0,
                12.0, 3.0,
            ],
            "surface_reelle_bati": [
                60.0, None,
                70.0,
                80.0, 100.0,
                50.0, None,
            ],
        }
    )

    result = clean_apartment_sales(df)

    assert set(result["id_mutation"]) == {"m1", "m4"}
    assert len(result) == 2

    m1 = result.loc[result["id_mutation"] == "m1"].iloc[0]

    assert bool(m1["has_dependance"]) is True
    assert bool(m1["has_nan_type_local"]) is False
    assert m1["surface_terrain_total"] == 15.0
    assert m1["nb_lignes_mutation"] == 2

    m4 = result.loc[result["id_mutation"] == "m4"].iloc[0]

    assert bool(m4["has_dependance"]) is False
    assert bool(m4["has_nan_type_local"]) is True
    assert m4["surface_terrain_total"] == 15.0

def test_filter_allowed_property_types_excludes_all_null_mutation():
    df = pd.DataFrame(
        {
            "id_mutation": ["m1", "m1", "m2"],
            "type_local": [
                None,
                None,
                "Appartement",
            ],
        }
    )

    result = filter_allowed_property_types(df)

    assert set(result["id_mutation"]) == {"m2"}