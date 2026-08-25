import pandas as pd


def filter_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conserve uniquement les mutations de nature 'Vente'.
    """

    df_sales = df.loc[
        df["nature_mutation"] == "Vente"
    ].copy()

    return df_sales


def filter_mutations_with_apartment(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Conserve uniquement les mutations contenant
    au moins une ligne de type 'Appartement'.
    """

    apartment_mutation_ids = (
        df.loc[
            df["type_local"] == "Appartement",
            "id_mutation",
        ]
        .dropna()
        .unique()
    )

    df_apartment_mutations = df.loc[
        df["id_mutation"].isin(apartment_mutation_ids)
    ].copy()

    return df_apartment_mutations


ALLOWED_PROPERTY_TYPES = {
    "Appartement",
    "Dépendance",
}


def filter_allowed_property_types(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Conserve les mutations dont les types de locaux non nuls
    sont uniquement 'Appartement' ou 'Dépendance'.

    Les valeurs manquantes de type_local sont tolérées.
    """

    valid_mutation_ids = (
        df.groupby("id_mutation")["type_local"]
        .apply(
            lambda values: set(values.dropna()).issubset(
                ALLOWED_PROPERTY_TYPES
            )
        )
    )

    valid_mutation_ids = valid_mutation_ids[
        valid_mutation_ids
    ].index

    df_filtered = df.loc[
        df["id_mutation"].isin(valid_mutation_ids)
    ].copy()

    return df_filtered

def filter_single_apartment_mutations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Conserve uniquement les mutations contenant
    exactement un appartement.

    Les dépendances et les valeurs manquantes de type_local
    peuvent rester associées à la mutation.
    """

    apartment_count = (
        df["type_local"]
        .eq("Appartement")
        .groupby(df["id_mutation"])
        .sum()
    )

    valid_mutation_ids = apartment_count[
        apartment_count == 1
    ].index

    df_filtered = df.loc[
        df["id_mutation"].isin(valid_mutation_ids)
    ].copy()

    return df_filtered

def aggregate_apartment_mutations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produit une ligne par mutation/appartement.

    La ligne principale conservée est celle de l'appartement.
    Des informations calculées au niveau de la mutation sont ajoutées.
    """

    apartment_rows = (
        df.loc[df["type_local"] == "Appartement"]
        .copy()
    )

    mutation_features = (
        df.groupby("id_mutation")
        .agg(
            has_dependance=(
                "type_local",
                lambda values: values.eq("Dépendance").any(),
            ),
            has_nan_type_local=(
                "type_local",
                lambda values: values.isna().any(),
            ),
            surface_terrain_total=(
                "surface_terrain",
                "sum",
            ),
            nb_lignes_mutation=(
                "id_mutation",
                "size",
            ),
        )
        .reset_index()
    )

    df_aggregated = apartment_rows.merge(
        mutation_features,
        on="id_mutation",
        how="left",
        validate="one_to_one",
    )

    return df_aggregated


def clean_apartment_sales(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Applique l'ensemble des règles métier permettant
    d'obtenir une ligne par appartement vendu.

    Pipeline :
    1. ventes uniquement ;
    2. mutations contenant au moins un appartement ;
    3. exclusion des mutations mixtes ;
    4. exactement un appartement par mutation ;
    5. agrégation au niveau de la mutation.
    """

    df = filter_sales(df)

    df = filter_mutations_with_apartment(df)

    df = filter_allowed_property_types(df)

    df = filter_single_apartment_mutations(df)

    df = aggregate_apartment_mutations(df)

    return df