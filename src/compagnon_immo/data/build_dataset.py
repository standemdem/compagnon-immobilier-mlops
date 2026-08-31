import pandas as pd


def add_price_per_square_meter(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ajoute la variable cible prix_m2.

    Le prix au m² correspond à :
    valeur_fonciere / surface_reelle_bati.

    Les lignes dont la surface bâtie est nulle, négative
    ou manquante sont exclues avant le calcul.
    """

    required_columns = {
        "valeur_fonciere",
        "surface_reelle_bati",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes requises manquantes : "
            f"{sorted(missing_columns)}"
        )

    df_target = df.loc[
        df["surface_reelle_bati"].notna()
        & (df["surface_reelle_bati"] > 0)
        & df["valeur_fonciere"].notna()
        & (df["valeur_fonciere"] > 0)
    ].copy()

    df_target["prix_m2"] = (
        df_target["valeur_fonciere"]
        / df_target["surface_reelle_bati"]
    )

    return df_target

def filter_price_outliers(
    df: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Exclut les valeurs extrêmes de prix_m2
    selon deux quantiles.

    Par défaut :
    - 1er percentile
    - 99e percentile
    """

    if "prix_m2" not in df.columns:
        raise ValueError(
            "La colonne 'prix_m2' est absente du DataFrame."
        )

    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError(
            "Les quantiles doivent vérifier : "
            "0 <= lower_quantile < upper_quantile <= 1"
        )

    lower_bound = df["prix_m2"].quantile(
        lower_quantile
    )

    upper_bound = df["prix_m2"].quantile(
        upper_quantile
    )

    df_filtered = df.loc[
        df["prix_m2"].between(
            lower_bound,
            upper_bound,
            inclusive="both",
        )
    ].copy()

    return df_filtered

def filter_valid_coordinates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Conserve uniquement les observations disposant
    d'une latitude et d'une longitude.
    """

    required_columns = {
        "latitude",
        "longitude",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes requises manquantes : "
            f"{sorted(missing_columns)}"
        )

    df_filtered = df.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    return df_filtered


def filter_metropolitan_france(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Conserve uniquement les observations situées
    en France métropolitaine à partir du code département.

    Sont exclus :
    - départements et régions d'outre-mer ;
    - collectivités d'outre-mer.
    """

    if "code_departement" not in df.columns:
        raise ValueError(
            "La colonne 'code_departement' est absente du DataFrame."
        )

    department = (
        df["code_departement"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    metropolitan_mask = (
        department.str.fullmatch(
            r"(0[1-9]|[1-8][0-9]|9[0-5]|2A|2B)"
        )
        .fillna(False)
    )

    df_filtered = df.loc[
        metropolitan_mask
    ].copy()

    return df_filtered

def filter_paris(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Conserve uniquement les observations situées à Paris
    à partir du code département.
    """

    if "code_departement" not in df.columns:
        raise ValueError(
            "La colonne 'code_departement' est absente du DataFrame."
        )

    department = (
        df["code_departement"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    paris_mask = (
        department == "75"
    ).fillna(False)

    df_filtered = df.loc[
        paris_mask
    ].copy()

    return df_filtered

def add_mutation_year(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ajoute la colonne annee_mutation à partir de date_mutation.
    """

    if "date_mutation" not in df.columns:
        raise ValueError(
            "La colonne 'date_mutation' est absente du DataFrame."
        )

    df_with_year = df.copy()

    df_with_year["date_mutation"] = pd.to_datetime(
        df_with_year["date_mutation"],
        errors="coerce",
    )

    if df_with_year["date_mutation"].isna().any():
        raise ValueError(
            "Certaines valeurs de date_mutation sont invalides."
        )

    df_with_year["annee_mutation"] = (
        df_with_year["date_mutation"].dt.year
    )

    return df_with_year

MODEL_COLUMNS = [
    "date_mutation",
    "annee_mutation",
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "latitude",
    "longitude",
    "has_dependance",
    "nom_commune",
    "prix_m2",
]

def select_model_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sélectionne uniquement les colonnes nécessaires
    à la modélisation et à l'évaluation temporelle.
    """

    missing_columns = (
        set(MODEL_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Colonnes modèle manquantes : "
            f"{sorted(missing_columns)}"
        )

    return df[MODEL_COLUMNS].copy()


def build_base_model_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prépare un dataset annuel pour la modélisation.

    Étapes :
    1. création de prix_m2 ;
    2. suppression des coordonnées manquantes ;
    3. filtre France métropolitaine ;
    4. ajout de l'année de mutation.

    Le filtre des valeurs extrêmes de prix_m2
    n'est volontairement pas appliqué ici.
    """

    df_model = add_price_per_square_meter(df)

    df_model = filter_valid_coordinates(df_model)

    df_model = filter_metropolitan_france(df_model)

    df_model = add_mutation_year(df_model)

    df_model = select_model_columns(df_model)

    return df_model

def build_paris_model_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prépare un dataset annuel limité à Paris
    pour la modélisation.

    Étapes :
    1. création de prix_m2 ;
    2. suppression des coordonnées manquantes ;
    3. filtre France métropolitaine ;
    4. filtre Paris ;
    5. ajout de l'année de mutation ;
    6. sélection des colonnes modèle.
    """

    df_model = add_price_per_square_meter(df)

    df_model = filter_valid_coordinates(df_model)

    df_model = filter_metropolitan_france(df_model)

    df_model = filter_paris(df_model)

    df_model = add_mutation_year(df_model)

    df_model = select_model_columns(df_model)

    return df_model

def concatenate_years(
    datasets: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Concatène plusieurs datasets annuels.

    Vérifie que tous possèdent exactement les mêmes colonnes.
    """

    if not datasets:
        raise ValueError(
            "Aucun dataset fourni pour la concaténation."
        )

    reference_columns = list(datasets[0].columns)

    for index, df in enumerate(datasets[1:], start=2):
        if list(df.columns) != reference_columns:
            raise ValueError(
                f"Le dataset n°{index} n'a pas "
                "le même schéma que le premier."
            )

    df_combined = pd.concat(
        datasets,
        ignore_index=True,
    )

    return df_combined

