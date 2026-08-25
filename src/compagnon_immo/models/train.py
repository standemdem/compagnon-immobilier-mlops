import pandas as pd


def temporal_train_test_split(
    df: pd.DataFrame,
    test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Réalise un split temporel.

    Toutes les observations antérieures à test_year
    constituent le jeu d'entraînement.

    Les observations de test_year constituent
    le jeu de test.
    """

    if "annee_mutation" not in df.columns:
        raise ValueError(
            "La colonne 'annee_mutation' est absente."
        )

    train = df.loc[
        df["annee_mutation"] < test_year
    ].copy()

    test = df.loc[
        df["annee_mutation"] == test_year
    ].copy()

    if train.empty:
        raise ValueError(
            "Le dataset d'entraînement est vide."
        )

    if test.empty:
        raise ValueError(
            f"Aucune observation trouvée pour {test_year}."
        )

    return train, test

def compute_target_bounds(
    train_df: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[float, float]:
    """
    Calcule les bornes de la cible prix_m2
    uniquement à partir du jeu d'entraînement.
    """

    if "prix_m2" not in train_df.columns:
        raise ValueError(
            "La colonne 'prix_m2' est absente du train."
        )

    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError(
            "Les quantiles doivent vérifier : "
            "0 <= lower_quantile < upper_quantile <= 1"
        )

    lower_bound = float(
        train_df["prix_m2"].quantile(lower_quantile)
    )

    upper_bound = float(
        train_df["prix_m2"].quantile(upper_quantile)
    )

    return lower_bound, upper_bound


def apply_target_bounds(
    df: pd.DataFrame,
    lower_bound: float,
    upper_bound: float,
) -> pd.DataFrame:
    """
    Conserve uniquement les observations dont prix_m2
    est compris entre les bornes fournies.
    """

    if "prix_m2" not in df.columns:
        raise ValueError(
            "La colonne 'prix_m2' est absente."
        )

    df_filtered = df.loc[
        df["prix_m2"].between(
            lower_bound,
            upper_bound,
            inclusive="both",
        )
    ].copy()

    return df_filtered

MODEL_FEATURES = [
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "latitude",
    "longitude",
    "has_dependance",
    "nom_commune",
]

TARGET = "prix_m2"


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare les variables explicatives de la cible.
    """

    required_columns = set(MODEL_FEATURES + [TARGET])

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes nécessaires manquantes : "
            f"{sorted(missing_columns)}"
        )

    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].copy()

    return X, y
