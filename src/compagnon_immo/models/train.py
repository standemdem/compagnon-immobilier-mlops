import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor)
from sklearn.pipeline import Pipeline

from compagnon_immo.models.transformers import (
    CommuneSalesEncoder,
    FeatureSelector,
)

MODEL_FEATURES = [
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "latitude",
    "longitude",
    "has_dependance",
    "nom_commune",
]

FINAL_MODEL_FEATURES = [
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "latitude",
    "longitude",
    "has_dependance",
    "nb_ventes_commune",
]

TARGET = "prix_m2"

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


def build_model_pipeline(
    model_type: str,
    model_params: dict,
) -> Pipeline:
    """
    Construit le pipeline sklearn complet.

    Étapes :
    1. création de nb_ventes_commune ;
    2. sélection des features numériques finales ;
    3. entraînement du modèle configuré.
    """

    if model_type == "random_forest":
        estimator = RandomForestRegressor(
            **model_params
        )

    elif model_type == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            **model_params
        )

    else:
        raise ValueError(
            f"Type de modèle non supporté : {model_type}"
        )

    pipeline = Pipeline(
        steps=[
            (
                "commune_sales_encoder",
                CommuneSalesEncoder(),
            ),
            (
                "feature_selector",
                FeatureSelector(
                    features=FINAL_MODEL_FEATURES
                ),
            ),
            (
                "model",
                estimator,
            ),
        ]
    )

    return pipeline

def save_model(
    pipeline: Pipeline,
    output_path: Path,
) -> None:
    """
    Sauvegarde le pipeline sklearn entraîné.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        output_path,
    )

    print(f"Modèle sauvegardé : {output_path}")


def save_model_metadata(
    metadata: dict,
    output_path: Path,
) -> None:
    """
    Sauvegarde les métadonnées du modèle au format JSON.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Métadonnées sauvegardées : {output_path}")