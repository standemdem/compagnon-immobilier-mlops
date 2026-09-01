import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CommuneSalesEncoder(BaseEstimator, TransformerMixin):
    """
    Ajoute une variable nb_ventes_commune correspondant
    au nombre d'observations de chaque commune dans le jeu
    utilisé lors du fit.

    Pour une commune inconnue au moment du transform,
    la médiane des volumes communaux du train est utilisée.
    """

    def __init__(self):
        self.commune_counts_ = None
        self.median_ = None

    def fit(self, X: pd.DataFrame, y=None):
        if "nom_commune" not in X.columns:
            raise ValueError(
                "La colonne 'nom_commune' est absente."
            )

        if X["nom_commune"].dropna().empty:
            raise ValueError(
                "La colonne 'nom_commune' ne contient aucune valeur exploitable."
            )

        counts = X.groupby("nom_commune").size()

        self.commune_counts_ = counts
        self.median_ = float(counts.median())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.commune_counts_ is None:
            raise RuntimeError(
                "CommuneSalesEncoder doit être fit avant transform."
            )

        if "nom_commune" not in X.columns:
            raise ValueError(
                "La colonne 'nom_commune' est absente."
            )

        X_transformed = X.copy()

        X_transformed["nb_ventes_commune"] = (
            X_transformed["nom_commune"]
            .map(self.commune_counts_)
            .fillna(self.median_)
        )

        return X_transformed

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Sélectionne uniquement les colonnes définies
    lors de l'initialisation.
    """

    def __init__(self, features: list[str]):
        self.features = features

    def fit(self, X: pd.DataFrame, y=None):
        missing_columns = (
            set(self.features) - set(X.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Colonnes requises manquantes : "
                f"{sorted(missing_columns)}"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing_columns = (
            set(self.features) - set(X.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Colonnes requises manquantes : "
                f"{sorted(missing_columns)}"
            )

        return X[self.features].copy()