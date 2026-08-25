from pathlib import Path
import pandas as pd

from compagnon_immo.models.evaluate import (
    evaluate_regression,
)
from compagnon_immo.models.train import (
    apply_target_bounds,
    build_model_pipeline,
    compute_target_bounds,
    split_features_target,
    temporal_train_test_split,
    save_model,
    save_model_metadata,
)

DATASET_PATH = Path(
    "data/prod/"
    "dvf_appartements_model_base_2020_2024.parquet.gz"
)

METADATA_PATH = Path(
    "data/models/"
    "prix_m2_pipeline_2020_2023.metadata.json"
)

TEST_YEAR = 2024

MODEL_PATH = Path(
    "data/models/"
    "prix_m2_pipeline_2020_2023.joblib"
)

def main() -> None:
    print("=== Chargement du dataset ===")

    df = pd.read_parquet(DATASET_PATH)

    print(
        f"Dataset : {len(df)} lignes / "
        f"{len(df.columns)} colonnes"
    )

    # --------------------------------------------------
    # Split temporel
    # --------------------------------------------------

    print("\n=== Split temporel ===")

    train, test = temporal_train_test_split(
        df,
        test_year=TEST_YEAR,
    )

    print(f"Train : {len(train)} lignes")
    print(f"Test  : {len(test)} lignes")

    # --------------------------------------------------
    # Bornes de la cible
    # --------------------------------------------------

    print("\n=== Filtrage de la cible ===")

    lower_bound, upper_bound = compute_target_bounds(
        train
    )

    print(f"Borne basse : {lower_bound:.2f}")
    print(f"Borne haute : {upper_bound:.2f}")

    train = apply_target_bounds(
        train,
        lower_bound,
        upper_bound,
    )

    test = apply_target_bounds(
        test,
        lower_bound,
        upper_bound,
    )

    print(f"Train filtré : {len(train)} lignes")
    print(f"Test filtré  : {len(test)} lignes")

    # --------------------------------------------------
    # X / y
    # --------------------------------------------------

    print("\n=== Préparation des features ===")

    X_train, y_train = split_features_target(train)
    X_test, y_test = split_features_target(test)

    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")

    # --------------------------------------------------
    # Pipeline
    # --------------------------------------------------

    print("\n=== Construction du pipeline ===")
    pipeline = build_model_pipeline(
        n_estimators=50,
        random_state=42,
        n_jobs=2,
        max_depth=20,
        min_samples_leaf=2,
    )


    # --------------------------------------------------
    # Entraînement
    # --------------------------------------------------

    print("\n=== Entraînement ===")

    pipeline.fit(
        X_train,
        y_train,
    )

    print("Entraînement terminé.")

    # --------------------------------------------------
    # Évaluation
    # --------------------------------------------------

    print("\n=== Évaluation sur 2024 ===")

    predictions = pipeline.predict(X_test)

    metrics = evaluate_regression(
        y_test,
        predictions,
    )

    print(f"MAE  : {metrics['mae']:.4f}")
    print(f"RMSE : {metrics['rmse']:.4f}")
    print(f"R²   : {metrics['r2']:.4f}")

    metadata = {
        "model_name": "prix_m2_pipeline",
        "model_type": "RandomForestRegressor",

        "training_years": [
            2020,
            2021,
            2022,
            2023,
        ],
        "test_year": TEST_YEAR,

        "target": "prix_m2",

        "input_features": [
            "surface_reelle_bati",
            "nombre_pieces_principales",
            "latitude",
            "longitude",
            "has_dependance",
            "nom_commune",
        ],

        "model_features": [
            "surface_reelle_bati",
            "nombre_pieces_principales",
            "latitude",
            "longitude",
            "has_dependance",
            "nb_ventes_commune",
        ],

        "target_filter": {
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        },

        "dataset": {
            "train_rows": len(train),
            "test_rows": len(test),
        },

        "metrics": metrics,

        "random_forest_parameters": {
            "n_estimators": 50,
            "random_state": 42,
            "n_jobs": 2,
            "max_depth": 20,
            "min_samples_leaf": 2,
        },
    }


    print("\n=== Sauvegarde du modèle ===")

    save_model(
        pipeline=pipeline,
        output_path=MODEL_PATH,
    )

    save_model_metadata(
        metadata=metadata,
        output_path=METADATA_PATH,
    )
    
if __name__ == "__main__":
    main()