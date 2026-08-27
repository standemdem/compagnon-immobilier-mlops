import os
import yaml
import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
from mlflow.models import infer_signature

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

###########################################
# Constantes
###########################################

PARAMS_PATH = Path("params.yaml")

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MLFLOW_EXPERIMENT_NAME = "compagnon-immobilier"

DATASET_PATH = Path(
    "data/prod/"
    "dvf_appartements_model_base_2020_2024.parquet.gz"
)

METADATA_PATH = Path(
    "data/models/"
    "prix_m2_pipeline_2020_2023.metadata.json"
)

MODEL_PATH = Path(
    "data/models/"
    "prix_m2_pipeline_2020_2023.joblib"
)

MLFLOW_REGISTERED_MODEL_NAME = "compagnon-immobilier-prix-m2"

############################################

with PARAMS_PATH.open("r", encoding="utf-8",) as file:
    config = yaml.safe_load(file)

model_config = config["model"]
model_params = model_config["params"]

test_year = config["training"]["test_year"]

lower_quantile = (config["target_filter"]["lower_quantile"])
upper_quantile = (config["target_filter"]["upper_quantile"])

def main() -> None:

    # --------------------------------------------------
    # MLFlow
    # --------------------------------------------------
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )
    # --------------------------------------------------
    # Chargement du dataset
    # --------------------------------------------------
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
        test_year=test_year,
    )

    print(f"Train : {len(train)} lignes")
    print(f"Test  : {len(test)} lignes")

    # --------------------------------------------------
    # Bornes de la cible
    # --------------------------------------------------

    print("\n=== Filtrage de la cible ===")

    lower_bound, upper_bound = compute_target_bounds(
        train, 
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
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
    print("\n=== MLflow ===")
    print(f"Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"Experiment   : {MLFLOW_EXPERIMENT_NAME}")

    with mlflow.start_run() as run:

        print(f"Run ID       : {run.info.run_id}")
        print("\n=== Construction du pipeline ===")
        pipeline = build_model_pipeline(**model_params)


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

        print(f"\n=== Évaluation sur {test_year} ===")

        predictions = pipeline.predict(X_test)

        signature = infer_signature(
            X_test,
            predictions,
        )

        metrics = evaluate_regression(
            y_test,
            predictions,
        )
        mlflow.log_metrics(
            {
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            }
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
            "test_year": test_year,

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
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },

            "dataset": {
                "train_rows": len(train),
                "test_rows": len(test),
            },

            "metrics": metrics,

            "model": {
                "type": model_config["type"],
                "parameters": model_params,
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
        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model",
        )

        mlflow.log_artifact(
            str(METADATA_PATH),
            artifact_path="metadata",
        )
        mlflow.log_params(
            {
                "model_type": model_config["type"],
                **model_params,
                "test_year": test_year,
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "train_rows": len(train),
                "test_rows": len(test),
            }
        )

        print("\n=== Enregistrement MLflow Model Registry ===")

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=X_test.head(5),
            registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
            serialization_format="cloudpickle",
        )

        client = MlflowClient()

        model_versions = client.search_model_versions(
            filter_string=(
                f"name='{MLFLOW_REGISTERED_MODEL_NAME}' "
                f"AND run_id='{run.info.run_id}'"
            )
        )

        if len(model_versions) != 1:
            raise RuntimeError(
                "Impossible d'identifier de manière unique "
                "la version MLflow créée par ce run."
            )

        registered_version = model_versions[0]

        
        print(
            f"Modèle MLflow enregistré : "
            f"{MLFLOW_REGISTERED_MODEL_NAME}"
        )

        print(
            f"Version MLflow : "
            f"{registered_version.version}"
        )

        
if __name__ == "__main__":
    main()