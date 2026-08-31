from datetime import datetime

import docker

from airflow.sdk import DAG, Param, task


DVC_CONTAINER_NAME = "compagnon_dvc"


def run_in_dvc_container(
    command: list[str],
    environment: dict[str, str] | None = None,
) -> None:
    client = docker.from_env()

    container = client.containers.get(
        DVC_CONTAINER_NAME
    )

    result = container.exec_run(
        command,
        workdir="/workspace",
        environment=environment,
    )

    output = result.output.decode()

    if output:
        print(output)

    if result.exit_code != 0:
        raise RuntimeError(
            f"Commande échouée : {' '.join(command)}"
        )


with DAG(
    dag_id="compagnon_immobilier_pipeline",
    description="Orchestration du pipeline MLOps Compagnon Immobilier",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["compagnon-immobilier", "mlops"],
    max_active_runs=1,
    params={
        "scope": Param(
            "paris",
            type="string",
            enum=[
                "france",
                "paris",
            ],
            title="Périmètre géographique",
            description="Périmètre utilisé pour l'entraînement du modèle.",
            section="Données",
        ),
        "model_type": Param(
            "random_forest",
            type="string",
            enum=[
                "random_forest",
                "hist_gradient_boosting",
            ],
            title="Type de modèle",
            description="Algorithme à entraîner.",
            section="Modèle",
        ),
        "experiment_name": Param(
            "rf-experiment",
            type="string",
            minLength=1,
            title="Nom de l'expérience",
            description="Nom utilisé pour identifier l'expérience DVC.",
            section="Modèle",
        ),

        # Random Forest
        "rf_n_estimators": Param(
            50,
            type="integer",
            minimum=1,
            maximum=500,
            title="Nombre d'arbres",
            description="Nombre d'arbres du Random Forest.",
            section="Random Forest",
        ),
        "rf_max_depth": Param(
            20,
            type="integer",
            minimum=1,
            maximum=100,
            title="Profondeur maximale",
            description="Profondeur maximale des arbres.",
            section="Random Forest",
        ),
        "rf_min_samples_leaf": Param(
            2,
            type="integer",
            minimum=1,
            title="Minimum d'échantillons par feuille",
            description="Minimum d'échantillons dans une feuille.",
            section="Random Forest",
        ),

        # HistGradientBoosting
        "hgb_max_iter": Param(
            100,
            type="integer",
            minimum=1,
            maximum=1000,
            title="Nombre maximal d'itérations",
            description="Nombre maximal d'itérations du HistGradientBoosting.",
            section="HistGradientBoosting",
        ),
        "hgb_learning_rate": Param(
            0.1,
            type="number",
            exclusiveMinimum=0,
            maximum=1,
            title="Learning rate",
            description="Taux d'apprentissage du HistGradientBoosting.",
            section="HistGradientBoosting",
        ),
        "hgb_max_depth": Param(
            20,
            type="integer",
            minimum=1,
            maximum=100,
            title="Profondeur maximale",
            description="Profondeur maximale du HistGradientBoosting.",
            section="HistGradientBoosting",
        ),
        "hgb_min_samples_leaf": Param(
            20,
            type="integer",
            minimum=1,
            title="Minimum d'échantillons par feuille",
            description="Minimum d'échantillons dans une feuille.",
            section="HistGradientBoosting",
        ),
    },
) as dag:

    @task
    def dvc_pull():
        run_in_dvc_container(
            ["dvc", "pull"]
        )

    @task(retries=0)
    def dvc_experiment(**context):
        params = context["params"]

        scope = params["scope"]
        model_type = params["model_type"]
        experiment_name = params["experiment_name"]

        command = [
            "dvc",
            "exp",
            "run",
            "--name",
            experiment_name,
            "--set-param",
            f"training.scope={scope}",
            "--set-param",
            f"model.type={model_type}",
        ]

        if model_type == "random_forest":
            command += [
                "--set-param",
                f"model.random_forest.n_estimators={params['rf_n_estimators']}",
                "--set-param",
                f"model.random_forest.max_depth={params['rf_max_depth']}",
                "--set-param",
                f"model.random_forest.min_samples_leaf={params['rf_min_samples_leaf']}",
            ]

        elif model_type == "hist_gradient_boosting":
            command += [
                "--set-param",
                f"model.hist_gradient_boosting.max_iter={params['hgb_max_iter']}",
                "--set-param",
                f"model.hist_gradient_boosting.learning_rate={params['hgb_learning_rate']}",
                "--set-param",
                f"model.hist_gradient_boosting.max_depth={params['hgb_max_depth']}",
                "--set-param",
                f"model.hist_gradient_boosting.min_samples_leaf={params['hgb_min_samples_leaf']}",
            ]

        else:
            raise ValueError(
                f"Type de modèle non supporté : {model_type}"
            )

        print("Commande DVC :", " ".join(command))

        run_in_dvc_container(
            command,
            environment={
                "MLFLOW_RUN_NAME": experiment_name,
            },
        )

    @task
    def dvc_push():
        run_in_dvc_container(
            ["dvc", "push"]
        )

    @task(trigger_rule="all_done")
    def restore_workspace():
        print("=== Restauration du workspace DVC ===")

        run_in_dvc_container(
            [
                "git",
                "restore",
                "--source=HEAD",
                "--",
                "params.yaml",
                "dvc.lock",
            ]
        )

        run_in_dvc_container(
            [
                "dvc",
                "checkout",
                "train",
                "--force",
            ]
        )

        print("Workspace DVC restauré.")

    dvc_pull() >> dvc_experiment() >> dvc_push() >> restore_workspace()
