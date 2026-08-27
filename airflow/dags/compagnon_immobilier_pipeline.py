from datetime import datetime

import docker

from airflow.sdk import DAG, task


DVC_CONTAINER_NAME = "compagnon_dvc"


def run_in_dvc_container(command: list[str]) -> None:
    client = docker.from_env()

    container = client.containers.get(
        DVC_CONTAINER_NAME
    )

    result = container.exec_run(
        command,
        workdir="/workspace",
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
) as dag:

    @task
    def dvc_pull():
        run_in_dvc_container(
            ["dvc", "pull"]
        )

    @task
    def dvc_repro():
        run_in_dvc_container(
            ["dvc", "repro"]
        )

    @task
    def dvc_push():
        run_in_dvc_container(
            ["dvc", "push"]
        )

    dvc_pull() >> dvc_repro() >> dvc_push()