from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator


def collect():
    from src.data.collect import main

    main()


def clean():
    from src.data.clean import main

    main()


def train():
    from src.training.trainer import main

    main()


def evaluate():
    from src.evaluation.metrics import main

    main()


def check_model_quality(**conect):
    from pathlib import Path

    import yaml

    import mlflow

    with open(Path("confid/training_config.yaml")) as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(f"sqlite:///{config['mlflow_db']}")
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["airflow"]["metrics.val_loss ASC"],
        max_results=1,
    )

    if not runs:
        return "skip_deploy"

    best_val_loss = runs[0].data.metrics.get("val_loss", float("inf"))
    print(f"Best val_loss: {best_val_loss:.4f}")

    if best_val_loss < config.get("min_val_loss_threshold", 1.0):
        return "deploy"
    return "skip_deploy"


with DAG(
    dag_id="energy_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["energy", "ml"],
) as dag:
    t1 = PythonOperator(task_id="collect", python_callable=collect)
    t2 = PythonOperator(task_id="clean", python_callable=clean)
    t3 = PythonOperator(task_id="train", python_callable=train)
    t4 = PythonOperator(task_id="evaluate", python_callable=evaluate)

    t5 = BranchPythonOperator(
        task_id="check_quality",
        python_callable=check_model_quality,
        provide_context=True,
    )

    t6 = EmptyOperator(task_id="deploy")
    t7 = EmptyOperator(task_id="skip_deploy")

    t1 >> t2 >> t3 >> t4 >> t5 >> [t6, t7]
