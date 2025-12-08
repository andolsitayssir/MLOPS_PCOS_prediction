from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default Arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'pcos_mlops_pipeline',
    default_args=default_args,
    description='Orchestration du pipeline MLOps PCOS avec Airflow',
    schedule_interval=timedelta(days=1), # Run once a day
    catchup=False,
) as dag:

    # Task 1: Check System Health / Git Status
    check_system = BashOperator(
        task_id='check_system',
        bash_command='git status',
    )

    # Task 2: Pull latest data from DVC Remote (S3/DAGsHub)
    pull_data = BashOperator(
        task_id='pull_data',
        bash_command='dvc pull',
    )

    # Task 3: Run the Training Pipeline (Reproduction)
    run_pipeline = BashOperator(
        task_id='run_pipeline',
        bash_command='dvc repro',
    )

    # Task 4: Push new models/data to Remote
    push_data = BashOperator(
        task_id='push_data',
        bash_command='dvc push',
    )

    # Task Dependencies (The Orchestration Flow)
    check_system >> pull_data >> run_pipeline >> push_data
