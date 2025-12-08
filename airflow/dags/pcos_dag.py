from airflow import DAG
from airflow.operators.bash import BashOperator  # fixed import
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='pcos_mlops_pipeline',
    default_args=default_args,
    description='Orchestration du pipeline MLOps PCOS avec Airflow',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    check_system = BashOperator(
        task_id='check_system',
        bash_command='git status',
    )

    pull_data = BashOperator(
        task_id='pull_data',
        bash_command='dvc pull',
    )

    run_pipeline = BashOperator(
        task_id='run_pipeline',
        bash_command='dvc repro',
    )

    push_data = BashOperator(
        task_id='push_data',
        bash_command='dvc push',
    )

    check_system >> pull_data >> run_pipeline >> push_data