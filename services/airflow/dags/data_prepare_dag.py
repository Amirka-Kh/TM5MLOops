from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 7, 4),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Define the DAG
with DAG(
    'data_prepare_dag',
    default_args=default_args,
    description='A DAG to run the ZenML pipeline after data extraction pipeline is successful',
    schedule_interval=timedelta(minutes=5),
    catchup=False,
) as dag:

    # Task to wait for the completion of data extraction pipeline
    wait_for_data_extraction = ExternalTaskSensor(
        task_id='wait_for_data_extraction',
        external_dag_id='data_extract_dag',
        external_task_id=None,
        mode='poke',
        timeout=600,
        poke_interval=60,
    )

    # Task to run the ZenML pipeline
    run_zenml_pipeline = BashOperator(
        task_id='run_zenml_pipeline',
        bash_command='source $PYTHONPATH/venv/bin/activate && python $PYTHONPATH/src/prepare_data.py',
        cwd='/mnt/c/Users/amira/PycharmProjects/MLOps/',
    )

    # Define task dependencies
    wait_for_data_extraction >> run_zenml_pipeline
