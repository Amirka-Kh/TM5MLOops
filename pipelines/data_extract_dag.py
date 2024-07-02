from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

from src.sample_data import sample_data
from src.validate_data import validate_initial_data
from src.version_data import version_data

default_args = {
    'owner': 'team5',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'data_extract_dag',
    default_args=default_args,
    description='A simple data extraction pipeline',
    schedule_interval=timedelta(minutes=5),
    start_date=days_ago(0),
    catchup=False,
) as dag:

    # Task 1: Extract a new sample of the data
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=sample_data,
    )

    # Task 2: Validate the sample using Great Expectations
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_initial_data,
    )

    # Task 3: Version the sample
    version_task = PythonOperator(
        task_id='version_data',
        python_callable=version_data,
    )

    # Task 4: Load the sample to the data store
    load_task = BashOperator(
        task_id='load_data',
        bash_command='dvc push; echo "Data Preparation Finished"',
    )

    # Define the task dependencies
    extract_task >> validate_task >> version_task >> load_task
