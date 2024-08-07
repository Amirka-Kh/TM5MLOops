from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os

os.environ['HYDRA_FULL_ERROR'] = '1'

default_args = {
    'owner': 'team5',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    'data_extract_dag',
    default_args=default_args,
    description='A simple data extraction pipeline',
    schedule_interval=timedelta(minutes=7),
    start_date=days_ago(0),
    catchup=False,
) as dag:

    # Task 1: Extract a new sample of the data
    extract_task = BashOperator(
        task_id='extract_data',
        bash_command='source $PYTHONPATH/venv/bin/activate && python $PYTHONPATH/src/data.py',
    )

    # Task 2: Validate the sample using Great Expectations
    # validate_task = PythonOperator(
    #     task_id='validate_data',
    #     python_callable=validate_initial_data,
    # )

    # Task 3: Version the sample
    version_task = BashOperator(
        task_id='version_data',
        bash_command='source $PYTHONPATH/venv/bin/activate && python $PYTHONPATH/src/version_data.py',
    )

    # Task 4: Load the sample to the data store
    # load_task = BashOperator(
    #     task_id='load_data',
    #     bash_command='dvc push; echo "Data Preparation Finished"',
    # )

    # Define the task dependencies
    # extract_task >> validate_task >> version_task >> load_task
    extract_task >> version_task
