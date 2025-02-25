# ./dags/birdclef_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def run_unpack_to_raw():
    """
    We'll directly import and call main() from your preprocess_to_raw.py
    """
    from scripts.unpack_to_raw import main
    main()

def run_preprocess_audiofiles_to_staging():
    """
    We'll directly import and call main() from ingest_csv_from_minio_to_mongodb.py
    """
    from scripts.preprocess_audiofiles_to_staging import main
    main()
    
def run_preprocess_metadata_to_staging():
    """
    We'll directly import and call main() from ingest_csv_from_minio_to_mongodb.py
    """
    from scripts.preprocess_metadata_to_staging import main
    main()

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'start_date': datetime(2025, 2, 1),
}

with DAG(
    dag_id='birdclef_data_pipeline',
    default_args=default_args,
    schedule_interval=None,  # or "0 0 * * *" if you want a daily schedule
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id='preprocess_and_upload_to_minio',
        python_callable=run_unpack_to_raw
    )

    t2 = PythonOperator(
        task_id='ingest_to_staging',
        python_callable=run_preprocess_audiofiles_to_staging
    )
    
    t3 = PythonOperator(
        task_id='ingest_metadata_into_mongo',
        python_callable=run_preprocess_metadata_to_staging
    )

    t1 >> t2 >> t3
