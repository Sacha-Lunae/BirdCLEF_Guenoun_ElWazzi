# ./dags/birdclef_dag_fast.py
import os
import time as t
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def run_unpack_to_raw_fast():
    """
    Appelle directement main() depuis votre module fast de preprocess_to_raw.
    """
    os.system(f"curl -X http://localhost:8000/log?time={t.time()}&func={run_unpack_to_raw_fast.__name__}&status=started")
    from scripts.unpack_to_raw_fast import main
    os.system(f"curl -X http://localhost:8000/log?time={t.time()}&func={run_unpack_to_raw_fast.__name__}&status=started")
    main()

def run_preprocess_audiofiles_to_staging_fast():
    """
    Appelle directement main() depuis votre module fast de preprocess_audiofiles_to_staging.
    """
    os.system(f"curl -X http://localhost:8000/log?time={t.time()}&func={run_preprocess_audiofiles_to_staging_fast.__name__}&status=started")
    from scripts.preprocess_audiofiles_to_staging_fast import main
    os.system(f"curl -X http://localhost:8000/log?time={t.time()}&func={run_preprocess_audiofiles_to_staging_fast.__name__}&status=ended")
    main()
    
def run_preprocess_metadata_to_staging_fast():
    """
    Appelle directement main() depuis votre module fast de preprocess_metadata_to_staging.
    """
    os.system(f"curl -X http://localhost:8000/log?time={t.time()}&func={run_preprocess_metadata_to_staging_fast.__name__}&status=started")
    from scripts.preprocess_metadata_to_staging_fast import main
    os.system(f"curl -X http://localhost:8000/log?time={t.time()}&func={run_preprocess_metadata_to_staging_fast.__name__}&status=ended")
    main()

def run_staging_to_curated_fast():
    """
    Appelle directement main() depuis votre module fast de process_to_curated.
    """
    os.system(f"curl -X http://localhost:8000/log?time={t.time()}&func={run_staging_to_curated_fast.__name__}&status=started")
    from scripts.process_to_curated_fast import main
    os.system(f"curl -X http://localhost:8000/log?time={t.time()}&func={run_staging_to_curated_fast.__name__}&status=ended")
    main()

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'start_date': datetime(2025, 2, 1),
}

with DAG(
    dag_id='birdclef_data_pipeline_fast',
    default_args=default_args,
    schedule_interval=None,  # ou "0 0 * * *" selon vos besoins
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id='preprocess_and_upload_to_minio_fast',
        python_callable=run_unpack_to_raw_fast
    )

    t2 = PythonOperator(
        task_id='ingest_to_staging_fast',
        python_callable=run_preprocess_audiofiles_to_staging_fast
    )
    
    t3 = PythonOperator(
        task_id='ingest_metadata_into_mongo_fast',
        python_callable=run_preprocess_metadata_to_staging_fast
    )
    
    t4 = PythonOperator(
        task_id='process_to_curated_fast',
        python_callable=run_staging_to_curated_fast
    )

    t1 >> t2 >> t3 >> t4
