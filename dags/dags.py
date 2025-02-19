from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import pandas as pd
import boto3
import librosa
import soundfile as sf
import requests

# MinIO configuration
MINIO_ENDPOINT = "http://minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

# Boto3 S3 client
s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)

# Paths
DATA_DIR = "/opt/airflow/dags/data"
RAW_BUCKET = "raw-bucket"
STAGING_BUCKET = "staging-bucket"
CURATED_BUCKET = "curated-bucket"

# Load CSV Data
def ingest_raw_data():
    """Ingest CSV metadata and audio files into MinIO raw bucket."""
    train_csv = os.path.join(DATA_DIR, "trained_data.csv")
    df = pd.read_csv(train_csv)

    for index, row in df.iterrows():
        audio_file = os.path.join(DATA_DIR, "train_audio", row["filename"])
        if os.path.exists(audio_file):
            # Upload audio to MinIO Raw Bucket
            s3_client.upload_file(audio_file, RAW_BUCKET, row["filename"])
            print(f"Uploaded {row['filename']} to {RAW_BUCKET}")

    # Upload CSV to Raw Bucket
    s3_client.upload_file(train_csv, RAW_BUCKET, "trained_data.csv")
    print("Uploaded trained_data.csv to MinIO Raw Bucket")

def transform_raw_to_staging():
    """Convert OGG to WAV and extract audio metadata."""
    objects = s3_client.list_objects_v2(Bucket=RAW_BUCKET)
    for obj in objects.get("Contents", []):
        filename = obj["Key"]
        if filename.endswith(".ogg"):
            # Download OGG file
            local_file = os.path.join(DATA_DIR, filename)
            s3_client.download_file(RAW_BUCKET, filename, local_file)

            # Convert OGG to WAV
            y, sr = librosa.load(local_file, sr=None)
            wav_file = local_file.replace(".ogg", ".wav")
            sf.write(wav_file, y, sr)

            # Upload WAV to Staging Bucket
            s3_client.upload_file(wav_file, STAGING_BUCKET, os.path.basename(wav_file))
            print(f"Converted and uploaded {wav_file} to {STAGING_BUCKET}")

            # Save extracted metadata
            metadata = {
                "filename": filename,
                "duration": librosa.get_duration(y=y, sr=sr),
                "sample_rate": sr
            }
            metadata_csv = os.path.join(DATA_DIR, "audio_metadata.csv")
            pd.DataFrame([metadata]).to_csv(metadata_csv, index=False)

            # Upload metadata CSV to Staging Bucket
            s3_client.upload_file(metadata_csv, STAGING_BUCKET, "audio_metadata.csv")
            print("Uploaded audio metadata to MinIO Staging Bucket")

def process_staging_to_curated():
    """Merge metadata, finalize processed data, and move to curated bucket."""
    metadata_csv = os.path.join(DATA_DIR, "audio_metadata.csv")
    curated_csv = os.path.join(DATA_DIR, "final_dataset.csv")

    s3_client.download_file(STAGING_BUCKET, "audio_metadata.csv", metadata_csv)
    df_metadata = pd.read_csv(metadata_csv)

    # Merge with trained_data.csv
    trained_csv = os.path.join(DATA_DIR, "trained_data.csv")
    s3_client.download_file(RAW_BUCKET, "trained_data.csv", trained_csv)
    df_trained = pd.read_csv(trained_csv)

    df_final = df_trained.merge(df_metadata, on="filename", how="left")
    df_final.to_csv(curated_csv, index=False)

    # Upload final processed dataset to Curated Bucket
    s3_client.upload_file(curated_csv, CURATED_BUCKET, "final_dataset.csv")
    print("Uploaded final processed dataset to MinIO Curated Bucket")

# Airflow DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "birdclef_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)

task_ingest_raw = PythonOperator(
    task_id="ingest_raw_data",
    python_callable=ingest_raw_data,
    dag=dag,
)

task_transform_to_staging = PythonOperator(
    task_id="transform_raw_to_staging",
    python_callable=transform_raw_to_staging,
    dag=dag,
)

task_process_to_curated = PythonOperator(
    task_id="process_staging_to_curated",
    python_callable=process_staging_to_curated,
    dag=dag,
)

task_ingest_raw >> task_transform_to_staging >> task_process_to_curated
