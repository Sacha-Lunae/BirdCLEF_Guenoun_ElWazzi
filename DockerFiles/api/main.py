import os
import boto3
from pymongo import MongoClient
import uuid
import requests
from datetime import datetime
from flask import Flask, jsonify, request
import time

app = Flask(__name__)

class DatabaseConnections:
    def __init__(self):
        # Minio
        self.minio_client = boto3.client(
            's3',
            endpoint_url='http://minio:9000',
            aws_access_key_id='minioadmin', 
            aws_secret_access_key='minioadmin',
            region_name='us-east-1'
        )
        
        # MongoDB
        self.mongo_uri = 'mongodb://mongodb:27017/'
        self.mongo_client = MongoClient(self.mongo_uri)
        self.mongo_db = self.mongo_client['birdclef']

db = DatabaseConnections()


AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow:8080")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
DAG_ID = "birdclef_data_pipeline"


@app.route("/ingest", methods=["POST"])
def trigger_dag():
    # Unpause the DAG
    patch_url = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}"
    patch_payload = {"is_paused": False}
    patch_resp = requests.patch(
        patch_url,
        json=patch_payload,
        auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
        headers={"Content-Type": "application/json"},
    )
    if patch_resp.status_code != 200:
        return jsonify({"error": patch_resp.text}), patch_resp.status_code

    # Now trigger the DAG run
    run_id = f"trigger_via_api_{uuid.uuid4()}"
    trigger_url = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns"
    trigger_payload = {"conf": {}, "dag_run_id": run_id}
    trigger_resp = requests.post(
        trigger_url,
        json=trigger_payload,
        auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
        headers={"Content-Type": "application/json"},
    )


    if trigger_resp.status_code == 200:
        return jsonify({"message": "DAG triggered", "run_id": run_id}), 200
    else:
        return jsonify({"error": trigger_resp.text}), trigger_resp.status_code

@app.get("/health")
def health_check():
    """
    Vérifie la santé de l'API et des connexions aux bases de données
    """
    status = {
        "api_status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connections": {}
    }
    
    try:
        # Test Minio
        db.minio_client.list_buckets()
        status["connections"]["minio"] = True
    except Exception as e:
        status["connections"]["minio"] = False
    
    try:
        # Test MongoDB
        db.mongo_client.server_info()
        status["connections"]["mongodb"] = True
    except:
        status["connections"]["mysql"] = False
    
    return status


@app.route("/stats", methods=["GET"])
def stats():
    # Statistiques sur les buckets Minio
    minio_descriptions = []
    try:
        buckets_response = db.minio_client.list_buckets()
        for bucket in buckets_response.get("Buckets", []):
            bucket_name = bucket["Name"]
            objects_response = db.minio_client.list_objects(Bucket=bucket_name)
            count = 0
            total_size = 0
            if 'Contents' in objects_response:
                for obj in objects_response['Contents']:
                    count += 1
                    total_size += obj.get("Size", 0)
            description = (
                f"Le bucket '{bucket_name}' de minio contient {count} fichiers pour un poids "
                f"total de {total_size} octets"
            )
            minio_descriptions.append(description)
    except Exception as e:
        minio_descriptions = [f"Erreur lors de la récupération des stats minio : {str(e)}"]

    # Statistiques sur MongoDB
    mongo_descriptions = []
    try:
        collections = db.mongo_db.list_collection_names()
        for coll in collections:
            collection = db.mongo_db[coll]
            count = collection.count_documents({})
            coll_stats = db.mongo_db.command("collstats", coll)
            size = coll_stats.get("size", 0)
            description = (
                f"La collection '{coll}' de MongoDB contient {count} documents pour une taille "
                f"totale de {size} octets"
            )
            mongo_descriptions.append(description)
    except Exception as e:
        mongo_descriptions = [f"Erreur lors de la récupération des stats MongoDB : {str(e)}"]


    return jsonify({
        "minio_stats": minio_descriptions,
        "mongo_stats": mongo_descriptions,
        "timestamp": datetime.now().isoformat()
    }), 200



if __name__ == "__main__":
    # Listen on 0.0.0.0 so Docker can map the port
    app.run(host="0.0.0.0", port=8000, debug=True)
