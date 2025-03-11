import os
import uuid
import requests
from flask import Flask, jsonify, request
import time
import boto3
from pymongo import MongoClient
from datetime import datetime
import time

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

app = Flask(__name__)

AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow:8080")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
DAG_ID = "birdclef_data_pipeline"
DAG_FAST_ID = "birdclef_data_pipeline_fast"

@app.route("/self_health", methods=["GET"])
def self_health():
    return jsonify({"status": "ok"}), 200


def wait_for_dag_run(dag_id, run_id, timeout=600, poll_interval=10):
    """
    Interroge l'API Airflow jusqu'à ce que le DAG run soit terminé.
    Retourne les informations du run, ainsi que des statistiques sur les tâches.
    """
    start_time = time.time()
    while True:
        status_url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns/{run_id}"
        status_resp = requests.get(status_url, auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD))
        if status_resp.status_code == 200:
            data = status_resp.json()
            state = data.get("state")
            if state in ["success", "failed", "upstream_failed"]:
                # Récupérer les statistiques des tâches
                task_url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances"
                task_resp = requests.get(task_url, auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD))
                if task_resp.status_code == 200:
                    tasks = task_resp.json().get("task_instances", [])
                    stats = {}
                    for task in tasks:
                        task_state = task.get("state")
                        stats[task_state] = stats.get(task_state, 0) + 1
                    data["task_stats"] = stats
                return data
        if time.time() - start_time > timeout:
            return {"state": "timeout"}
        time.sleep(poll_interval)

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

    if trigger_resp.status_code != 200:
        return jsonify({"error": trigger_resp.text}), trigger_resp.status_code

    # Attendre la fin de l'exécution du DAG et récupérer les stats
    run_data = wait_for_dag_run(DAG_ID, run_id)
    return jsonify({"message": "DAG completed", "run_id": run_id, "run_data": run_data}), 200

@app.route("/ingest_fast", methods=["POST"])
def trigger_dag_fast():
    # Unpause the DAG fast
    patch_url = f"{AIRFLOW_URL}/api/v1/dags/{DAG_FAST_ID}"
    patch_payload = {"is_paused": False}
    patch_resp = requests.patch(
        patch_url,
        json=patch_payload,
        auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
        headers={"Content-Type": "application/json"},
    )
    if patch_resp.status_code != 200:
        return jsonify({"error": patch_resp.text}), patch_resp.status_code

    # Now trigger the DAG run for the fast version
    run_id = f"trigger_via_api_{uuid.uuid4()}"
    trigger_url = f"{AIRFLOW_URL}/api/v1/dags/{DAG_FAST_ID}/dagRuns"
    trigger_payload = {"conf": {}, "dag_run_id": run_id}
    trigger_resp = requests.post(
        trigger_url,
        json=trigger_payload,
        auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
        headers={"Content-Type": "application/json"},
    )

    if trigger_resp.status_code != 200:
        return jsonify({"error": trigger_resp.text}), trigger_resp.status_code

    # Attendre la fin de l'exécution du DAG fast et récupérer les stats
    run_data = wait_for_dag_run(DAG_FAST_ID, run_id)
    return jsonify({"message": "DAG fast completed", "run_id": run_id, "run_data": run_data}), 200

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
@app.route("/compare", methods=["GET"])
def compare():
    # 1. Déclencher simultanément les deux DAGs
    
    # Générer des identifiants uniques pour chaque run
    ingest_run_id = f"trigger_via_api_{uuid.uuid4()}"
    ingest_fast_run_id = f"trigger_via_api_{uuid.uuid4()}"
    
    # Unpause les deux DAGs
    patch_url_ingest = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}"
    patch_payload = {"is_paused": False}
    requests.patch(patch_url_ingest, json=patch_payload, auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
                     headers={"Content-Type": "application/json"})
    
    patch_url_fast = f"{AIRFLOW_URL}/api/v1/dags/{DAG_FAST_ID}"
    requests.patch(patch_url_fast, json=patch_payload, auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
                     headers={"Content-Type": "application/json"})
    
    # Déclencher les DAGs
    trigger_url_ingest = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns"
    trigger_payload_ingest = {"conf": {}, "dag_run_id": ingest_run_id}
    resp_ingest = requests.post(trigger_url_ingest, json=trigger_payload_ingest,
                                auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
                                headers={"Content-Type": "application/json"})
    
    trigger_url_fast = f"{AIRFLOW_URL}/api/v1/dags/{DAG_FAST_ID}/dagRuns"
    trigger_payload_fast = {"conf": {}, "dag_run_id": ingest_fast_run_id}
    resp_fast = requests.post(trigger_url_fast, json=trigger_payload_fast,
                              auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
                              headers={"Content-Type": "application/json"})
    
    if resp_ingest.status_code != 200 or resp_fast.status_code != 200:
        return jsonify({"error": "Erreur lors du déclenchement des DAGs"}), 500
    
    # 2. Attendre la fin des exécutions
    run_data_ingest = wait_for_dag_run(DAG_ID, ingest_run_id)
    run_data_fast = wait_for_dag_run(DAG_FAST_ID, ingest_fast_run_id)
    
    # 3. Récupérer les durées d'exécution de chaque tâche pour chaque DAG run
    def get_task_durations(dag_id, run_id):
        task_url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances"
        task_resp = requests.get(task_url, auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD))
        durations = {}
        if task_resp.status_code == 200:
            tasks = task_resp.json().get("task_instances", [])
            for task in tasks:
                task_id = task.get("task_id")
                start = task.get("start_date")
                end = task.get("end_date")
                if start and end:
                    # Convertir les dates ISO en objets datetime
                    dt_start = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    dt_end = datetime.fromisoformat(end.replace("Z", "+00:00"))
                    durations[task_id] = (dt_end - dt_start).total_seconds()
        return durations
    
    durations_ingest = get_task_durations(DAG_ID, ingest_run_id)
    durations_fast = get_task_durations(DAG_FAST_ID, ingest_fast_run_id)
    
    # 4. Comparer les durées pour chaque tâche
    comparison = {}
    all_tasks = set(list(durations_ingest.keys()) + list(durations_fast.keys()))
    for task_id in all_tasks:
        d_ingest = durations_ingest.get(task_id)
        d_fast = durations_fast.get(task_id)
        comparison[task_id] = {
            "ingest_duration": d_ingest,
            "ingest_fast_duration": d_fast,
            "difference": (d_ingest - d_fast) if (d_ingest is not None and d_fast is not None) else None
        }
    
    # 5. Retourner la comparaison
    return jsonify({
        "ingest_run_data": run_data_ingest,
        "ingest_fast_run_data": run_data_fast,
        "task_durations_ingest": durations_ingest,
        "task_durations_fast": durations_fast,
        "comparison": comparison,
        "timestamp": datetime.now().isoformat()
    }), 200
if __name__ == "__main__":
    # Listen on 0.0.0.0 so Docker can map the port
    app.run(host="0.0.0.0", port=8000, debug=True)
