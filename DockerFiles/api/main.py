import os
import uuid
import requests
from flask import Flask, jsonify, request
import time

app = Flask(__name__)

AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow:8080")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
DAG_ID = "birdclef_data_pipeline"
DAG_FAST_ID = "birdclef_data_pipeline_fast"

@app.route("/self_health", methods=["GET"])
def health():
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

@app.route("/health", methods=["POST"])
def health():
    pass

@app.route("/stats", methods=["POST"])
def stats():
    pass

if __name__ == "__main__":
    # Listen on 0.0.0.0 so Docker can map the port
    app.run(host="0.0.0.0", port=8000, debug=True)
