import os
import uuid
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow:8080")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
DAG_ID = "birdclef_data_pipeline"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

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

if __name__ == "__main__":
    # Listen on 0.0.0.0 so Docker can map the port
    app.run(host="0.0.0.0", port=8000, debug=True)
