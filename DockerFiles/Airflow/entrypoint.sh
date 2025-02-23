#!/bin/bash
set -e

echo "Initializing Airflow database (ignoring errors if already initialized)..."
airflow db init || true
airflow db upgrade || true

echo "Creating default connections (ignoring errors)..."
airflow connections create-default-connections || true

echo "Creating admin user (ignoring errors if already exists)..."
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com || true

echo "Starting the scheduler in background..."
airflow scheduler &

echo "Starting the Airflow webserver on port 8080..."
exec airflow webserver --port 8080 --host 0.0.0.0

