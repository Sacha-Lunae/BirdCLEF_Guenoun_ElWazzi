# Dockerfile
FROM apache/airflow:2.7.1

# 1) Switch to root for system installs
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# 2) Switch back to airflow user for pip installs
USER airflow
RUN pip install --no-cache-dir \
    apache-airflow-providers-mysql \
    apache-airflow-providers-postgres \
    apache-airflow-providers-mongo \
    apache-airflow-providers-amazon \
    boto3 \
    pymongo \
    elasticsearch \
    numba \
    numpy

# The base image already has WORKDIR /opt/airflow
# and a default CMD "airflow version". You can override it if you like:
CMD ["airflow", "--help"]
