FROM apache/airflow:latest-python3.11

WORKDIR /project

COPY airflow.requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r airflow.requirements.txt --upgrade

USER root

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends vim curl git rsync unzip \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG DOCKER_UID=1000

# Create a new user with the specified UID and GID=0
RUN adduser --disabled-password --gecos "" --uid ${DOCKER_UID} --gid 0 custom_airflow \
    && echo "Created user custom_airflow with UID ${DOCKER_UID} and GID 0"

# Ensure the new user has the same permissions as the airflow user
RUN usermod -aG root custom_airflow \
    && chown -R custom_airflow:root /project

EXPOSE 8080

USER custom_airflow

CMD ["airflow", "standalone"]
