#!/usr/bin/env bash

. .env

# Stop a local MLFlow server instance

echo "Host: ${MLFLOW_LOCAL_HOST}" 
echo "Port: ${MLFLOW_LOCAL_PORT}"

fuser -k "${MLFLOW_LOCAL_PORT}/tcp"