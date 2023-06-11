#!/usr/bin/env bash

. .env

###################################################################
# Functions
###################################################################

usage() {
    echo """
    Usage: ${0} [pipeline name] [params]
    Parameters:
        pipeline_directory  The directory for MLflow pipelines.
        params              Pipeline parameters.
    """
}

help() {
    if [ "${1}" = "-h" ] || [ "${1}" = "-help" ]; then
        usage
        exit 1
    fi
}

###################################################################
# Main
###################################################################

help "${1}"

# Exports
export PROJECT_NAME
export MLFLOW_TRACKING_URI
export MLFLOW_EXPERIMENT_NAME="${PROJECT_NAME}_${1}"
echo "MLFlow address: ${MLFLOW_TRACKING_URI}"
echo "MLFlow experiment: ${MLFLOW_EXPERIMENT_NAME}"

mlflow run ${MLFLOW_CONDA} "./experiments/${1}" ${2} 