#!/usr/bin/env bash

. .env

###################################################################
# Main
###################################################################

echo $CR_PAT | sudo docker login ghcr.io -u USERNAME --password-stdin

# Define the source image and target repository
SOURCE_IMAGE="ghcr.io/jbris/stan-cmdstanr-docker:latest"
TARGET_REPO="ghcr.io/junqi108/stan-cmdstanr-docker:latest"

# Tag the Docker image
docker tag $SOURCE_IMAGE $TARGET_REPO

# Push the Docker image
docker push $TARGET_REPO

