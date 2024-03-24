#!/usr/bin/env bash

###################################################################
# Main
###################################################################

docker compose down
docker compose pull
docker compose up -d