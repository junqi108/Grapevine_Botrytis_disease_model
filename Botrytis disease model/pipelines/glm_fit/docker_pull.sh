#!/usr/bin/env bash

###################################################################
# Main
###################################################################
docker-compose down
docker pull ghcr.io/junqi108/stan-cmdstanr-docker:latest
docker-compose up -d


