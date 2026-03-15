#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
docker build -t data-provision:v2 .
minikube image load data-provision:v2