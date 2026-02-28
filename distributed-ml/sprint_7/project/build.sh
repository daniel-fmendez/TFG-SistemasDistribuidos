#!/bin/bash

docker build -t data-provision:v2 -f dataset/Dockerfile .
docker build -t deployer:v2 -f deployer/Dockerfile .
docker build -t my-master:v6 -f master/Dockerfile .
docker build -t my-worker:v6 -f worker/Dockerfile .
sleep 5
echo "Escribiendo en minikube..."
minikube image load data-provision:v2
minikube image load deployer:v2 
minikube image load my-master:v6
minikube image load my-worker:v6

echo "Escritura completa!"