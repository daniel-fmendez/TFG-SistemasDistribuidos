#!/bin/bash
MINIKUBE_IP=$(minikube ip)
echo "Abriendo túneles..."
echo "  local:16443 -> remoto:6443 (k3s API)"
echo "  remoto:50051 -> $MINIKUBE_IP:50051 (master gRPC)"

ssh -p 30122 -L 16443:127.0.0.1:6443 -N -o StrictHostKeyChecking=no danfer@***REMOVED*** &
SSH1=$!

ssh -p 30122 -R 50051:$MINIKUBE_IP:30051 -N -o StrictHostKeyChecking=no danfer@***REMOVED*** &
SSH2=$!

echo "Túneles activos (PIDs: $SSH1 $SSH2)"
echo "Para cerrarlos: kill $SSH1 $SSH2"
wait