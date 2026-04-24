#!/bin/bash

echo "Iniciando port-forwards de monitoring..."

pkill -f "kubectl port-forward" 2>/dev/null
sleep 2

kubectl port-forward -n monitoring svc/prometheus-server 9090:80 --address=0.0.0.0 &
PROM_PID=$!

kubectl port-forward -n monitoring svc/grafana 3000:80 --address=0.0.0.0 &
GRAF_PID=$!

echo "Prometheus en http://localhost:9090 (PID: $PROM_PID)"
echo "Grafana en http://localhost:3000 (PID: $GRAF_PID)"