#!/bin/bash

docker build -t data-provision:v2 -f dataset/Dockerfile .
docker build -t deployer:v2 -f deployer/Dockerfile .
docker build -t my-master:v6 -f master/Dockerfile .
docker build -t my-worker:v6 -f worker/Dockerfile .
sleep 5

echo "Escribiendo en k3s lcoal..."
docker save data-provision:v2 | sudo k3s ctr images import -
docker save deployer:v2 | sudo k3s ctr images import -
docker save my-master:v6 | sudo k3s ctr images import -
docker save my-worker:v6 | sudo k3s ctr images import -

echo "Escritura completa!"

# Remote

:' REMOTE_USER=danfer
REMOTE_HOST=***REMOVED***
REMOTE_PORT=30122

echo "Exportando imagen worker para remoto..."
docker save my-worker:v6 -o /tmp/my-worker-v6.tar
docker save data-provision:v2 -o /tmp/data-provision-v2.tar

echo "Copiando al servidor remoto..."
scp -P $REMOTE_PORT /tmp/my-worker-v6.tar $REMOTE_USER@$REMOTE_HOST:/tmp/
scp -P $REMOTE_PORT /tmp/data-provision-v2.tar $REMOTE_USER@$REMOTE_HOST:/tmp/

echo "Importando en k3s remoto..."
ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST "sudo k3s ctr images import /tmp/my-worker-v6.tar"
ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST "sudo k3s ctr images import /tmp/data-provision-v2.tar"

echo "Limpiando..."
rm /tmp/my-worker-v6.tar /tmp/data-provision-v2.tar
ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST "rm /tmp/my-worker-v6.tar /tmp/data-provision-v2.tar"

echo "Escritura completa en remoto!"

'