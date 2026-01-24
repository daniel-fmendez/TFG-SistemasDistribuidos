# TFG: Sistema de Entrenamiento Distribuido

## Sprint 1 - Completado

### Arquitectura Actual
- Master (Flask): Recibe Métricas en puerto 5000
- Workers (Python): envian jmétricas cada 5 segundos a modo de simulación.
- Comunicación: HTTP/v1 REST

### Cómo ejecutar
bash
minikube start
kubectl apply -f <Nombre deployment>
kubectl get pods

### Comandos útiles
bash
# Ver logs
kubectl logs -f deployment/master-deployment
kubectl logs -f deployment/worker-deployment

# Acceder desde el navegador
kubectl port-forward service/master-service 5000:5000

# Escalar workers
kubectl scale deployment worker-deployment --replicas=5

## Próximos pasos
- [ ] Migrar a gRPC
- [ ] Añadir PyTorch
- [ ] Primeros entrenamientos