import time 
import heartbeat_pb2
import heartbeat_pb2_grpc
import grpc

class HeartbeatServicer:
    def __init__(self, worker_registry, metrics_collector):
        self.registry = worker_registry
        self.metrics = metrics_collector

    def Heartbeat(self, request, context):
        worker_id = request.worker_id
        print(f"[Heartbeat] Pulso recibido de {worker_id}")
        now = time.time()
        # Actualizar metricas
        self.metrics.record_stats(
            worker_id=worker_id,
            memory_usage=request.memory_usage,
            memory_mb=request.memory_mb,
            cpu_usage=request.cpu_usage
        )
        try:
            self.registry.update_alive(worker_id)
            return heartbeat_pb2.HeartbeatResponse(
                timestamp = int(now),
                ack = True
            )
            
        except KeyError:
            print(f"[Master] Heartbeat de worker desconocido: {worker_id}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Worker {worker_id} no figura en el registro activo.")
            return heartbeat_pb2.HeartbeatResponse(
                timestamp = int(now),
                ack = False
            )
        
    
class HeartbeatMonitor:
    def __init__(self, config, worker_registry):
        self.cfg = config
        self.registry = worker_registry

    def monitor_loop(self):
        while len(self.registry.worker_last_seen) < self.registry.num_workers:
            time.sleep(2)
        print("[HeartbeatMonitor] Todos registrados, empezando monitorización")
        
        while True:
            time.sleep(self.cfg.heartbeat_interval)
            now = time.time() 
            self.registry.check_dead(now)

    
