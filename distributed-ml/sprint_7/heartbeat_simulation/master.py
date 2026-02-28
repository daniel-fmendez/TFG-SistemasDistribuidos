
import os
import grpc
import logging
import time 
import datetime
import threading
import subprocess
import heartbeat_pb2
import heartbeat_pb2_grpc

from concurrent import futures
# Objetivo de la simulacion: 
#   - Conectar pods
#   - Mandar conexion
#   - Realizar heartbeat

MASTER_HOST = "localhost"
MASTER_PORT = "4001"

PUSH_HEARTBEAT = 10 # 5s
NUM_WORKERS = 3
class HeartbeatMonitor:
    def __init__(self):
        self.active_workers = {}
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.num_workers = NUM_WORKERS

    def RegisterWorker(self, request, context):
        worker_id = request.worker_id

        with self.condition:
            self.active_workers[worker_id] = time.time()
            current = len(self.active_workers)
            print(f"[Master] Worker conectado: {worker_id} ({current}/{self.num_workers})")

            if current == self.num_workers:
                print(f"[Master] ✅ Todos los workers conectados: {list(self.active_workers.keys())}")


            while len(self.active_workers) < self.num_workers:
                self.condition.wait()

            self.condition.notify_all()

        return heartbeat_pb2.StartResponse(
            ready=True,
        )

    def Heartbeat(self, request, context):
        worker_id = request.worker_id
        now = time.time()

        with self.lock:
            if worker_id not in self.active_workers:
                print(f"[Master] Heartbeat de worker desconocido: {worker_id}")
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Worker no registrado")
                return heartbeat_pb2.HeartbeatResponse(timestamp=int(now), ack=False)

            self.active_workers[worker_id] = now
            print(f"[Master] Heartbeat recibido de {worker_id} @ {datetime.datetime.fromtimestamp(now)}")

        return heartbeat_pb2.HeartbeatResponse(
            timestamp=int(now),
            ack=True
        )
    def _monitor_loop(self):
        """Detecta workers caídos por timeout."""
        while True:
            time.sleep(PUSH_HEARTBEAT)
            now = time.time()
            with self.lock:
                caidos = [
                    wid for wid, last_seen in self.active_workers.items()
                    if now - last_seen > (PUSH_HEARTBEAT * 3)
                ]
                for wid in caidos:
                    print(f"[Master] ⚠️  Worker caído (timeout): {wid}")
                    del self.active_workers[wid]

def serve():
    # Crear service y conectarlo a un server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    heartbeat_service = HeartbeatMonitor()
    heartbeat_pb2_grpc.add_HeartbeatServiceServicer_to_server(heartbeat_service, server)

    monitor_thread = threading.Thread(target=heartbeat_service._monitor_loop, daemon=True)
    monitor_thread.start()

    server.add_insecure_port(f'[::]:{MASTER_PORT}')
    server.start()
    print(f"Master gRPC escuchando en puerto {MASTER_PORT}")
    server.wait_for_termination()
    
if __name__ == "__main__":
    serve()