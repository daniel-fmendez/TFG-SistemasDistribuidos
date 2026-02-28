import sys
import socket
import grpc
import time
import random
import threading
import heartbeat_pb2
import heartbeat_pb2_grpc

from datetime import datetime
# Objetivo de la simulacion: 
#   - Conectar con mast
#   - Conectartse al master
#   - Simular trabajo y hacer heartbeat
MASTER_HOST = "localhost"
MASTER_PORT = "4001"

PUSH_HEARTBEAT = 10

class WorkerSimulator:
    def __init__(self, worker_id=None):
        self.worker_id = worker_id if worker_id else socket.gethostname()
        self.is_running = True

        channel = grpc.insecure_channel(f'{MASTER_HOST}:{MASTER_PORT}')
        self.stub = heartbeat_pb2_grpc.HeartbeatServiceStub(channel=channel)
    
    def simulate(self):
        epoch = 0
        while self.is_running:
            epoch += 1
            print(f"[Worker] Epoch {epoch} iniciando...")
            time.sleep(random.uniform(3, 6))  # simula trabajo
            print(f"[Worker] Epoch {epoch} completado")

    def heartbeat_loop(self):
        while self.is_running:
            try:
                # Espera pulsacion
                time.sleep(PUSH_HEARTBEAT)
                
                # Envia pulsacion
                response = simulator.stub.Heartbeat(
                    heartbeat_pb2.HeartbeatRequest(
                        timestamp = int(datetime.now().timestamp()),
                        worker_id = self.worker_id
                    )
                )
                # Hacer algo con la respuesta
            except grpc.RpcError as e:
                # Verificamos la causa del error
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    print("[Heartbeat] ¡Servidor caído o inalcanzable!")
                else:
                    print(f"[Heartbeat] Error de gRPC: {e.code()} - {e.details()}")
                print("Salvando datos actuales!!!")
                self.is_running = False
            except Exception as e:
                print(f"[Heartbeat] Error: {e}")
                self.is_running = False
if __name__ == "__main__":
    simulator = WorkerSimulator()
    print(f"Esperando al servidor....")
    worker_id = sys.argv[1] if len(sys.argv) > 1 else None
    simulator = WorkerSimulator(worker_id=worker_id)
    print(f"[Worker {simulator.worker_id}] Esperando al servidor...")

    try:
        signal = simulator.stub.RegisterWorker(
            heartbeat_pb2.StartRequest(worker_id=simulator.worker_id), 
            timeout=240
        )
        
        if signal.ready:
            print(f"Señal recibida. Iniciando simulacion...")
            work_thread = threading.Thread(target=simulator.heartbeat_loop, daemon=True)
            work_thread.start()
            simulator.simulate()
            print(f"Entrenamiento completado")
    except grpc.RpcError as e:
        print(f"Error en la comunicación gRPC: {e}")
    except Exception as e:
        print(f"Error: {e}")