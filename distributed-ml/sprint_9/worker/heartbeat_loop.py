
import time
import grpc
import extra_functions
class HeartbeatLoop:
    def __init__(self, config, worker_id, grpc_client, stop_event):
        self.cfg = config
        self.worker_id = worker_id
        self.client = grpc_client
        self.stop_event = stop_event
        
    def heartbeat_loop(self):
        while not self.stop_event.is_set():
            try:
                
                response = self.client.send_heartbeat(
                    timestamp=int(time.time()),
                    worker_id=self.worker_id, 
                )
                if not response.ack:
                    print("[Heartbeat] Master rechazó el heartbeat, deteniendo worker")
                    self.stop_event.set()
                #Gestionar la respuesta

            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    print("[Heartbeat] Servidor caído o inalcanzable, deteniendo worker")
                else:
                    print(f"[Heartbeat] Error gRPC: {e.code()} - {e.details()}")
                self.stop_event.set()

            except Exception as e:
                print(f"[Heartbeat] Error inesperado: {e}")
                self.stop_event.set()
            time.sleep(self.cfg.heartbeat_interval)