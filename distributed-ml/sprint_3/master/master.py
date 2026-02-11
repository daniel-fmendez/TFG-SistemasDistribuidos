import grpc
from concurrent import futures
import threading
import metrics_pb2
import metrics_pb2_grpc
import logging
import os
import time 
import datetime

MODEL_BASE_DIR = "/app/models"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsService(metrics_pb2_grpc.MetricsService):
    def __init__(self, server, idle_timeout=300, grace_period=60):
        self.server = server
        # Dict: worker_id : metrics
        self.metrics = {}
        self.workers_connected = {}
        self.workers_finished = set()
        self.last_activity = time.time()  
        self.shutdown_lock = threading.Lock()
        self.shutdown_initiated = False

        self.idle_timeout = idle_timeout
        self.grace_period = grace_period
        logger.info(f"Master inicializado. Esperando workers")
        logger.info(f"Configuración - Idle timeout: {idle_timeout}s, Grace period: {grace_period}s")

    def ReportMetrics(self, request, context):
        self.last_activity = time.time() 
        logger.info(f"Worker: {request.worker_id}\t, Epoch: {request.epoch}, Step: {request.step}, Loss: {request.loss:.4f}, Acc: {request.accuracy:.4f}")

        self.metrics[request.worker_id].append({
            'worker_id': request.worker_id,
            'epoch': request.epoch,
            'step': request.step,
            'loss': request.loss,
            'accuracy': request.accuracy,
            'timestamp': request.timestamp,
            'model_type': request.model_type
        })

        return metrics_pb2.Ack(success=True, message="200 Ok")
    
    def ReportTrainingComplete(self, request, context):
        self.last_activity = time.time()

        print("\n=============================================")
        logger.info("Entrenamiento completado: ")

        logger.info(f"- Worker: {request.worker_id}")
        logger.info(f"- Model: {request.model_type}")
        logger.info(f"- Final Loss: {request.final_loss}")
        logger.info(f"- Final Accuracy: {request.final_accuracy}")
        logger.info(f"- Total Epochs: {request.total_epochs}")
        logger.info(f"- Model Path: {request.model_path}")
        print("\n=============================================")

        total_workers = len(self.workers_connected)
        finished_workers = len(self.workers_finished)
        
        if finished_workers == total_workers and total_workers > 0:
            logger.info("Todos los workers han completado su entrenamiento")
            logger.info(f"Esperando {self.grace_period} segundos antes de cerrar...")
            
            # Iniciar thread de shutdown con grace period
            threading.Thread(target=self._graceful_shutdown, daemon=True).start()

        return metrics_pb2.Ack(success=True, message="200 Ok")
    
    def StartSignal(self, request, context):
        self.last_activity = time.time()

        worker_id = request.worker_id
        logger.info(f"Worker conectado: {worker_id}")
        
        self.workers_connected[worker_id] = True
        self.metrics[worker_id] = []

        # Se actualiza la lista de workers
        logger.info(f"Worker {worker_id} aceptado. Enviando señal de inicio...")
        logger.info(f"Total workers conectados: {len(self.workers_connected)}")

        return metrics_pb2.StartResponse(ready=True, report_step=10)
    
    def _graceful_shutdown(self):
        """Espera el grace period y luego cierra el servidor ordenadamente"""
        with self.shutdown_lock:
            if self.shutdown_initiated:
                return
            self.shutdown_initiated = True
        
        logger.info(f"Iniciando proceso de cierre...")
        time.sleep(self.grace_period)
        
        logger.info("Guardando resumen final de métricas...")
        self._save_final_summary()
        
        logger.info("Cerrando servidor gRPC...")
        self.server.stop(grace=10)
        logger.info("Servidor cerrado exitosamente")
        
    def _save_final_summary(self):
        """Guarda un resumen de todas las métricas recopiladas"""
        try:
            summary_file = f"/app/models/training_summary_{int(time.time())}.txt"
            with open(summary_file, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("RESUMEN DE ENTRENAMIENTO\n")
                f.write(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Total de workers: {len(self.workers_connected)}\n")
                
                for worker_id, metrics_list in self.metrics.items():
                    if metrics_list:
                        f.write(f"\nWorker: {worker_id}\n")
                        f.write(f"  - Total métricas reportadas: {len(metrics_list)}\n")
                        f.write(f"  - Modelo: {metrics_list[0].get('model_type', 'N/A')}\n")
                        
                        if len(metrics_list) > 0:
                            final_metric = metrics_list[-1]
                            f.write(f"  - Última Loss: {final_metric.get('loss', 'N/A'):.4f}\n")
                            f.write(f"  - Última Accuracy: {final_metric.get('accuracy', 'N/A'):.4f}\n")
                
                f.write("\n" + "=" * 60 + "\n")
            
            logger.info(f"Resumen guardado en: {summary_file}")
        except Exception as e:
            logger.error(f"Error al guardar resumen: {e}")

    def check_idle_timeout(self):
        """Verifica si ha pasado demasiado tiempo sin actividad"""
        if len(self.workers_connected) == 0:
            # No hay workers, no aplicar timeout
            return
        
        idle_time = time.time() - self.last_activity
        
        if idle_time > self.idle_timeout:
            logger.warning(f"Timeout de inactividad alcanzado ({idle_time:.0f}s sin actividad)")
            logger.info("Cerrando servidor por inactividad...")
            
            with self.shutdown_lock:
                if not self.shutdown_initiated:
                    self.shutdown_initiated = True
                    self._save_final_summary()
                    self.server.stop(grace=10)
def listModels():
    models = os.listdir(MODEL_BASE_DIR)

    if len(models) == 0:
        print(f"Modelos no encontrados")
    else:
        print(f"Modelos encontrados: {models}")

def serve():
    listModels()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    metrics_service = MetricsService(
        server, 
        idle_timeout=180,  # 3 minutos sin actividad
        grace_period=60    # 1 minuto después del último worker
    )

    metrics_service = MetricsService(server)
    metrics_pb2_grpc.add_MetricsServiceServicer_to_server(metrics_service, server)

    
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("Master gRPC escuchando en puerto 50051")
    logger.info(f"Esperando worker(s)...")
    def monitor_idle():
        while True:
            time.sleep(30)  # Verificar cada 30 segundos
            if metrics_service.shutdown_initiated:
                break
            metrics_service.check_idle_timeout()
    
    monitor_thread = threading.Thread(target=monitor_idle, daemon=True)
    monitor_thread.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Interrupción recibida")
        metrics_service._save_final_summary()
        server.stop(grace=10)
        logger.info("Servidor detenido")

if __name__ == '__main__':
    print("Iniciando Server...")
    serve()
    listModels()