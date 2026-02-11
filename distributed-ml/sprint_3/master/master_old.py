import grpc
from concurrent import futures
import threading
import metrics_pb2
import metrics_pb2_grpc
import logging
import os

MODEL_BASE_DIR = "/app/models"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsService(metrics_pb2_grpc.MetricsService):
    def __init__(self, server, required_workers=1):
        self.server = server
        self.metrics = []
        self.required_workers = required_workers
        self.workers_connected = {}
        self.workers_finished = set()
        self.start_event = threading.Event()
        self.lock = threading.Lock()
        logger.info(f"Master inicializado. Esperando {required_workers} workers")

    def ReportMetrics(self, request, context):
        logger.info(f"Worker: {request.worker_id}, Epoch: {request.epoch}, Step: {request.step}, Loss: {request.loss:.4f}, Acc: {request.accuracy:.4f}")
        self.metrics.append({
            'worker_id': request.worker_id,
            'epoch': request.epoch,
            'step': request.step,
            'loss': request.loss,
            'accuracy': request.accuracy,
            'timestamp': request.timestamp,
            'model_type': request.model_type
        })

        return metrics_pb2.Ack(success=True, message="200 Ok")
    
    def GetAllMetrics(self, request, context):
        logger.info("Metricas solicitadas")
        metric_list = [
            metrics_pb2.MetricData(**m) for m in self.metrics
        ]
        return metrics_pb2.MetricsList(metrics=metric_list)
    
    def ReportTrainingComplete(self, request, context):
        print("\n=============================================")
        logger.info("Entrenamiento completado: ")

        logger.info(f"- Worker: {request.worker_id}")
        logger.info(f"- Model: {request.model_type}")
        logger.info(f"- Final Loss: {request.final_loss}")
        logger.info(f"- Final Accuracy: {request.final_accuracy}")
        logger.info(f"- Total Epochs: {request.total_epochs}")
        logger.info(f"- Model Path: {request.model_path}")
        print("\n=============================================")
        with self.lock:
            self.workers_finished.add(request.worker_id)
            finished = len(self.workers_finished)
            logger.info(f"Workers terminados: {finished}/{self.required_workers}")

            if finished >= self.required_workers:
                logger.info("Todos los workers han terminado. Apagando server...")
                # Apagado limpio
                threading.Thread(
                    target=lambda: self.server.stop(0),
                    daemon=True
                ).start()
        return metrics_pb2.Ack(success=True, message="200 Ok")
    
    def StartSignal(self, request, context):
        worker_id = request.worker_id
        logger.info(f"Worker conectado: {worker_id}")
        
        with self.lock:
            self.workers_connected[worker_id] = True
            count = len(self.workers_connected)
            logger.info(f"Workers conectados: {count}/{self.required_workers}")
            
            if count >= self.required_workers:
                logger.info("¡Barrera alcanzada! Liberando workers para comenzar entrenamiento...")
                self.start_event.set()

        # El worker espera aquí hasta que todos estén conectados
        logger.info(f"Worker {worker_id} esperando en barrera...")
        self.start_event.wait()
        logger.info(f"Worker {worker_id} liberado. Enviando señal de inicio...")

        # Cambiar yield por return (RPC unario)
        return metrics_pb2.StartResponse(ready=True, report_step=10)

def listModels():
    models = os.listdir(MODEL_BASE_DIR)

    if len(models) == 0:
        print(f"Modelos no encontrados")
    else:
        print(f"Modelos encontrados: {models}")

def serve():
    listModels()
    required_workers = int(os.getenv('REQUIRED_WORKERS', '1'))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    metrics_service = MetricsService(server,required_workers)
    metrics_pb2_grpc.add_MetricsServiceServicer_to_server(metrics_service, server)

    
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("Master gRPC escuchando en puerto 50051")
    logger.info(f"Esperando {required_workers} worker(s)...")
    server.wait_for_termination()

if __name__ == '__main__':
    print("Iniciando Server...")
    serve()
    listModels()