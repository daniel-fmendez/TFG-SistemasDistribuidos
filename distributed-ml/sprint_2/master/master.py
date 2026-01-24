import grpc
from concurrent import futures
import metrics_pb2
import metrics_pb2_grpc
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsService(metrics_pb2_grpc.MetricsService):
    def __init__(self):
        self.metrics = []

    def ReportMetrics(self, request, context):
        logger.info(f"Worker: {request.worker_id}, Loss {request.loss: .4f}")
        self.metrics.append ({
            'worker_id': request.worker_id,
            'iteration': request.iteration,
            'loss': request.loss,
            'accuracy': request.accuracy,
            'timestamp': request.timestamp
        })

        return metrics_pb2.Ack(success=True, message="200 Ok")
    
    def GetAllMetrics(self, request, context):
        logger.info("Metricas solicitadas")
        metric_list = [
            metrics_pb2.MetricData(**m) for m in self.metrics
        ]
        return metrics_pb2.MetricsList(metrics=metric_list)
    

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    metrics_pb2_grpc.add_MetricsServiceServicer_to_server(MetricsService(), server)
    
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("Master gRPC escuchando en puerto 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()