import threading
from concurrent import futures

import grpc
from k8s_orchestrator import KubernetesOrchestrator
import training_pb2_grpc
import heartbeat_pb2_grpc
from aggregation_service import AggregationService
from config_loader import TrainingConfig
from model_persistence import ModelPersistence
from training_servicer import TrainingServicer
from worker_registry import WorkerRegistry
from heartbeat import HeartbeatServicer, HeartbeatMonitor
from metrics_collector import MetricsCollector

METRICS_PORT = 8000

def serve():
    cfg = TrainingConfig()
    stop_event = threading.Event()
    metrics_collector = MetricsCollector()
    metrics_collector.start(port=METRICS_PORT)
    metrics_collector.set_total_epoch(cfg.epochs)
    # Construir dependencias
    aggregator = AggregationService(cfg, metrics_collector)
    persistence = ModelPersistence(cfg)
    orchestrator = KubernetesOrchestrator()
    
    def on_all_finished():
        persistence.save_final_model(
            global_weights=aggregator.global_weights,
            metrics=registry.get_metrics()
        )
        stop_event.set()

    registry = WorkerRegistry(cfg, metrics_collector,on_all_finished=on_all_finished)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    training_servicer = TrainingServicer(cfg, persistence, aggregator, registry, orchestrator, metrics_collector)
    training_pb2_grpc.add_TrainingServiceServicer_to_server(training_servicer, server)
    

    heartbeat_servicer = HeartbeatServicer(registry, metrics_collector)
    heartbeat_pb2_grpc.add_HeartbeatServiceServicer_to_server(heartbeat_servicer, server)

    server.add_insecure_port(f'[::]:{cfg.master_port}')
    server.start()
    print(f"Master gRPC escuchando en puerto {cfg.master_port}")

    stop_event.wait()
    server.stop(grace=10)

if __name__ == "__main__":
    serve()