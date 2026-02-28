import threading
from concurrent import futures

import grpc
from k8s_orchestrator import KubernetesOrchestrator
import training_pb2_grpc
from aggregation_service import AggregationService
from config_loader import TrainingConfig
from model_persistence import ModelPersistence
from training_servicer import TrainingServicer
from worker_registry import WorkerRegistry


def serve():
    cfg = TrainingConfig()
    stop_event = threading.Event()

    # Construir dependencias
    aggregator = AggregationService(cfg)
    persistence = ModelPersistence(cfg)
    orchestrator = KubernetesOrchestrator()

    aggregator.initialize_weights()
    def on_all_finished():
        persistence.save_final_model(
            global_weights=aggregator.global_weights,
            metrics=registry.get_metrics()
        )
        stop_event.set()

    registry = WorkerRegistry(cfg, on_all_finished=on_all_finished)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    servicer = TrainingServicer(cfg, persistence, aggregator, registry, orchestrator)
    training_pb2_grpc.add_TrainingServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{cfg.master_port}')
    server.start()
    print(f"Master gRPC escuchando en puerto {cfg.master_port}")

    stop_event.wait()
    server.stop(grace=10)

if __name__ == "__main__":
    serve()