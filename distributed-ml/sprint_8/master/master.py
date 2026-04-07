import threading
import time
from concurrent import futures
import torch
import random
import numpy as np
import grpc
from k8s_orchestrator import KubernetesOrchestrator
import training_pb2_grpc
import heartbeat_pb2_grpc
import dataset_pb2_grpc
from aggregation_service import AggregationService
from config_loader import TrainingConfig
from model_persistence import ModelPersistence
from training_servicer import TrainingServicer
from worker_registry import WorkerRegistry
from heartbeat import HeartbeatServicer, HeartbeatMonitor
from dataset_servicer import DatasetServicer
from metrics_collector import MetricsCollector
from aggregator import Aggregator
from compressor import Compressor
METRICS_PORT = 8000

def serve():
    cfg = TrainingConfig()

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    stop_event = threading.Event()
    pause_event = threading.Event()
    metrics_collector = MetricsCollector()
    metrics_collector.start(port=METRICS_PORT)
    metrics_collector.set_total_epoch(cfg.epochs)
    
    persistence = ModelPersistence(cfg)
    
    def on_all_finished():
        persistence.save_final_model(
            global_weights=aggregator_service.global_weights,
            metrics=registry.get_metrics()
        )
        print("Esperando scrape final de Prometheus...")
        time.sleep(90)
        stop_event.set()

    registry = WorkerRegistry(
        cfg, 
        metrics_collector,
        on_all_finished=on_all_finished,
        pause_event = pause_event
    )
    compressor = Compressor(
        strategy=cfg.compression_strategy, 
        k_ratio=cfg.top_k_ratio,            
        quantization_bits=cfg.quantization_bits
    )
    aggregator = Aggregator(registry=registry, strategy=cfg.aggregation_strategy, compressor=compressor)
    aggregator_service = AggregationService(cfg, registry, metrics_collector,aggregator)
    aggregator_service.initialize_weights()
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    training_servicer = TrainingServicer(cfg, persistence, aggregator_service, registry, metrics_collector)
    training_pb2_grpc.add_TrainingServiceServicer_to_server(training_servicer, server)
    
    dataset_servicer = DatasetServicer(cfg.dataset_path)
    dataset_pb2_grpc.add_DatasetServiceServicer_to_server(dataset_servicer, server)

    heartbeat_servicer = HeartbeatServicer(registry, metrics_collector)
    heartbeat_pb2_grpc.add_HeartbeatServiceServicer_to_server(heartbeat_servicer, server)

    server.add_insecure_port(f'[::]:{cfg.master_port}')
    server.start()
    print(f"Master gRPC escuchando en puerto {cfg.master_port}")

    monitor = HeartbeatMonitor(cfg, registry)
    monitor_thread = threading.Thread(target=monitor.monitor_loop, daemon=True)
    monitor_thread.start()
    print(f"HeartbeatMonitor arrancado")
    
    stop_event.wait()
    server.stop(grace=10)

if __name__ == "__main__":
    serve()