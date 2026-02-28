
from aggregation_service import AggregationService
from worker_registry import WorkerRegistry
from model_persistence import ModelPersistence
from k8s_orchestrator import KubernetesOrchestrator
import training_pb2
import training_pb2_grpc

class TrainingServicer:
    def __init__(self, config, model_persistence, aggregation_service, worker_registry, orchestrator):
        self.cfg = config
        self.persistence = model_persistence
        self.aggregator = aggregation_service
        self.registry = worker_registry
        self.orchestrator = orchestrator
        # TODO ver si va aqui
        # self.heartbeat = heartbeat_service

        self.orchestrator.create_workers(self.cfg)
    
    def RegisterWorker(self, request, context):
        index = self.registry.register_worker(request.worker_id)
        shard = self.registry.get_shard(index)
        
        return training_pb2.StartResponse(
            ready=True,
            start=shard["start"],
            end=shard["end"],
            report_step=self.cfg.report_step
        )
    
    def GetInitialWeights(self, request, context):
        worker_id = request.worker_id
        print(f"Worker {worker_id} solicitando pesos iniciales")
    
        return training_pb2.WeightResponse(
            weights_path=self.aggregator.initialize_weights()
        )

    def PushWeights(self, request, context):
        worker_id = request.worker_id
        step = request.step
        self.aggregator.aggregate(worker_id, step)
        return training_pb2.Ack(success=True, message="Gradientes recibidos")
    
    def GetUpdatedWeights(self, request, context):
        path = self.aggregator.get_updated_weights()
        print(f"Worker {request.worker_id} solicitando pesos actualizados")
        
        return training_pb2.WeightResponse(
            weights_path=path
        )
    
    def ReportMetrics(self, request, context):
        payload = {
            'worker_id': request.worker_id,
            'epoch': request.epoch,
            'step': request.step,
            'loss': request.loss,
            'accuracy': request.accuracy,
            'timestamp': request.timestamp
        }
        self.registry.save_metrics(payload)
        return training_pb2.Ack(success=True, message="Ok")
    
    def FinishTraining(self, request, context):
        self.registry.mark_finished(request.worker_id)
        return training_pb2.Ack(success=True, message=f"Worker {request.worker_id} finalizado")
        
    