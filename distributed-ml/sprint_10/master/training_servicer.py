import io
import os
from aggregation_service import AggregationService
from worker_registry import WorkerRegistry
from model_persistence import ModelPersistence
import training_pb2
import training_pb2_grpc

WORKER_WEIGHTS_DIR = "/data/worker_weights"
CHUNK_SIZE = 3 * 1024 * 1024 

class TrainingServicer:
    def __init__(self, config, model_persistence, aggregation_service, worker_registry, metrics_collector):
        self.cfg = config
        self.persistence = model_persistence
        self.aggregator = aggregation_service
        self.registry = worker_registry
        self.metrics_collector = metrics_collector
    
    def RegisterWorker(self, request, context):
        index = self.registry.register_worker(request.worker_id, is_remote=request.is_remote)
        shard = self.registry.get_shard(index)
        
        return training_pb2.StartResponse(
            ready=True,
            start=shard["start"],
            end=shard["end"],
            report_step=self.cfg.report_step,
            worker_index = index
        )
    
    def GetInitialWeights(self, request, context):
        self.registry.update_alive(request.worker_id)
        self.registry.wait_all_ready_to_train() 
        path = self.aggregator.get_updated_weights()
        yield from self._stream_weights_file(path, request.worker_id)

    def PushWeights(self, request_iterator, context):
        if self.registry.pause_event and self.registry.pause_event.is_set():
            for _ in request_iterator:
                pass
            return training_pb2.Ack(success=False, message="Sistema pausado")

        worker_id = None
        step = None
        buf = io.BytesIO()

        for chunk in request_iterator:
            if worker_id is None:
                worker_id=chunk.worker_id
                step=chunk.step
            buf.write(chunk.data)

        step_dir = os.path.join(WORKER_WEIGHTS_DIR, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        worker_index = self.registry.get_worker_index(worker_id)
        weights_path = os.path.join(step_dir, f"worker_{worker_index}.pt")

        buf.seek(0)
        with open(weights_path, "wb") as f:
            f.write(buf.read())

        self.registry.update_alive(worker_id)
        self.aggregator.aggregate(worker_id, step)

        return training_pb2.Ack(success=True, message="Pesos recibidos")
    
    def GetUpdatedWeights(self, request, context):
        self.registry.update_alive(request.worker_id)
        print(f"Worker {request.worker_id} solicitando pesos actualizados")
        path = self.aggregator.get_updated_weights()
        yield from self._stream_weights_file(path, request.worker_id)
    
    def ReportMetrics(self, request, context):
        self.registry.update_alive(request.worker_id)
        payload = {
            'worker_id': request.worker_id,
            'epoch': request.epoch,
            'step': request.step,
            'loss': request.loss,
            'accuracy': request.accuracy,
            'timestamp': request.timestamp
        }
        self.registry.save_metrics(payload)
        self.metrics_collector.record_epoch(worker_id=request.worker_id, epoch=request.epoch)

        self.metrics_collector.record_metric(
            worker_id=request.worker_id,
            loss=request.loss,
            accuracy=request.accuracy
        )
        return training_pb2.Ack(success=True, message="Ok")
    
    def FinishTraining(self, request, context):
        self.metrics_collector.record_epoch(request.worker_id,self.cfg.epochs)
        self.registry.mark_finished(request.worker_id)
        return training_pb2.Ack(success=True, message=f"Worker {request.worker_id} finalizado")
        
    def CheckRebalance(self, request, context):
        worker_id = request.worker_id
        was_rebalanced = self.registry.consume_rebalance()
        paused = self.registry.pause_event.is_set() if self.registry.pause_event else False

        if was_rebalanced:
            shard = self.registry.get_shard_for_worker(worker_id)
            if shard:
                return training_pb2.RebalanceResponse(
                    rebalanced=True,
                    pause=paused,
                    new_start=shard["start"],
                    new_end=shard["end"]
                )

        return training_pb2.RebalanceResponse(
            rebalanced=False,
            pause=paused,
            new_start=0,
            new_end=0
        )
    def SyncEpoch(self, request, context):
        self.registry.wait_epoch_sync(request.worker_id, request.epoch)
        self.metrics_collector.record_epoch_end(
            worker_id=request.worker_id,
            timestamp=request.timestamp,
            epoch=request.epoch,
            samples_in_epoch=request.total_samples
        )
        return training_pb2.Ack(success=True, message="Época sincronizada")
    
    def ReportEpochStart(self, request, context):
        self.metrics_collector.record_epoch_start(
            worker_id=request.worker_id,
            timestamp=request.timestamp
        )
        return training_pb2.Ack(success=True, message=f"Epoch {request.epoch} start recorded")
    
    def ReportSyncMetrics(self, request, context):
        self.metrics_collector.record_sync_duration(worker_id=request.worker_id, duration_seconds=request.last_sync_duration)
        self.metrics_collector.record_bytes_sent(worker_id=request.worker_id, bytes_sent=request.last_bytes_sent)
        return training_pb2.Ack(success=True, message="Sync metrics")
    
    def _stream_weights_file(self, path, worker_id):
        total = os.path.getsize(path)
        sent = 0
        chunk_index = 0
        with open(path, "rb") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                sent += len(chunk)
                yield training_pb2.WeightChunk(
                    data=chunk,
                    chunk_index=chunk_index,
                    is_last=sent >= total,
                    worker_id=worker_id,
                    step=0
                )
                chunk_index += 1