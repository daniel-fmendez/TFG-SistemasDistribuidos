import os
import grpc
import socket
import training_pb2
import training_pb2_grpc
import heartbeat_pb2
import heartbeat_pb2_grpc
import dataset_pb2
import dataset_pb2_grpc
import extra_functions

CHUNK_SIZE = 3 * 1024 * 1024

class GrpcClient:
    def __init__(self, host, port):
        try:
            ip = socket.getaddrinfo(host, port, socket.AF_INET)[0][4][0]
            address = f'{ip}:{port}'
            print(f"[gRPC] Resolviendo {host}:{port} -> {address}")
        except Exception:
            address = f'{host}:{port}' 
        channel = grpc.insecure_channel(address)
        self.training_stub = training_pb2_grpc.TrainingServiceStub(channel=channel)
        self.heartbeat_stub = heartbeat_pb2_grpc.HeartbeatServiceStub(channel=channel)
        self.dataset_stub = dataset_pb2_grpc.DatasetServiceStub(channel=channel)

    def register(self, worker_id, timeout):
        return self.training_stub.RegisterWorker(
            training_pb2.WorkerRequest(worker_id=worker_id), 
            timeout=timeout
        )

    def get_initial_weights(self, worker_id):
        return self.training_stub.GetInitialWeights(
            training_pb2.WeightRequest(worker_id=worker_id)
        )
    
    def get_updated_weights(self, worker_id):
        return self.training_stub.GetUpdatedWeights(
            training_pb2.WeightRequest(worker_id=worker_id)
        )

    def push_weights(self, worker_id, step, weights_path):
        def generate_chunks():
            with open(weights_path, "rb") as f:
                total = os.path.getsize(weights_path)
                sent = 0
                chunk_index = 0
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    sent += len(chunk)
                    is_last = sent >= total
                    yield training_pb2.WeightChunk(
                        data=chunk,
                        chunk_index=chunk_index,
                        is_last=is_last,
                        worker_id=worker_id,
                        step=step
                    )
                    chunk_index += 1

        return self.training_stub.PushWeights(generate_chunks())

    def report_metrics(self, metric_data):
        return self.training_stub.ReportMetrics(metric_data)

    def finish_training(self, worker_id):
        return self.training_stub.FinishTraining(
            training_pb2.FinishRequest(worker_id=worker_id)
        )
    
    def check_rebalance(self, worker_id):
        return self.training_stub.CheckRebalance(
            training_pb2.WorkerRequest(worker_id=worker_id)
        )
    
    def sync_epoch(self, worker_id, epoch):
        return self.training_stub.SyncEpoch(
            training_pb2.EpochSyncRequest(worker_id=worker_id, epoch=epoch)
        )

    # Heartbeat
    def send_heartbeat(self, worker_id, timestamp):
        metrics = extra_functions.get_system_metrics()
        return self.heartbeat_stub.Heartbeat(
            heartbeat_pb2.HeartbeatRequest(
                    timestamp=timestamp, 
                    worker_id=worker_id,
                    cpu_usage=metrics['cpu_usage'],
                    memory_usage=metrics['memory_usage'],
                    memory_mb=metrics['memory_mb']
                )
        )

    # Dataaset
    def download_dataset(self, worker_id):
        return self.dataset_stub.DownloadDataset(
            dataset_pb2.DatasetRequest(
                worker_id=worker_id
            )
        )