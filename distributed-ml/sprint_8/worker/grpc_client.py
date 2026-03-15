import grpc
import training_pb2
import training_pb2_grpc
import heartbeat_pb2
import heartbeat_pb2_grpc
import extra_functions
class GrpcClient:
    def __init__(self, host, port):
        channel = grpc.insecure_channel(f'{host}:{port}')
        self.training_stub = training_pb2_grpc.TrainingServiceStub(channel=channel)
        self.heartbeat_stub = heartbeat_pb2_grpc.HeartbeatServiceStub(channel=channel)

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

    def push_weights(self, worker_id, step):
        return self.training_stub.PushWeights(
            training_pb2.WeightData(
                worker_id=worker_id,
                step=step,
            )
        )

    def report_metrics(self, metric_data):
        return self.training_stub.ReportMetrics(metric_data)

    def finish_training(self, worker_id):
        return self.training_stub.FinishTraining(
            training_pb2.FinishRequest(worker_id=worker_id)
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
    
    def check_rebalance(self, worker_id):
        return self.training_stub.CheckRebalance(
            training_pb2.WeightRequest(worker_id=worker_id)
        )