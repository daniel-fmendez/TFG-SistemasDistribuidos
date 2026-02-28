import grpc
import training_pb2
import training_pb2_grpc

class GrpcClient:
    def __init__(self, host, port):
        channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = training_pb2_grpc.TrainingServiceStub(channel=channel)

    def register(self, worker_id, timeout):
        return self.stub.RegisterWorker(
            training_pb2.StartRequest(worker_id=worker_id), 
            timeout=timeout
        )

    def get_initial_weights(self, worker_id):
        return self.stub.GetInitialWeights(
            training_pb2.WeightRequest(worker_id=worker_id)
        )

    def get_updated_weights(self, worker_id):
        return self.stub.GetUpdatedWeights(
            training_pb2.WeightRequest(worker_id=worker_id)
        )

    def push_weights(self, worker_id, step):
        return self.stub.PushWeights(
            training_pb2.WeightData(
                worker_id=worker_id,
                step=step,
            )
        )

    def report_metrics(self, metric_data):
        return self.stub.ReportMetrics(metric_data)

    def finish_training(self, worker_id):
        return self.stub.FinishTraining(
            training_pb2.FinishRequest(worker_id=worker_id)
        )