import grpc
import metrics_pb2 
import metrics_pb2_grpc

def test_connection():
    channel = grpc.insecure_channel('localhost:50051')
    stub = metrics_pb2_grpc.MetricsServiceStub(channel)

    metric = metrics_pb2.MetricData(
        worker_id="test-worker",
        iteration=1,
        loss=0.5,
        accuracy=0.95,
        timestamp="2025-01-18T10:00:00"
    )
    
    response = stub.ReportMetrics(metric)
    print(f"Test ReportMetric: {response.message}")

    metrics = stub.GetAllMetrics(metrics_pb2.Empty())
    print(f"Total métricas: {len(metrics.metrics)}")

if __name__ == "__main__":
    test_connection()