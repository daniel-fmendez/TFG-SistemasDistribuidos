import socket
import time 
import os
import grpc
import metrics_pb2
import metrics_pb2_grpc
import datetime

MASTER_HOST = os.getenv('MASTER_HOST', 'master-service')
MASTER_PORT = os.getenv('MASTER_PORT', '50051')


def main():
    worker_id = socket.gethostname()
    channel = grpc.insecure_channel(f'{MASTER_HOST}:{MASTER_PORT}')
    stub = metrics_pb2_grpc.MetricsServiceStub(channel=channel)

    print(f"Worker {worker_id} conectado a {MASTER_HOST}:{MASTER_PORT}")

    iteration = 0

    while True:
        iteration += 1
        loss = 1.0 / (iteration + 1) 

        metric = metrics_pb2.MetricData (
            worker_id=worker_id,
            iteration=iteration,
            loss=loss,
            accuracy=0.9,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        try:
            response = stub.ReportMetrics(metric)
            print(f"Iteración {iteration}: Loss={loss:.4f}")
        except grpc.RpcError as e:
            print(f"Error gRPC: {e}")

        time.sleep(10)

if __name__ == "__main__":
    main()