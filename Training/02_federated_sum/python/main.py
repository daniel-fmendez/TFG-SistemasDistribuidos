import grpc
import federated_sum_pb2
import federated_sum_pb2_grpc

def brain():
    numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = federated_sum_pb2_grpc.ProcessorStub(channel)

        print("Cerebro (Python) enviando datos al Sumador (Go)...")
        request = federated_sum_pb2.SumRequest(numbers=numeros)

        response = stub.ComputeSum(request= request)
        print(f"Resultado recibido desde Go: {response.result}")

if __name__ == '__main__':
    brain()