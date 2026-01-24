import grpc
from concurrent import futures
import hello_pb2
import hello_pb2_grpc

class Greeter(hello_pb2_grpc.GreeterService):
    def SayHello(self, request, context):
        return hello_pb2.HelloReply(message=f"Hello {request.name}")
    
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
hello_pb2_grpc.add_GreeterServiceServicer_to_server(Greeter(), server)

server.add_insecure_port('[::]:50051')

server.start()
print("Server listening on port 50051")
server.wait_for_termination()