import grpc
import federated_grad_pb2
import federated_grad_pb2_grpc

from concurrent import futures
import sys
# Training
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

LEARNING_RATE = 0.01

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
model = SimpleNet()
criterion = nn.CrossEntropyLoss()

class WorkerServicer (federated_grad_pb2_grpc.WorkerServicer):
    def TrainBatch(self, request, context):
        batch_size = request.batch_size
        images = torch.tensor(list(request.data)).reshape(batch_size, 1, 28, 28)
        labels = torch.tensor(list(request.labels), dtype=torch.long)

        print(f"   Recibido batch de tamaño: {batch_size}")
        print(f"   Imágenes shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_values = param.grad.flatten().tolist()
                grad_shape = list(param.grad.shape)
                gradients[name] = federated_grad_pb2.Gradients(
                    values=grad_values,
                    shape=grad_shape
                )
                print(f"      Gradiente {name}: shape={grad_shape}, primeros valores={grad_values[:5]}")

        model.zero_grad()
        print(f"      Loss: {loss.item():.4f}\n")

        return federated_grad_pb2.TrainResponse(
            gradients=gradients,
            loss=loss.item()
        )
    
    def UpdateWeights(self, request, context):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in request.weights:
                    shape = tuple(request.weights[name].shape)

                    new_weights = torch.tensor(
                        list(request.weights[name].values)
                    ).reshape(shape)

                    param.copy_(new_weights)
        return federated_grad_pb2.WeightAck(success=True)

def serve(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=3))
    federated_grad_pb2_grpc.add_WorkerServicer_to_server(
        WorkerServicer(), 
        server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    print(f"   Worker server iniciado en puerto {port}")
    print(f"   Modelo: {model}")
    print(f"   Parámetros del modelo:")
    for name, param in model.named_parameters():
        print(f"      - {name}: {list(param.shape)}")
    print("\n  Esperando peticiones del coordinador...\n")
    
    server.wait_for_termination()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 50051
    serve(port=port)