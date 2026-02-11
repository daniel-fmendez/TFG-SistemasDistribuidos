# Worker temmpplate

# El servidor los invoca en el kubernete
# Se inicializan y se conectan al master
# Carga su parte del dataset
#
# Por cada epoch solicitan pesos
# Ejecutan el entrenamiento por cada epoch:
#   - Calcular outpus
#   - Calcular loss backward (gradiente)
#   - Vaciar gradientes 
#   - Repetir
#
# Durante el entrenamiento enviar cada X steps metricas de cada worker avg/loss , accuracy....
import socket
import grpc
import os
import torch
import datetime
import io
import training_pb2
import training_pb2_grpc

from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MASTER_HOST = os.getenv('MASTER_HOST', 'master-service')
MASTER_PORT = os.getenv('MASTER_PORT', '50051')
MODEL_DIR = "/data"
BATCH_SIZE = 8
MICRO_BATCH_SIZE = 2

class DistributedTrainer():
    def __init__(self):
        self.worker_id = socket.gethostname()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Conexion gRPC
        channel = grpc.insecure_channel(f'{MASTER_HOST}:{MASTER_PORT}')
        self.stub = training_pb2_grpc.TrainingServiceStub(channel=channel)

    def train(self, start, end, report_step, epochs=5):
        # Cargar datos
        dataset = load_from_disk("/data/train")
        dataset = dataset.select(range(start, end))
        
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        # Pedir pesos iniciales
        weight_request = training_pb2.WeightRequest(worker_id=self.worker_id)
        weight_response = self.stub.GetInitialWeights(weight_request)

        weights_path = weight_response.weights_path

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=4
        ).to(self.device)
        model.load_state_dict(torch.load(weights_path, map_location=self.device))

        train_loader = DataLoader(dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True)
        optim = AdamW(model.parameters(), lr=5e-5)
        accumulation_steps = BATCH_SIZE // MICRO_BATCH_SIZE

        for epoch in range(epochs):
            epoch_loss = 0.0
            steps = 0
            correct_predictions = 0
            total_samples = 0
            
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                if (step + 1)  % accumulation_steps == 0:
                    # Enviar gradientes al master
                    self._sync_with_master(model, step)                    
                    optim.step()
                    optim.zero_grad()

                
                #  Enviar metricas 
                if (step % report_step == 0):
                    # Metricas
                    epoch_loss += loss.item()
                    steps += 1

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_samples += labels.size(0)
                    
                    step_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

                    metric = training_pb2.MetricData (
                        worker_id = self.worker_id,
                        epoch = epoch,
                        step = step,
                        loss = loss.item(),
                        accuracy = step_accuracy,
                        timestamp = datetime.datetime.now().isoformat(),
                    )

                    try:
                        response = self.stub.ReportMetrics(metric)
                        print(f"Step {step}: Loss={loss:.4f}: Accuracy={step_accuracy}")
                    except grpc.RpcError as e:
                        print(f"Error gRPC: {e}")

        if any(param.grad is not None for param in model.parameters()):
            self._sync_with_master(model, step=-1)
            optim.zero_grad()

        finish_request = training_pb2.FinishRequest(
            worker_id=self.worker_id,
        )
        response = self.stub.FinishTraining(finish_request)

    def _sync_with_master(self, model, step):
        gradients = {
            name: param.grad.clone() 
            for name, param in model.named_parameters()
            if param.grad is not None    
        }
        buffer = io.BytesIO()
        torch.save(gradients, buffer)
        buffer.seek(0)

        grad_data = training_pb2.GradientData(
            worker_id=self.worker_id,
            step=step,
            serialized_gradients=buffer.read()
        )
        
        self.stub.PushGradients(grad_data)
        
        # Pedir pesos actualizados
        weight_request = training_pb2.WeightRequest(worker_id=self.worker_id)
        weight_response = self.stub.GetUpdatedWeights(weight_request)

        weights_path = weight_response.weights_path
        updated_weights = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(updated_weights)

if __name__ == "__main__":
    trainer = DistributedTrainer()
    print(f"Esperando al servidor....")
    
    try:
        signal = trainer.stub.RegisterWorker(
            training_pb2.StartRequest(worker_id=trainer.worker_id), 
            timeout=240
        )
        
        if signal.ready:
            print(f"Señal recibida. Iniciando entrenamiento...")
            # Cargar datos
            shard_start = signal.start
            shard_end = signal.end
            print(f"{trainer.worker_id}: Inicio: {shard_start} - Fin: {shard_end}")

            trainer.train(start=shard_start, end=shard_end, report_step=signal.report_step, epochs=6)
            print(f"Entrenamiento completado")
    except grpc.RpcError as e:
        print(f"Error en la comunicación gRPC: {e}")
    except Exception as e:
        print(f"Error: {e}")