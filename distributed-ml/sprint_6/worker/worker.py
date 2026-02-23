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

from config_loader import TrainingConfig
from model_factory import ModelFactory
from dataset_factory import DatasetFactory
MASTER_HOST = os.getenv('MASTER_HOST', 'master-service')
MASTER_PORT = os.getenv('MASTER_PORT', '50051')
MODEL_DIR = "/data"
WORKER_WEIGHTS_DIR = "/data/worker_weights"


class DistributedTrainer():
    def __init__(self):
        self.cfg = TrainingConfig()
        self.worker_id = socket.gethostname()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Conexion gRPC
        channel = grpc.insecure_channel(f'{self.cfg.master_host}:{self.cfg.master_port}')
        self.stub = training_pb2_grpc.TrainingServiceStub(channel=channel)

    def train(self, start, end, report_step):
        # Cargar datos
        start = int(start)
        end = int(end)
        report_step = int(report_step)

        dataset = DatasetFactory.load("/data/train")
        dataset_info = DatasetFactory.get_info(self.cfg.dataset_name)

        print(f"Seleccionando samples [{start}:{end}]")
        dataset = dataset.select(range(start, end))

        if dataset_info["type"] == "text_classification":
            dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        elif dataset_info["type"] == "image_classification":
            dataset.set_format(type="torch", columns=["pixel_values", "label"])

        # Pedir pesos iniciales
        weight_request = training_pb2.WeightRequest(worker_id=self.worker_id)
        weight_response = self.stub.GetInitialWeights(weight_request)

        dataset_info = DatasetFactory.get_info(self.cfg.dataset_name)
        num_labels = dataset_info["num_labels"]

        model = ModelFactory.build(
            self.cfg.model_type, 
            self.cfg.model_name, 
            num_labels
        ).to(self.device)
        model.load_state_dict(torch.load(weight_response.weights_path, 
                            map_location=self.device, weights_only=True))

        train_loader = DataLoader(dataset, batch_size=self.cfg.micro_batch_size, shuffle=True)
        optim = AdamW(model.parameters(), lr=self.cfg.learning_rate)
        accumulation_steps = self.cfg.batch_size // self.cfg.micro_batch_size

        batch_counter = 0
        for epoch in range(self.cfg.epochs):
            sync_every = self.cfg.sync_every_early if epoch == 0 else self.cfg.sync_every
            correct_predictions = 0
            total_samples = 0

            for step, batch in enumerate(train_loader):
                labels = batch["label"].to(self.device)

                if dataset_info["type"] == "text_classification":
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                elif dataset_info["type"] == "image_classification":
                    pixel_values = batch["pixel_values"].to(self.device)
                    outputs = model(pixel_values)
                    # Para modelos de visión, calcular loss manualmente
                    import torch.nn.functional as F
                    loss = F.cross_entropy(outputs.logits if hasattr(outputs, 'logits') else outputs, labels)
                    # Crear objeto similar a outputs de transformers
                    class OutputWrapper:
                        def __init__(self, logits, loss):
                            self.logits = logits
                            self.loss = loss
                    outputs = OutputWrapper(outputs if not hasattr(outputs, 'logits') else outputs.logits, loss)
                
                loss = outputs.loss
                loss.backward()

                if (step + 1)  % accumulation_steps == 0:
                    batch_counter += 1
                    # Enviar gradientes al master
                    if batch_counter % sync_every == 0:
                        self._sync_with_master(model, step)
                    
                    optim.step()
                    optim.zero_grad()
                
                #  Enviar metricas 
                if step % report_step == 0:
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_samples += labels.size(0)
                    
                    step_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
                    
                    metric = training_pb2.MetricData(
                        worker_id=self.worker_id,
                        epoch=epoch,
                        step=step,
                        loss=loss.item(),
                        accuracy=step_accuracy,
                        timestamp=datetime.datetime.now().isoformat(),
                    )
                    
                    try:
                        self.stub.ReportMetrics(metric)
                        print(f"Epoch {epoch}, Step {step}: Loss={loss:.4f}, Acc={step_accuracy:.4f}")
                    except grpc.RpcError as e:
                        print(f"Error gRPC: {e}")
                    

        if batch_counter % sync_every != 0:
            optim.step()
            self._sync_with_master(model, batch_counter)

        finish_request = training_pb2.FinishRequest(
            worker_id=self.worker_id,
        )
        self.stub.FinishTraining(finish_request)

    def _sync_with_master(self, model, step):
        # Crear directorio para este step
        step_dir = os.path.join(WORKER_WEIGHTS_DIR, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        
        
        worker_index = int(self.worker_id.split('-')[1])
        weights_path = os.path.join(step_dir, f"worker_{worker_index}.pt")
        torch.save(model.state_dict(), weights_path)
        
        
        # Notificar al master (mensaje ligero con worker_id completo)
        weight_data = training_pb2.WeightData(
            worker_id=self.worker_id,  # String completo para el mapeo
            step=step,
        )
        self.stub.PushWeights(weight_data)
        
        # Esperar pesos actualizados...
        weight_request = training_pb2.WeightRequest(worker_id=self.worker_id)
        weight_response = self.stub.GetUpdatedWeights(weight_request)

        weights_path = weight_response.weights_path
        updated_weights = torch.load(weights_path, map_location=self.device, weights_only=True)
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

            shard_start = int(shard_start)
            shard_end = int(shard_end)
            report_step = int(signal.report_step)
            trainer.train(start=shard_start, end=shard_end, report_step=report_step)
            print(f"Entrenamiento completado")
    except grpc.RpcError as e:
        print(f"Error en la comunicación gRPC: {e}")
    except Exception as e:
        print(f"Error: {e}")