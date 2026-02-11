import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from torch.optim import AdamW

# gRPC
import socket
import time 
import os
import grpc
import metrics_pb2
import metrics_pb2_grpc
import datetime

# MASTER_HOST = os.getenv('MASTER_HOST', 'localhost')
MASTER_HOST = os.getenv('MASTER_HOST', 'master-service')
MASTER_PORT = os.getenv('MASTER_PORT', '50051')
MODEL_TYPE = os.getenv('MODEL_TYPE', 'general') 
MODEL_BASE_DIR = "/app/models"

class DistributedTrainer:
    def __init__(self):
        self.worker_id = socket.gethostname()
        self.model_type = MODEL_TYPE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Worker: {self.worker_id} - Tipo: {self.model_type}")
        print(f"Device: {self.device}")
        print(f"Intentando conectar a {MASTER_HOST}:{MASTER_PORT}...")
        
        # Crear canal gRPC
        channel = grpc.insecure_channel(f'{MASTER_HOST}:{MASTER_PORT}')
        self.stub = metrics_pb2_grpc.MetricsServiceStub(channel=channel)
        print(f"✓ Canal gRPC creado para {MASTER_HOST}:{MASTER_PORT}")
    
    def load_data(self):
        print(f"Cargando dataset para modelo {self.model_type}...")
        
        if self.model_type == "general":
            dataset = load_dataset("imdb", split="train", revision="main")
            dataset = dataset.shuffle(seed=42).select(range(5000))
        elif self.model_type == "technical":
            # Por ahora usamos el mismo, luego lo cambiaremos
            dataset = load_dataset("imdb", split="train",revision="main")
            dataset = dataset.shuffle(seed=43).select(range(2000, 4000))
        elif self.model_type == "legal":
            dataset = load_dataset("imdb", split="train", revision="main")
            dataset = dataset.shuffle(seed=44).select(range(4000, 6000))
        else:
            dataset = load_dataset("imdb", split="train", revision="main")
            dataset = dataset.shuffle(seed=45).select(range(1000))

        print(f"Dataset size: {len(dataset)}")
        print(f"Ejemplo: {dataset[0]}")
        print(f"Labels únicos: {set(dataset['label'])}")

        return dataset
    
    def prepare_model(self):
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
        ).to(self.device)
        
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        return model, tokenizer        
    
    def train(self, epochs=3, batch_size=8, report_step=10):
        model, tokenizer = self.prepare_model()
        dataset = self.load_data()

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
        tokenized = dataset.map(tokenize_function, batched=True)
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        
        train_size = int(0.8 * len(tokenized))
        val_size = len(tokenized) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            tokenized, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model.train()
        optim = AdamW(model.parameters(), lr=5e-5)

        best_val_accuracy = 0.0
        final_accuracy = 0.0
        final_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            steps = 0
            correct_predictions = 0
            total_samples = 0
            for step, batch in enumerate(train_loader, 1):
                optim.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits

                loss.backward()
                optim.step()
                # Metricas
                epoch_loss += loss.item()
                steps += 1

                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                step_accuracy = correct_predictions / total_samples
                # Mandar metricas
                if (step % report_step == 0):
                    metric = metrics_pb2.MetricData (
                        worker_id = self.worker_id,
                        epoch = epoch,
                        step = step,
                        loss = loss.item(),
                        accuracy = step_accuracy,
                        timestamp = datetime.datetime.now().isoformat(),
                        model_type = self.model_type
                    )

                    try:
                        response = self.stub.ReportMetrics(metric)
                        print(f"Step {step}: Loss={loss:.4f}: Accuracy={step_accuracy}")
                    except grpc.RpcError as e:
                        print(f"Error gRPC: {e}")


            avg_loss = epoch_loss / steps
            train_accuracy = correct_predictions / total_samples

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
                    predictions = torch.argmax(outputs.logits, dim=1)
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)
            
            val_accuracy = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)

            print(f"\nEpoch {epoch} - {datetime.datetime.now().isoformat()}")
            print(f"TRAIN - Loss: {avg_loss:.4f} | Accuracy: {train_accuracy:.4f}")
            print(f"VAL   - Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.4f}")
            
            # ===== Guardar modelo =====
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

                save_dir = os.path.join(
                    MODEL_BASE_DIR,
                    self.model_type,
                    self.worker_id,
                    f"best_model-{datetime.datetime.now().isoformat()}"
                )

                os.makedirs(save_dir, exist_ok=True)

                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

                print(f"Nuevo mejor modelo guardado en: {save_dir}")
            
            model.train()

            final_loss = avg_val_loss
            final_accuracy = val_accuracy

        result = metrics_pb2.TrainingResult(
            worker_id = self.worker_id,
            model_type = self.model_type,
            final_loss = final_loss,
            final_accuracy = final_accuracy,
            total_epochs = epochs,
            model_path = save_dir
        )

        self.stub.ReportTrainingComplete(result)


if __name__ == "__main__":
    trainer = DistributedTrainer()
    print(f"Esperando al servidor....")
    
    
    try:
        # Cambiar iteración por llamada simple
        signal = trainer.stub.StartSignal(
            metrics_pb2.StartRequest(worker_id=trainer.worker_id), 
            timeout=120
        )
        
        if signal.ready:
            print(f"✓ Señal recibida. Iniciando entrenamiento...")
            trainer.train(epochs=3, report_step=signal.report_step)
            print(f"✓ Entrenamiento completado")
    except grpc.RpcError as e:
        print(f"✗ Error en la comunicación gRPC: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")
        