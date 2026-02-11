import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
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

MASTER_HOST = os.getenv('MASTER_HOST', 'master-service')
MASTER_PORT = os.getenv('MASTER_PORT', '50051')
MODEL_TYPE = os.getenv('MODEL_TYPE', 'general') 
MODEL_BASE_DIR = "/app/models"

MIN_FREE_MEMORY_GB = float(os.getenv('MIN_FREE_MEMORY_GB', '0.8'))  # Memoria mínima requerida
MAX_WAIT_TIME = int(os.getenv('MAX_WAIT_TIME', '600'))  # Máximo 10 minutos esperando
MEMORY_CHECK_INTERVAL = int(os.getenv('MEMORY_CHECK_INTERVAL', '10')) 

class DistributedTrainer:
    def __init__(self):
        self.worker_id = socket.gethostname()
        self.model_type = MODEL_TYPE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Worker: {self.worker_id} - Tipo: {self.model_type}")
        print(f"Device: {self.device}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Configurar asignación de memoria expandible
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        print(f"Intentando conectar a {MASTER_HOST}:{MASTER_PORT}...")
        
        # Crear canal gRPC
        channel = grpc.insecure_channel(f'{MASTER_HOST}:{MASTER_PORT}')
        self.stub = metrics_pb2_grpc.MetricsServiceStub(channel=channel)
        print(f"✓ Canal gRPC creado para {MASTER_HOST}:{MASTER_PORT}")
    
    def load_data(self):

        print(f"Cargando dataset para modelo {self.model_type}...")
        target_column = "text"

        if self.model_type == "general": 
            try:
                dataset = load_dataset("wikimedia/wikipedia", "20231101.es", split="train")
                dataset = dataset.shuffle(seed=42).select(range(6000))
            except Exception as e:
                print(f"Error cargando Wikipedia: {e}")
                print("Usando dataset alternativo...")
                dataset = load_dataset("mrm8488/spanish_news", split="train")
                dataset = dataset.shuffle(seed=42).select(range(min(4000, len(dataset))))
        elif self.model_type == "technical":
            # Por ahora usamos el mismo, luego lo cambiaremos
            dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
            dataset = dataset.shuffle(seed=43).select(range(min(5500, len(dataset))))
            dataset = dataset.rename_column("output", target_column)
        elif self.model_type == "legal":
            try:
                dataset = load_dataset("joelito/legal_es_1M", split="train")
                dataset = dataset.shuffle(seed=44).select(range(2000))  # Reducido de 6000 a 2000
            except Exception as e:
                print(f"Error cargando dataset legal: {e}")
                print("Usando dataset alternativo español...")
                dataset = load_dataset("mrm8488/spanish_news", split="train")
                dataset = dataset.shuffle(seed=44).select(range(min(2000, len(dataset))))
        else:
            dataset = load_dataset("imdb", split="train")
            dataset = dataset.shuffle(seed=45).select(range(4000))

        print(f"Dataset size: {len(dataset)}")

        columns_to_remove = [col for col in dataset.column_names if col != target_column]
        return dataset.remove_columns(columns_to_remove)
    
    def get_gpu_memory_info(self):
        """Obtiene información de memoria GPU disponible"""
        if not torch.cuda.is_available():
            return None
        
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        allocated_memory = torch.cuda.memory_allocated(0) / 1e9  # GB
        reserved_memory = torch.cuda.memory_reserved(0) / 1e9  # GB
        free_memory = total_memory - reserved_memory  # GB
        
        return {
            'total': total_memory,
            'allocated': allocated_memory,
            'reserved': reserved_memory,
            'free': free_memory
        }
    def wait_for_gpu_memory(self, required_gb=MIN_FREE_MEMORY_GB):
        """Espera hasta que haya suficiente memoria GPU disponible"""
        if not torch.cuda.is_available():
            print("No hay GPU disponible, usando CPU")
            return True
        
        wait_time = 0
        attempt = 0
        
        while wait_time < MAX_WAIT_TIME:
            attempt += 1
            mem_info = self.get_gpu_memory_info()
            
            print(f"\nIntento {attempt} - Verificando memoria GPU:")
            print(f"   Total: {mem_info['total']:.2f} GB")
            print(f"   Reservada: {mem_info['reserved']:.2f} GB")
            print(f"   Libre: {mem_info['free']:.2f} GB")
            print(f"   Requerida: {required_gb:.2f} GB")
            
            if mem_info['free'] >= required_gb:
                print(f"✓ Memoria suficiente disponible ({mem_info['free']:.2f} GB >= {required_gb:.2f} GB)")
                return True
            
            print(f"Memoria insuficiente. Esperando {MEMORY_CHECK_INTERVAL}s... ({wait_time}/{MAX_WAIT_TIME}s transcurridos)")
            time.sleep(MEMORY_CHECK_INTERVAL)
            wait_time += MEMORY_CHECK_INTERVAL
        
        print(f"⚠ Timeout: No se liberó suficiente memoria después de {MAX_WAIT_TIME}s")
        print(f"   Procediendo con memoria disponible ({mem_info['free']:.2f} GB)")
        return False
    
    def prepare_model(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Configuración de modelos con fallbacks
        model_configs = {
            "general": [
                "distilbert-base-multilingual-cased",
                "dccuchile/bert-base-spanish-wwm-uncased",
                "distilbert-base-uncased"
            ],
            "technical": [
                "distilbert-base-multilingual-cased",
                "distilbert-base-uncased"
            ],
            "legal": [
                "distilbert-base-multilingual-cased",  # Más pequeño que RoBERTa
                "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
                "distilbert-base-uncased"
            ]
        }
        
        # Obtener lista de modelos a probar
        model_names = model_configs.get(self.model_type, ["distilbert-base-uncased"])
        
        # Intentar cargar modelos en orden
        for model_name in model_names:
            try:
                print(f"Intentando cargar modelo: {model_name}")
                
                # Forzar uso de safetensors para evitar problemas de seguridad
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2,
                    use_safetensors=True,  # Forzar safetensors
                    trust_remote_code=False  # Seguridad adicional
                ).to(self.device)
                
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    print("✓ Gradient checkpointing habilitado")
                
                model = model.to(self.device)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                print(f"✓ Modelo cargado exitosamente: {model_name}")
                if torch.cuda.is_available():
                    print(f"  GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
                    print(f"  GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
                return model, tokenizer
                
            except Exception as e:
                print(f"✗ Error cargando {model_name}: {e}")
                if model_name == model_names[-1]:
                    # Si es el último modelo, intentar sin safetensors
                    print(f"Intentando cargar sin forzar safetensors...")
                    try:
                        model = AutoModelForSequenceClassification.from_pretrained(
                            model_name,
                            num_labels=2,
                            trust_remote_code=False
                        ).to(self.device)

                        if hasattr(model, 'gradient_checkpointing_enable'):
                            model.gradient_checkpointing_enable()

                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        print(f"✓ Modelo cargado (sin safetensors): {model_name}")
                        return model, tokenizer
                    except Exception as e2:
                        print(f"✗ Error final: {e2}")
                        raise RuntimeError(f"No se pudo cargar ningún modelo para tipo '{self.model_type}'")
                else:
                    print(f"Probando siguiente modelo alternativo...")
                    continue     
    
    def train(self, epochs=3, target_batch_size=16, micro_batch_size=4, report_step=10):
        accumulation_steps = target_batch_size // micro_batch_size
        model, tokenizer = self.prepare_model()
        dataset = self.load_data()

        def tokenize_function(examples):
            # Crear labels dummy (0 o 1 aleatorio) ya que no tenemos labels reales
            import random
            labels = [random.randint(0, 1) for _ in range(len(examples["text"]))]
            tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
            tokenized["label"] = labels
            return tokenized
        
        tokenized = dataset.map(tokenize_function, batched=True)
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        
        train_size = int(0.8 * len(tokenized))
        val_size = len(tokenized) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            tokenized, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=micro_batch_size)

        model.train()
        optim = AdamW(model.parameters(), lr=5e-5)
        total_steps = len(train_loader) * epochs // accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=0.1 * total_steps,
            num_training_steps=total_steps
        )
        best_val_accuracy = 0.0
        final_accuracy = 0.0
        final_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            steps = 0
            correct_predictions = 0
            total_samples = 0
            for step, batch in enumerate(train_loader, 1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss = loss / accumulation_steps

                loss.backward()
                if step  % accumulation_steps == 0:
                    optim.step()
                    scheduler.step()
                    optim.zero_grad()
                    if torch.cuda.is_available() and step % 50 == 0:
                        torch.cuda.empty_cache()
                # Metricas
                epoch_loss += loss.item()
                steps += 1

                logits = outputs.logits
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
        signal = trainer.stub.StartSignal(
            metrics_pb2.StartRequest(worker_id=trainer.worker_id), 
            timeout=120
        )
        
        if signal.ready:
            print(f"✓ Señal recibida. Iniciando entrenamiento...")
            trainer.train(epochs=10, report_step=signal.report_step)
            print(f"✓ Entrenamiento completado")
    except grpc.RpcError as e:
        print(f"✗ Error en la comunicación gRPC: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")