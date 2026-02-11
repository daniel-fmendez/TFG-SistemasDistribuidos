# Master Template

# Main
# Se inicializa el server:
#   - Se establecen pesos mediante random pytorch
#   - Se preparan los workers y se encolan hasta que todos esten listos
#   - Una vez encolados todos se divide el dataset en N muestras. N nºservers
#   - Se les indica iniciar.
#
# Cada N steps los workers comunican gradientes, el server calcula la agregacion y promedios y se envian nuevos pesos
# Al finalizar el msater guarda UN modelo unico y los workers se mueren sin guardar nad (Son calculadoras)

# LLamadas gRPC
#   - Start signal
#   - Send metrics
#   - Update Weigths
#   - Finish training

import os
import torch
import grpc
from concurrent import futures
import logging
import os
import time 
import datetime
import threading
import yaml
import subprocess
import io 
import torch.nn as nn
import json
import training_pb2
import training_pb2_grpc

from datasets import load_dataset_builder
from templates import get_pvc_template, get_dataset_init_job_template, get_worker_job_template
from kubernetes import client, config
from kubernetes.utils import create_from_dict
from kubernetes.client.rest import ApiException
from transformers import DistilBertForSequenceClassification

# Donde se guardará el modelo
MODEL_DIR = "/data"
WEIGHTS_DIR = "/data/weights"

PVC_NAME = "server-pvc"
WORKERS_FILE = "worker-job.yaml"
TOTAL_SAMPLES  = 8000
REPORT_STEP = 40

# Eliminar warning de hugging face
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4
    )
    return model

def init_weights_random(module):
    if isinstance(module, nn.Linear):
        # Inicialización Xavier para capas lineales
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Embedding):
        # Inicialización normal para embeddings
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    elif isinstance(module, nn.LayerNorm):
        # LayerNorm: weights a 1, bias a 0
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)

def apply_manifest(file):
    try:
        config.load_incluster_config()
        k8s_client = client.ApiClient()

        # Leer el YAML
        with open(f"{file}.yaml", "r") as f:
            docs = list(yaml.safe_load_all(f))

        for doc in docs:
            if not doc:
                continue

            # Crea o actualiza el recurso
            create_from_dict(k8s_client, data=doc, verbose=True)

        print("Manifiesto aplicado correctamente")

    except FileNotFoundError:
        print("No se encontró el archivo YAML.")
    except ApiException as e:
        print("Error de la API de Kubernetes:", e)
    except Exception as e:
        print("Error al aplicar el manifiesto:", str(e))


class TrainingService:
    def __init__(self, num_workers, server):
        self.server = server
        self.num_workers = num_workers
        self.registered_workers = []
        self.workers_gradient = {}
        self.shards = []
        self.metrics = [[] for _ in range(num_workers)]
        self.finished_workers = set()
        self.training_completed = False 
        self.version_counter = 0

        # Inicializar modelo global
        self.model = build_model()
        self.model.apply(init_weights_random)
        self.global_weights = self.model.state_dict()

        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

        os.makedirs(WEIGHTS_DIR, exist_ok=True)

        self._save_weights_to_pvc()
        # Metodos al inicio
        self.prepare_training()
        self.createWorkers()
        logger.info(f"Esperando Workers....")


    def _save_weights_to_pvc(self):
        weights_path = os.path.join(WEIGHTS_DIR, f"global_weights_v{self.version_counter}.pt")
        torch.save(self.global_weights, weights_path)
        self.current_weights_path = weights_path
        self.version_counter += 1
        logger.info(f"Pesos guardados en {weights_path}")
        return weights_path
    # ===================================================
    # LLamadas gRPC
    # ===================================================

    # Llamada gRPC (cuando arranca el worker)
    def RegisterWorker(self, request, context):
        worker_id = request.worker_id

        with self.condition:
            if worker_id not in self.registered_workers:
                self.registered_workers.append(worker_id)
                print(f"Worker {worker_id} registrado.")

            index = self.registered_workers.index(worker_id)
            while len(self.registered_workers) < self.num_workers:
                self.condition.wait()

            self.condition.notify_all()

            config = self.shards[index]
        return training_pb2.StartResponse(
            ready=True,
            start = config["start"],
            end = config["end"],
            report_step = REPORT_STEP
        )

    def ReportMetrics(self, request, context):
        logger.info(f"Worker: {request.worker_id}\t, Epoch: {request.epoch}, Step: {request.step}, Loss: {request.loss:.4f}, Acc: {request.accuracy:.4f}")

        self.metrics[request.worker_id].append({
            'worker_id': request.worker_id,
            'epoch': request.epoch,
            'step': request.step,
            'loss': request.loss,
            'accuracy': request.accuracy,
            'timestamp': request.timestamp,
        })

        return training_pb2.Ack(success=True, message="200 Ok")

    def GetInitialWeights(self, request, context):
        worker_id = request.worker_id
        logger.info(f"Worker {worker_id} solicitando pesos iniciales")

        return training_pb2.WeightResponse(
            weights_path=self.current_weights_path
        )
    
    def PushGradients(self, request, context):
        worker_id = request.worker_id
        buffer = io.BytesIO(request.serialized_gradients)
        gradients = torch.load(buffer)
        
        with self.lock:
            self.workers_gradient[worker_id] = gradients
            
            if len(self.workers_gradient) == self.num_workers:
                self._aggregate_gradients()

                self._save_weights_to_pvc()
                self.workers_gradient.clear()
        
        return training_pb2.Ack(success=True, message="Gradientes recibidos")
    
    # Cada N steps los workers comunican gradientes, agrefa y acctualiza
    def _aggregate_gradients(self):
        logger.info("Agregando gradientes de todos los workers")
        
        aggregated = {}
        
        for param_name in self.global_weights.keys():
            # Recolectar gradientes de todos los workers para este parámetro
            grads = [worker_grads[param_name] for worker_grads in self.workers_gradient.values()]
            
            # Promedio simple
            aggregated[param_name] = torch.stack(grads).mean(dim=0)
        
        # Actualizar pesos globales (SGD simple)
        learning_rate = 0.001
        with torch.no_grad():
            for name in self.global_weights.keys():
                self.global_weights[name] -= learning_rate * aggregated[name]

    def GetUpdatedWeights(self, request, context):
        worker_id = request.worker_id
        logger.info(f"Worker {worker_id} solicitando pesos actualizados")
        
        return training_pb2.WeightResponse(
            weights_path=self.current_weights_path
        )

    def FinishTraining(self, request, context):
        worker_id = request.worker_id

        with self.lock:
            # Registrar worker como terminado
            self.finished_workers.add(worker_id)
            logger.info(f"Worker {worker_id} ha finalizado. ({len(self.finished_workers)}/{self.num_workers})")
            
            # Si todos los workers han terminado
            if len(self.finished_workers) == self.num_workers:
                if self.workers_gradient:
                    logger.info("Agregando gradientes finales...")
                    self._aggregate_gradients()
                    self._save_weights_to_pvc()
                
                self._save_final_model()
                
                # Marcar entrenamiento como completado
                self.training_completed = True
                self.server.stop(grace=10)  # Espera 10s antes de forzar cierre
        
        return training_pb2.Ack(
            success=True, 
            message=f"Worker {worker_id} finalizado correctamente"
        )
    
    def _save_final_model(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"final_model_{timestamp}")
        
        # Crear directorio si no existe
        os.makedirs(model_path, exist_ok=True)
        
        # Reconstruir modelo con pesos finales
        final_model = build_model()
        final_model.load_state_dict(self.global_weights)
        
        # Guardar modelo completo y pesos
        final_model.save_pretrained(model_path)
        
        torch.save(
            self.global_weights, 
            os.path.join(model_path, "pytorch_model_weights.pt")
        )
        
        logger.info(f"Modelo guardado exitosamente en {model_path}")
        self._save_training_metrics(model_path)
    
    def _save_training_metrics(self, model_path):
        metrics_file = os.path.join(model_path, "training_metrics.json")
        
        # Guardar métricas de todos los workers
        all_metrics = {
            f"worker_{i}": self.metrics[i] 
            for i in range(self.num_workers)
        }
        
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)
        
        logger.info(f"Métricas guardadas en {metrics_file}")
    # ===================================================

    # Preparalos splits del modelo para cada worker
    def prepare_training(self):
        # Se crea al inicio
        shard_len = TOTAL_SAMPLES // self.num_workers
        for i in range(self.num_workers):
            start = i * shard_len
            end = start + shard_len
            shard = {
                "start": start,
                "end": end
            }
            self.shards.append(shard)

        print("Shards preparados....")

    # Crea los workers
    def createWorkers(self):
        for i in range(self.num_workers):
            worker_manifest = get_worker_job_template(
                worker_id=i,
                master_host='master-service',
                master_port=50051,
                pvc_name=PVC_NAME
            )
            name = f"worker-{i}"
            with open(f"{name}.yaml", "w") as f:
                yaml.dump(worker_manifest, f, default_flow_style=False)
            
            apply_manifest(name)
    
    

    
def serve():
    # Crear service y conectarlo a un server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    trainer_service = TrainingService(
        num_workers=4,
        server=server
    )
    training_pb2_grpc.add_TrainingServiceServicer_to_server(trainer_service, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("Master gRPC escuchando en puerto 50051")
    server.wait_for_termination()
    
if __name__ == "__main__":
    serve()
    