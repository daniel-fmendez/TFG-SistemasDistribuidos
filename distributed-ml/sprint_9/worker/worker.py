import socket
import os
import threading
import random
import numpy as np
import grpc
import torch
from config_loader import TrainingConfig
from dataset_factory import DatasetFactory
from grpc_client import GrpcClient
from model_factory import ModelFactory
from training_loop import TrainingLoop
from weight_synchronizer import WeightSynchronizer
from heartbeat_loop import HeartbeatLoop

class DistributedTrainer:
    def __init__(self):
        self.cfg = TrainingConfig()
        self.worker_id = socket.gethostname()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        master_host = os.environ.get("MASTER_HOST") or self.cfg.master_host
        master_port = int(os.environ.get("MASTER_PORT") or self.cfg.master_port)
        self.client = GrpcClient(master_host, master_port)

    def _set_seed(self):
        torch.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def run(self):
        
        try:
            stop_event = threading.Event()
            
            # Insertar heartbeat
            heartbeat = HeartbeatLoop(self.cfg, self.worker_id, self.client, stop_event)
            heartbeat_thread = threading.Thread(target=heartbeat.heartbeat_loop, daemon=True)
            heartbeat_thread.start()
            signal = self.client.register(self.worker_id, timeout=600)
            if not signal.ready:
                return
            

            shard_len = self.cfg.total_samples // self.cfg.num_workers
            worker_index = signal.worker_index

            synchronizer = WeightSynchronizer(
                config=self.cfg,
                worker_id=self.worker_id,
                worker_index=worker_index,
                device=self.device,
                grpc_client=self.client,
                is_remote=False
            )


            dataset_info = DatasetFactory.get_info(self.cfg.dataset_name)
            print(f"dataset_info: {dataset_info}") 
            dataset = DatasetFactory.load_shard("/data/train", int(signal.start), int(signal.end), dataset_info)

            self._set_seed()
            model = self._build_model(dataset_info)
            synchronizer.load_initial_weights(model)
            
            
            loop = TrainingLoop(self.cfg, self.device, self.client, synchronizer)
            loop.run(self.worker_id, model, dataset, dataset_info, signal, stop_event)

            self.client.finish_training(self.worker_id)
            print("Entrenamiento completado")

        except grpc.RpcError as e:
            print(f"Error gRPC: {e}")
        except Exception as e:
            print(f"Error: {e}")
    
    def _build_model(self, dataset_info):
        return ModelFactory.build(
            self.cfg.model_type,
            self.cfg.model_name,
            dataset_info["num_labels"]
        ).to(self.device)
    
if __name__ == "__main__":
    trainer = DistributedTrainer()
    print("Esperando al servidor....")
    trainer.run()