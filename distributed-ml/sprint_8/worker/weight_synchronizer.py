import os
import time
import torch

WORKER_WEIGHTS_DIR = "/data/worker_weights"

class WeightSynchronizer:
    def __init__(self, config, worker_id, worker_index, device, grpc_client):
        self.cfg = config
        self.worker_id = worker_id
        self.worker_index = worker_index
        self.device = device
        self.client = grpc_client

        self.is_paused = False 
    def sync(self, model, step):
        self._save_weights_to_disk(model, step)
        response = self.client.push_weights(self.worker_id, step)
        if not response.success:
            self.is_paused = True
            print(f"[Worker] Sistema pausado, esperando reanudación...")
            self._wait_for_resume()
            self.is_paused = False
            return
        self._load_updated_weights(model)

    def _wait_for_resume(self):
        while True:
            time.sleep(5)
            response = self.client.check_rebalance(self.worker_id)
            if not response.pause:
                print(f"[Worker] Sistema reanudado, continuando...")
                return
            print(f"[Worker] Aún pausado, esperando...")
    
    def load_initial_weights(self, model):
        response = self.client.get_initial_weights(self.worker_id)
        weights = torch.load(response.weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(weights)

    def _save_weights_to_disk(self, model, step):
        step_dir = os.path.join(WORKER_WEIGHTS_DIR, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        weights_path = os.path.join(step_dir, f"worker_{self.worker_index}.pt")
        torch.save(model.state_dict(), weights_path)

    def _load_updated_weights(self, model):
        response = self.client.get_updated_weights(self.worker_id)
        weights = torch.load(response.weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(weights)

    def check_rebalance(self):
        response = self.client.check_rebalance(self.worker_id)
        
        if response.pause:
            print(f"[Worker {self.worker_id}] Pausado por el master, esperando...")
            while True:
                time.sleep(self.cfg.heartbeat_interval)
                response = self.client.check_rebalance(self.worker_id)
                if not response.pause:
                    print(f"[Worker {self.worker_id}] Reanudando...")
                    break

        return response
        