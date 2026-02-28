import os

import torch

WORKER_WEIGHTS_DIR = "/data/worker_weights"

class WeightSynchronizer:
    def __init__(self, worker_id, worker_index, device, grpc_client):
        self.worker_id = worker_id
        self.worker_index = worker_index
        self.device = device
        self.client = grpc_client

    def sync(self, model, step):
        self._save_weights_to_disk(model, step)
        self.client.push_weights(self.worker_id, step)
        self._load_updated_weights(model)

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
    