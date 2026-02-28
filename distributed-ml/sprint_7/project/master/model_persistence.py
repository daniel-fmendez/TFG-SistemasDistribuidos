import json
import os
import threading
from datetime import datetime

import torch
from model_factory import ModelFactory

MODEL_DIR = "/data"
METRICS_FILE = "training_metrics.json"
WEIGHTS_FILE = "pytorch_model_weights.pt"

class ModelPersistence:
    def __init__(self, config):
        self.cfg = config
        self.lock = threading.Lock()

        os.makedirs(MODEL_DIR, exist_ok=True)

    def save_final_model(self, global_weights, metrics):
        with self.lock:
            model_path = self._build_model_path()
            os.makedirs(model_path, exist_ok=True)
            self._save_weights(model_path, global_weights)
            self._save_pretrained(model_path, global_weights)
            self._save_metrics(model_path, metrics)
            print(f"Modelo guardado exitosamente en {model_path}")
    
    def _build_model_path(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(MODEL_DIR, f"final_model_{timestamp}")

    def _save_weights(self, model_path, global_weights):
        torch.save(global_weights, os.path.join(model_path, WEIGHTS_FILE))

    def _save_pretrained(self, model_path, global_weights):
        if self.cfg.model_type not in ["distilbert", "bert", "roberta"]:
            return
        
        model = ModelFactory.build(
            model_type=self.cfg.model_type,
            model_name=self.cfg.model_name,
            num_labels=self.cfg.num_labels
        )
        model.load_state_dict(global_weights)
        model.save_pretrained(model_path)
        del model

    def _save_metrics(self, model_path, metrics):
        metrics_file = os.path.join(model_path, METRICS_FILE)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Métricas guardadas en {metrics_file}")