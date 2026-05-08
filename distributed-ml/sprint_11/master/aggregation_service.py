import os
import shutil
import threading

import torch
from dataset_factory import DatasetFactory
from model_factory import ModelFactory

WORKER_WEIGHTS_DIR = "/data/worker_weights"
WEIGHTS_DIR = "/data/weights"

METRICS_FILE = "training_metrics.json"
WEIGHTS_FILE = "pytorch_model_weights.pt"

class AggregationService:
    def __init__(self, config, registry, metrics_collector, aggregator):
        self.cfg = config
        self.registry = registry
        self.metrics_collector = metrics_collector
        self.aggregator = aggregator
        
        self.workers_weights = {}
        self.global_weights = None
        self.current_weights_path = None
        self.version_counter = 0  
        self.lock = threading.Lock()
        self.running = True
        
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        os.makedirs(WORKER_WEIGHTS_DIR, exist_ok=True)

    def _aggregate_weights_from_disk(self, step):
        step_dir = os.path.join(WORKER_WEIGHTS_DIR, f"step_{step}")

        with self.registry._lock:
            alive_list = list(self.registry.alive_workers)
            
        self.global_weights = self.aggregator.fed_average(step_dir, alive_list)
        self._save_weights_to_pvc()
    
    def initialize_weights(self):
        dataset_info = DatasetFactory.get_info(self.cfg.dataset_name)
        num_labels = dataset_info["num_labels"]
        model = ModelFactory.build(
            model_display_name=self.cfg.model_name, 
            num_labels=num_labels
        )
        self.global_weights = model.state_dict()
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return self._save_weights_to_pvc()

    def _save_weights_to_pvc(self):
        with self.lock: 
            final_path = os.path.join(WEIGHTS_DIR, f"global_weights_v{self.version_counter}.pt")
            temp_path = final_path + ".tmp"

            torch.save(self.global_weights, temp_path)
            os.replace(temp_path, final_path) 

            if self.version_counter > 0:
                old_path = os.path.join(WEIGHTS_DIR, f"global_weights_v{self.version_counter-1}.pt")
                if os.path.exists(old_path):
                    os.remove(old_path)

            self.current_weights_path = final_path
            self.version_counter += 1
            print(f"Pesos guardados en {final_path}")
            return final_path

    def _cleanup_weights(self, step):
        step_dir = os.path.join(WORKER_WEIGHTS_DIR, f"step_{step}")
        if os.path.exists(step_dir):
            shutil.rmtree(step_dir)
            print(f"Pesos del step {step} eliminados")

    def aggregate(self, worker_id, step):
        if self.registry.pause_event and self.registry.pause_event.is_set():
            print(f"[Aggregator] Sistema pausado, descartando agregación del step {step}")
            return
        
        should_aggregate = False

        with self.lock:
            if step not in self.workers_weights:
                self.workers_weights[step] = set()
            self.workers_weights[step].add(worker_id)
         
            # Cuando todos los vivos esten listos
            alive_count = len(self.registry.alive_workers)

            if len(self.workers_weights[step]) == alive_count:
                del self.workers_weights[step]
                should_aggregate = True

        if should_aggregate:
            if self.registry.pause_event and self.registry.pause_event.is_set():
                print(f"[Aggregator] Pausa detectada, descartando agregación del step {step}")
                return
            self._aggregate_weights_from_disk(step)
            self._cleanup_weights(step)
            self.metrics_collector.record_aggregation()
            
    def get_updated_weights(self):
        with self.lock:
            return self.current_weights_path 
        
    