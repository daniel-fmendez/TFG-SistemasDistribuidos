import os

import yaml

CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/config/config.yaml")

class TrainingConfig:
    def __init__(self, path=CONFIG_PATH):
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        
        # Model
        self.model_type = raw["model"]["type"]
        self.model_name = raw["model"]["name"]
        self.num_labels = int(raw["model"]["num_labels"])

        # Dataset
        self.dataset_name = raw["dataset"]["name"]
        self.total_samples = int(raw["dataset"]["total_samples"])
        self.split = raw["dataset"]["split"]

        # Training
        self.epochs = int(raw["training"]["epochs"])
        self.batch_size = int(raw["training"]["batch_size"])
        self.micro_batch_size = int(raw["training"]["micro_batch_size"])
        self.sync_every = int(raw["training"]["sync_every"])
        self.sync_every_early = int(raw["training"]["sync_every_early"])
        self.learning_rate = float(raw["training"]["learning_rate"])
        self.report_step = int(raw["training"]["report_step"])
        
        # Workers
        self.num_workers = int(raw["workers"]["num_workers"])
        self.master_host = raw["workers"]["master_host"]
        self.master_port = int(raw["workers"]["master_port"])
        self.pvc_name = raw["workers"]["pvc_name"]
        
    def log(self, logger):
        logger.info(f"Config cargada:")
        logger.info(f"  Modelo: {self.model_name} ({self.num_labels} labels)")
        logger.info(f"  Dataset: {self.dataset_name} ({self.total_samples} samples)")
        logger.info(f"  Workers: {self.num_workers}")
        logger.info(f"  Epochs: {self.epochs}, Batch: {self.batch_size}, Sync cada: {self.sync_every}")
    