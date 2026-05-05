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
        

        # Dataset
        self.dataset_name = raw["dataset"]["name"]
        self.dataset_path = raw["dataset"]["path"]
        self.total_samples = int(raw["dataset"]["total_samples"])
        self.num_labels = int(raw["dataset"]["num_labels"])
        self.split = raw["dataset"]["split"]

        # Training
        self.seed = int(raw["training"]["seed"])
        self.epochs = int(raw["training"]["epochs"])
        self.batch_size = int(raw["training"]["batch_size"])
        self.micro_batch_size = int(raw["training"]["micro_batch_size"])
        self.sync_every = int(raw["training"]["sync_every"])
        self.sync_every_early = int(raw["training"]["sync_every_early"])
        self.learning_rate = float(raw["training"]["learning_rate"])
        self.report_step = int(raw["training"]["report_step"])
        self.aggregation_strategy = raw["training"]["aggregation_strategy"]
        self.compression_strategy = raw["training"]["compression_strategy"]
        self.quantization_bits = raw["training"]["quantization_bits"]
        self.top_k_ratio = raw["training"]["top_k_ratio"]
        
        # Workers
        self.num_local_workers = int(raw["workers"]["num_local_workers"])
        self.num_lan_workers = int(raw["workers"]["num_lan_workers"])
        self.num_remote_workers = int(raw["workers"]["num_remote_workers"])
        self.num_workers = self.num_local_workers + self.num_remote_workers + self.num_lan_workers
        self.remote_shard_ratio = float(raw["workers"].get("remote_shard_ratio", 1.0))
        self.master_host = raw["workers"]["master_host"]
        self.master_ip = raw["workers"]["master_ip"]
        self.master_port = int(raw["workers"]["master_port"])
        self.pvc_name = raw["workers"]["pvc_name"]
        self.lan_pvc_name = raw["workers"]["lan_pvc_name"]
        self.remote_pvc_name = raw["workers"]["remote_pvc_name"]

        # Heartbeat
        self.heartbeat_interval = int(raw["heartbeat"]["interval"])
        self.heartbeat_multiplier = int(raw["heartbeat"]["multiplier"])
        
    def log(self, logger):
        logger.info(f"Config cargada:")
        logger.info(f"  Modelo: {self.model_name} ({self.num_labels} labels)")
        logger.info(f"  Dataset: {self.dataset_name} ({self.total_samples} samples)")
        logger.info(f"  Workers: {self.num_workers}")
        logger.info(f"  Epochs: {self.epochs}, Batch: {self.batch_size}, Sync cada: {self.sync_every}")
    