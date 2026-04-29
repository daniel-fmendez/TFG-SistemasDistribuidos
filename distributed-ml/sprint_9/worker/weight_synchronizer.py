import os
import io
import time
import torch
from compressor import Compressor

WORKER_WEIGHTS_DIR = "/data/worker_weights"
CHUNK_SIZE = 3 * 1024 * 1024 

class WeightSynchronizer:
    def __init__(self, config, worker_id, worker_index, device, grpc_client, is_remote):
        self.cfg = config
        self.worker_id = worker_id
        self.worker_index = worker_index
        self.device = device
        self.client = grpc_client
        self.is_remote = is_remote

        self.compressor = Compressor(
            strategy=config.compression_strategy, 
            k_ratio=config.top_k_ratio,            
            quantization_bits=config.quantization_bits
        )

    def load_initial_weights(self, model):
        chunks = self.client.get_initial_weights(self.worker_id)
        weights = self._chunks_to_state_dict(chunks)
        model.load_state_dict(weights)

    def sync(self, model, step):
        sync_start = time.time()

        weights_path = self._push_weights(model, step)

        bytes_sent = os.path.getsize(weights_path)

        response = self.client.push_weights(self.worker_id, step, weights_path)
        if not response.success:
            self._block_until_resumed()

        self._pull_updated_weights(model)
        sync_duration = time.time() - sync_start

        self.client.report_sync_metrics(self.worker_id, bytes_sent, sync_duration)
        
    def consume_rebalance(self):
        response = self.client.check_rebalance(self.worker_id)
        
        if response.pause:
            self._block_until_resumed()
            response = self.client.check_rebalance(self.worker_id)

        if response.rebalanced:
            return {"start": response.new_start, "end": response.new_end}
                
        return None

    
            
    def _push_weights(self, model, step):
        step_dir = os.path.join(WORKER_WEIGHTS_DIR, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)

        state_dict = model.state_dict()
        compressed, metadata = self.compressor.compress(state_dict)

        weights_path = os.path.join(step_dir, f"worker_{self.worker_index}.pt")
        torch.save({"weights": compressed, "metadata": metadata}, weights_path)
        return weights_path
    
    
    def _pull_updated_weights(self, model):
        chunks = self.client.get_updated_weights(self.worker_id)
        weights = self._chunks_to_state_dict(chunks)
        model.load_state_dict(weights)

    def _chunks_to_state_dict(self, chunks):
        buf = io.BytesIO()
        for chunk in chunks:
            buf.write(chunk.data)
        buf.seek(0)
        return torch.load(buf, map_location=self.device, weights_only=True)

    def _block_until_resumed(self):
        print(f"[Worker {self.worker_id}] Pausado, esperando reanudación...")
        while True:
            time.sleep(self.cfg.heartbeat_interval)
            response = self.client.check_rebalance(self.worker_id)
            if not response.pause:
                print(f"[Worker {self.worker_id}] Reanudado.")
                return
