import threading
import time


# TODO adapt to heartbeat
# Maneja workers, metricas y shards
class WorkerRegistry:
    def __init__(self, config, on_all_finished=None):
        self.cfg = config
        self._on_all_finished = on_all_finished
        # Diccionario de workers vivos
        self.registered_workers = {}
        self.num_workers = self.cfg.num_workers
        self.metrics = {}
        self.shards = []
        self.finished_workers = set()
        self._lock = threading.Lock()
        self._all_ready = threading.Condition(self._lock)

        self._prepare_training()

    def register_worker(self, worker_id):
        with self._all_ready:
            if worker_id not in self.registered_workers:
                index = len(self.registered_workers)
                self.registered_workers[worker_id] = time.time()
                print(f"Worker {worker_id} registrado con índice {index}.")
            else:
                index = list(self.registered_workers.keys()).index(worker_id)
            
            while len(self.registered_workers) < self.num_workers:
                self._all_ready.wait()

            self._all_ready.notify_all()
            return index

    def get_shard(self, index):
        return self.shards[index]


    # Preparalos splits del modelo para cada worker
    def _prepare_training(self):
        # Se crea al inicio
        shard_len = self.cfg.total_samples // self.num_workers
        for i in range(self.num_workers):
            start = i * shard_len
            end = start + shard_len
            shard = {
                "start": start,
                "end": end
            }
            self.shards.append(shard)

        print("Shards preparados")

    def save_metrics(self, data):
        worker_id = data.get('worker_id')
        if worker_id not in self.metrics:
            self.metrics[worker_id] = []

        self.metrics[worker_id].append(data)
        print(f"Métricas guardadas para el worker {worker_id}")
    
    def mark_finished(self, worker_id):
        with self._lock:
            self.finished_workers.add(worker_id)
            print(f"Worker {worker_id} finalizado. ({len(self.finished_workers)}/{self.num_workers})")
            all_done = len(self.finished_workers) == self.num_workers

        if all_done and self._on_all_finished:
            self._on_all_finished()

    def get_metrics(self):
        with self._lock:
            return dict(self.metrics)