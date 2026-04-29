import threading
import time

# Maneja workers, metricas y shards
class WorkerRegistry:
    def __init__(self, config, metrics_collector, on_all_finished=None, pause_event=None):
        self.cfg = config
        self.metrics_collector = metrics_collector
        self._on_all_finished = on_all_finished
        self.pause_event = pause_event
        # Diccionario de workers
        self.worker_last_seen = {}
        self.alive_workers = set()
        self.dead_workers = set()
        self.worker_index_map = {}
        self.min_workers = 2
        self.num_workers = self.cfg.num_workers
        self.metrics = {}
        self.shards = []
        self.finished_workers = set()
        self._lock = threading.Lock()
        self._all_ready = threading.Condition(self._lock)
        self._training_barrier = threading.Barrier(self.num_workers)

        self._epoch_barriers = {}
        self._epoch_events = {}
        self._epoch_expected = {}
        
        self.rebalanced = False 
        self._prepare_training()

    def register_worker(self, worker_id):
        with self._all_ready:
            if worker_id not in self.worker_index_map:
                self.worker_index_map[worker_id] = len(self.worker_index_map)
            
            if worker_id not in self.worker_last_seen:
                grace = self.cfg.heartbeat_interval * self.cfg.heartbeat_multiplier * 3
                self.worker_last_seen[worker_id] = time.time() + grace
                self.alive_workers.add(worker_id)
                registered_count = len(self.worker_last_seen)
                self.metrics_collector.set_workers_registered(registered_count)
                print(f"Worker {worker_id} registrado con índice {self.worker_index_map[worker_id]}.")
                self._all_ready.notify_all()

            if self.pause_event and self.pause_event.is_set():
                if len(self.alive_workers) >= self.min_workers:
                    self.pause_event.clear()
                    print(f"[Registry] Suficientes workers ({len(self.alive_workers)}), reanudando...")

            while len(self.worker_last_seen) < self.num_workers and not self.pause_event.is_set():
                self._all_ready.wait(timeout=5.0) 

            self._all_ready.notify_all()
            return self.worker_index_map[worker_id]

    def get_shard(self, index):
        with self._lock:
            if index < len(self.shards):
                return self.shards[index]
            return None
        
    def get_shard_for_worker(self, worker_id):
        with self._lock:
            for shard in self.shards:
                if shard.get("worker_id") == worker_id:
                    return shard
            return None
        
    # Prepara los splits del modelo para cada worker
    def _prepare_training(self):
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
        all_done = False
        with self._lock:
            self.finished_workers.add(worker_id)
            self.alive_workers.discard(worker_id)
            finished = len(self.finished_workers)
            dead = len(self.dead_workers)
            self.metrics_collector.update_completion(finished, dead, self.num_workers)
            print(f"Worker {worker_id} finalizado. ({finished}/{self.num_workers})")
            all_done = (finished + dead) >= self.num_workers

            if not self.alive_workers and not all_done:
                print("[Registry] No quedan workers vivos, finalizando...")
                all_done = True

            for epoch, barrier in self._epoch_barriers.items():
                if epoch in self._epoch_expected:
                    self._epoch_expected[epoch] = min(
                        self._epoch_expected[epoch],
                        len(self.alive_workers) + len(self.finished_workers)
                    )
                    if len(barrier) >= self._epoch_expected[epoch]:
                        print(f"[Registry] Worker finalizado, liberando barrera época {epoch}")
                        self._epoch_events[epoch].set()

        if all_done and self._on_all_finished:
            self._on_all_finished()

    def get_metrics(self):
        with self._lock:
            return dict(self.metrics)
    
    def update_alive(self, worker_id):
        now = time.time()
        with self._lock:
            if worker_id not in self.worker_last_seen:
                raise KeyError(f"Worker {worker_id} no está registrado") 
            self.worker_last_seen[worker_id] = now
            self.alive_workers.add(worker_id)
            self.dead_workers.discard(worker_id)

    def check_dead(self, now):
        newly_dead = []
        with self._lock:
            dead = [
                wid for wid, last_seen in self.worker_last_seen.items()
                if now - last_seen > (self.cfg.heartbeat_interval * self.cfg.heartbeat_multiplier)
                and wid not in self.finished_workers
            ]
            for wid in dead:
                if wid not in self.dead_workers:
                    print(f"[Master] Worker caído (timeout): {wid}")
                    self.dead_workers.add(wid)
                    self.alive_workers.discard(wid)
                    self.metrics_collector.mark_worker_dead(wid)
                    del self.worker_last_seen[wid]
                    newly_dead.append(wid)

                    for epoch, barrier in self._epoch_barriers.items():
                        if epoch in self._epoch_expected:
                            self._epoch_expected[epoch] = min(
                                self._epoch_expected[epoch],
                                len(self.alive_workers) + len(barrier)
                            )
                            if len(barrier) >= self._epoch_expected[epoch]:
                                print(f"[Registry] Worker muerto, liberando barrera época {epoch}")
                                self._epoch_events[epoch].set()

            finished = len(self.finished_workers)
            dead_count = len(self.dead_workers)
            self.metrics_collector.update_completion(finished, dead_count, self.num_workers)

        for wid in newly_dead:
            if len(self.alive_workers) >= self.min_workers:
                print(f"[Master] Workers vivos suficientes, rebalanceando...")
                self._rebalance(len(self.alive_workers))
            else:
                print(f"[Master] Workers insuficientes, pausando...")
                self._pause()

    def _rebalance(self, alive_count):
        self.shards = []
        shard_len = self.cfg.total_samples // alive_count
        alive_list = list(self.alive_workers)
        
        for i, worker_id in enumerate(alive_list):
            start = i * shard_len
            end = start + shard_len
            self.shards.append({
                "worker_id": worker_id,
                "start": start,
                "end": end
            })
            self.rebalanced = True  
        print(f"[Master] Shards rebalanceados para {alive_count} workers")
        if self.pause_event and self.pause_event.is_set():
            self.pause_event.clear()
            with self._all_ready:
                self._all_ready.notify_all()

    def _pause(self):
        print(f"[Master] Pausando, workers vivos: {len(self.alive_workers)}/{self.min_workers}")
        if self.pause_event:
            self.pause_event.set()
            with self._all_ready:
                self._all_ready.notify_all()
    def consume_rebalance(self):
        with self._lock:
            was_rebalanced = self.rebalanced
            self.rebalanced = False
            return was_rebalanced
        
    def get_worker_index(self, worker_id):
        with self._lock:
            if worker_id in self.worker_index_map:
                return self.worker_index_map[worker_id]
            raise KeyError(f"Worker {worker_id} no encontrado")
    
    def wait_all_ready_to_train(self):
        print(f"[Registry] Worker esperando barrera de inicio...")
        try:
            self._training_barrier.wait(timeout=300)
            print(f"[Registry] Barrera superada, todos listos")
        except threading.BrokenBarrierError:
            raise RuntimeError("Barrera rota, algún worker no llegó a tiempo")
        
        
    def wait_epoch_sync(self, worker_id, epoch):
        with self._lock:
            if epoch not in self._epoch_barriers:
                self._epoch_barriers[epoch] = set()
                self._epoch_events[epoch] = threading.Event()
                self._epoch_expected[epoch] = len(self.alive_workers)
                print(f"[Registry] Barrera época {epoch}: esperando {self._epoch_expected[epoch]} workers")
            
            self._epoch_barriers[epoch].add(worker_id)
            received = len(self._epoch_barriers[epoch])
            expected = self._epoch_expected[epoch]
            print(f"[Registry] Worker {worker_id} en barrera época {epoch} ({received}/{expected})")
            
            if received >= expected:
                self._epoch_events[epoch].set()
        
        reached = self._epoch_events[epoch].wait(timeout=300)
        if not reached:
            print(f"[Registry] TIMEOUT barrera época {epoch}, liberando de todas formas")
            self._epoch_events[epoch].set()
        print(f"[Registry] Worker {worker_id} barrera época {epoch} superada")