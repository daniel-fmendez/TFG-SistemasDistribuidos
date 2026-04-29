from prometheus_client import Gauge, Counter, start_http_server
import time
from datetime import datetime

class MetricsCollector:
    def __init__(self):
        self.worker_loss = Gauge('worker_loss', 'Loss por worker', ['worker_id'])
        self.worker_accuracy = Gauge('worker_accuracy', 'Accuracy por worker', ['worker_id'])
        self.workers_registered = Gauge('workers_registered', 'Workers registrados')
        self.workers_finished = Gauge('workers_finished', 'Workers que han terminado')
        self.aggregations_total = Counter('aggregations_total', 'Total de agregaciones realizadas')
        self.memory_usage = Gauge('worker_memory_usage', 'Memory usage percentage', ['worker_id'])
        self.memory_mb = Gauge('worker_memory_usage_mb', 'Memory usage in MB', ['worker_id'])
        self.cpu_usage = Gauge('worker_cpu_usage', 'CPU usage percentage', ['worker_id'])
        self.total_epoch = Gauge('total_epochs', "Total de epochs a calcular")
        self.total_val = 0
        self.worker_progress = Gauge('worker_progress_ratio', 'Progreso 0-1 del worker', ['worker_id'])
        self.epoch = Gauge('epoch', 'Epoch actual')
        self.last_epoch = {}
        # Sprint 9 nuevas metricas
        self.network_bytes_sent = Counter('network_bytes_sent', 'Bytes enviados por worker', ['worker_id'])
        self.epoch_duration_seconds = Gauge('epoch_duration_seconds', 'Duración de la última epoch en segundos', ['worker_id', 'epoch'])
        self.sync_duration_seconds = Gauge('sync_duration_seconds', 'Duración del último ciclo de sincronización (push+pull) en segundos', ['worker_id'])
        self.throughput_samples_per_second = Gauge('throughput_samples_per_second', 'Muestras procesadas por segundo en la última epoch', ['worker_id'])

        self._epoch_start_times = {}

    def start(self, port=8000):
        start_http_server(port)
        print(f"Prometheus metrics en puerto {port}")

    def record_metric(self, worker_id, loss, accuracy):
        self.worker_loss.labels(worker_id=worker_id).set(loss)
        self.worker_accuracy.labels(worker_id=worker_id).set(accuracy)

    def record_stats(self, worker_id, memory_usage, memory_mb, cpu_usage):
        self.memory_usage.labels(worker_id=worker_id).set(memory_usage)
        self.memory_mb.labels(worker_id=worker_id).set(memory_mb)
        self.cpu_usage.labels(worker_id=worker_id).set(cpu_usage)
        
    def record_aggregation(self):
        self.aggregations_total.inc()

    def set_workers_registered(self, n):
        self.workers_registered.set(n)

    def set_workers_finished(self, n):
        self.workers_finished.set(n)

    def set_total_epoch(self, n):
        self.total_epoch.set(n)
        self.total_val = n

    def record_epoch(self, worker_id, epoch):
        total = self.total_val
        if worker_id not in self.last_epoch or epoch > self.last_epoch[worker_id]:
            self.last_epoch[worker_id] = epoch
            self.epoch.set(epoch)
            progreso = epoch / total if total > 0 else 0
            self.worker_progress.labels(worker_id=worker_id).set(progreso)

    def mark_worker_dead(self, worker_id):
        self.worker_loss.labels(worker_id=worker_id).set(0)
        self.worker_accuracy.labels(worker_id=worker_id).set(0)
        self.cpu_usage.labels(worker_id=worker_id).set(0)
        self.memory_usage.labels(worker_id=worker_id).set(0)
        self.memory_mb.labels(worker_id=worker_id).set(0)

    def update_completion(self, finished, dead, total):
        self.workers_finished.set(finished + dead)
        self.workers_registered.set(total - finished - dead)

    
    def record_epoch_start(self, worker_id, timestamp):
        self._epoch_start_times[worker_id] = timestamp

    def record_epoch_end(self, worker_id, timestamp, epoch, samples_in_epoch):
        start = self._epoch_start_times.get(worker_id)
        end = timestamp
        if start is None:
            return
        duration = end - start
        self.epoch_duration_seconds.labels(worker_id=worker_id, epoch=str(epoch)).set(duration)
        if duration > 0:
            self.throughput_samples_per_second.labels(worker_id=worker_id).set(
                samples_in_epoch / duration
            )

    def record_sync_duration(self, worker_id, duration_seconds):
        self.sync_duration_seconds.labels(worker_id=worker_id).set(duration_seconds)

    def record_bytes_sent(self, worker_id, bytes_sent):
        self.network_bytes_sent.labels(worker_id=worker_id).inc(bytes_sent)
