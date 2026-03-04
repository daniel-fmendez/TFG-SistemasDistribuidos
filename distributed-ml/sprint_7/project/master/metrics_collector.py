from prometheus_client import Gauge, Counter, start_http_server

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
        self.epoch = Gauge('epoch', 'Epoch actual')
        self.last_epoch = 0

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

    def record_epoch(self, n):
        if n > self.last_epoch:
            self.last_epoch = n
            self.epoch.set(n)