import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.utils import create_from_dict
from templates import get_worker_job_template

class KubernetesOrchestrator:
    def __init__(self) -> None:
        config.load_incluster_config()
        self._client = client.ApiClient()
        self._batch = client.BatchV1Api()

    def apply(self, manifest):
        try:
            create_from_dict(self._client, data=manifest, verbose=True)
            print("Manifiesto aplicado correctamente")
        except ApiException as e:
            print(f"Error de la API de Kubernetes: {e}")
        except Exception as e:
            print(f"Error al aplicar manifiesto: {e}")

    def create_workers(self, cfg):
        for i in range(cfg.num_workers):
            manifest = get_worker_job_template(
                worker_id=i,
                master_host=cfg.master_host,
                master_port=cfg.master_port,
                pvc_name=cfg.pvc_name
            )
            self.apply(manifest)

    def wait_job_completion(self, job_name, namespace="default", interval=5):
        while True:
            job = self._batch.read_namespaced_job_status(job_name, namespace)
            if job.status.succeeded:
                print(f"Job {job_name} completado")
                return
            if job.status.failed:
                raise RuntimeError(f"Job {job_name} falló")
            time.sleep(interval)