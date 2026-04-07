import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.utils import create_from_dict
from templates import get_worker_job_template

class KubernetesOrchestrator:
    def __init__(self):
        config.load_incluster_config()
        self._client = client.ApiClient()
        self._batch = client.BatchV1Api()
        self._core = client.CoreV1Api() 

    def apply(self, manifest):
        try:
            create_from_dict(self._client, data=manifest, verbose=True)
            print("Manifiesto aplicado correctamente")
        except ApiException as e:
            if e.status == 409:
                print(f"Recurso ya existe, ignorando: {manifest['metadata']['name']}")
            else:
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

    def ensure_remote_configmap(self, k3s_core_client):
        local_cm = self._core.read_namespaced_config_map(
            name="training-config",
            namespace="default"
        )

        remote_cm = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(name="training-config"),
            data=local_cm.data
        )

        try:
            k3s_core_client.create_namespaced_config_map(
                namespace="default",
                body=remote_cm
            )
            print("ConfigMap creado en k3s")
        except client.exceptions.ApiException as e:
            if e.status == 409:  # ya existe, actualizar
                k3s_core_client.replace_namespaced_config_map(
                    name="training-config",
                    namespace="default",
                    body=remote_cm
                )
                print("ConfigMap actualizado en k3s")

    @staticmethod
    def build_remote(kubeconfig_path):
        orchestrator = KubernetesOrchestrator.__new__(KubernetesOrchestrator)
        config.load_kube_config(config_file=kubeconfig_path)
        orchestrator._client = client.ApiClient()
        orchestrator._batch = client.BatchV1Api()
        orchestrator._core = client.CoreV1Api() 
        return orchestrator