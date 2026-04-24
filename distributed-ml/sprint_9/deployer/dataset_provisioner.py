
from k8s_orchestrator import KubernetesOrchestrator
from config_loader import TrainingConfig
from dataset_factory import DatasetFactory
from model_factory import ModelFactory
from templates import get_pvc_template, get_dataset_init_job_template

class DatasetProvisioner:
    def __init__(self, config, orchestrator, pvc_name, node_role):
        self.cfg = config
        self.orchestrator = orchestrator
        self.pvc_name = pvc_name
        self.node_role = node_role

    def provision(self):
        self._create_pvc()
        self._launch_init_job()
        self.orchestrator.wait_job_completion(f"dataset-init-{self.node_role}")

    def _create_pvc(self):
        size_in_gb = DatasetFactory.calculate_storage_size(self.cfg.dataset_name)
        storage_class = f"local-path-{self.node_role}" 
        pvc = get_pvc_template(
            name=self.pvc_name, 
            size_gi=size_in_gb,
            storage_class=storage_class
        )
        self.orchestrator.apply(pvc)
        print(f"PVC {self.pvc_name} creado ({size_in_gb:.1f}Gi) en {storage_class}")

    def _launch_init_job(self):
        job_manifest = get_dataset_init_job_template(
            job_name=f"dataset-init-{self.node_role}",
            pvc_name=self.pvc_name,
            image="data-provision:v2",
            node_role=self.node_role
        )
        self.orchestrator.apply(job_manifest)