
from k8s_orchestrator import KubernetesOrchestrator
from config_loader import TrainingConfig
from dataset_factory import DatasetFactory
from model_factory import ModelFactory
from templates import get_pvc_template, get_dataset_init_job_template, \
                     get_nfs_pv_template, get_nfs_pvc_template

class DatasetProvisioner:
    def __init__(self, config, orchestrator):
        self.cfg = config
        self.orchestrator = orchestrator

    def provision(self):
        self._create_pvc()
        self._launch_init_job()
        self.orchestrator.wait_job_completion("dataset-init")

    def _create_pvc(self):
        if self.cfg.nfs_server:
            pv = get_nfs_pv_template(
                pv_name=self.cfg.nfs_pv_name,
                nfs_server=self.cfg.nfs_server,
                nfs_path=self.cfg.nfs_path
            )
            pvc = get_nfs_pvc_template(
                pvc_name=self.cfg.pvc_name,
                pv_name=self.cfg.nfs_pv_name
            )
            self.orchestrator.apply(pv)
            self.orchestrator.apply(pvc)
            print("PV y PVC NFS creados")
        else:
            size_in_gb = DatasetFactory.calculate_storage_size(self.cfg.dataset_name)
            pvc = get_pvc_template(name=self.cfg.pvc_name, size_gi=size_in_gb)
            self.orchestrator.apply(pvc)
            print("PVC local creado")

    def _launch_init_job(self):
        job_manifest = get_dataset_init_job_template(
            job_name="dataset-init",
            pvc_name=self.cfg.pvc_name,
            image="data-provision:v2"
        )
        self.orchestrator.apply(job_manifest)