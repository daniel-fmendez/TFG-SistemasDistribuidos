from config_loader import TrainingConfig
from k8s_orchestrator import KubernetesOrchestrator
from dataset_provisioner import DatasetProvisioner
from master_launcher import MasterLauncher

def deploy():
    cfg = TrainingConfig()
    k8s = KubernetesOrchestrator()

    provisioner = DatasetProvisioner(cfg, k8s)
    launcher = MasterLauncher(cfg, k8s)

    print(f"Creando PVC {cfg.pvc_name} con el dataset")
    provisioner.provision()

    print("Creando el master...")
    launcher.launch()


if __name__ == "__main__":
    deploy()