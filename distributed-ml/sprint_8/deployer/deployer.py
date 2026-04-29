import threading
import time
from config_loader import TrainingConfig
from k8s_orchestrator import KubernetesOrchestrator
from dataset_provisioner import DatasetProvisioner
from master_launcher import MasterLauncher

REMOTE_KUBECONFIG = "/app/config/k3s-remote.yaml"

def deploy():
    cfg = TrainingConfig()
    local_k8s = KubernetesOrchestrator()
    # remote_k8s = KubernetesOrchestrator.build_remote(REMOTE_KUBECONFIG)

    # local_k8s.ensure_remote_configmap(remote_k8s._core

    local_provisioner = DatasetProvisioner(cfg, local_k8s)
    # remote_provisioner = DatasetProvisioner(cfg, remote_k8s)
    
    errors = []
    def run(provisioner, label):
        try:
            provisioner.provision()
        except Exception as e:
            errors.append(f"{label}: {e}")

    t_local = threading.Thread(target=run, args=(local_provisioner, "local"))
    # t_remote = threading.Thread(target=run, args=(remote_provisioner, "remoto"))

    print("Provisionando dataset en local y remoto en paralelo...")
    t_local.start()
    # t_remote.start()
    t_local.join()
    # t_remote.join()

    if errors:
        raise RuntimeError(f"Falló el provisionamiento: {errors}")
    
    print("Dataset listo en ambos clústeres.")

    launcher = MasterLauncher(cfg, local_k8s)
    print("Creando master...")
    launcher.launch()

    print("Creando workers locales...")
    time.sleep(10)
    local_k8s.create_workers(cfg)

    print("Creando workers remotos...")
    # remote_k8s.create_workers(cfg, worker_type="remote")


if __name__ == "__main__":
    deploy()