import threading
import time
import os
import yaml
from config_loader import TrainingConfig
from k8s_orchestrator import KubernetesOrchestrator
from dataset_provisioner import DatasetProvisioner
from master_launcher import MasterLauncher
from templates import get_master_rbac_templates

REMOTE_KUBECONFIG = "./k3s-remote.yaml"
LOCAL_KUBECONFIG = os.path.expanduser("~/.kube/config")
def _create_local_configmap(k8s, config_path):
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "training-config"
        },
        "data": {
            "config.yaml": open(config_path).read()
        }
    }
    k8s.apply_or_update_configmap(configmap)
    print("ConfigMap creado/actualizado")

def _create_master_serviceaccount(k8s):
    sa, role, binding = get_master_rbac_templates()
    k8s.apply(sa)
    k8s.apply(role)
    k8s.apply(binding)
    print("ServiceAccount master-sa creado")
def deploy():
    cfg = TrainingConfig()
    local_k8s = KubernetesOrchestrator()
    _create_master_serviceaccount(local_k8s)
    config_path = os.getenv("CONFIG_PATH")
    # remote_k8s = KubernetesOrchestrator.build_remote(REMOTE_KUBECONFIG)

    # local_k8s.ensure_remote_configmap(remote_k8s._core)
    _create_local_configmap(local_k8s, config_path)
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