import threading
import time
import os
import yaml
import subprocess
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
def _cleanup(k8s):
    """Limpia recursos anteriores antes de desplegar"""
    
    resources = [
        ("job", "dataset-init"),
        ("job", "master"),
        ("service", "master-service"),
    ]
    
    # Workers dinámicos
    result = subprocess.run(
        ["kubectl", "get", "jobs", "-o", "name", "-n", "default"],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if "worker-" in line:
            name = line.split("/")[-1]
            resources.append(("job", name))
    
    for kind, name in resources:
        subprocess.run(
            ["kubectl", "delete", kind, name, 
             "--ignore-not-found", "--grace-period=0", "--force"],
            capture_output=True
        )
    
    # Esperar a que los pods desaparezcan
    print("Esperando limpieza de pods...")
    for _ in range(30):
        result = subprocess.run(
            ["kubectl", "get", "pods", "--no-headers"],
            capture_output=True, text=True
        )
        if not result.stdout.strip():
            break
        time.sleep(2)
    print("Limpieza completa")
    
def deploy():
    cfg = TrainingConfig()
    local_k8s = KubernetesOrchestrator()
    remote_k8s = KubernetesOrchestrator.build_remote(REMOTE_KUBECONFIG)

    _cleanup(local_k8s)
    _create_master_serviceaccount(local_k8s)

    config_path = os.getenv("CONFIG_PATH")
    _create_local_configmap(local_k8s, config_path)
    local_k8s.ensure_remote_configmap(remote_k8s._core)

    local_provisioner = DatasetProvisioner(
        cfg, local_k8s,
        pvc_name=cfg.pvc_name,
        node_role="local"
    )
    lan_provisioner = DatasetProvisioner(
        cfg, local_k8s,
        pvc_name=cfg.lan_pvc_name,
        node_role="lan"
    )
    remote_provisioner = DatasetProvisioner(
        cfg, remote_k8s,
        pvc_name=cfg.remote_pvc_name, 
        node_role="remote"
    )
    
    errors = []
    def run(provisioner, label):
        try:
            provisioner.provision()
        except Exception as e:
            errors.append(f"{label}: {e}")

    t_local = threading.Thread(target=run, args=(local_provisioner, "local"))
    t_lan = threading.Thread(target=run, args=(lan_provisioner, "lan"))
    # t_remote = threading.Thread(target=run, args=(remote_provisioner, "remoto"))

    print("Provisionando dataset en local y remoto en paralelo...")
    t_local.start()
    t_lan.start()
    # t_remote.start()

    t_local.join()
    t_lan.join()
    # t_remote.join()

    if errors:
        raise RuntimeError(f"Falló el provisionamiento: {errors}")
    
    print("Dataset listo en ambos clústeres.")

    launcher = MasterLauncher(cfg, local_k8s)
    print("Creando master...")
    launcher.launch()

    print("Creando workerss...")
    time.sleep(10)
    local_k8s.create_workers(cfg, remote_orchestrator=remote_k8s)


if __name__ == "__main__":
    deploy()