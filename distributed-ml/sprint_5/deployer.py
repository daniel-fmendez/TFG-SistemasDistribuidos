# Se encarga de crear el PVC con el dataset e invocar el master
#   - Reservar espacio de PVC
#   - Invocar master

import time
import yaml

from datasets import load_dataset_builder
from templates import get_pvc_template, get_master_job_template, get_master_service_template, get_dataset_init_job_template
from kubernetes import client, config
from kubernetes.utils import create_from_dict
from kubernetes.client.rest import ApiException

DATASET_REPO = "ag_news"
PVC_NAME = "server-pvc"

def reserve_space():
    # Lanza job de creacion de pv
    builder = load_dataset_builder(DATASET_REPO)

    size_in_bytes = builder.info.splits['train'].num_bytes
    size_in_gb = size_in_bytes / (1024**3)
    # Damos un margen
    size_in_gb = size_in_gb * 1.5
    pvc = get_pvc_template(
        name=PVC_NAME,
        size_gi= size_in_gb
    )

    with open(f"{PVC_NAME}.yaml", "w") as f:
        yaml.dump(pvc, f, default_flow_style=False)
    
    apply_manifest(PVC_NAME)
    
    launch_dataset_init_job(PVC_NAME)
    wait_job_completion("dataset-init")

def apply_manifest(file):
    try:
        config.load_incluster_config()
        k8s_client = client.ApiClient()

        # Leer el YAML
        with open(f"{file}.yaml", "r") as f:
            docs = list(yaml.safe_load_all(f))

        for doc in docs:
            if not doc:
                continue

            # Crea o actualiza el recurso
            create_from_dict(k8s_client, data=doc, verbose=True)

        print("Manifiesto aplicado correctamente")

    except FileNotFoundError:
        print("No se encontró el archivo YAML.")
    except ApiException as e:
        print("Error de la API de Kubernetes:", e)
    except Exception as e:
        print("Error al aplicar el manifiesto:", str(e))

def launch_dataset_init_job(pvc_name):
    job_manifest = get_dataset_init_job_template(
        job_name="dataset-init",
        pvc_name=pvc_name,
        image="data-provision:v1"
    )

    with open("dataset-init.yaml", "w") as f:
        yaml.dump(job_manifest, f)

    apply_manifest("dataset-init")

def wait_job_completion(job_name):
    config.load_incluster_config()
    batch_v1 = client.BatchV1Api()

    while True:
        job = batch_v1.read_namespaced_job_status(job_name, "default")
        status = job.status

        if status.succeeded:
            print("Dataset provisionado")
            return

        if status.failed:
            raise RuntimeError("Fallo al provisionar dataset")

        time.sleep(5)
def create_master():
    master_template = get_master_job_template(
        job_name="master",
        image="my-master:v5",
        pvc_name=PVC_NAME
    )
    master_name = "master-job"
    with open(f"{master_name}.yaml", "w") as f:
        yaml.dump(master_template, f)

    apply_manifest(master_name)

    service_template = get_master_service_template(
        service_name="master-service"
    )
    
    service_name = "master-service"
    with open(f"{service_name}.yaml", "w") as f:
        yaml.dump(service_template, f)

    apply_manifest(service_name)

if __name__ == "__main__":
    # Primero reserva espacio
    print(f"Creando PVC {PVC_NAME} con el dataset")
    reserve_space()

    # Despues crea el master
    print(f"Creando el master...")
    create_master()