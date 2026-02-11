
def get_pvc_template(name, size_gi, storage_class="standard"):
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": name,
        },
        "spec": {
            "accessModes": ["ReadWriteMany"],
            "storageClassName": storage_class,
            "resources": {
                "requests": {
                    "storage": f"{size_gi}Gi"
                }
            }
        }
    }

def get_dataset_init_job_template(job_name, pvc_name, image):
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
        },
        "spec": {
            "backoffLimit": 2,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "dataset-loader",
                            "image": image,
                            "volumeMounts": [
                                {
                                    "name": "dataset-vol",
                                    "mountPath": "/data"
                                }
                            ]
                        }
                    ],
                    "volumes": [
                        {
                            "name": "dataset-vol",
                            "persistentVolumeClaim": {
                                "claimName": pvc_name
                            }
                        }
                    ]
                }
            }
        }
    }

def get_worker_job_template(worker_id, master_host, master_port, pvc_name):
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"worker-{worker_id}",
        },
        "spec": {
            "backoffLimit": 2,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "distributed-training",
                        "role": "worker",
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "worker",
                            "image": "my-worker:v5",
                            "env": [
                                {
                                    "name": "MASTER_HOST",
                                    "value": master_host
                                },
                                {
                                    "name": "MASTER_PORT",
                                    "value": str(master_port)
                                },
                                {
                                    "name": "PYTHONUNBUFFERED",
                                    "value": "1"
                                }
                            ],
                            "volumeMounts": [
                                {
                                    "name": "dataset-storage",
                                    "mountPath": "/data"
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "2Gi",
                                    "cpu": "1000m"
                                },
                                "limits": {
                                    "memory": "4Gi",
                                    "cpu": "2000m"
                                }
                            }
                        }
                    ],
                    "volumes": [
                        {
                            "name": "dataset-storage",
                            "persistentVolumeClaim": {
                                "claimName": pvc_name
                            }
                        }
                    ]
                }
            }
        }
    }

def get_master_job_template(job_name, image, pvc_name):
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "labels": {
                "app": "master"
            }
        },
        "spec": {
            "template": {
                "metadata": {
                    "labels": {
                        "app": "master"
                    }
                },
                "spec": {
                    "serviceAccountName": "master-sa",
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "master",
                            "image": image,
                            "imagePullPolicy": "Never",
                            "ports": [
                                {
                                    "containerPort": 50051
                                }
                            ],
                            "env": [
                                {
                                    "name": "PYTHONUNBUFFERED",
                                    "value": "1"
                                }
                            ],
                            "volumeMounts": [
                                {
                                    "name": "master-storage",
                                    "mountPath": "/data"
                                }
                            ]
                        }
                    ],
                    "volumes": [
                        {
                            "name": "master-storage",
                            "persistentVolumeClaim": {
                                "claimName": pvc_name
                            }
                        }
                    ],
                }
            }
        }
    }


def get_master_service_template(service_name):
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": service_name
        },
        "spec": {
            "selector": {
                "app": "master"
            },
            "ports": [
                {
                    "port": 50051,
                    "targetPort": 50051
                }
            ]
        }
    }