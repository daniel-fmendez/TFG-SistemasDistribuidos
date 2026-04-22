
def get_pvc_template(name, size_gi, storage_class="local-path"):
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
            "ttlSecondsAfterFinished": 30,
            "backoffLimit": 2,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "imagePullPolicy": "Never",
                    "nodeSelector": {
                        "role": "local" 
                    },
                    "containers": [
                        {
                            "name": "dataset-loader",
                            "image": image,
                            "volumeMounts": [
                                {
                                    "name": "dataset-vol",
                                    "mountPath": "/data"
                                },
                                {
                                    "name": "config-volume",
                                    "mountPath": "/app/config"
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
                        },
                        {
                            "name": "config-volume",
                            "configMap": {
                                "name": "training-config"
                            }
                        }
                    ]
                }
            }
        }
    }

def get_worker_job_template(worker_id, master_host, master_port, pvc_name, worker_type="local"):
    
    # DNS config solo para workers LAN y remotos
    if worker_type in ("lan", "remote"):
        dns_policy = "None"
        dns_config = {
            "nameservers": ["8.8.8.8", "1.1.1.1"],
            "searches": ["default.svc.cluster.local", "svc.cluster.local"],
            "options": [{"name": "ndots", "value": "5"}]
        }
    else:
        dns_policy = "ClusterFirst"
        dns_config = {}

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"worker-{worker_id}",
        },
        "spec": {
            "ttlSecondsAfterFinished": 30,
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
                    "nodeSelector": {
                        "role": worker_type
                    },
                    "dnsPolicy": dns_policy,
                    "dnsConfig": dns_config,
                    "containers": [
                        {
                            "name": "worker",
                            "image": "my-worker:v6",
                            "imagePullPolicy": "Never",
                            "env": [
                                {"name": "MASTER_HOST", "value": master_host},
                                {"name": "MASTER_PORT", "value": str(master_port)},
                                {"name": "PYTHONUNBUFFERED", "value": "1"}
                            ],
                            "volumeMounts": [
                                {"name": "dataset-storage", "mountPath": "/data"},
                                {"name": "config-volume", "mountPath": "/app/config"}
                            ],
                            "resources": {
                                "requests": {"memory": "2Gi", "cpu": "1000m"},
                                "limits": {"memory": "4Gi", "cpu": "2000m"}
                            }
                        }
                    ],
                    "volumes": [
                        {
                            "name": "dataset-storage",
                            "persistentVolumeClaim": {"claimName": pvc_name}
                        },
                        {
                            "name": "config-volume",
                            "configMap": {"name": "training-config"}
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
            "ttlSecondsAfterFinished": 30,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "master"
                    }
                },
                "spec": {
                    "serviceAccountName": "master-sa",
                    "restartPolicy": "Never",
                    "nodeSelector": {
                        "role": "local" 
                    },
                    "containers": [
                        {
                            "name": "master",
                            "image": image,
                            "imagePullPolicy": "Never",
                            "ports": [
                                {   
                                    "name": "grpc",
                                    "containerPort": 50051
                                },
                                {   
                                    "name": "metrics",
                                    "containerPort": 8000
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
                                },
                                {
                                    "name": "config-volume",
                                    "mountPath": "/app/config"
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
                        },
                        {
                            "name": "config-volume",
                            "configMap": {
                                "name": "training-config"
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
            "type": "NodePort",
            "selector": {
                "app": "master"
            },
            "ports": [
                {
                    "name": "grpc",
                    "port": 50051,
                    "targetPort": 50051,
                    "nodePort": 30051 
                },
                {
                    "name": "metrics",
                    "port": 8000,
                    "targetPort": 8000
                }
            ]
        }
    }

def get_master_rbac_templates():
    sa = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {
            "name": "master-sa",
            "namespace": "default"
        }
    }
    role = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {
            "name": "master-role",
            "namespace": "default"
        },
        "rules": [
            {
                "apiGroups": [""],
                "resources": ["persistentvolumeclaims"],
                "verbs": ["get", "list", "create", "delete", "watch"]
            },
            {
                "apiGroups": [""],
                "resources": ["configmaps"],
                "verbs": ["get", "list", "watch"]
            },
            {
                "apiGroups": ["batch"],
                "resources": ["jobs", "jobs/status"],
                "verbs": ["get", "list", "create", "delete", "watch"]
            },
            {
                "apiGroups": [""],
                "resources": ["pods", "pods/status", "pods/log"],
                "verbs": ["get", "list", "create", "delete", "watch"]
            },
            {
                "apiGroups": [""],
                "resources": ["services"],
                "verbs": ["get", "list", "create", "delete"]
            }
        ]
    }
    role_binding = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {
            "name": "master-rolebinding",
            "namespace": "default"
        },
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "Role",
            "name": "master-role"
        },
        "subjects": [
            {
                "kind": "ServiceAccount",
                "name": "master-sa",
                "namespace": "default"
            }
        ]
    }
    return sa, role, role_binding

def get_nfs_pv_template(pv_name, nfs_server, nfs_path, size_gi=15):
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolume",
        "metadata": {"name": pv_name},
        "spec": {
            "capacity": {"storage": f"{size_gi}Gi"},
            "accessModes": ["ReadWriteMany"],
            "persistentVolumeReclaimPolicy": "Retain",
            "mountOptions": [
                "soft",        # no bloquear indefinidamente
                "timeo=30",    # timeout 3 segundos (unidades de 0.1s)
                "retrans=3",   # 3 reintentos
                "nolock"       # evitar bloqueos NFS
            ],
            "nfs": {
                "server": nfs_server,
                "path": nfs_path
            }
        }
    }

def get_nfs_pvc_template(pvc_name, pv_name, size_gi=15):
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": pvc_name,
            "namespace": "default"
        },
        "spec": {
            "accessModes": ["ReadWriteMany"],
            "resources": {
                "requests": {
                    "storage": f"{size_gi}Gi"
                }
            },
            "volumeName": pv_name,
            "storageClassName": ""
        }
    }