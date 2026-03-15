from templates import get_master_job_template, get_master_service_template
class MasterLauncher:
    def __init__(self, config, orchestrator):
        self.cfg = config
        self.k8s = orchestrator

    def launch(self):
        self._create_job()
        self._create_service()

    def _create_job(self):
        manifest = get_master_job_template(
            job_name="master",
            image="my-master:v6",
            pvc_name=self.cfg.pvc_name
        )
        self.k8s.apply(manifest)

    def _create_service(self):
        manifest = get_master_service_template(service_name="master-service")
        self.k8s.apply(manifest)