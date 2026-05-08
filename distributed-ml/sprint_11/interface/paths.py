from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

BUILD_SH      = ROOT / "build.sh"
RUN_SH        = ROOT / "run.sh"
MONITORING_SH = ROOT / "monitoring.sh"
DEPLOYER_PY   = ROOT / "deployer.py"

CONFIG_YAML   = ROOT / "interface" / "config.yaml"
REGISTRY_JSON = ROOT / "shared" / "registry.json"  # nuevo, solo de la UI

CONFIG_LOADER   = ROOT / "shared" / "config_loader.yaml"

SHARED_DIR    = ROOT / "shared"
DEPLOYER_DIR  = ROOT / "deployer"