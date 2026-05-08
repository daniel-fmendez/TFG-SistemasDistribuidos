export PYTHONPATH="$(pwd)/master:$(pwd)/worker:$(pwd)/shared:$(pwd)/deployer:$(pwd)/dataset:$(pwd)"
export CONFIG_PATH="$(pwd)/shared/config.yaml"
python deployer.py