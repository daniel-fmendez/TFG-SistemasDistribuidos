from dataset_factory import DatasetFactory
from config_loader import TrainingConfig

def prepare_dataset():
    cfg = TrainingConfig()
    
    DatasetFactory.prepare_and_save(
        dataset_name=cfg.dataset_name,
        output_dir="/data/train",
        max_length=512
    )

if __name__ == "__main__":
    prepare_dataset()