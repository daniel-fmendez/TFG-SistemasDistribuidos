import os

from datasets import load_dataset, load_dataset_builder, load_from_disk
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image

class DatasetFactory:
    
    SUPPORTED = {
        "ag_news": {
            "type": "text_classification",
            "num_labels": 4,
            "hf_name": "ag_news",
            "text_column": "text",
            "label_column": "label",
            "tokenizer": "distilbert-base-uncased"
        },
        "imdb": {
            "type": "text_classification", 
            "num_labels": 2,
            "hf_name": "imdb",
            "text_column": "text",
            "label_column": "label",
            "tokenizer": "distilbert-base-uncased"
        },
        "sst2": {
            "type": "text_classification",
            "num_labels": 2,
            "hf_name": "glue",
            "subset": "sst2",
            "text_column": "sentence",
            "label_column": "label",
            "tokenizer": "bert-base-uncased"
        },
        "cifar10": {
            "type": "image_classification",
            "num_labels": 10,
            "hf_name": "cifar10",
            "label_column": "label",
        }
    }
    
    @staticmethod
    def get_info(dataset_name):
        if dataset_name not in DatasetFactory.SUPPORTED:
            raise ValueError(f"Dataset no soportado: {dataset_name}")
        return DatasetFactory.SUPPORTED[dataset_name]
    
    @staticmethod
    def calculate_storage_size(dataset_name):
        info = DatasetFactory.get_info(dataset_name)
        
        # Cargar builder para obtener tamaño
        if "subset" in info:
            builder = load_dataset_builder(info["hf_name"], info["subset"])
        else:
            builder = load_dataset_builder(info["hf_name"])
        
        size_in_bytes = builder.info.splits['train'].num_bytes
        size_in_gb = size_in_bytes / (1024**3)
        size_in_gb = (size_in_gb * 2.5) + 1.5
        
        print(f"Dataset {dataset_name}: {size_in_gb:.2f} GB necesarios")
        return size_in_gb
    
    @staticmethod
    def prepare_and_save(dataset_name, output_dir="/data/train", max_length=512):
        info = DatasetFactory.get_info(dataset_name)

        print(f"Descargando dataset {dataset_name}...")
        if "subset" in info:
            dataset = load_dataset(info["hf_name"], info["subset"], split="train")
        else:
            dataset = load_dataset(info["hf_name"], split="train")

        print(f"Dataset cargado: {len(dataset)} muestras")

        if info["type"] == "text_classification":
            tokenizer = AutoTokenizer.from_pretrained(info["tokenizer"])
            def tokenize_function(examples):
                return tokenizer(
                    examples[info["text_column"]],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
            dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=[info["text_column"]],
                desc="Tokenizando"
            )

        elif info["type"] == "image_classification":
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                )
            ])

            def preprocess_images(examples):
                examples["pixel_values"] = [
                    transform(img.convert("RGB")).numpy()
                    for img in examples["img"]
                ]
                return examples

            dataset = dataset.map(
                preprocess_images,
                batched=True,
                remove_columns=["img"],
                desc="Procesando imágenes"
            )

        os.makedirs(os.path.dirname(output_dir) if os.path.dirname(output_dir) else ".", exist_ok=True)
        dataset.save_to_disk(output_dir)
        print(f"Dataset guardado en {output_dir}")
        return dataset
    
    @staticmethod
    def load(output_dir="/data/train"):
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Dataset no encontrado en {output_dir}")
        
        print(f"Cargando dataset desde {output_dir}")
        return load_from_disk(output_dir)
    
    @staticmethod
    def load_shard(path, start, end, dataset_info):
        dataset = DatasetFactory.load(path)
        dataset = dataset.select(range(start, end))

        if dataset_info["type"] == "text_classification":
            dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        elif dataset_info["type"] == "image_classification":
            dataset.set_format(type="torch", columns=["pixel_values", "label"])

        return dataset