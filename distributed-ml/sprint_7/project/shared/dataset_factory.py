import os

from datasets import load_dataset, load_dataset_builder, load_from_disk
from transformers import AutoTokenizer


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
        
        # Margen de seguridad: dataset crudo + tokenizado + pesos modelo
        # Dataset tokenizado suele ser ~2x el tamaño original
        # Pesos del modelo ~1GB
        # Total: 2.5x + 1GB de margen
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
        print(f"Columnas originales: {dataset.column_names}")
        
        if info["type"] == "text_classification":
            # Tokenizar
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
            
            # Verificar columna label
            if info["label_column"] in dataset.column_names:
                print(f"Columna '{info['label_column']}' verificada")
            else:
                raise ValueError(f"ERROR: No se encontró columna '{info['label_column']}'")
        
        print(f"Columnas finales: {dataset.column_names}")
        print(f"Ejemplo de muestra procesada:")
        print(dataset[0])
        
        # Guardar
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        dataset.save_to_disk(output_dir)
        print(f"Dataset guardado en {output_dir}")
        
        return dataset
    
    @staticmethod
    def load(output_dir="/data/train"):
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Dataset no encontrado en {output_dir}")
        
        print(f"Cargando dataset desde {output_dir}")
        return load_from_disk(output_dir)