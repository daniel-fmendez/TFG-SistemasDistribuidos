import os
from datasets import load_dataset
from transformers import DistilBertTokenizer

DATASET_NAME = "ag_news"
OUTPUT_DIR = "/data/train"
MAX_LENGTH = 512

def prepare_dataset():
    dataset = load_dataset(DATASET_NAME, split="train")

    print(f"Dataset cargado: {len(dataset)} muestras")
    print(f"Columnas originales: {dataset.column_names}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],  # Eliminar columna de texto original
        desc="Tokenizando"
    )
    # tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    if "label" in tokenized_dataset.column_names:
        print("Columna 'label' ya existe")
    else:
        print("ERROR: No se encontró columna 'label'")
        return

    print(f"Columnas finales: {tokenized_dataset.column_names}")
    print(f"Ejemplo de muestra tokenizada:")
    print(tokenized_dataset[0])

    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    tokenized_dataset.save_to_disk(OUTPUT_DIR)

    print(f"Dataset guardado en {OUTPUT_DIR}")
if __name__ == "__main__":
    prepare_dataset()