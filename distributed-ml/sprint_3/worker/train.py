from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
import torch

def load_data():
    dataset = load_dataset("imdb", split="train[:1000]")

    print(f"Dataset cargado: {len(dataset)} ejemplos")
    return dataset

def train_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = load_data()

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    print(" Configuración:")
    print(f"  - Ejemplos: {len(tokenized)}")
    print(f"  - Device: {torch.cuda.is_available() and 'CUDA' or 'CPU'}")
    print(f"  - Modelo: {model.config.model_type}")
    
    return model, tokenized

if __name__ == "__main__":
    train_model()