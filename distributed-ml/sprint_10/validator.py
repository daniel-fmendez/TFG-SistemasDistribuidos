import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoTokenizer

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_path, "shared"))

from dataset_factory import DatasetFactory
from model_factory import ModelFactory



def validate(weights_path, dataset_name, model_type, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    dataset_info = DatasetFactory.get_info(dataset_name)
    num_labels = dataset_info["num_labels"]

    print(f"Cargando modelo {model_type} ({model_name})...")
    model = ModelFactory.build(model_type, model_name, num_labels).to(device)
    weights = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    print(f"Descargando split de test para {dataset_name}...")
    if dataset_info["type"] == "image_classification":
        if "subset" in dataset_info:
            raw = load_dataset(dataset_info["hf_name"], dataset_info["subset"], split="test")
        else:
            raw = load_dataset(dataset_info["hf_name"], split="test")
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
        def preprocess(examples):
            examples["pixel_values"] = [
                transform(img.convert("RGB")).numpy()
                for img in examples["img"]
            ]
            return examples

        dataset = raw.map(preprocess, batched=True, remove_columns=["img"])
        dataset.set_format(type="torch", columns=["pixel_values", "label"])
        def collate(batch):
            return {
                "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
                "label": torch.tensor([b["label"] for b in batch])
            }
    elif dataset_info["type"] == "text_classification":
        tokenizer = AutoTokenizer.from_pretrained(dataset_info["tokenizer"])

        split = "validation" if dataset_name == "sst2" else "test"
        if "subset" in dataset_info:
            raw = load_dataset(dataset_info["hf_name"], dataset_info["subset"], split=split)
        else:
            raw = load_dataset(dataset_info["hf_name"], split=split)

        def tokenize(examples):
            return tokenizer(
                examples[dataset_info["text_column"]],
                padding="max_length",
                truncation=True,
                max_length=512
            )

        dataset = raw.map(tokenize, batched=True, remove_columns=[dataset_info["text_column"]])
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        def collate(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                "label": torch.tensor([b["label"] for b in batch])
            }
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate)

    correct = 0
    total = 0
    total_loss = 0.0

    print("Validando...")
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)

            if dataset_info["type"] == "image_classification":
                outputs = model(batch["pixel_values"].to(device))
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = F.cross_entropy(logits, labels)

            elif dataset_info["type"] == "text_classification":
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=labels
                )
                logits = outputs.logits
                loss = outputs.loss

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
    accuracy = correct / total
    avg_loss = total_loss / len(loader)
    print(f"\n--- Resultados ---")
    print(f"Dataset:  {dataset_name} (test)")
    print(f"Modelo:   {model_type} / {model_name}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Loss:     {avg_loss:.4f}")
    print(f"Muestras: {correct}/{total}")
    return accuracy, avg_loss


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Uso: python validate.py <weights.pt> <dataset> <model_type> <model_name>")
        print("Ejemplo: python validate.py model.pt cifar10 resnet18 resnet18")
        print("Ejemplo: python validate.py model.pt ag_news distilbert distilbert-base-uncased")
        sys.exit(1)

    validate(
        weights_path=sys.argv[1],
        dataset_name=sys.argv[2],
        model_type=sys.argv[3],
        model_name=sys.argv[4]
    )