import torch.nn as nn
from transformers import (
    DistilBertForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification
)
from torchvision import models

class ModelFactory:

    @staticmethod
    def build(model_type, model_name, num_labels, **kwargs):
        if model_type == "distilbert":
            return DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        elif model_type == "bert":
            return BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        elif model_type == "roberta":
            return RobertaForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        elif model_type == "resnet18":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_labels)
            return model
        elif model_type == "resnet50":
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_labels)
            return model
        elif model_type == "mobilenet":
            model = models.mobilenet_v2(pretrained=True)
            # Reemplazar clasificador
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_labels)
            return model
        else:
            raise ValueError(f"Modelo no soportado: {model_type}")
    
    @staticmethod
    def get_tokenizer(model_type, model_name):
        if model_type in ["distilbert", "bert", "roberta"]:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_name)
        return None  # Modelos de visión no necesitan tokenizer