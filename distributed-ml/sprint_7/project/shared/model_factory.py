import torch.nn as nn
from torchvision import models
from transformers import (
    DistilBertForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification
)


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
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_labels)
            return model
        elif model_type == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_labels)
            return model
        elif model_type == "mobilenet":
            weights = models.MobileNet_V2_Weights.DEFAULT
            model = models.mobilenet_v2(weights=weights)
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