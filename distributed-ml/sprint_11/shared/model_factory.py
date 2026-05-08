import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os

class ModelFactory:
    @staticmethod
    def _get_registry():
        path = "registry.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {"models": {}, "datasets": {}}
    
    @staticmethod
    def build(model_display_name, num_labels):
        registry = ModelFactory._get_registry()
        config = registry["models"].get(model_display_name)

        if not config:
            raise ValueError(f"Modelo '{model_display_name}' no encontrado en el registro.")
        
        m_type = config["type"].lower()
        m_name = config["name"]

        nlp_keywords = ["transformer", "bert", "roberta", "distilbert", "albert"]
        if any(k in m_type for k in nlp_keywords):
            return AutoModelForSequenceClassification.from_pretrained(
                m_name, num_labels=num_labels
            )
    
        try:
            model_setup_func = getattr(models, m_type)
            try:
                weights_name = f"{m_type.capitalize().replace('net', 'Net')}_Weights.DEFAULT"
                model = model_setup_func(weights='DEFAULT')
            except:
                model = model_setup_func(pretrained=True)

            if hasattr(model, 'fc'): # Caso ResNet
                model.fc = nn.Linear(model.fc.in_features, num_labels)
            elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_labels)
            elif hasattr(model, 'heads'):
                model.heads.head = nn.Linear(model.heads.head.in_features, num_labels)
            
            return model

        except AttributeError:
            raise ValueError(f"La arquitectura '{m_type}' no existe en HuggingFace ni en Torchvision.")
    @staticmethod
    def get_tokenizer(model_display_name):
        registry = ModelFactory._get_registry()
        config = registry["models"].get(model_display_name)
        nlp_keywords = ["transformer", "bert", "roberta", "distilbert", "albert"]
        if any(k in config["type"].lower() for k in nlp_keywords):
            return AutoTokenizer.from_pretrained(config["name"])
        
        return None