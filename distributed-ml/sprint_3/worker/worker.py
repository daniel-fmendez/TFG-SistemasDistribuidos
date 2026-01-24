import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset

# gRPC
import socket
import time 
import os
import grpc
import metrics_pb2
import metrics_pb2_grpc
import datetime

MASTER_HOST = os.getenv('MASTER_HOST', 'master-service')
MASTER_PORT = os.getenv('MASTER_PORT', '50051')
MODEL_TYPE = os.getenv('MODEL_TYPE', 'general') 

class DistributedTrainer:
    def __init__(self):
        self.worker_id = socket.gethostname()
        self.model = MODEL_TYPE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pass