from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

load_dataset("imdb")
DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
