import os
import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(weights_path):
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

def get_path(model_name):
    return os.path.join(BASE_DIR, "..", "results", model_name)

model_fed_avg = load_model(get_path("model_fed_avg.pt"))
model_fed_median = load_model(get_path("model_fed_median.pt"))
model_fed_trimmed_mean = load_model(get_path("model_fed_trimmed_mean.pt"))

def predict(image, chosen_model):
    tensor = transform(image).unsqueeze(0)
    if chosen_model == "Fed Average":
        model = model_fed_avg
    elif chosen_model == "Fed Median":
        model = model_fed_median
    else: 
        model = model_fed_trimmed_mean
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
    return {CLASSES[i]: float(probs[i]) for i in range(10)}

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Radio(
            choices=["Fed Average", "Fed Median", "Trimmed Mean"],
            label="Selecciona modelo a usar"
        )
    ],
    outputs=gr.Label(num_top_classes=3),
    title="CIFAR-10 — Federated ResNet18",
    description="Modelo entrenado con federated learning distribuido en Kubernetes"
)

interface.launch()
