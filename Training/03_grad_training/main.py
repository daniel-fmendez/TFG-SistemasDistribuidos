import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

NUMBER_OF_EPOCHS = 10
LEARNING_RATE = 0.01

train_loader = torch.utils.data.DataLoader (
    dataset=datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    ),
    batch_size=32,
    shuffle=True
)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleNet()

criterion = nn.CrossEntropyLoss()

for epoch in range(NUMBER_OF_EPOCHS):
    epoch_loss = 0.0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= LEARNING_RATE * param.grad

        model.zero_grad()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUMBER_OF_EPOCHS}] - Loss: {avg_loss:.4f}")

print("Entrenamiento Finalzado")