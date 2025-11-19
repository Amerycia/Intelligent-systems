import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

print("=== Перевірка форм датасету ===")
print("Train images shape:", train_dataset.data.shape)
print("Train labels shape:", train_dataset.targets.shape)
print("Test images shape:", test_dataset.data.shape)
print("Test labels shape:", test_dataset.targets.shape)
print("\nУнікальні мітки в train_dataset:")
print(sorted(train_dataset.targets.unique().tolist()))
print("\nПоказ кількох зображень з тренувального набору...")

images, labels = next(iter(train_loader))

for i in range(6):
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.title(f"Label (істинна мітка): {labels[i].item()}")
    plt.axis("off")
    plt.show()

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
epochs = 3
print("\n=== Починаємо навчання ===")
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f"\nTest accuracy: {test_accuracy:.4f}")