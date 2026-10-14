"""Train a small CNN on MNIST and save it for the digit recognizer app.

Usage:
    python train.py            # train with default settings
    python train.py --epochs 5
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "mnist_cnn.pt"


class DigitCNN(nn.Module):
    """Same architecture as the TensorFlow.js web version."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 13x13 -> 11x11
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.dropout2 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # -> 13x13
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # -> 5x5
        x = self.dropout1(torch.flatten(x, 1))
        x = self.dropout2(F.relu(self.fc1(x)))
        return self.fc2(x)  # raw logits; apply softmax for probabilities


def load_model(path=MODEL_PATH):
    """Load a trained model from disk in evaluation mode."""
    model = DigitCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def evaluate(model, loader):
    """Return the accuracy of the model on the given data loader."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


def train(epochs=3, batch_size=128):
    """Download MNIST, train the CNN, print test accuracy, and save the weights."""
    torch.manual_seed(0)
    to_tensor = transforms.ToTensor()  # pixels scaled to [0, 1], white digit on black
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=to_tensor)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=to_tensor)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000)

    model = DigitCNN()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (images, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch}/{epochs}  step {step}/{len(train_loader)}  "
                      f"loss {running_loss / 100:.4f}")
                running_loss = 0.0
        print(f"Epoch {epoch} done. Test accuracy: {evaluate(model, test_loader) * 100:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MNIST digit CNN.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)
