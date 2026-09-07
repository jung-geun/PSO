"""MNIST dataset gradient baseline (PyTorch standard backprop optimizer, non-PSO)."""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 5 * 5, 2048)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.8)

        self.fc2 = nn.Linear(2048, 1024)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.8)

        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.drop1(self.pool1(self.relu1(self.conv1(x))))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.drop3(self.relu3(self.fc1(self.drop2(x))))
        x = self.drop4(self.relu4(self.fc2(x)))
        x = self.fc3(x)
        return x


def get_data(download: bool = True):
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=download
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=download
    )
    return train_dataset, test_dataset


def get_device() -> torch.device:
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = get_device()
    print(f"Selected device: {device}")

    train_dataset, test_dataset = get_data(download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MNISTModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(10):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item() * bx.size(0)
                total += bx.size(0)

        val_loss /= total

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            loss = criterion(out, by)
            test_loss += loss.item() * bx.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

    test_loss /= total
    test_acc = correct / total
    print(f"Final test loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
