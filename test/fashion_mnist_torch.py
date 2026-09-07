"""Fashion-MNIST dataset gradient baseline (PyTorch standard backprop optimizer, non-PSO)."""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.sig1 = nn.Sigmoid()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.sig2 = nn.Sigmoid()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.sig3 = nn.Sigmoid()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.sig1(self.conv1(x)))
        x = self.pool2(self.sig2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.sig3(self.fc1(x))
        x = self.fc2(x)
        return x


def get_data(download: bool = True):
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, transform=transform, download=download
    )
    test_dataset = datasets.FashionMNIST(
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = FashionMNISTModel().to(device)
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
