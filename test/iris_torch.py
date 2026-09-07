"""Iris dataset gradient baseline (PyTorch standard backprop optimizer, non-PSO)."""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class IrisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
        )

    def forward(self, x):
        return self.net(x)


def load_data(seed: int = 42):
    iris = load_iris()
    X = iris.data.astype("float32")
    y = iris.target.astype("int64")

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=seed
    )
    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.int64),
        torch.tensor(y_test, dtype=torch.int64),
    )


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

    x_train, x_test, y_train, y_test = load_data(seed=42)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=32, shuffle=False
    )

    model = IrisModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(200):
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
            for bx, by in val_loader:
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
        for bx, by in val_loader:
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
