import pytest
import torch
import torch.nn as nn


@pytest.fixture
def xor_data():
    """
    Returns deterministic XOR input features (4, 2) and labels (4, 1) as float32 torch tensors on CPU.
    """
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    return x, y


@pytest.fixture
def model_factory():
    """
    Factory fixture producing deterministic, PyTorch nn.Module models.
    Supports units tuning, zero initialization, and input/output dimension changes.
    """

    def _create_model(
        units: int = 4,
        zero_init: bool = False,
        input_dim: int = 2,
        output_dim: int = 1,
    ) -> nn.Module:
        torch.manual_seed(42)
        layers = [
            nn.Linear(input_dim, units),
            nn.ReLU(),
            nn.Linear(units, output_dim),
        ]
        model = nn.Sequential(*layers)
        if zero_init:
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        return model

    return _create_model
