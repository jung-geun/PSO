"""Behavioral tests for the offline CIFAR/ResNet convergence adapter."""

from __future__ import annotations

import builtins
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test import post_training_resnet_convergence as resnet  # noqa: E402
from test.post_training_model_convergence import (  # noqa: E402
    CandidateEndpoint,
    ObjectiveResult,
    ProtocolError,
    SelectedResidualCodec,
    StudyConfig,
    prepare_run,
    select_endpoint,
)


class _TinyBlock(nn.Module):
    def __init__(self, channels: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(value)))


class _TinyResNet(nn.Module):
    """Small module with the same prefix/layer4/suffix contract as ResNet."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.maxpool = nn.Identity()
        self.layer1 = nn.Sequential(_TinyBlock())
        self.layer2 = nn.Sequential(_TinyBlock())
        self.layer3 = nn.Sequential(_TinyBlock())
        self.layer4 = nn.Sequential(_TinyBlock(), _TinyBlock())
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2, 3)
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.maxpool(self.relu(self.bn1(self.conv1(value))))
        value = self.layer1(value)
        value = self.layer2(value)
        value = self.layer3(value)
        value = self.layer4(value)
        return self.fc(torch.flatten(self.avgpool(value), 1))


def _synthetic_cifar() -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Make valid-shaped, deterministic pixels without constructing a dataset."""
    count = resnet.TRAIN_SAMPLES
    labels = np.repeat(np.arange(10, dtype=np.int64), count // 10)
    images = np.zeros((count, 32, 32, 3), dtype=np.uint8)
    encoded = np.arange(count, dtype=np.uint32).view(np.uint8).reshape(count, 4)
    images[:, 0, 0, :] = encoded[:, :3]
    rng = np.random.default_rng(20260908)
    initial_assignment = np.full(count, "", dtype=object)
    for cls in range(10):
        members = np.flatnonzero(labels == cls)
        members = members[rng.permutation(len(members))]
        initial_assignment[members[:3500]] = "bp_train"
        initial_assignment[members[3500:4000]] = "refine_search"
        initial_assignment[members[4000:5000]] = "selection_val"
    first = 0
    second = next(
        index
        for index in range(1, count // 10)
        if initial_assignment[index] != initial_assignment[first]
    )
    images[second] = images[first]
    return images, labels, (first, second)


def test_cifar_manifest_is_deterministic_disjoint_and_group_safe() -> None:
    images, labels, duplicate_pair = _synthetic_cifar()
    first = resnet.build_cifar_manifests(images, labels, split_seed=20260908)
    second = resnet.build_cifar_manifests(images, labels, split_seed=20260908)

    assert first == second
    roles = first["roles"]
    role_sets = {role: set(indices) for role, indices in roles.items()}
    assert sum(len(indices) for indices in role_sets.values()) == len(labels)
    for role, values in role_sets.items():
        for other, other_values in role_sets.items():
            if role != other:
                assert values.isdisjoint(other_values)
    assert set().union(*role_sets.values()) == set(range(len(labels)))

    owner_roles = [role for role, values in role_sets.items() if duplicate_pair[0] in values]
    assert len(owner_roles) == 1
    assert duplicate_pair[1] in role_sets[owner_roles[0]]
    assert set(first["objective"]).issubset(role_sets["refine_search"])
    assert len(first["objective"]) == resnet.OBJECTIVE_SAMPLES
    assert len(set(first["objective"])) == resnet.OBJECTIVE_SAMPLES
    objective_labels = labels[np.asarray(first["objective"])]
    assert np.bincount(objective_labels, minlength=10).tolist() == [103, 103, 103, 103, 102, 102, 102, 102, 102, 102]
    assert first["normalization_scope"] == "bp_train_only"


def test_real_resnet_selected_suffix_topology_without_downloads() -> None:
    try:
        import torchvision  # noqa: F401
    except Exception as exc:  # torchvision is optional on lightweight CI workers.
        pytest.skip(f"torchvision unavailable: {exc}")

    for architecture, block in (("resnet18", "layer4.1"), ("resnet50", "layer4.2")):
        model = resnet.make_cifar_resnet(architecture, seed=501)
        assert model.conv1.in_channels == 3
        assert model.conv1.out_channels == 64
        assert model.conv1.kernel_size == (3, 3)
        assert model.conv1.stride == (1, 1)
        assert isinstance(model.maxpool, nn.Identity)

        names = resnet.selected_parameter_names(model, architecture)
        expected = tuple(
            name
            for name, parameter in model.named_parameters()
            if name.startswith(block + ".") and parameter.is_floating_point()
        )
        assert names == expected
        assert names and all(name.startswith(block + ".") for name in names)
        assert resnet.head_parameter_names(model) == ("fc.weight", "fc.bias")


def test_cached_suffix_parity_and_residual_zero_nonzero_restoration() -> None:
    torch.manual_seed(7)
    model = _TinyResNet()
    images = torch.randn(5, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 1, 0])
    model.eval()
    cache = resnet.ResNetCache.build(model, images, labels, block_index=1, batch_size=2)
    names = tuple(name for name, _ in model.named_parameters() if name.startswith("layer4.1."))
    codec = SelectedResidualCodec(model, names, projection_seed=resnet.PROJECTION_SEED)
    zero = codec.zero_residual()
    nonzero = torch.full((codec.dimension,), 0.4)

    base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    base_logits = resnet.CachedSuffixEvaluator(model, cache, "cpu").logits()
    assert torch.equal(codec.decode(zero), torch.cat([value.reshape(-1) for value in codec.base_values]))
    assert torch.count_nonzero(codec.decode_delta(zero)) == 0
    assert torch.count_nonzero(codec.decode_delta(nonzero)) > 0

    with codec.applied(model, zero):
        assert torch.equal(resnet.CachedSuffixEvaluator(model, cache, "cpu").logits(), base_logits)
    assert all(torch.equal(value, base_state[name]) for name, value in model.state_dict().items())

    model.train()
    with pytest.raises(RuntimeError, match="candidate failure"):
        with codec.applied(model, nonzero):
            selected = dict(model.named_parameters())
            assert any(not torch.equal(selected[name], base_state[name]) for name in names)
            assert all(torch.equal(selected[name], base_state[name]) for name in selected if name not in names)
            assert all(torch.equal(value, base_state[name]) for name, value in model.named_buffers())
            raise RuntimeError("candidate failure")
    assert model.training
    assert all(torch.equal(value, base_state[name]) for name, value in model.state_dict().items())

    parity = resnet.cached_residual_parity(model, images, cache, codec, nonzero)
    assert parity["passed"] is True
    assert parity["samples"] == len(images)
    assert parity["max_abs_difference"] <= 1e-6
    assert resnet.cached_full_parity(model, images, cache)["passed"] is True



def test_endpoint_selection_ties_are_stable() -> None:
    objective = ObjectiveResult(loss=0.25, samples=4)
    endpoints = (
        CandidateEndpoint(20, 1, torch.ones(64), objective),
        CandidateEndpoint(10, 0, torch.zeros(64), objective),
    )
    assert select_endpoint(endpoints, {10: 0.5, 20: 0.5}).generation == 10
    assert select_endpoint(endpoints, {10: 0.8, 20: 0.8}, maximize=True).generation == 10
    with pytest.raises(ProtocolError):
        select_endpoint(endpoints, {10: float("nan"), 20: 0.5})


def test_ensemble_fit_uses_objective_pool_and_apply_does_not_refit() -> None:
    pytest.importorskip("scipy")
    rng = np.random.default_rng(19)
    objective_probs = rng.uniform(0.01, 1.0, size=(3, 9, 3))
    objective_probs /= objective_probs.sum(axis=-1, keepdims=True)
    selection_probs = np.roll(objective_probs, shift=1, axis=1).copy()
    objective_labels = np.arange(9, dtype=np.int64) % 3
    selection_labels = np.roll(objective_labels, 2)

    fitted = resnet.run_ensemble_methods(objective_probs, objective_labels, swarm_seeds=(601,))
    fitted_snapshot = copy.deepcopy(fitted)
    applied = resnet.evaluate_fitted_ensemble(fitted, selection_probs, selection_labels)
    assert fitted == fitted_snapshot

    uniform = np.full(3, 1 / 3)
    expected_uniform = np.einsum("m,mnk->nk", uniform, selection_probs)
    expected_nll = float(-np.log(np.clip(expected_uniform[np.arange(9), selection_labels], 1e-300, 1.0)).mean())
    assert applied["uniform"]["metrics"]["nll"] == pytest.approx(expected_nll, abs=1e-12)
    assert fitted["uniform"]["metrics"]["nll"] == pytest.approx(
        float(-np.log(np.clip(np.einsum("m,mnk->nk", uniform, objective_probs)[np.arange(9), objective_labels], 1e-300, 1.0)).mean()),
        abs=1e-12,
    )

    for candidate in applied["ensemble_pso"]:
        weights = np.asarray(candidate["weights"], dtype=np.float64)
        mixed = np.einsum("m,mnk->nk", weights, selection_probs)
        expected = float(-np.log(np.clip(mixed[np.arange(9), selection_labels], 1e-300, 1.0)).mean())
        assert candidate["selection_metrics"]["nll"] == pytest.approx(expected, abs=1e-12)


def test_official_test_loader_refuses_pre_freeze_without_importing_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    prepare_run(run_root, StudyConfig())
    imported = False
    original_import = builtins.__import__

    def reject_torchvision(name: str, *args: object, **kwargs: object):
        nonlocal imported
        if name.startswith("torchvision"):
            imported = True
            raise AssertionError("official dataset import must be behind the frozen seal")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torchvision)
    with pytest.raises(resnet.TestSealError, match="forbidden before frozen"):
        resnet.load_official_test_data(tmp_path / "data", run_root, allow_download=False)
    assert imported is False
