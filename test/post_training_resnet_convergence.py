"""CIFAR-10 ResNet adapters for the post-training model-convergence protocol.

The module deliberately keeps torchvision imports lazy: importing the common
registry must not construct a dataset or download anything.  All persistence
is scoped to a caller supplied run root and all public-test access is guarded
by the common frozen-manifest seal.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from test.post_training_model_convergence import (
    BASE_SEEDS,
    PROJECTION_SEED,
    OBJECTIVE_CHECKPOINTS,
    PSO_GENERATIONS,
    RANDOM_CANDIDATES,
    RESIDUAL_DIMENSION,
    SWARM_SEEDS,
    AuditResult,
    ObjectiveResult,
    ProtocolError,
    ResourceCounters,
    SealError,
    SelectedResidualCodec,
    StudyConfig,
    StudyState,
    StudyStateMachine,
    atomic_write_bytes,
    atomic_write_json,
    begin_confirmation,
    canonical_json,
    fingerprint_file,
    fingerprint_module,
    fingerprint_paths,
    fingerprint_nonselected_state,
    finish_confirmation,
    freeze_run,
    load_frozen_manifest,
    load_state,
    persist_state,
    prepare_run,
    run_equal_budget_random,
    run_residual_pso,
    run_state_neutral_audit,
    select_endpoint,
    sha256_bytes,
    verify_frozen_manifest,
)


WORKLOAD_IDS = ("cifar10_resnet18", "cifar10_resnet50")
ARCHITECTURES = {"cifar10_resnet18": "resnet18", "cifar10_resnet50": "resnet50"}
TRAIN_SAMPLES = 50_000
SPLIT_COUNTS = {"bp_train": 35_000, "refine_search": 5_000, "selection_val": 10_000}
OBJECTIVE_SAMPLES = 1_024
BATCH_SIZE = 128
TRAIN_EPOCHS = 100
SMOKE_EPOCHS = 2
SMOKE_BATCH_SIZE = 8
SMOKE_OBJECTIVE_SAMPLES = 16
SMOKE_SEARCH_GENERATIONS = 2
SMOKE_SEARCH_PARTICLES = 12


class TestSealError(SealError):
    """Raised when official CIFAR test data is opened before confirmation."""


class CIFARSubset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, indices: Sequence[int], *, augment: bool = False) -> None:
        self.images = images
        self.labels = labels
        self.indices = tuple(int(i) for i in indices)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[self.indices[index]]
        if self.augment:
            # CIFAR augmentation is intentionally implemented without a global
            # torchvision transform object so loader RNG ownership is explicit.
            pad = F.pad(image.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
            top = int(torch.randint(0, 9, ()).item())
            left = int(torch.randint(0, 9, ()).item())
            image = pad[:, top : top + 32, left : left + 32]
            if bool(torch.rand(()) < 0.5):
                image = image.flip(-1)
        return image, self.labels[self.indices[index]]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_cifar_resnet(architecture: str, seed: int = 501) -> nn.Module:
    """Construct the exact scratch CIFAR stem and requested torchvision model."""
    if architecture not in {"resnet18", "resnet50"}:
        raise ProtocolError(f"unsupported ResNet architecture: {architecture}")
    from torchvision import models

    _seed_everything(seed)
    constructor = getattr(models, architecture)
    model = constructor(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    if model.conv1.kernel_size != (3, 3) or model.conv1.stride != (1, 1):
        raise ProtocolError("CIFAR stem was not installed")
    return model


def selected_parameter_names(model: nn.Module, architecture: str) -> tuple[str, ...]:
    block = "layer4.1" if architecture == "resnet18" else "layer4.2"
    names = tuple(name for name, value in model.named_parameters() if name.startswith(block + ".") and value.is_floating_point())
    if not names:
        raise ProtocolError(f"selected block has no floating parameters: {block}")
    if any(name.startswith("layer4.") and not name.startswith(block + ".") for name in names):
        raise ProtocolError("selected-name topology mismatch")
    return names


def head_parameter_names(model: nn.Module) -> tuple[str, ...]:
    names = tuple(name for name, _ in model.named_parameters() if name in {"fc.weight", "fc.bias"})
    if names != ("fc.weight", "fc.bias"):
        raise ProtocolError(f"unexpected CIFAR head parameters: {names}")
    return names


def _canonical_pixel_hash(image: np.ndarray) -> str:
    value = np.asarray(image, dtype=np.uint8)
    if value.shape != (32, 32, 3):
        raise ProtocolError(f"unexpected CIFAR image shape: {value.shape}")
    payload = b"RGB32\0" + canonical_json((32, 32, 3)) + value.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _initial_stratified_assignment(labels: np.ndarray, seed: int) -> tuple[list[int], dict[int, str]]:
    generator = np.random.default_rng(seed)
    bucket = np.full(len(labels), "", dtype=object)
    global_order: list[int] = []
    for cls in range(10):
        members = np.flatnonzero(labels == cls)
        members = members[generator.permutation(len(members))]
        global_order.extend(int(i) for i in members)
        bucket[members[:3500]] = "bp_train"
        bucket[members[3500:4000]] = "refine_search"
        bucket[members[4000:5000]] = "selection_val"
    if any(not item for item in bucket):
        raise ProtocolError("stratified CIFAR assignment did not cover train set")
    rank = {index: position for position, index in enumerate(global_order)}
    return [int(i) for i in global_order], {int(i): str(bucket[i]) for i in range(len(labels))}


def build_cifar_manifests(images: np.ndarray, labels: Sequence[int], *, split_seed: int = 20260908) -> dict[str, Any]:
    """Build deterministic duplicate-group-aware CIFAR role manifests."""
    pixels = np.asarray(images)
    y = np.asarray(labels, dtype=np.int64)
    if pixels.shape != (TRAIN_SAMPLES, 32, 32, 3) or y.shape != (TRAIN_SAMPLES,):
        raise ProtocolError(f"expected CIFAR train shape (50000,32,32,3), got {pixels.shape}, {y.shape}")
    if int(y.min()) != 0 or int(y.max()) != 9:
        raise ProtocolError("CIFAR labels must be in [0,9]")
    order, assignment = _initial_stratified_assignment(y, split_seed)
    groups: dict[str, list[int]] = {}
    for index in range(len(y)):
        groups.setdefault(_canonical_pixel_hash(pixels[index]), []).append(index)
    rank = {index: position for position, index in enumerate(order)}
    invalid: list[dict[str, Any]] = []
    moved = 0
    for digest, members in groups.items():
        classes = {int(y[i]) for i in members}
        if len(classes) != 1:
            invalid.append({"fingerprint": digest, "indices": members, "labels": sorted(classes)})
            continue
        owner = assignment[min(members, key=lambda i: rank[i])]
        for index in members:
            if assignment[index] != owner:
                moved += 1
                assignment[index] = owner
    if invalid:
        raise ProtocolError(f"duplicate pixels have conflicting labels: {invalid[:2]}")
    split_indices = {role: [index for index in order if assignment[index] == role] for role in SPLIT_COUNTS}
    # Group ownership is primary; exact cardinality may therefore change.
    for role, expected in SPLIT_COUNTS.items():
        if not split_indices[role]:
            raise ProtocolError(f"empty CIFAR role: {role}")
        if role == "bp_train" and len(split_indices[role]) < 30_000:
            raise ProtocolError("duplicate grouping changed bp_train implausibly")
    refine_by_class = {cls: [i for i in split_indices["refine_search"] if int(y[i]) == cls] for cls in range(10)}
    objective: list[int] = []
    for cls in range(10):
        need = 103 if cls < 4 else 102
        if len(refine_by_class[cls]) < need:
            raise ProtocolError(f"insufficient refine_search class {cls} examples")
        objective.extend(refine_by_class[cls][:need])
    if len(objective) != OBJECTIVE_SAMPLES:
        raise ProtocolError("CIFAR objective does not contain exactly 1024 examples")
    return {
        "dataset": "cifar10",
        "split_seed": int(split_seed),
        "source_train_samples": TRAIN_SAMPLES,
        "roles": {role: [int(i) for i in values] for role, values in split_indices.items()},
        "objective": [int(i) for i in objective],
        "counts": {role: len(values) for role, values in split_indices.items()},
        "duplicate_groups": len(groups),
        "duplicate_members_moved": moved,
        "duplicate_group_hashes": sorted(groups),
        "normalization_scope": "bp_train_only",
    }


def prepare_cifar_data(data_root: str | os.PathLike[str], *, split_seed: int = 20260908, allow_download: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    """Load only official CIFAR training data and create the development manifest."""
    from torchvision.datasets import CIFAR10

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    dataset = CIFAR10(root=str(root), train=True, download=allow_download)
    raw_images = np.asarray(dataset.data, dtype=np.uint8)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    manifest = build_cifar_manifests(raw_images, labels, split_seed=split_seed)
    bp = np.asarray(manifest["roles"]["bp_train"], dtype=np.int64)
    pixels = torch.from_numpy(raw_images).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    mean = pixels[bp].mean(dim=(0, 2, 3))
    std = pixels[bp].std(dim=(0, 2, 3), unbiased=False).clamp_min(1e-12)
    normalized = (pixels - mean[None, :, None, None]) / std[None, :, None, None]
    provenance = {
        "source": "torchvision.datasets.CIFAR10(train=True)",
        "train_samples": len(dataset),
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "normalization_scope": "bp_train_only",
        "data_sha256": hashlib.sha256(raw_images.tobytes(order="C")).hexdigest(),
    }
    manifest["normalization"] = provenance
    return normalized, torch.from_numpy(labels), manifest, provenance


def _assert_test_sealed(run_root: Path) -> None:
    state = load_state(run_root)
    if state.state not in {StudyState.FROZEN, StudyState.CONFIRMING, StudyState.COMPLETED}:
        raise TestSealError("official CIFAR test access is forbidden before frozen state")
    verify_frozen_manifest(run_root)


def load_official_test_data(data_root: str | os.PathLike[str], run_root: str | os.PathLike[str], *, allow_download: bool = False, mean: Sequence[float] | None = None, std: Sequence[float] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the official test dataset exactly behind the frozen seal."""
    _assert_test_sealed(Path(run_root))
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=str(data_root), train=False, download=allow_download)
    if len(dataset) != 10_000:
        raise ProtocolError(f"official CIFAR test must contain 10000 samples, got {len(dataset)}")
    images = torch.from_numpy(np.asarray(dataset.data, dtype=np.uint8)).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    labels = torch.as_tensor(dataset.targets, dtype=torch.long)
    if mean is not None and std is not None:
        images = (images - torch.as_tensor(mean)[None, :, None, None]) / torch.as_tensor(std)[None, :, None, None]
    return images, labels


def audit_test_duplicates(train_images: np.ndarray, test_images: np.ndarray) -> dict[str, Any]:
    train_hashes = {_canonical_pixel_hash(image) for image in np.asarray(train_images)}
    test_hashes = [_canonical_pixel_hash(image) for image in np.asarray(test_images)]
    overlap = sorted(train_hashes.intersection(test_hashes))
    return {"train_unique": len(train_hashes), "test_unique": len(set(test_hashes)), "exact_duplicate_groups": len(overlap), "exact_duplicate": bool(overlap), "overlap_hashes": overlap}


def _prefix(model: nn.Module, x: torch.Tensor, block_index: int) -> torch.Tensor:
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    for index in range(block_index):
        x = model.layer4[index](x)
    return x


def _suffix(model: nn.Module, x: torch.Tensor, block_index: int) -> torch.Tensor:
    for index in range(block_index, len(model.layer4)):
        x = model.layer4[index](x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return model.fc(x)


def _avgpool_features(model: nn.Module, x: torch.Tensor, block_index: int) -> torch.Tensor:
    for index in range(block_index, len(model.layer4)):
        x = model.layer4[index](x)
    return torch.flatten(model.avgpool(x), 1)


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    logits_cpu = logits.detach().cpu().to(torch.float64)
    log_prob = F.log_softmax(logits_cpu, dim=1)
    target = labels.detach().cpu().to(torch.long)
    nll = -log_prob[torch.arange(target.numel()), target].mean()
    accuracy = (logits_cpu.argmax(dim=1) == target).to(torch.float64).mean()
    return {
        "nll": float(nll),
        "accuracy": float(accuracy),
        "samples": int(target.numel()),
    }


def _probability_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.ndim != 2 or len(p) != len(y):
        raise ProtocolError("probability/label shape mismatch")
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-300, 1.0)).mean()
    return {"nll": float(nll), "accuracy": float((p.argmax(axis=1) == y).mean()), "samples": int(len(y))}


def evaluate_logits(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, device: torch.device | str, *, batch_size: int = 512) -> tuple[dict[str, float], np.ndarray]:
    model_device = torch.device(device)
    model.to(model_device).eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            logits = model(images[start : start + batch_size].to(model_device))
            probs.append(
                torch.softmax(logits, dim=1)
                .cpu()
                .to(torch.float64)
                .numpy()
            )
    values = np.concatenate(probs, axis=0)
    return _probability_metrics(values, labels.cpu().numpy()), values


@dataclasses.dataclass
class ResNetCache:
    prefixes: tuple[torch.Tensor, ...]
    labels: torch.Tensor
    block_index: int
    source_fingerprint: str
    batch_size: int

    @classmethod
    def build(cls, model: nn.Module, images: torch.Tensor, labels: torch.Tensor, block_index: int, *, batch_size: int = 64) -> "ResNetCache":
        chunks: list[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                chunks.append(_prefix(model, images[start : start + batch_size].to(next(model.parameters()).device), block_index).detach().cpu().clone())
        return cls(tuple(chunks), labels.detach().cpu().clone(), block_index, hashlib.sha256(images.detach().cpu().contiguous().numpy().tobytes()).hexdigest(), batch_size)

    @property
    def samples(self) -> int:
        return int(self.labels.numel())


class CachedSuffixEvaluator:
    """Evaluate the real mutable block and suffix against a detached prefix cache."""
    def __init__(self, model: nn.Module, cache: ResNetCache, device: torch.device | str) -> None:
        self.model = model
        self.cache = cache
        self.device = torch.device(device)
        self.model.to(self.device)

    def logits(self, *, grad: bool = False) -> torch.Tensor:
        output: list[torch.Tensor] = []
        context = contextlib.nullcontext() if grad else torch.no_grad()
        with context:
            for prefix in self.cache.prefixes:
                output.append(_suffix(self.model, prefix.to(self.device), self.cache.block_index))
        return torch.cat(output, dim=0)

    def objective(self, residual: torch.Tensor | None = None, codec: SelectedResidualCodec | None = None) -> ObjectiveResult:
        try:
            if residual is not None and codec is not None:
                with codec.applied(self.model, residual):
                    logits = self.logits()
            else:
                logits = self.logits()
            stats = classification_metrics(logits, self.cache.labels.to(self.device))
            return ObjectiveResult(stats["nll"], self.cache.samples, forward_passes=len(self.cache.prefixes))
        finally:
            if codec is not None:
                codec.restore_base(self.model)

    def validation(self, residual: torch.Tensor | None = None, codec: SelectedResidualCodec | None = None) -> AuditResult:
        if residual is not None and codec is not None:
            with codec.applied(self.model, residual):
                stats = classification_metrics(self.logits(), self.cache.labels.to(self.device))
        else:
            stats = classification_metrics(self.logits(), self.cache.labels.to(self.device))
        return AuditResult(stats["nll"], stats["accuracy"], self.cache.samples, metadata={"nll": stats["nll"], "accuracy": stats["accuracy"]})


def cached_residual_parity(model: nn.Module, images: torch.Tensor, cache: ResNetCache, codec: SelectedResidualCodec, residual: torch.Tensor, *, atol: float = 1e-6, rtol: float = 1e-5) -> dict[str, Any]:
    """Compare one nonzero candidate's cached suffix with a full forward."""
    dev = next(model.parameters()).device
    with codec.applied(model, residual):
        with torch.no_grad():
            full = model(images.to(dev)).cpu()
            cached = torch.cat([_suffix(model, item.to(dev), cache.block_index).cpu() for item in cache.prefixes])
    difference = float((full - cached).abs().max()) if full.numel() else 0.0
    return {"passed": bool(torch.allclose(full, cached, atol=atol, rtol=rtol)), "max_abs_difference": difference, "samples": len(images)}



def _save_torch(path: Path, value: Any) -> str:
    stream = io.BytesIO()
    torch.save(value, stream)
    atomic_write_bytes(path, stream.getvalue())
    return fingerprint_file(path)


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_baseline(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, manifest: Mapping[str, Any], *, seed: int, device: torch.device | str, epochs: int = TRAIN_EPOCHS, batch_size: int = BATCH_SIZE, checkpoint_path: Path | None = None, counters: ResourceCounters | None = None) -> dict[str, Any]:
    _seed_everything(seed)
    dev = torch.device(device)
    model.to(dev)
    bp_indices = manifest["roles"]["bp_train"]
    selection_indices = manifest["roles"]["selection_val"]
    train_loader = DataLoader(CIFARSubset(images, labels, bp_indices, augment=True), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + 1000), num_workers=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    telemetry: list[dict[str, Any]] = []
    audit_indices = bp_indices[: min(1024, len(bp_indices))]
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(dev))
            loss = criterion(logits, batch_y.to(dev))
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite baseline loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            if counters is not None:
                counters.base_training_samples += int(batch_y.numel())
                counters.base_training_forward_passes += 1
                counters.base_training_backward_passes += 1
        scheduler.step()
        if epoch >= max(1, epochs - 10) or epochs <= SMOKE_EPOCHS:
            def audit() -> dict[str, Any]:
                train_stats, _ = evaluate_logits(model, images[audit_indices], labels[audit_indices], dev, batch_size=batch_size)
                val_stats, _ = evaluate_logits(model, images[selection_indices], labels[selection_indices], dev, batch_size=batch_size)
                return {"epoch": epoch, "audit": train_stats, "selection": val_stats, "lr": float(optimizer.param_groups[0]["lr"])}
            telemetry.append(run_state_neutral_audit(model, audit))
    final_state = _state_cpu(model)
    checkpoint_hash = None
    if checkpoint_path is not None:
        checkpoint_hash = _save_torch(checkpoint_path, {"state_dict": final_state, "seed": seed, "epoch": epochs, "architecture": model.__class__.__name__})
    losses = [float(item["selection"]["nll"]) for item in telemetry]
    metrics = [float(item["selection"]["accuracy"]) for item in telemetry]
    mean_loss = max(abs(float(np.mean(losses))), 1e-12) if losses else 1.0
    plateau = bool(losses and (max(losses) - min(losses)) / mean_loss <= 0.01 and (max(metrics) - min(metrics)) <= 0.005)
    return {"seed": seed, "epochs": epochs, "checkpoint": str(checkpoint_path) if checkpoint_path else None, "checkpoint_hash": checkpoint_hash, "state_dict": final_state, "telemetry": telemetry, "baseline_plateau": plateau, "final_metrics": telemetry[-1] if telemetry else None, "batch_size": batch_size}


def _feature_cache_objective(model: nn.Module, cache: ResNetCache, codec: SelectedResidualCodec, residual: torch.Tensor) -> ObjectiveResult:
    evaluator = CachedSuffixEvaluator(model, cache, next(model.parameters()).device)
    return evaluator.objective(residual, codec)


def _head_cache(model: nn.Module, images: torch.Tensor, block_index: int, device: torch.device | str, batch_size: int = 64) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    features: list[torch.Tensor] = []
    model.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            prefix = _prefix(model, images[start : start + batch_size].to(device), block_index)
            features.append(_avgpool_features(model, prefix, block_index).detach().cpu())
    return tuple(features), images


def run_feature_adam(model: nn.Module, cache: ResNetCache, names: Sequence[str], *, device: torch.device | str, updates: int = 40, lr: float = 1e-3) -> dict[str, Any]:
    dev = torch.device(device)
    selected = dict(model.named_parameters())
    base = [selected[name].detach().to(dev).clone() for name in names]
    scales = [0.05 * max(float(torch.sqrt(torch.mean(item * item))), 0.01) for item in base]
    delta = nn.Parameter(torch.zeros(sum(item.numel() for item in base), device=dev))
    optimizer = torch.optim.AdamW([delta], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    offsets = np.cumsum([0] + [item.numel() for item in base])
    block_prefix = names[0].split(".")[:2]
    block = model.layer4[int(block_prefix[1])]
    local_names = [name.split(".", 2)[2] for name in names]
    trajectory: list[dict[str, Any]] = []
    for update in range(updates + 1):
        if update:
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=dev, dtype=torch.float32)
            for prefix, labels in zip(cache.prefixes, cache.labels.split([len(p) for p in cache.prefixes])):
                overrides: dict[str, torch.Tensor] = {}
                for index, local_name in enumerate(local_names):
                    part = delta[offsets[index] : offsets[index + 1]].view_as(base[index])
                    bound = scales[index]
                    overrides[local_name] = base[index] + part.clamp(-bound, bound)
                mutable = torch.func.functional_call(block, overrides, (prefix.to(dev),))
                x = mutable
                for index in range(int(block_prefix[1]) + 1, len(model.layer4)):
                    x = model.layer4[index](x)
                logits = model.fc(torch.flatten(model.avgpool(x), 1))
                target = labels.to(dev)
                total = total + F.cross_entropy(
                    logits,
                    target,
                    reduction="sum",
                )
            (total / cache.samples).backward()
            optimizer.step()
        with torch.no_grad():
            values = []
            for index, item in enumerate(base):
                part = delta[
                    offsets[index] : offsets[index + 1]
                ].view_as(item)
                values.append(
                    item
                    + part.clamp(-scales[index], scales[index])
                )
            for name, value in zip(names, values):
                selected[name].copy_(value)
        stats = CachedSuffixEvaluator(model, cache, dev).validation()
        trajectory.append({"update": update, "objective": stats.loss, "accuracy": stats.primary_metric})
    final_parameters = [value.detach().cpu().clone() for value in values]
    with torch.no_grad():
        for name, value in zip(names, base):
            selected[name].copy_(value)
    return {"method": "feature_adam", "updates": updates, "trajectory": trajectory, "best_residual": delta.detach().cpu().tolist(), "final_parameters": final_parameters, "final": trajectory[-1]}


def run_head_adam(model: nn.Module, features: tuple[torch.Tensor, ...], labels: torch.Tensor, *, device: torch.device | str, updates: int = 40, lr: float = 1e-3) -> dict[str, Any]:
    dev = torch.device(device)
    weight = model.fc.weight.detach().to(dev).clone()
    bias = model.fc.bias.detach().to(dev).clone()
    delta_w = nn.Parameter(torch.zeros_like(weight))
    delta_b = nn.Parameter(torch.zeros_like(bias))
    optimizer = torch.optim.AdamW([delta_w, delta_b], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    trajectory: list[dict[str, Any]] = []
    chunks = labels.split([len(x) for x in features])
    for update in range(updates + 1):
        if update:
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=dev, dtype=torch.float32)
            for feature, target in zip(features, chunks):
                logits = F.linear(feature.to(dev), weight + delta_w, bias + delta_b)
                target = target.to(dev)
                total = total + F.cross_entropy(
                    logits,
                    target,
                    reduction="sum",
                )
            (total / len(labels)).backward()
            optimizer.step()
        with torch.no_grad():
            model.fc.weight.copy_(weight + delta_w)
            model.fc.bias.copy_(bias + delta_b)
        logits = torch.cat([F.linear(feature.to(dev), model.fc.weight, model.fc.bias) for feature in features])
        stats = classification_metrics(logits, labels.to(dev))
        trajectory.append(
            {
                "update": update,
                "objective": stats["nll"],
                "accuracy": stats["accuracy"],
            }
        )
    final_weight = (weight + delta_w).detach().cpu().clone()
    final_bias = (bias + delta_b).detach().cpu().clone()
    with torch.no_grad():
        model.fc.weight.copy_(weight)
        model.fc.bias.copy_(bias)
    return {"method": "head_adam", "updates": updates, "trajectory": trajectory, "final_weight": final_weight, "final_bias": final_bias, "final": trajectory[-1]}


def _smoke_search(objective: Any, *, seed: int, device: torch.device | str) -> dict[str, Any]:
    """Two-generation real PSO movement used only by smoke (production is 12x60)."""
    rng = np.random.default_rng(seed)
    positions = [np.zeros(64, dtype=np.float32)]
    for _ in range(5):
        value = rng.uniform(-0.25, 0.25, 64).astype(np.float32)
        positions.extend([value, -value])
    positions.append(rng.uniform(-0.25, 0.25, 64).astype(np.float32))
    velocities = [np.zeros(64, dtype=np.float32) for _ in positions]
    pbest = [value.copy() for value in positions]
    pbest_loss = [math.inf] * len(positions)
    gbest = positions[0].copy()
    gbest_loss = math.inf
    evaluations = 0
    for generation in range(1, SMOKE_SEARCH_GENERATIONS + 1):
        snapshot = [value.copy() for value in positions]
        for index, candidate in enumerate(snapshot):
            result = ObjectiveResult.coerce(objective(torch.from_numpy(candidate)))
            evaluations += 1
            if result.loss < pbest_loss[index]:
                pbest_loss[index], pbest[index] = result.loss, candidate.copy()
            if result.loss < gbest_loss:
                gbest_loss, gbest = result.loss, candidate.copy()
        if generation < SMOKE_SEARCH_GENERATIONS:
            for index in range(len(positions)):
                r1, r2 = rng.random(64), rng.random(64)
                velocities[index] = 0.7298 * (velocities[index] + 2.05 * r1 * (pbest[index] - snapshot[index]) + 2.05 * r2 * (gbest - snapshot[index]))
                positions[index] = np.clip(snapshot[index] + velocities[index], -1.0, 1.0).astype(np.float32)
    return {"best_objective": float(gbest_loss), "best_residual": gbest.tolist(), "evaluations": evaluations, "generations": SMOKE_SEARCH_GENERATIONS, "movement": "constriction"}


def _ensemble_metrics(probabilities: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    mixed = np.einsum("m,mnk->nk", weights, probabilities)
    return _probability_metrics(mixed, labels)


def _fit_temperature(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    try:
        from scipy.optimize import minimize_scalar
    except ImportError as exc:
        raise ProtocolError("scipy is required for temperature ensemble fitting") from exc
    uniform = np.asarray(probs, dtype=np.float64).mean(axis=0)
    evaluations = 0
    best_probabilities: np.ndarray | None = None

    def objective(temp: float) -> float:
        nonlocal evaluations, best_probabilities
        evaluations += 1
        logits = np.log(np.clip(uniform, 1e-300, 1.0)) / float(temp)
        shifted = logits - logits.max(axis=1, keepdims=True)
        scaled = np.exp(shifted)
        best_probabilities = scaled / scaled.sum(axis=1, keepdims=True)
        return _probability_metrics(best_probabilities, labels)["nll"]

    result = minimize_scalar(objective, method="bounded", bounds=(0.01, 10.0))
    if best_probabilities is None:
        raise ProtocolError("temperature solver returned no candidate")
    return {"temperature": float(result.x), "metrics": _probability_metrics(best_probabilities, labels), "evaluations": evaluations, "success": bool(result.success), "status": int(result.status)}


def cached_avgpool_parity(model: nn.Module, images: torch.Tensor, *, block_index: int, batch_size: int = 64, atol: float = 1e-6, rtol: float = 1e-5) -> dict[str, Any]:
    """Compare an avgpool cache against the complete feature path."""
    dev = next(model.parameters()).device
    model.eval()
    cached: list[torch.Tensor] = []
    direct: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(dev)
            prefix = _prefix(model, batch, block_index)
            cached.append(_avgpool_features(model, prefix, block_index).cpu())
            full_prefix = _prefix(model, batch, 0)
            direct.append(_avgpool_features(model, full_prefix, 0).cpu())
    lhs, rhs = torch.cat(cached), torch.cat(direct)
    difference = float((lhs - rhs).abs().max()) if lhs.numel() else 0.0
    return {"passed": bool(torch.allclose(lhs, rhs, atol=atol, rtol=rtol)), "max_abs_difference": difference, "samples": len(images)}


def cached_full_parity(model: nn.Module, images: torch.Tensor, cache: ResNetCache, *, atol: float = 1e-6, rtol: float = 1e-5) -> dict[str, Any]:
    """Compare cached suffix logits with a complete model forward."""
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        full = model(images.to(dev)).cpu()
        cached = torch.cat([_suffix(model, item.to(dev), cache.block_index).cpu() for item in cache.prefixes])
    difference = float((full - cached).abs().max()) if full.numel() else 0.0
    return {"passed": bool(torch.allclose(full, cached, atol=atol, rtol=rtol)), "max_abs_difference": difference, "samples": len(images)}


def _fit_slsqp(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ProtocolError("scipy is required for SLSQP ensemble fitting") from exc
    evaluations = 0

    def objective(weights: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return _ensemble_metrics(probs, labels, weights)["nll"]

    result = minimize(
        objective,
        np.full(len(probs), 1 / len(probs)),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(probs),
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    weights = np.asarray(result.x, dtype=np.float64)
    return {"weights": weights.tolist(), "metrics": _ensemble_metrics(probs, labels, weights), "evaluations": evaluations, "success": bool(result.success), "status": int(result.status), "message": str(result.message)}


def _ensemble_pso(probs: np.ndarray, labels: np.ndarray, *, seed: int, generations: int = 20, particles: int = 12) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    positions = [np.zeros(3, dtype=np.float64)]
    for _ in range(5):
        value = rng.uniform(-0.25, 0.25, 3)
        positions.extend([value, -value])
    positions.append(rng.uniform(-0.25, 0.25, 3))
    velocities = [np.zeros(3, dtype=np.float64) for _ in positions]
    pbest = [x.copy() for x in positions]
    pbest_loss = [math.inf] * particles
    gbest = positions[0].copy()
    gbest_loss = math.inf
    trajectory: list[dict[str, Any]] = []
    for generation in range(1, generations + 1):
        for index, candidate in enumerate(positions):
            weights = np.exp(np.clip(candidate, -5, 5) - np.max(np.clip(candidate, -5, 5)))
            weights /= weights.sum()
            loss = _ensemble_metrics(probs, labels, weights)["nll"]
            if loss < pbest_loss[index]:
                pbest_loss[index], pbest[index] = loss, candidate.copy()
            if loss < gbest_loss:
                gbest_loss, gbest = loss, candidate.copy()
        trajectory.append({"generation": generation, "objective": gbest_loss})
        if generation == generations:
            break
        old = [x.copy() for x in positions]
        for index in range(particles):
            r1, r2 = rng.random(3), rng.random(3)
            velocities[index] = 0.7298 * (velocities[index] + 2.05 * r1 * (pbest[index] - old[index]) + 2.05 * r2 * (gbest - old[index]))
            positions[index] = np.clip(old[index] + velocities[index], -5, 5)
    logits = np.exp(gbest - np.max(gbest)); weights = logits / logits.sum()
    return {"seed": seed, "weights": weights.tolist(), "metrics": _ensemble_metrics(probs, labels, weights), "trajectory": trajectory, "queries": generations * particles}
def evaluate_fitted_ensemble(fitted: Mapping[str, Any], pool_probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Audit objective-fitted ensemble candidates on selection data without refitting."""
    probabilities = np.asarray(pool_probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    uniform = np.full(3, 1 / 3)
    result: dict[str, Any] = {"uniform": {"metrics": _ensemble_metrics(probabilities, labels, uniform), "weights": uniform.tolist()}}
    temp_entry = fitted["uniform_temperature"]
    temp = float(temp_entry["temperature"])
    uniform_probs = probabilities.mean(axis=0)
    logits = np.log(np.clip(uniform_probs, 1e-300, 1.0)) / temp
    scaled = np.exp(logits - logits.max(axis=1, keepdims=True)); scaled /= scaled.sum(axis=1, keepdims=True)
    result["uniform_temperature"] = {"metrics": _probability_metrics(scaled, labels), "temperature": temp}
    slsqp_entry = fitted["slsqp_weights"]
    slsqp_weights = np.asarray(slsqp_entry["weights"], dtype=np.float64)
    result["slsqp_weights"] = {"metrics": _ensemble_metrics(probabilities, labels, slsqp_weights), "weights": slsqp_weights.tolist()}
    pso_candidates = []
    for candidate in fitted["ensemble_pso"]:
        weights = np.asarray(candidate["weights"], dtype=np.float64)
        pso_candidates.append({**candidate, "selection_metrics": _ensemble_metrics(probabilities, labels, weights)})
    result["ensemble_pso"] = pso_candidates
    return result



def run_ensemble_methods(pool_probabilities: np.ndarray, labels: np.ndarray, *, swarm_seeds: Sequence[int] = SWARM_SEEDS) -> dict[str, Any]:
    probabilities = np.asarray(pool_probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[0] != 3:
        raise ProtocolError("ResNet ensemble pool must have shape (3,N,10)")
    uniform_weights = np.full(3, 1 / 3)
    result: dict[str, Any] = {"uniform": {"weights": uniform_weights.tolist(), "metrics": _ensemble_metrics(probabilities, labels, uniform_weights)}}
    result["uniform_temperature"] = _fit_temperature(probabilities, labels)
    result["slsqp_weights"] = _fit_slsqp(probabilities, labels)
    result["ensemble_pso"] = [_ensemble_pso(probabilities, labels, seed=int(seed)) for seed in swarm_seeds]
    return result


def _selection_metric(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, manifest: Mapping[str, Any], device: torch.device | str, *, maximize: bool = False) -> dict[str, float]:
    indices = manifest["roles"]["selection_val"]
    metrics, _ = evaluate_logits(model, images[indices], labels[indices], device, batch_size=128)
    return {"nll": float(metrics["nll"]), "accuracy": float(metrics["accuracy"]), "maximize": bool(maximize)}

class ResNetConvergenceAdapter:
    def __init__(self, *, workload_id: str, config: StudyConfig | Mapping[str, Any], run_root: str | os.PathLike[str], data_root: str | os.PathLike[str], device: str | torch.device = "cpu", allow_download: bool = False) -> None:
        if workload_id not in WORKLOAD_IDS:
            raise ProtocolError(f"unsupported ResNet workload: {workload_id}")
        self.workload_id = workload_id
        self.architecture = ARCHITECTURES[workload_id]
        self.config = config if isinstance(config, StudyConfig) else StudyConfig.from_dict(config)
        self.run_root = Path(run_root)
        self.data_root = Path(data_root)
        self.device = torch.device(device)
        self.allow_download = bool(allow_download)
        if str(self.device) not in {"cpu", "mps", "cuda"}:
            raise ProtocolError(
                "ResNet device must be cpu, mps, or cuda"
            )

    def _result_path(self) -> Path:
        return self.run_root / "workloads" / self.workload_id / "result.json"

    def _write_result(self, result: Mapping[str, Any]) -> dict[str, Any]:
        path = self._result_path()
        atomic_write_json(path, result)
        return dict(result)

    def _base_result(self) -> dict[str, Any]:
        return {"workload_id": self.workload_id, "family": "classification", "config": self.config.to_dict(), "manifests": {}, "provenance": {"architecture": self.architecture, "device": str(self.device)}, "baselines": {}, "arms": {}, "ensemble": {}, "development_selection": {}, "confirmation": {}, "integrity": {"official_test_opened": False}, "leakage_counters": {"official_test_data_loaded_before_freeze": False, "official_test_evaluations_before_freeze": 0, "official_test_construction": 0, "official_test_evaluations": 0}, "resource_ledger": {}, "artifact_hashes": {}}

    def run_prepare(self) -> dict[str, Any]:
        state = prepare_run(self.run_root, self.config) if not (self.run_root / "state.json").exists() else load_state(self.run_root)
        workload_root = self.run_root / "workloads" / self.workload_id
        workload_root.mkdir(parents=True, exist_ok=True)
        images, labels, manifest, provenance = prepare_cifar_data(self.data_root, split_seed=self.config.split_seed, allow_download=self.allow_download)
        atomic_write_json(workload_root / "manifest.json", manifest)
        atomic_write_json(workload_root / "provenance.json", provenance)
        result = self._base_result()
        result["manifests"], result["provenance"] = manifest, provenance | {"architecture": self.architecture, "device": str(self.device)}
        result["artifact_hashes"] = fingerprint_paths(self.run_root, [workload_root / "manifest.json", workload_root / "provenance.json"])
        self._write_result(result)
        persist_state(self.run_root, state)
        return result

    def _load_prepared(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any], dict[str, Any]]:
        workload_root = self.run_root / "workloads" / self.workload_id
        manifest_path = workload_root / "manifest.json"
        if not manifest_path.exists():
            self.run_prepare()
        images, labels, manifest, provenance = prepare_cifar_data(self.data_root, split_seed=self.config.split_seed, allow_download=False)
        result = json.loads(self._result_path().read_text()) if self._result_path().exists() else self._base_result()
        return images, labels, manifest, provenance, result

    def run_smoke(self) -> dict[str, Any]:
        images, labels, manifest, provenance, result = self._load_prepared()
        model = make_cifar_resnet(self.architecture, seed=BASE_SEEDS[0])
        checkpoint = self.run_root / "workloads" / self.workload_id / "smoke-baseline.pt"
        baseline = train_baseline(model, images, labels, manifest, seed=BASE_SEEDS[0], device=self.device, epochs=SMOKE_EPOCHS, batch_size=SMOKE_BATCH_SIZE, checkpoint_path=checkpoint)
        reloaded = make_cifar_resnet(self.architecture, seed=BASE_SEEDS[0] + 1)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        reloaded.load_state_dict(payload["state_dict"])
        block_index = 1 if self.architecture == "resnet18" else 2
        names = selected_parameter_names(reloaded, self.architecture)
        codec = SelectedResidualCodec(reloaded, names, projection_seed=PROJECTION_SEED)
        tiny = manifest["objective"][:SMOKE_OBJECTIVE_SAMPLES]
        cache = ResNetCache.build(reloaded, images[tiny], labels[tiny], block_index, batch_size=SMOKE_BATCH_SIZE)
        evaluator = CachedSuffixEvaluator(reloaded, cache, self.device)
        zero = torch.zeros(64)
        nonzero = (torch.arange(64, dtype=torch.float32) % 5 - 2) / 10
        residuals = [zero, nonzero, -nonzero]
        objective_results = [evaluator.objective(value, codec).to_dict() for value in residuals]
        parity = cached_full_parity(reloaded, images[tiny], cache)
        avgpool_parity = cached_avgpool_parity(reloaded, images[tiny], block_index=block_index, batch_size=SMOKE_BATCH_SIZE)
        search = _smoke_search(lambda residual: evaluator.objective(residual, codec), seed=SWARM_SEEDS[0], device=self.device)
        _, pool_probs = evaluate_logits(reloaded, images[tiny], labels[tiny], self.device, batch_size=SMOKE_BATCH_SIZE)
        ensemble = run_ensemble_methods(np.stack([pool_probs, pool_probs, pool_probs]), labels[tiny].numpy())
        result["integrity"] = {"cache_parity": parity, "avgpool_cache_parity": avgpool_parity, "objective_results": objective_results, "nonzero_changed": len({item["loss"] for item in objective_results}) > 1, "codec_names": list(names), "checkpoint_roundtrip": fingerprint_module(reloaded) == fingerprint_module(model)}
        result["resource_ledger"]["smoke"] = {"search_queries": search["evaluations"], "real_model": True, "backward_epochs": SMOKE_EPOCHS, "checkpoint": str(checkpoint.relative_to(self.run_root)), "ensemble_methods": sorted(ensemble)}
        result["leakage_counters"]["official_test_construction"] = 0
        self._write_result(result)
        return result

    def run_develop(self) -> dict[str, Any]:
        state = load_state(self.run_root)
        if state.state == StudyState.PREPARED:
            state.transition(StudyState.DEVELOPING)
        elif state.state != StudyState.DEVELOPING:
            raise ProtocolError(f"develop requires prepared/developing state, found {state.state.value}")
        persist_state(self.run_root, state)
        images, labels, manifest, provenance, result = self._load_prepared()
        workload_root = self.run_root / "workloads" / self.workload_id
        block_index = 1 if self.architecture == "resnet18" else 2
        for seed in BASE_SEEDS:
            model = make_cifar_resnet(self.architecture, seed=seed)
            checkpoint = workload_root / f"baseline-{seed}.pt"
            baseline = train_baseline(model, images, labels, manifest, seed=seed, device=self.device, checkpoint_path=checkpoint)
            state_dict = baseline.pop("state_dict")
            result["baselines"][str(seed)] = {**baseline, "checkpoint": str(checkpoint.relative_to(self.run_root)), "state_fingerprint": fingerprint_module(model)}
            names = selected_parameter_names(model, self.architecture)
            codec = SelectedResidualCodec(model, names, projection_seed=PROJECTION_SEED)
            objective_indices = manifest["objective"]
            objective_cache = ResNetCache.build(model, images[objective_indices], labels[objective_indices], block_index, batch_size=64)
            selection_cache = ResNetCache.build(model, images[manifest["roles"]["selection_val"]], labels[manifest["roles"]["selection_val"]], block_index, batch_size=128)
            evaluator = CachedSuffixEvaluator(model, objective_cache, self.device)
            validation = CachedSuffixEvaluator(model, selection_cache, self.device)
            for swarm_seed in SWARM_SEEDS:
                pso = run_residual_pso(lambda residual: evaluator.objective(residual, codec), codec, seed=swarm_seed, model=model, device=self.device, validation=lambda residual: validation.validation(residual, codec), objective_samples=OBJECTIVE_SAMPLES)
                random_search = run_equal_budget_random(lambda residual: evaluator.objective(residual, codec), codec, seed=swarm_seed, model=model, device=self.device, validation=lambda residual: validation.validation(residual, codec), objective_samples=OBJECTIVE_SAMPLES)
                pso_record = {"method": "feature_pso", "base_seed": seed, "swarm_seed": swarm_seed, "best_objective": pso.best_objective.to_dict(), "best_residual": pso.best_residual.cpu().tolist(), "endpoints": [endpoint.to_dict(include_vector=True) for endpoint in pso.endpoints if endpoint.generation in OBJECTIVE_CHECKPOINTS], "initial_residual": [0.0] * RESIDUAL_DIMENSION, "trajectory": [dict(item) for item in pso.trajectory], "counters": pso.counters.to_dict(), "failures": list(pso.failures)}
                random_record = {"method": "feature_random", "base_seed": seed, "swarm_seed": swarm_seed, "best_objective": random_search.best_objective.to_dict(), "best_residual": random_search.best_residual.cpu().tolist(), "endpoints": [endpoint.to_dict(include_vector=True) for endpoint in random_search.endpoints if endpoint.generation in OBJECTIVE_CHECKPOINTS], "initial_residual": [0.0] * RESIDUAL_DIMENSION, "trajectory": [dict(item) for item in random_search.trajectory], "counters": random_search.counters.to_dict(), "failures": list(random_search.failures)}
                result["arms"].setdefault("feature_pso", {}).setdefault(str(seed), {})[str(swarm_seed)] = pso_record
                result["arms"].setdefault("feature_random", {}).setdefault(str(seed), {})[str(swarm_seed)] = random_record
                pso_path = workload_root / f"feature-pso-{seed}-{swarm_seed}.json"
                random_path = workload_root / f"feature-random-{seed}-{swarm_seed}.json"
                atomic_write_json(pso_path, pso_record)
                atomic_write_json(random_path, random_record)
            # Controls each start from an untouched, independently reconstructed baseline.
            control_model = make_cifar_resnet(self.architecture, seed=seed + 10_000)
            control_model.load_state_dict(state_dict)
            control_cache = ResNetCache.build(control_model, images[objective_indices], labels[objective_indices], block_index, batch_size=64)
            feature_adam = run_feature_adam(control_model, control_cache, names, device=self.device)
            with torch.no_grad():
                for name, value in zip(names, feature_adam["final_parameters"]):
                    dict(control_model.named_parameters())[name].copy_(value.to(next(control_model.parameters()).device))
            head_model = make_cifar_resnet(self.architecture, seed=seed + 20_000)
            head_model.load_state_dict(state_dict)
            head_features, _ = _head_cache(head_model, images[objective_indices], block_index, self.device)
            head_adam = run_head_adam(head_model, head_features, labels[objective_indices], device=self.device)
            with torch.no_grad():
                head_model.fc.weight.copy_(head_adam["final_weight"].to(next(head_model.parameters()).device))
                head_model.fc.bias.copy_(head_adam["final_bias"].to(next(head_model.parameters()).device))
            result["arms"].setdefault("feature_adam", {})[str(seed)] = {**feature_adam, "base_seed": seed, "selection_final": _selection_metric(control_model, images, labels, manifest, self.device, maximize=False)}
            result["arms"].setdefault("head_adam", {})[str(seed)] = {**head_adam, "base_seed": seed, "selection_final": _selection_metric(head_model, images, labels, manifest, self.device, maximize=False)}
            feature_control_checkpoint = workload_root / f"feature-adam-{seed}.pt"
            head_control_checkpoint = workload_root / f"head-adam-{seed}.pt"
            result["arms"]["feature_adam"][str(seed)]["checkpoint"] = str(feature_control_checkpoint.relative_to(self.run_root))
            result["arms"]["head_adam"][str(seed)]["checkpoint"] = str(head_control_checkpoint.relative_to(self.run_root))
            result["arms"]["feature_adam"][str(seed)]["checkpoint_hash"] = _save_torch(feature_control_checkpoint, {"state_dict": _state_cpu(control_model), "base_seed": seed})
            result["arms"]["head_adam"][str(seed)]["checkpoint_hash"] = _save_torch(head_control_checkpoint, {"state_dict": _state_cpu(head_model), "base_seed": seed})
            atomic_write_json(workload_root / f"feature-adam-{seed}.json", result["arms"]["feature_adam"][str(seed)])
            atomic_write_json(workload_root / f"head-adam-{seed}.json", result["arms"]["head_adam"][str(seed)])
            result["integrity"][str(seed)] = {"selected_names": list(names), "selected_total_numel": codec.total_numel, "nonselected_state_fingerprint": fingerprint_nonselected_state(model, names), "objective_cache": objective_cache.source_fingerprint, "selection_cache": selection_cache.source_fingerprint}
        # Ensemble fit is objective-only; selection_val is audited independently.
        objective_pool: list[np.ndarray] = []
        selection_pool: list[np.ndarray] = []
        objective_labels = labels[manifest["objective"]]
        selection_labels = labels[manifest["roles"]["selection_val"]]
        for seed in BASE_SEEDS:
            model = make_cifar_resnet(self.architecture, seed=seed)
            payload = torch.load(workload_root / f"baseline-{seed}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(payload["state_dict"])
            _, objective_probs = evaluate_logits(model, images[manifest["objective"]], objective_labels, self.device)
            _, selection_probs = evaluate_logits(model, images[manifest["roles"]["selection_val"]], selection_labels, self.device)
            objective_pool.append(objective_probs)
            selection_pool.append(selection_probs)
        objective_ensemble = run_ensemble_methods(np.stack(objective_pool), objective_labels.numpy())
        selection_ensemble = evaluate_fitted_ensemble(objective_ensemble, np.stack(selection_pool), selection_labels.numpy())
        result["ensemble"] = {}
        for method in ("uniform", "uniform_temperature", "slsqp_weights", "ensemble_pso"):
            result["ensemble"][method] = {"objective": objective_ensemble[method], "selection": selection_ensemble[method], "fit_scope": "refine_search"}
        pso_selection = selection_ensemble["ensemble_pso"]
        selected_ensemble_pso = min(pso_selection, key=lambda item: (float(item["selection_metrics"]["nll"]), int(item["seed"])))
        result["development_selection"] = {"ensemble_pso": {"seed": int(selected_ensemble_pso["seed"]), "selection_nll": float(selected_ensemble_pso["selection_metrics"]["nll"]), "weights": list(selected_ensemble_pso["weights"])}}
        for seed in BASE_SEEDS:
            model = make_cifar_resnet(self.architecture, seed=seed)
            payload = torch.load(workload_root / f"baseline-{seed}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(payload["state_dict"])
            selected_for_seed: dict[str, Any] = {}
            for method in ("feature_pso", "feature_random"):
                candidates = result["arms"][method][str(seed)]
                ranked: list[
                    tuple[float, float, int, int, list[float], Mapping[str, Any]]
                ] = []
                for swarm_seed, record in candidates.items():
                    trajectory = record.get("trajectory", [])
                    if not trajectory:
                        raise ProtocolError(
                            f"missing trajectory for {method} base {seed}, "
                            f"swarm {swarm_seed}"
                        )
                    endpoint_by_generation = {
                        int(endpoint["generation"]): endpoint["residual"]
                        for endpoint in record.get("endpoints", [])
                    }
                    initial = trajectory[0].get("initial_validation")
                    if isinstance(initial, Mapping):
                        ranked.append(
                            (
                                float(initial["loss"]),
                                -float(initial.get("primary_metric") or 0.0),
                                0,
                                int(swarm_seed),
                                list(record["initial_residual"]),
                                record,
                            )
                        )
                    for row in trajectory:
                        validation_record = row.get("validation")
                        generation = int(row.get("generation", -1))
                        if not isinstance(validation_record, Mapping):
                            continue
                        if generation not in endpoint_by_generation:
                            raise ProtocolError(
                                f"missing endpoint vector for generation "
                                f"{generation}"
                            )
                        ranked.append(
                            (
                                float(validation_record["loss"]),
                                -float(
                                    validation_record.get("primary_metric")
                                    or 0.0
                                ),
                                generation,
                                int(swarm_seed),
                                list(endpoint_by_generation[generation]),
                                record,
                            )
                        )
                if not ranked or any(
                    not math.isfinite(item[0]) for item in ranked
                ):
                    raise ProtocolError(
                        f"missing finite selection checkpoints for {method} "
                        f"base {seed}"
                    )
                ranked.sort(key=lambda item: item[:4])
                (
                    selected_loss,
                    negative_accuracy,
                    selected_generation,
                    selected_swarm,
                    selected_residual,
                    _,
                ) = ranked[0]
                codec = SelectedResidualCodec(
                    model,
                    names,
                    projection_seed=PROJECTION_SEED,
                )
                residual = torch.as_tensor(
                    selected_residual,
                    dtype=torch.float32,
                )
                codec.apply_residual(model, residual)
                selected_checkpoint = (
                    workload_root / f"selected-{method}-{seed}.pt"
                )
                selected_hash = _save_torch(
                    selected_checkpoint,
                    {
                        "state_dict": _state_cpu(model),
                        "residual": residual,
                        "base_seed": seed,
                        "swarm_seed": selected_swarm,
                        "generation": selected_generation,
                    },
                )
                selected_for_seed[method] = {
                    "method": method,
                    "base_seed": seed,
                    "swarm_seed": selected_swarm,
                    "generation": selected_generation,
                    "selection_nll": selected_loss,
                    "selection_accuracy": -negative_accuracy,
                    "residual": residual.tolist(),
                    "checkpoint": str(
                        selected_checkpoint.relative_to(self.run_root)
                    ),
                    "checkpoint_hash": selected_hash,
                    "all_checkpoint_selection_nll": {
                        f"{item[3]}:{item[2]}": item[0] for item in ranked
                    },
                }
                codec.restore_base(model)
            result["development_selection"][str(seed)] = selected_for_seed
        arm_files = sorted(path for path in workload_root.glob("*.json") if path.name not in {"result.json", "manifest.json", "provenance.json"})
        artifact_paths = [workload_root / "manifest.json", workload_root / "provenance.json"] + [workload_root / f"baseline-{seed}.pt" for seed in BASE_SEEDS] + [workload_root / f"selected-feature_pso-{seed}.pt" for seed in BASE_SEEDS] + [workload_root / f"selected-feature_random-{seed}.pt" for seed in BASE_SEEDS] + [workload_root / f"feature-adam-{seed}.pt" for seed in BASE_SEEDS] + [workload_root / f"head-adam-{seed}.pt" for seed in BASE_SEEDS] + arm_files
        relative_artifacts = [str(path.relative_to(self.run_root)) for path in artifact_paths]
        result["artifact_hashes"] = fingerprint_paths(self.run_root, relative_artifacts)
        result["integrity"]["official_test_opened"] = False
        self._write_result(result)
        persist_state(self.run_root, state)
        return result

    def run_confirm(self) -> dict[str, Any]:
        state = load_state(self.run_root)
        begin_confirmation(self.run_root, state)
        persist_state(self.run_root, state)
        images, labels, manifest, provenance, result = self._load_prepared()
        mean, std = provenance["mean"], provenance["std"]
        test_images, test_labels = load_official_test_data(self.data_root, self.run_root, allow_download=self.allow_download, mean=mean, std=std)
        workload_root = self.run_root / "workloads" / self.workload_id
        # The fingerprint audit uses the decoded test bytes exactly once, after the seal.
        train_raw = ((images * torch.as_tensor(std)[None, :, None, None] + torch.as_tensor(mean)[None, :, None, None]) * 255.0).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        test_raw = ((test_images * torch.as_tensor(std)[None, :, None, None] + torch.as_tensor(mean)[None, :, None, None]) * 255.0).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        confirmation: dict[str, Any] = {"test_samples": len(test_labels), "base": {}, "selected_feature_pso": {}, "selected_feature_random": {}, "arms": {}, "ensemble": {}, "test_construction": 1, "duplicate_audit": audit_test_duplicates(train_raw, test_raw)}
        test_pool: list[np.ndarray] = []
        test_forward_passes = 0
        for seed in BASE_SEEDS:
            base_model = make_cifar_resnet(self.architecture, seed=seed)
            payload = torch.load(workload_root / f"baseline-{seed}.pt", map_location="cpu", weights_only=True)
            base_model.load_state_dict(payload["state_dict"])
            base_metrics, base_probs = evaluate_logits(base_model, test_images, test_labels, self.device)
            test_forward_passes += 1
            test_pool.append(base_probs)
            base_prediction_path = workload_root / f"test-predictions-base-{seed}.pt"
            confirmation["base"][str(seed)] = {"metrics": base_metrics, "prediction_artifact": str(base_prediction_path.relative_to(self.run_root))}
            _save_torch(base_prediction_path, {"probabilities": base_probs, "targets": test_labels.cpu()})
            selected_metrics_by_method: dict[str, dict[str, float]] = {}
            for method in ("feature_pso", "feature_random"):
                selected_payload = torch.load(workload_root / f"selected-{method}-{seed}.pt", map_location="cpu", weights_only=True)
                selected_model = make_cifar_resnet(self.architecture, seed=seed + 1)
                selected_model.load_state_dict(selected_payload["state_dict"])
                selected_metrics, selected_probs = evaluate_logits(selected_model, test_images, test_labels, self.device)
                test_forward_passes += 1
                selected_metrics_by_method[method] = selected_metrics
                prediction_path = workload_root / f"test-predictions-selected-{method}-{seed}.pt"
                confirmation.setdefault(f"selected_{method}", {})[str(seed)] = {"metrics": selected_metrics, "prediction_artifact": str(prediction_path.relative_to(self.run_root))}
                _save_torch(prediction_path, {"probabilities": selected_probs, "targets": test_labels.cpu(), "residual": selected_payload["residual"]})
            arm_confirmation: dict[str, Any] = {"feature_pso": {}, "feature_random": {}, "feature_adam": {}, "head_adam": {}}
            for method in ("feature_adam", "head_adam"):
                control_payload = torch.load(workload_root / f"{method.replace('_', '-')}-{seed}.pt", map_location="cpu", weights_only=True)
                control_model = make_cifar_resnet(self.architecture, seed=seed)
                control_model.load_state_dict(control_payload["state_dict"])
                arm_metrics, arm_probs = evaluate_logits(control_model, test_images, test_labels, self.device)
                test_forward_passes += 1
                prediction_path = workload_root / f"test-predictions-{method}-{seed}.pt"
                arm_confirmation[method] = {"metrics": arm_metrics, "base_seed": seed, "prediction_artifact": str(prediction_path.relative_to(self.run_root))}
                _save_torch(prediction_path, {"probabilities": arm_probs, "targets": test_labels.cpu()})
            for method in ("feature_pso", "feature_random"):
                selected = result["development_selection"][str(seed)][method]
                prediction_path = workload_root / f"test-predictions-selected-{method}-{seed}.pt"
                arm_confirmation[method][str(selected["swarm_seed"])] = {"metrics": selected_metrics_by_method[method], "base_seed": seed, "swarm_seed": selected["swarm_seed"], "prediction_artifact": str(prediction_path.relative_to(self.run_root))}
            confirmation["arms"][str(seed)] = arm_confirmation
        test_pool_array = np.stack(test_pool)
        test_labels_np = test_labels.numpy()
        pool_prediction_path = workload_root / "test-predictions-base-pool.pt"
        _save_torch(pool_prediction_path, {"probabilities": test_pool_array, "targets": test_labels.cpu()})
        pool_artifact = str(pool_prediction_path.relative_to(self.run_root))
        uniform = np.full(3, 1 / 3)
        uniform_probs = np.einsum("m,mnk->nk", uniform, test_pool_array)
        uniform_path = workload_root / "test-predictions-ensemble-uniform.pt"
        _save_torch(uniform_path, {"probabilities": uniform_probs, "targets": test_labels.cpu(), "weights": uniform})
        confirmation["ensemble"]["uniform"] = {"metrics": _probability_metrics(uniform_probs, test_labels_np), "weights": uniform.tolist(), "prediction_artifact": str(uniform_path.relative_to(self.run_root))}
        for method, entry in result.get("ensemble", {}).items():
            if method == "uniform":
                continue
            objective_entry = entry.get("objective", {}) if isinstance(entry, Mapping) else {}
            if method == "slsqp_weights" and objective_entry.get("weights"):
                weights = np.asarray(objective_entry["weights"], dtype=np.float64)
                mixed = np.einsum("m,mnk->nk", weights, test_pool_array)
                path = workload_root / "test-predictions-ensemble-slsqp_weights.pt"
                _save_torch(path, {"probabilities": mixed, "targets": test_labels.cpu(), "weights": weights})
                confirmation["ensemble"][method] = {"metrics": _probability_metrics(mixed, test_labels_np), "weights": weights.tolist(), "prediction_artifact": str(path.relative_to(self.run_root))}
            elif method == "uniform_temperature" and objective_entry.get("temperature"):
                temp = float(objective_entry["temperature"])
                logits = np.log(np.clip(uniform_probs, 1e-300, 1.0)) / temp
                scaled = np.exp(logits - logits.max(axis=1, keepdims=True)); scaled /= scaled.sum(axis=1, keepdims=True)
                path = workload_root / "test-predictions-ensemble-uniform_temperature.pt"
                _save_torch(path, {"probabilities": scaled, "targets": test_labels.cpu(), "temperature": temp})
                confirmation["ensemble"][method] = {"metrics": _probability_metrics(scaled, test_labels_np), "temperature": temp, "prediction_artifact": str(path.relative_to(self.run_root))}
            elif method == "ensemble_pso" and objective_entry:
                candidates = objective_entry if isinstance(objective_entry, list) else [objective_entry]
                selected_seed = int(result.get("development_selection", {}).get("ensemble_pso", {}).get("seed", SWARM_SEEDS[0]))
                selected = next((item for item in candidates if int(item.get("seed", -1)) == selected_seed), candidates[0])
                weights = np.asarray(selected.get("weights", uniform), dtype=np.float64)
                mixed = np.einsum("m,mnk->nk", weights, test_pool_array)
                path = workload_root / "test-predictions-ensemble-pso.pt"
                _save_torch(path, {"probabilities": mixed, "targets": test_labels.cpu(), "weights": weights, "seed": selected.get("seed")})
                confirmation["ensemble"][method] = {"metrics": _probability_metrics(mixed, test_labels_np), "weights": weights.tolist(), "seed": selected.get("seed"), "prediction_artifact": str(path.relative_to(self.run_root))}
        result["confirmation"] = confirmation
        result["leakage_counters"]["official_test_construction"] = 1
        result["leakage_counters"]["official_test_evaluations"] = test_forward_passes
        prediction_files = [str(path.relative_to(self.run_root)) for path in workload_root.glob("test-predictions-*.pt")]
        result["artifact_hashes"].update(fingerprint_paths(self.run_root, prediction_files))
        self._write_result(result)
        finish_confirmation(state, success=True)
        persist_state(self.run_root, state)
        return result

    def run_phase(self, phase: str) -> dict[str, Any]:
        if phase == "prepare":
            return self.run_prepare()
        if phase == "smoke":
            return self.run_smoke()
        if phase == "develop":
            return self.run_develop()
        if phase == "confirm":
            return self.run_confirm()
        if phase == "publish":
            return json.loads(self._result_path().read_text(encoding="utf-8"))
        raise ProtocolError(f"unsupported ResNet phase: {phase}")


def create_adapter(*, workload_id: str, config: StudyConfig | Mapping[str, Any], run_root: str | os.PathLike[str], data_root: str | os.PathLike[str], device: str | torch.device = "cpu", allow_download: bool = False, **_: Any) -> ResNetConvergenceAdapter:
    return ResNetConvergenceAdapter(workload_id=workload_id, config=config, run_root=run_root, data_root=data_root, device=device, allow_download=allow_download)


__all__ = [
    "BATCH_SIZE", "BASE_SEEDS", "CIFARSubset", "CachedSuffixEvaluator", "ResNetCache", "ResNetConvergenceAdapter", "SMOKE_EPOCHS", "TestSealError", "audit_test_duplicates", "build_cifar_manifests", "cached_avgpool_parity", "cached_full_parity", "cached_residual_parity", "classification_metrics", "create_adapter", "evaluate_fitted_ensemble", "evaluate_logits", "head_parameter_names", "load_official_test_data", "make_cifar_resnet", "prepare_cifar_data", "run_ensemble_methods", "run_feature_adam", "run_head_adam", "selected_parameter_names", "train_baseline",
]
