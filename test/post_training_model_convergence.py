"""Common protocol and search machinery for post-training convergence studies.

This module deliberately contains no model or dataset implementation.  Workload
adapters register factories at runtime (the YOLO adapter is therefore imported
only when its factory is invoked).  The search code operates on a residual
vector and a scalar objective so that classification and detection adapters can
share exactly the same accounting and lifecycle rules.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import datetime as _datetime
import hashlib
import io
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import tempfile
import sys
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence, runtime_checkable
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import torch
import torch.nn as nn

from pso.optimizer import _RandomSource
from pso.plugins import ConstrictionMovement, IterationContext, SwarmState


PROTOCOL_VERSION = "post-training-model-convergence-1.0.0"
DEFAULT_WORKLOAD_IDS = (
    "cifar10_resnet18",
    "cifar10_resnet50",
    "voc_yolo11n",
)
BASE_SEEDS = (501, 502, 503)
SWARM_SEEDS = (601, 602, 603)
SPLIT_SEED = 20260908
PROJECTION_SEED = 20260909
BOOTSTRAP_SEED = 20260910
PARTICLE_COUNT = 12
PSO_GENERATIONS = 60
RANDOM_CANDIDATES = PARTICLE_COUNT * PSO_GENERATIONS
RESIDUAL_DIMENSION = 64
RESIDUAL_BOUND = 1.0
INITIAL_RADIUS = 0.25
OBJECTIVE_CHECKPOINTS = (0, 10, 20, 30, 40, 50, 60)


class ProtocolError(ValueError):
    """Raised when an artifact or callback violates the study contract."""


class ObjectiveEvaluationError(RuntimeError):
    """Raised when a candidate objective cannot be evaluated."""


class StateTransitionError(RuntimeError):
    """Raised for an invalid or repeated study lifecycle transition."""


class SealError(RuntimeError):
    """Raised when a frozen artifact set cannot be confirmed."""


@dataclasses.dataclass(frozen=True)
class StudyConfig:
    """Immutable protocol configuration shared by all workload adapters."""

    protocol_version: str = PROTOCOL_VERSION
    split_seed: int = SPLIT_SEED
    base_seeds: tuple[int, ...] = BASE_SEEDS
    swarm_seeds: tuple[int, ...] = SWARM_SEEDS
    projection_seed: int = PROJECTION_SEED
    bootstrap_seed: int = BOOTSTRAP_SEED
    particle_count: int = PARTICLE_COUNT
    pso_generations: int = PSO_GENERATIONS
    residual_dimension: int = RESIDUAL_DIMENSION
    residual_bound: float = RESIDUAL_BOUND
    initial_radius: float = INITIAL_RADIUS
    objective_checkpoints: tuple[int, ...] = OBJECTIVE_CHECKPOINTS
    device: str = "cpu"
    workload_ids: tuple[str, ...] = DEFAULT_WORKLOAD_IDS

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ProtocolError("protocol_version does not match the approved protocol")
        if tuple(self.base_seeds) != BASE_SEEDS:
            raise ProtocolError("base_seeds are fixed at 501, 502, 503")
        if tuple(self.swarm_seeds) != SWARM_SEEDS:
            raise ProtocolError("swarm_seeds are fixed at 601, 602, 603")
        for name, value in (
            ("split_seed", self.split_seed),
            ("projection_seed", self.projection_seed),
            ("bootstrap_seed", self.bootstrap_seed),
            ("particle_count", self.particle_count),
            ("pso_generations", self.pso_generations),
            ("residual_dimension", self.residual_dimension),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ProtocolError(f"{name} must be an integer")
        if self.split_seed != SPLIT_SEED:
            raise ProtocolError(f"split_seed is fixed at {SPLIT_SEED}")
        if self.projection_seed != PROJECTION_SEED:
            raise ProtocolError(
                f"projection_seed is fixed at {PROJECTION_SEED}"
            )
        if self.bootstrap_seed != BOOTSTRAP_SEED:
            raise ProtocolError(
                f"bootstrap_seed is fixed at {BOOTSTRAP_SEED}"
            )
        if self.particle_count != PARTICLE_COUNT or self.pso_generations != PSO_GENERATIONS:
            raise ProtocolError("the primary PSO budget is fixed at 12x60")
        if self.residual_dimension != RESIDUAL_DIMENSION:
            raise ProtocolError("the residual dimension is fixed at 64")
        if self.residual_bound != RESIDUAL_BOUND or self.initial_radius != INITIAL_RADIUS:
            raise ProtocolError("residual bounds and initialization radius are fixed")
        if tuple(self.objective_checkpoints) != OBJECTIVE_CHECKPOINTS:
            raise ProtocolError("objective checkpoints are fixed")
        if self.device not in {"cpu", "mps", "cuda"}:
            raise ProtocolError("device must be cpu, mps, or cuda")
        if not self.workload_ids:
            raise ProtocolError("at least one workload must be registered")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StudyConfig":
        data = dict(value)
        for key in ("base_seeds", "swarm_seeds", "objective_checkpoints", "workload_ids"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class ObjectiveResult:
    """One finite scalar objective evaluation and its exact sample accounting."""

    loss: float
    samples: int
    forward_passes: int = 0
    backward_passes: int = 0
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.loss, bool) or not isinstance(self.loss, (int, float)):
            raise ProtocolError("objective loss must be a scalar number")
        if not math.isfinite(float(self.loss)):
            raise FloatingPointError(f"non-finite objective loss: {self.loss!r}")
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 0:
            raise ProtocolError("objective samples must be a nonnegative integer")
        for field in ("forward_passes", "backward_passes"):
            count = getattr(self, field)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ProtocolError(f"{field} must be a nonnegative integer")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def coerce(cls, value: "ObjectiveResult | float | int | torch.Tensor") -> "ObjectiveResult":
        if isinstance(value, cls):
            return value
        if torch.is_tensor(value):
            if value.ndim != 0:
                raise ProtocolError("objective tensor result must be scalar")
            value = value.detach().item()
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ProtocolError("objective callback must return ObjectiveResult or scalar float")
        return cls(loss=float(value), samples=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": float(self.loss),
            "samples": self.samples,
            "forward_passes": self.forward_passes,
            "backward_passes": self.backward_passes,
            "metadata": _jsonable(self.metadata),
        }


@dataclasses.dataclass(frozen=True)
class AuditResult:
    """Read-only validation/audit result; it is never used by the optimizer."""

    loss: float
    primary_metric: float | None = None
    samples: int = 0
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.loss)):
            raise FloatingPointError("audit loss must be finite")
        if self.primary_metric is not None and not math.isfinite(float(self.primary_metric)):
            raise FloatingPointError("audit metric must be finite")
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 0:
            raise ProtocolError("audit samples must be a nonnegative integer")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": float(self.loss),
            "primary_metric": self.primary_metric,
            "samples": self.samples,
            "metadata": _jsonable(self.metadata),
        }


@runtime_checkable
class ObjectiveCallback(Protocol):
    def __call__(self, residual: torch.Tensor) -> ObjectiveResult | float: ...


@runtime_checkable
class AuditCallback(Protocol):
    def __call__(self, residual: torch.Tensor) -> AuditResult | Mapping[str, Any] | float: ...


@dataclasses.dataclass
class ResourceCounters:
    """Counters for every expensive operation, including failed candidates."""

    objective_queries: int = 0
    objective_samples: int = 0
    objective_failures: int = 0
    objective_forward_passes: int = 0
    objective_backward_passes: int = 0
    validation_evaluations: int = 0
    validation_samples: int = 0
    gradient_updates: int = 0
    gradient_samples: int = 0
    full_forward_passes: int = 0
    test_forward_passes: int = 0
    cache_forward_passes: int = 0
    fusion_evaluations: int = 0
    solver_evaluations: int = 0
    bytes_written: int = 0
    wall_time_seconds: float = 0.0

    def record_objective(self, result: ObjectiveResult | None, *, failed: bool, fallback_samples: int = 0) -> None:
        self.objective_queries += 1
        if failed:
            self.objective_failures += 1
        if result is None:
            self.objective_samples += fallback_samples
            return
        self.objective_samples += result.samples
        self.objective_forward_passes += result.forward_passes
        self.objective_backward_passes += result.backward_passes

    def merge(self, other: "ResourceCounters") -> None:
        for field in dataclasses.fields(self):
            setattr(self, field.name, getattr(self, field.name) + getattr(other, field.name))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CandidateEndpoint:
    generation: int
    particle_index: int
    residual: torch.Tensor
    objective: ObjectiveResult

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.residual):
            raise ProtocolError("endpoint residual must be a tensor")
        if self.residual.ndim != 1:
            raise ProtocolError("endpoint residual must be one-dimensional")
        object.__setattr__(self, "residual", self.residual.detach().clone())

    def to_dict(self, *, include_vector: bool = False) -> dict[str, Any]:
        value = {
            "generation": self.generation,
            "particle_index": self.particle_index,
            "objective": self.objective.to_dict(),
        }
        if include_vector:
            value["residual"] = self.residual.detach().cpu().tolist()
        return value


@dataclasses.dataclass
class SearchResult:
    method: str
    seed: int
    generations: int
    particles: int
    best_residual: torch.Tensor | None
    best_objective: ObjectiveResult | None
    endpoints: tuple[CandidateEndpoint, ...]
    trajectory: tuple[dict[str, Any], ...]
    counters: ResourceCounters
    failures: tuple[str, ...] = ()

    @property
    def objective_queries(self) -> int:
        return self.counters.objective_queries

    def to_dict(self, *, include_vectors: bool = False) -> dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "generations": self.generations,
            "particles": self.particles,
            "best_objective": None if self.best_objective is None else self.best_objective.to_dict(),
            "best_residual": None
            if self.best_residual is None or not include_vectors
            else self.best_residual.detach().cpu().tolist(),
            "endpoints": [e.to_dict(include_vector=include_vectors) for e in self.endpoints],
            "trajectory": [_jsonable(row) for row in self.trajectory],
            "counters": self.counters.to_dict(),
            "failures": list(self.failures),
        }


# Integer arithmetic, rather than Python's process-randomized hash(), defines
# the projection and makes it reproducible across machines and processes.
def projection_salt(names: Sequence[str], seed: int = PROJECTION_SEED) -> int:
    payload = (str(seed) + "\0" + "\0".join(names)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0xFFFFFFFF


def _projection_arrays(total: int, dimension: int, salt: int) -> tuple[torch.Tensor, torch.Tensor]:
    indices = []
    signs = []
    for j in range(total):
        first = ((j + 1) * 0x9E3779B1 + salt * 0x85EBCA77) & 0xFFFFFFFF
        second = ((j + 1) * 0xC2B2AE3D + salt * 0x27D4EB2F) & 0xFFFFFFFF
        indices.append(first % dimension)
        signs.append(1.0 if (second & 1) == 0 else -1.0)
    return torch.tensor(indices, dtype=torch.long), torch.tensor(signs, dtype=torch.float32)


class SelectedResidualCodec:
    """Exact-name, immutable residual codec for one selected parameter layout.

    The projection is sparse by construction: each selected scalar reads one
    latent coordinate and one deterministic sign.  No dense ``D x 64`` matrix
    is allocated.  ``apply`` always starts and ends at the codec's fp32 base
    state, including when the callback raises.
    """

    dimension = RESIDUAL_DIMENSION

    def __init__(
        self,
        model: nn.Module,
        names: Sequence[str] | None = None,
        *,
        selected_names: Sequence[str] | None = None,
        projection_seed: int = PROJECTION_SEED,
    ) -> None:
        if names is None:
            names = selected_names
        elif selected_names is not None and tuple(names) != tuple(selected_names):
            raise ProtocolError("names and selected_names disagree")
        if names is None:
            raise ProtocolError("selected parameter names are required")
        if tuple(names) != tuple(dict.fromkeys(names)) or not names:
            raise ProtocolError("selected parameter names must be nonempty and unique")
        named = dict(model.named_parameters())
        unknown = [name for name in names if name not in named]
        if unknown:
            raise ProtocolError(f"selected parameter names are absent: {unknown}")
        tensors: list[torch.Tensor] = []
        shapes: list[tuple[int, ...]] = []
        offsets: list[int] = []
        scales: list[float] = []
        cursor = 0
        for name in names:
            parameter = named[name]
            if not parameter.is_floating_point():
                raise ProtocolError(f"selected parameter is not floating point: {name}")
            value = parameter.detach().to(device="cpu", dtype=torch.float32).clone()
            tensors.append(value)
            shape = tuple(value.shape)
            shapes.append(shape)
            offsets.append(cursor)
            cursor += value.numel()
            rms = float(torch.sqrt(torch.mean(value * value))) if value.numel() else 0.0
            scales.append(0.05 * max(rms, 0.01))
        indices, signs = _projection_arrays(cursor, self.dimension, projection_salt(names, projection_seed))
        self._names = tuple(names)
        self._shapes = tuple(shapes)
        self._offsets = tuple(offsets)
        self._base_values = tuple(tensors)
        self._scales = tuple(scales)
        self._projection_indices = indices
        self._projection_signs = signs
        self._total_numel = cursor
        self._projection_seed = int(projection_seed)
        self._name_to_position = MappingProxyType({name: i for i, name in enumerate(self._names)})
        self._nonselected_base = MappingProxyType(
            {
                name: parameter.detach().to(device="cpu").clone()
                for name, parameter in named.items()
                if name not in self._name_to_position
            }
        )

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        return self._shapes

    @property
    def offsets(self) -> tuple[int, ...]:
        return self._offsets

    @property
    def total_numel(self) -> int:
        return self._total_numel

    @property
    def projection_seed(self) -> int:
        return self._projection_seed

    @property
    def scales(self) -> tuple[float, ...]:
        return self._scales

    @property
    def base_values(self) -> tuple[torch.Tensor, ...]:
        return tuple(value.clone() for value in self._base_values)

    @property
    def projection_indices(self) -> torch.Tensor:
        return self._projection_indices.clone()

    def _validate_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(residual):
            raise ProtocolError("residual must be a tensor")
        if residual.ndim != 1 or residual.numel() != self.dimension:
            raise ProtocolError(f"residual must have shape ({self.dimension},)")
        if not residual.is_floating_point():
            residual = residual.float()
        if not bool(torch.isfinite(residual).all().item()):
            raise FloatingPointError("residual contains non-finite values")
        if bool(((residual < -RESIDUAL_BOUND) | (residual > RESIDUAL_BOUND)).any().item()):
            raise ProtocolError(f"residual must lie in [{-RESIDUAL_BOUND}, {RESIDUAL_BOUND}]")
        return residual

    def residual_values(self, residual: torch.Tensor) -> tuple[torch.Tensor, ...]:
        z = self._validate_residual(residual)
        device, dtype = z.device, z.dtype
        if not bool(torch.count_nonzero(z).item()):
            return tuple(base.to(device=device, dtype=dtype).clone() for base in self._base_values)
        indices = self._projection_indices.to(device=device)
        signs = self._projection_signs.to(device=device, dtype=dtype)
        values: list[torch.Tensor] = []
        for index, (base, offset) in enumerate(zip(self._base_values, self._offsets)):
            end = offset + base.numel()
            chosen = z.index_select(0, indices[offset:end]) * signs[offset:end]
            values.append(base.to(device=device, dtype=dtype).view(-1).add(chosen * self._scales[index]).view(base.shape))
        return tuple(values)

    def residual_delta(self, residual: torch.Tensor) -> torch.Tensor:
        z = self._validate_residual(residual)
        indices = self._projection_indices.to(device=z.device)
        signs = self._projection_signs.to(device=z.device, dtype=z.dtype)
        scale_parts = [
            torch.full((base.numel(),), self._scales[i], device=z.device, dtype=z.dtype)
            for i, base in enumerate(self._base_values)
        ]
        scales = torch.cat(scale_parts) if scale_parts else torch.empty(0, device=z.device, dtype=z.dtype)
        return z.index_select(0, indices) * signs * scales

    def decode(self, residual: torch.Tensor) -> torch.Tensor:
        """Return the selected parameters' candidate fp32/latent-device vector."""
        return torch.cat([value.reshape(-1) for value in self.residual_values(residual)])

    def decode_delta(self, residual: torch.Tensor) -> torch.Tensor:
        return self.residual_delta(residual)

    def zero_residual(self, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(self.dimension, device=device, dtype=dtype)

    def _check_model(self, model: nn.Module) -> dict[str, nn.Parameter]:
        named = dict(model.named_parameters())
        missing = [name for name in self._names if name not in named]
        if missing:
            raise ProtocolError(f"model is missing selected names: {missing}")
        return named

    def restore_base(self, model: nn.Module) -> None:
        named = self._check_model(model)
        with torch.no_grad():
            for name, base in zip(self._names, self._base_values):
                target = named[name]
                target.copy_(base.to(device=target.device, dtype=target.dtype).view_as(target))

    def apply_residual(self, model: nn.Module, residual: torch.Tensor) -> None:
        named = self._check_model(model)
        values = self.residual_values(residual)
        with torch.no_grad():
            for name, value in zip(self._names, values):
                target = named[name]
                target.copy_(value.to(device=target.device, dtype=target.dtype).view_as(target))

    def apply(self, model: nn.Module, residual: torch.Tensor) -> None:
        """Apply a candidate in-place; call ``restore_base`` after evaluation."""
        self.apply_residual(model, residual)

    @contextlib.contextmanager
    def applied(self, model: nn.Module, residual: torch.Tensor) -> Iterator[nn.Module]:
        """Apply one candidate without copying the frozen model per query.

        Selected parameters and buffers are restored with ``copy_`` and module
        modes are preserved. Non-selected parameter version counters provide a
        cheap mutation guard; adapters additionally verify full state hashes at
        run boundaries, as required by the protocol.
        """
        z = self._validate_residual(residual)
        parameters = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        missing = [name for name in self._names if name not in parameters]
        if missing:
            raise ProtocolError(f"model is missing selected names: {missing}")
        selected = set(self._names)
        nonselected_versions = {
            name: parameter._version
            for name, parameter in parameters.items()
            if name not in selected
        }
        buffer_snapshot = {
            name: value.detach().clone()
            for name, value in buffers.items()
        }
        modes = {name: child.training for name, child in model.named_modules()}
        try:
            self.restore_base(model)
            model.eval()
            self.apply_residual(model, z)
            if not bool(torch.count_nonzero(z).item()):
                for name, base in zip(self._names, self._base_values):
                    expected = base.to(
                        device=parameters[name].device,
                        dtype=parameters[name].dtype,
                    )
                    if not torch.equal(parameters[name].detach(), expected):
                        raise ProtocolError(
                            "zero residual did not preserve exact baseline parameters"
                        )
            yield model
            changed = [
                name
                for name, parameter in parameters.items()
                if name not in selected
                and parameter._version != nonselected_versions[name]
            ]
            if changed:
                raise ProtocolError(
                    f"candidate mutated non-selected parameters: {changed[:3]}"
                )
        finally:
            changed_now = [
                name
                for name, parameter in parameters.items()
                if name not in selected
                and parameter._version != nonselected_versions[name]
            ]
            with torch.no_grad():
                for name in changed_now:
                    parameters[name].copy_(
                        self._nonselected_base[name].to(
                            device=parameters[name].device,
                            dtype=parameters[name].dtype,
                        )
                    )
            with torch.no_grad():
                for name, saved in buffer_snapshot.items():
                    buffers[name].copy_(
                        saved.to(device=buffers[name].device, dtype=buffers[name].dtype)
                    )
            for name, child in model.named_modules():
                child.train(bool(modes[name]))
            self.restore_base(model)

    def evaluate(self, model: nn.Module, residual: torch.Tensor, callback: Callable[[], Any]) -> Any:
        with self.applied(model, residual):
            return callback()

    def selected_state_fingerprint(self, model: nn.Module) -> str:
        named = self._check_model(model)
        return _fingerprint_tensors((name, named[name]) for name in self._names)


@dataclasses.dataclass(frozen=True)
class _Evaluation:
    residual: torch.Tensor
    objective: ObjectiveResult | None
    error: str | None


def _evaluate_candidate(
    objective: ObjectiveCallback,
    residual: torch.Tensor,
    counters: ResourceCounters,
    failures: list[str],
    *,
    codec: SelectedResidualCodec | None,
    model: nn.Module | None,
    fallback_samples: int,
) -> _Evaluation:
    candidate = residual.detach().clone()
    try:
        if codec is not None and model is not None:
            with codec.applied(model, candidate):
                value = objective(candidate.clone())
        else:
            value = objective(candidate.clone())
        result = ObjectiveResult.coerce(value)
        counters.record_objective(result, failed=False)
        return _Evaluation(candidate, result, None)
    except Exception as exc:
        counters.record_objective(None, failed=True, fallback_samples=fallback_samples)
        message = f"{type(exc).__name__}: {exc}"
        failures.append(message)
        return _Evaluation(candidate, None, message)


def _initial_positions(rng: _RandomSource, particles: int, dimension: int, device: torch.device) -> list[torch.Tensor]:
    if particles != PARTICLE_COUNT:
        raise ProtocolError("the convergence protocol requires exactly 12 particles")
    positions = [torch.zeros(dimension, device=device, dtype=torch.float32)]
    for _ in range(5):
        sample = rng.uniform((dimension,), -INITIAL_RADIUS, INITIAL_RADIUS, device=device, dtype=torch.float32)
        positions.extend((sample, -sample))
    positions.append(rng.uniform((dimension,), -INITIAL_RADIUS, INITIAL_RADIUS, device=device, dtype=torch.float32))
    return [position.detach().clone() for position in positions]


def _is_strictly_better(loss: float, incumbent: float | None) -> bool:
    if not math.isfinite(loss):
        raise FloatingPointError("objective comparator received a non-finite loss")
    return incumbent is None or loss < incumbent


def _validation_snapshot(model: nn.Module | None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        snapshot["torch_cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    if model is not None:
        snapshot["state"] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        snapshot["training"] = {module: child.training for module, child in model.named_modules()}
    try:
        import numpy as np
        snapshot["numpy"] = copy.deepcopy(np.random.get_state())
    except ImportError:
        pass
    return snapshot


def _restore_validation_snapshot(model: nn.Module | None, snapshot: Mapping[str, Any]) -> None:
    random.setstate(snapshot["python"])
    torch.set_rng_state(snapshot["torch_cpu"])
    if "torch_cuda" in snapshot:
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])
    if "numpy" in snapshot:
        import numpy as np
        np.random.set_state(snapshot["numpy"])
    if model is not None:
        model.load_state_dict(snapshot["state"], strict=True)
        for name, child in model.named_modules():
            child.train(bool(snapshot["training"][name]))


@contextlib.contextmanager
def state_neutral_audit(model: nn.Module | None = None) -> Iterator[None]:
    """Preserve model parameters, buffers, modes, and host/device RNG state."""
    snapshot = _validation_snapshot(model)
    try:
        if model is not None:
            model.eval()
        with torch.no_grad():
            yield
    finally:
        _restore_validation_snapshot(model, snapshot)


def run_state_neutral_audit(
    model: nn.Module,
    callback: Callable[[], AuditResult | Mapping[str, Any] | float],
) -> AuditResult | Mapping[str, Any] | float:
    with state_neutral_audit(model):
        return callback()


def _call_validation(
    callback: AuditCallback | None,
    residual: torch.Tensor,
    counters: ResourceCounters,
    *,
    model: nn.Module | None,
) -> AuditResult | Mapping[str, Any] | float | None:
    if callback is None:
        return None
    snapshot = _validation_snapshot(model)
    try:
        if model is not None:
            model.eval()
        with torch.no_grad():
            result = callback(residual.detach().clone())
        if isinstance(result, AuditResult):
            counters.validation_evaluations += 1
            counters.validation_samples += result.samples
        elif isinstance(result, Mapping):
            counters.validation_evaluations += 1
        else:
            counters.validation_evaluations += 1
        return result
    finally:
        _restore_validation_snapshot(model, snapshot)


def _run_search(
    objective: ObjectiveCallback,
    *,
    seed: int,
    method: str,
    generations: int,
    particles: int,
    device: torch.device | str,
    codec: SelectedResidualCodec | None,
    model: nn.Module | None,
    validation: AuditCallback | None,
    fallback_samples: int,
    random_mode: bool,
) -> SearchResult:
    if generations != PSO_GENERATIONS or particles != PARTICLE_COUNT:
        raise ProtocolError("primary search budget must be exactly 12 particles x 60 generations")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ProtocolError("seed must be an integer")
    resolved = torch.device(device)
    rng = _RandomSource(seed=seed, device=resolved)
    positions = _initial_positions(rng, particles, RESIDUAL_DIMENSION, resolved)
    velocities = [torch.zeros_like(position) for position in positions]
    pbest_positions = [position.clone() for position in positions]
    pbest_losses: list[float | None] = [None] * particles
    gbest_position: torch.Tensor | None = None
    gbest_loss: float | None = None
    gbest_result: ObjectiveResult | None = None
    gbest_index = 0
    counters = ResourceCounters()
    failures: list[str] = []
    endpoints: list[CandidateEndpoint] = []
    trajectory: list[dict[str, Any]] = []
    movement = ConstrictionMovement(c0=2.05, c1=2.05)
    initial_validation = _call_validation(
        validation,
        positions[0],
        counters,
        model=model,
    )

    def evaluate_generation(gen: int, generation_positions: Sequence[torch.Tensor]) -> None:
        nonlocal gbest_position, gbest_loss, gbest_result, gbest_index
        generation_values: list[_Evaluation] = []
        for index, candidate in enumerate(generation_positions):
            evaluation = _evaluate_candidate(
                objective,
                candidate,
                counters,
                failures,
                codec=codec,
                model=model,
                fallback_samples=fallback_samples,
            )
            generation_values.append(evaluation)
            if evaluation.objective is None:
                continue
            loss = evaluation.objective.loss
            if _is_strictly_better(loss, pbest_losses[index]):
                pbest_losses[index] = loss
                pbest_positions[index] = candidate.detach().clone()
            if _is_strictly_better(loss, gbest_loss):
                gbest_loss = loss
                gbest_position = candidate.detach().clone()
                gbest_result = evaluation.objective
                gbest_index = index
        if gbest_position is None or gbest_loss is None or gbest_result is None:
            raise ObjectiveEvaluationError("all objective candidates failed")
        best_result = gbest_result
        if gen in OBJECTIVE_CHECKPOINTS:
            validation_result = _call_validation(validation, gbest_position, counters, model=model)
        else:
            validation_result = None
        velocity_norm = float(torch.stack([v.norm() for v in velocities]).mean().item())
        trajectory.append({
            "generation": gen,
            "objective_best": float(gbest_loss),
            "velocity_norm": velocity_norm,
            "displacement_norm": float(torch.stack([p.norm() for p in generation_positions]).mean().item()),
            "best_vector_norm": float(gbest_position.norm().item()),
            "validation": None if validation_result is None else _jsonable(validation_result.to_dict() if isinstance(validation_result, AuditResult) else validation_result),
        })
        if gen == 1:
            trajectory[-1]["initial_validation"] = (
                None
                if initial_validation is None
                else _jsonable(
                    initial_validation.to_dict()
                    if isinstance(initial_validation, AuditResult)
                    else initial_validation
                )
            )
        endpoints.append(CandidateEndpoint(gen, gbest_index, gbest_position, best_result))

    evaluate_generation(1, positions)
    for generation in range(2, generations + 1):
        state = SwarmState(
            positions=tuple(position.clone() for position in positions),
            velocities=tuple(velocity.clone() for velocity in velocities),
            pbest_positions=tuple(position.clone() for position in pbest_positions),
            pbest_scores=tuple((loss if loss is not None else math.inf, 0.0, 0.0) for loss in pbest_losses),
            gbest_position=gbest_position.clone(),
            gbest_score=(gbest_loss, 0.0, 0.0),
            pbest_improved=tuple(False for _ in positions),
        )
        proposed: list[torch.Tensor] = []
        proposed_velocities: list[torch.Tensor] = []
        for index in range(particles):
            if random_mode:
                velocity = velocities[index]
                candidate = rng.uniform((RESIDUAL_DIMENSION,), -RESIDUAL_BOUND, RESIDUAL_BOUND, device=resolved, dtype=torch.float32)
            else:
                context = IterationContext(
                    epoch=generation,
                    total_epochs=generations,
                    w=1.0,
                    particle_idx=index,
                    is_negative=False,
                    rng=rng,
                    optimizer=None,
                )
                _, velocity = movement.propose(index, state, context)
                candidate = positions[index] + velocity
                outside = (candidate < -RESIDUAL_BOUND) | (candidate > RESIDUAL_BOUND)
                candidate = torch.clamp(candidate, -RESIDUAL_BOUND, RESIDUAL_BOUND)
                velocity = torch.where(outside, torch.zeros_like(velocity), velocity)
            proposed.append(candidate.detach().clone())
            proposed_velocities.append(velocity.detach().clone())
        velocities = proposed_velocities
        positions = proposed
        evaluate_generation(generation, positions)

    assert gbest_position is not None and gbest_loss is not None
    best_endpoint = endpoints[-1]
    return SearchResult(
        method=method,
        seed=seed,
        generations=generations,
        particles=particles,
        best_residual=gbest_position.detach().clone(),
        best_objective=ObjectiveResult(gbest_loss, best_endpoint.objective.samples, best_endpoint.objective.forward_passes, best_endpoint.objective.backward_passes),
        endpoints=tuple(endpoints),
        trajectory=tuple(trajectory),
        counters=counters,
        failures=tuple(failures),
    )


def run_residual_pso(
    objective: ObjectiveCallback,
    codec: SelectedResidualCodec | None = None,
    *,
    seed: int,
    generations: int = PSO_GENERATIONS,
    particles: int = PARTICLE_COUNT,
    device: torch.device | str = "cpu",
    model: nn.Module | None = None,
    validation: AuditCallback | None = None,
    objective_samples: int = 0,
) -> SearchResult:
    """Run the fixed 12x60 constriction PSO with exactly 720 evaluations."""
    return _run_search(
        objective,
        seed=seed,
        method="feature_pso",
        generations=generations,
        particles=particles,
        device=device,
        codec=codec,
        model=model,
        validation=validation,
        fallback_samples=objective_samples,
        random_mode=False,
    )


def run_equal_budget_random(
    objective: ObjectiveCallback,
    codec: SelectedResidualCodec | None = None,
    *,
    seed: int,
    generations: int = PSO_GENERATIONS,
    particles: int = PARTICLE_COUNT,
    device: torch.device | str = "cpu",
    model: nn.Module | None = None,
    validation: AuditCallback | None = None,
    objective_samples: int = 0,
) -> SearchResult:
    """Run the equal-query random control: 12 initial + 708 U[-1,1]."""
    return _run_search(
        objective,
        seed=seed,
        method="feature_random",
        generations=generations,
        particles=particles,
        device=device,
        codec=codec,
        model=model,
        validation=validation,
        fallback_samples=objective_samples,
        random_mode=True,
    )


run_random_search = run_equal_budget_random


class StudyState(str, Enum):
    PREPARED = "prepared"
    DEVELOPING = "developing"
    FROZEN = "frozen"
    CONFIRMING = "confirming"
    COMPLETED = "completed"
    FAILED = "failed"


RunState = StudyState
_ALLOWED_TRANSITIONS = {
    StudyState.PREPARED: {StudyState.DEVELOPING, StudyState.FAILED},
    StudyState.DEVELOPING: {StudyState.FROZEN, StudyState.FAILED},
    StudyState.FROZEN: {StudyState.CONFIRMING, StudyState.FAILED},
    StudyState.CONFIRMING: {StudyState.COMPLETED, StudyState.FAILED},
    StudyState.COMPLETED: set(),
    StudyState.FAILED: set(),
}


@dataclasses.dataclass
class StudyStateMachine:
    state: StudyState = StudyState.PREPARED
    history: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    failure_reason: str | None = None

    def transition(self, target: StudyState | str, *, detail: str | None = None) -> StudyState:
        target_state = StudyState(target)
        if target_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise StateTransitionError(f"invalid transition {self.state.value} -> {target_state.value}")
        previous = self.state
        self.state = target_state
        if target_state == StudyState.FAILED:
            self.failure_reason = detail or "unspecified protocol failure"
        self.history.append({"from": previous.value, "to": target_state.value, "detail": detail})
        return self.state

    def fail(self, reason: str) -> StudyState:
        if not reason:
            raise ProtocolError("failure reason must not be empty")
        return self.transition(StudyState.FAILED, detail=reason)

    def require(self, expected: StudyState | str) -> None:
        target = StudyState(expected)
        if self.state != target:
            raise StateTransitionError(f"expected state {target.value}, found {self.state.value}")

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state.value, "history": list(self.history), "failure_reason": self.failure_reason}


@dataclasses.dataclass(frozen=True)
class WorkloadSpec:
    workload_id: str
    family: str
    adapter_factory: Callable[..., Any]
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.workload_id or not self.family or not callable(self.adapter_factory):
            raise ProtocolError("workload spec requires id, family, and callable factory")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


_WORKLOADS: dict[str, WorkloadSpec] = {}


def register_workload(spec: WorkloadSpec, *, replace: bool = False) -> WorkloadSpec:
    if spec.workload_id in _WORKLOADS and not replace:
        raise ProtocolError(f"workload already registered: {spec.workload_id}")
    _WORKLOADS[spec.workload_id] = spec
    return spec


def get_workload(workload_id: str) -> WorkloadSpec:
    try:
        return _WORKLOADS[workload_id]
    except KeyError as exc:
        raise ProtocolError(f"unknown workload: {workload_id}") from exc


def registered_workloads() -> tuple[WorkloadSpec, ...]:
    return tuple(_WORKLOADS[workload_id] for workload_id in sorted(_WORKLOADS))


def _lazy_adapter(module_name: str) -> Callable[..., Any]:
    def factory(*args: Any, **kwargs: Any) -> Any:
        module = importlib.import_module(module_name)
        creator = getattr(module, "create_adapter", None)
        if creator is None or not callable(creator):
            raise ProtocolError(f"adapter module {module_name!r} does not expose create_adapter")
        return creator(*args, **kwargs)
    return factory


register_workload(WorkloadSpec("cifar10_resnet18", "classification", _lazy_adapter("test.post_training_resnet_convergence"), {"model": "resnet18", "dataset": "cifar10"}))
register_workload(WorkloadSpec("cifar10_resnet50", "classification", _lazy_adapter("test.post_training_resnet_convergence"), {"model": "resnet50", "dataset": "cifar10"}))
register_workload(WorkloadSpec("voc_yolo11n", "detection", _lazy_adapter("test.post_training_yolo_convergence"), {"model": "yolo11n", "dataset": "voc2007+2012", "optional_dependency": "ultralytics==8.4.142"}))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fingerprint_tensors(tensors: Iterator[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(canonical_json(tuple(value.shape)))
        digest.update(value.numpy().tobytes() if value.device.type == "cpu" else bytes(value))
    return digest.hexdigest()


def fingerprint_module(model: nn.Module) -> str:
    return _fingerprint_tensors(iter(list(model.named_parameters()) + list(model.named_buffers())))


def fingerprint_nonselected_state(model: nn.Module, selected_names: Sequence[str]) -> str:
    excluded = set(selected_names)
    return _fingerprint_tensors((name, tensor) for name, tensor in list(model.named_parameters()) + list(model.named_buffers()) if name not in excluded)


def fingerprint_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_paths(root: str | os.PathLike[str], paths: Sequence[str | os.PathLike[str]]) -> dict[str, str]:
    base = Path(root).resolve()
    result: dict[str, str] = {}
    for item in paths:
        path = Path(item)
        if path.is_absolute():
            full = path.resolve()
        else:
            cwd_relative = path.resolve()
            full = (
                cwd_relative
                if base in cwd_relative.parents or cwd_relative == base
                else (base / path).resolve()
            )
        if base not in full.parents and full != base:
            raise SealError(f"artifact escapes run root: {item}")
        if not full.is_file():
            raise SealError(f"artifact is not a file: {full}")
        result[str(full.relative_to(base))] = fingerprint_file(full)
    return result


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    return destination


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> Path:
    return atomic_write_bytes(path, canonical_json(value) + b"\n")


@dataclasses.dataclass(frozen=True)
class FrozenManifest:
    protocol_version: str
    config: Mapping[str, Any]
    artifacts: Mapping[str, str]
    manifest_hash: str
    state: str = StudyState.FROZEN.value

    def payload_without_hash(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "config": _jsonable(self.config),
            "artifacts": dict(self.artifacts),
            "state": self.state,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload_without_hash(), "manifest_hash": self.manifest_hash}


_DEFER_MATRIX_FREEZE = False
_MATRIX_CONFIRMING = False
_DEFER_MATRIX_COMPLETION = False

def freeze_run(
    run_root: str | os.PathLike[str],
    config: StudyConfig,
    artifacts: Sequence[str | os.PathLike[str]],
    state_machine: StudyStateMachine,
) -> FrozenManifest:
    state_machine.require(StudyState.DEVELOPING)
    root = Path(run_root)
    hashes = fingerprint_paths(root, artifacts)
    provisional = {
        "protocol_version": config.protocol_version,
        "config": config.to_dict(),
        "artifacts": hashes,
        "state": StudyState.FROZEN.value,
    }
    manifest = FrozenManifest(config.protocol_version, config.to_dict(), hashes, sha256_bytes(canonical_json(provisional)))
    if not _DEFER_MATRIX_FREEZE:
        atomic_write_json(root / "frozen_manifest.json", manifest.to_dict())
        state_machine.transition(StudyState.FROZEN)
    return manifest


def load_frozen_manifest(run_root: str | os.PathLike[str]) -> FrozenManifest:
    path = Path(run_root) / "frozen_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        manifest = FrozenManifest(
            protocol_version=value["protocol_version"],
            config=value["config"],
            artifacts=value["artifacts"],
            manifest_hash=value["manifest_hash"],
            state=value["state"],
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SealError(f"invalid frozen manifest: {path}") from exc
    if manifest.state != StudyState.FROZEN.value or manifest.protocol_version != PROTOCOL_VERSION:
        raise SealError("frozen manifest has an invalid protocol or state")
    expected = sha256_bytes(canonical_json(manifest.payload_without_hash()))
    if expected != manifest.manifest_hash:
        raise SealError("frozen manifest self-hash mismatch")
    return manifest


def verify_frozen_manifest(run_root: str | os.PathLike[str], manifest: FrozenManifest | None = None) -> FrozenManifest:
    frozen = manifest or load_frozen_manifest(run_root)
    actual = fingerprint_paths(run_root, tuple(frozen.artifacts.keys()))
    if actual != dict(frozen.artifacts):
        raise SealError("frozen artifact hash mismatch")
    return frozen

def load_state(run_root: str | os.PathLike[str]) -> StudyStateMachine:
    path = Path(run_root) / "state.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        machine = StudyStateMachine(state=StudyState(value["state"]))
        machine.history.extend(value.get("history", []))
        machine.failure_reason = value.get("failure_reason")
        return machine
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise StateTransitionError(f"invalid persisted state: {path}") from exc


def publish_artifacts(
    source_root: str | os.PathLike[str],
    destination_root: str | os.PathLike[str],
    paths: Sequence[str | os.PathLike[str]],
) -> dict[str, str]:
    """Copy already-produced evidence through atomic replaces and hash it."""
    source = Path(source_root).resolve()
    destination = Path(destination_root)
    published: dict[str, str] = {}
    for item in paths:
        relative = Path(item)
        if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
            raise ProtocolError(f"publication path must be relative: {item}")
        source_path = source / relative
        if not source_path.is_file():
            raise SealError(f"publication source is not a file: {source_path}")
        data = source_path.read_bytes()
        target = atomic_write_bytes(destination / relative, data)
        published[str(relative)] = sha256_bytes(data)
        if fingerprint_file(target) != published[str(relative)]:
            raise SealError(f"published artifact hash mismatch: {target}")
    return published


def begin_confirmation(run_root: str | os.PathLike[str], state_machine: StudyStateMachine) -> FrozenManifest:
    if _MATRIX_CONFIRMING and state_machine.state == StudyState.CONFIRMING:
        return verify_frozen_manifest(run_root)
    state_machine.require(StudyState.FROZEN)
    manifest = verify_frozen_manifest(run_root)
    state_machine.transition(StudyState.CONFIRMING)
    return manifest


def finish_confirmation(state_machine: StudyStateMachine, *, success: bool, reason: str | None = None) -> None:
    state_machine.require(StudyState.CONFIRMING)
    if success:
        if not _DEFER_MATRIX_COMPLETION:
            state_machine.transition(StudyState.COMPLETED)
    else:
        state_machine.fail(reason or "confirmation failed")


def select_endpoint(
    endpoints: Sequence[CandidateEndpoint],
    validation_metric: Mapping[int, float],
    *,
    maximize: bool = False,
) -> CandidateEndpoint:
    if not endpoints:
        raise ProtocolError("cannot select from empty endpoints")
    ranked: list[tuple[float, int, CandidateEndpoint]] = []
    for endpoint in endpoints:
        value = validation_metric.get(endpoint.generation)
        if value is None or not math.isfinite(float(value)):
            raise ProtocolError(f"missing finite validation metric for generation {endpoint.generation}")
        ranked.append(((-float(value) if maximize else float(value)), endpoint.generation, endpoint))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2]


def prepare_run(run_root: str | os.PathLike[str], config: StudyConfig) -> StudyStateMachine:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    state = StudyStateMachine()
    atomic_write_json(root / "config.json", config.to_dict())
    atomic_write_json(root / "state.json", state.to_dict())
    return state


def persist_state(run_root: str | os.PathLike[str], state_machine: StudyStateMachine) -> None:
    atomic_write_json(Path(run_root) / "state.json", state_machine.to_dict())


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-training ResNet/YOLO convergence protocol")
    parser.add_argument("--phase", choices=["prepare", "smoke", "develop", "confirm", "publish", "all"], required=True)
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default="cpu",
    )
    parser.add_argument("--data-root", type=Path, default=Path("result/cache"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def _make_adapters(config: StudyConfig, root: Path, data_root: Path, allow_download: bool) -> list[Any]:
    if tuple(config.workload_ids) != DEFAULT_WORKLOAD_IDS:
        raise ProtocolError("the main matrix must contain all three workloads in fixed order")
    return [
        get_workload(workload_id).adapter_factory(
            workload_id=workload_id,
            config=config,
            run_root=root,
            data_root=data_root,
            device=config.device,
            allow_download=allow_download,
        )
        for workload_id in config.workload_ids
    ]


_REQUIRED_RESULT_FIELDS = (
    "workload_id", "family", "config", "manifests", "provenance", "baselines",
    "arms", "ensemble", "development_selection", "confirmation", "integrity",
    "leakage_counters", "resource_ledger", "artifact_hashes",
)


def _finite_record(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_record(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_record(item) for item in value)
    return True


def _completed_record(value: Any, label: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise SealError(f"missing completed record: {label}")
    if not _finite_record(value):
        raise SealError(f"non-finite record: {label}")
    failures = value.get("failures")
    if isinstance(failures, list) and failures:
        raise SealError(f"failed evaluations in record: {label}")
    if value.get("success") is False or value.get("completed") is False:
        raise SealError(f"unsuccessful record: {label}")
    numeric = _find_numeric(value, ("loss", "objective", "nll", "accuracy", "map", "metric", "queries"))
    if numeric is None:
        raise SealError(f"record has no finite completion metric: {label}")


def _find_numeric(value: Any, keys: Sequence[str]) -> float | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if any(token in str(key).lower() for token in keys) and _finite_number(item):
                return float(item)
            found = _find_numeric(item, keys)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_numeric(item, keys)
            if found is not None:
                return found
    return None


def _arm_cells(arms: Mapping[str, Any], method: str) -> dict[tuple[int, int], Any]:
    cells: dict[tuple[int, int], Any] = {}
    method_tree = arms.get(method)
    if isinstance(method_tree, Mapping):
        for base_key, swarm_tree in method_tree.items():
            if not str(base_key).isdigit() or not isinstance(swarm_tree, Mapping):
                continue
            for swarm_key, record in swarm_tree.items():
                if str(swarm_key).isdigit():
                    cells[(int(base_key), int(swarm_key))] = record
    for key, value in arms.items():
        text = str(key)
        if text.startswith(f"{method}/"):
            parts = text.split("/")
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                cells[(int(parts[1]), int(parts[2]))] = value
        if text.isdigit() and isinstance(value, Mapping):
            base = int(text)
            for child_key, child in value.items():
                child_text = str(child_key)
                if child_text.startswith(f"{method}:") and child_text.split(":", 1)[1].isdigit():
                    swarm = int(child_text.split(":", 1)[1])
                    cells[(base, swarm)] = child.get(method, child) if isinstance(child, Mapping) else child
    return cells


def _control_cells(arms: Mapping[str, Any], method: str) -> dict[int, Any]:
    cells: dict[int, Any] = {}
    method_tree = arms.get(method)
    if isinstance(method_tree, Mapping):
        for base_key, record in method_tree.items():
            if str(base_key).isdigit():
                cells[int(base_key)] = record
    for key, value in arms.items():
        text = str(key)
        if text.startswith(f"{method}/") and text.split("/")[-1].isdigit():
            cells[int(text.split("/")[-1])] = value
        if text.isdigit() and isinstance(value, Mapping) and method in value:
            cells[int(text)] = value[method]
    return cells


def _saved_record_count(value: Any) -> int:
    if isinstance(value, Mapping):
        keys = {str(key).lower() for key in value}
        if keys & {"predictions", "prediction", "metrics", "metric", "map50_95", "nll"}:
            return 1
        return sum(_saved_record_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_saved_record_count(item) for item in value)
    return 0


def _test_counter(result: Mapping[str, Any], token: str) -> int | None:
    values: list[int] = []
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                lowered = str(key).lower()
                if "test" in lowered and token in lowered and isinstance(item, int) and not isinstance(item, bool):
                    values.append(item)
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
    visit(result.get("leakage_counters", {}))
    return sum(values) if values else None


def _validate_development_result(
    workload_id: str,
    result: Mapping[str, Any],
) -> None:
    baselines = result["baselines"]
    if (
        not isinstance(baselines, Mapping)
        or set(map(str, baselines)) != set(map(str, BASE_SEEDS))
    ):
        raise SealError(
            f"{workload_id}: baseline matrix must contain exactly "
            "three base seeds"
        )
    for seed in BASE_SEEDS:
        _completed_record(
            baselines.get(str(seed)),
            f"{workload_id}/baseline/{seed}",
        )
    arms = result["arms"]
    if not isinstance(arms, Mapping):
        raise SealError(f"{workload_id}: missing arms")
    for method in ("feature_pso", "feature_random"):
        cells = _arm_cells(arms, method)
        expected = {
            (base, swarm)
            for base in BASE_SEEDS
            for swarm in SWARM_SEEDS
        }
        if set(cells) != expected:
            raise SealError(
                f"{workload_id}: {method} matrix is incomplete"
            )
        for cell, record in cells.items():
            _completed_record(
                record,
                f"{workload_id}/{method}/{cell[0]}/{cell[1]}",
            )
    for method in ("feature_adam", "head_adam"):
        cells = _control_cells(arms, method)
        if set(cells) != set(BASE_SEEDS):
            raise SealError(
                f"{workload_id}: {method} control matrix is incomplete"
            )
        for seed, record in cells.items():
            _completed_record(
                record,
                f"{workload_id}/{method}/{seed}",
            )
    ensemble = result["ensemble"]
    if not isinstance(ensemble, Mapping):
        raise SealError(f"{workload_id}: missing ensemble records")
    if result["family"] == "classification":
        for method in (
            "uniform",
            "uniform_temperature",
            "slsqp_weights",
        ):
            _completed_record(
                ensemble.get(method),
                f"{workload_id}/ensemble/{method}",
            )
        pso = ensemble.get("ensemble_pso")
        if isinstance(pso, Mapping):
            pso = pso.get("objective")
        if not isinstance(pso, list) or len(pso) != len(SWARM_SEEDS):
            raise SealError(
                f"{workload_id}: ensemble PSO matrix is incomplete"
            )
        for seed, record in zip(SWARM_SEEDS, pso):
            _completed_record(
                record,
                f"{workload_id}/ensemble_pso/{seed}",
            )
    else:
        aliases = {
            "uniform_wbf": ("uniform_wbf", "uniform"),
            "ensemble_pso_wbf": (
                "ensemble_pso_wbf",
                "pso_wbf",
                "ensemble_pso",
            ),
            "random_wbf": (
                "random_wbf",
                "feature_random_wbf",
                "ensemble_random",
                "random",
            ),
        }
        for method, names in aliases.items():
            records = next(
                (ensemble[name] for name in names if name in ensemble),
                None,
            )
            if method == "uniform_wbf":
                _completed_record(
                    records,
                    f"{workload_id}/ensemble/{method}",
                )
                continue
            if (
                not isinstance(records, list)
                or len(records) != len(SWARM_SEEDS)
            ):
                raise SealError(
                    f"{workload_id}: detection ensemble {method} "
                    "matrix is incomplete"
                )
            for seed, record in zip(SWARM_SEEDS, records):
                _completed_record(
                    record,
                    f"{workload_id}/ensemble/{method}/{seed}",
                )
    if (
        not isinstance(result["artifact_hashes"], Mapping)
        or not result["artifact_hashes"]
    ):
        raise SealError(
            f"{workload_id}: nonempty artifact hashes are required "
            "before freezing"
        )
    if result["integrity"].get("official_test_opened") is not False:
        raise SealError(
            f"{workload_id}: official_test_opened must be false "
            "before freezing"
        )
    if (
        _test_counter(result, "construction") not in (None, 0)
        or _test_counter(result, "forward") not in (None, 0)
    ):
        raise SealError(
            f"{workload_id}: pre-freeze test exposure is nonzero"
        )


def _validate_matrix_results(root: Path, *, require_confirmation: bool = False, strict_development: bool = False) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for workload_id in DEFAULT_WORKLOAD_IDS:
        path = root / "workloads" / workload_id / "result.json"
        if not path.is_file():
            raise SealError(f"missing workload result: {path}")
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SealError(f"invalid workload result: {path}") from exc
        if not isinstance(result, dict) or any(field not in result for field in _REQUIRED_RESULT_FIELDS):
            raise SealError(f"incomplete workload result: {workload_id}")
        if result.get("workload_id") != workload_id:
            raise SealError(f"workload result id mismatch: {workload_id}")
        if not _finite_record(result):
            raise SealError(f"non-finite workload result: {workload_id}")
        if strict_development:
            _validate_development_result(workload_id, result)
        if require_confirmation:
            confirmation = result["confirmation"]
            family = result.get("family")
            expected_confirmations = (
                len(BASE_SEEDS) * 5 + 4
                if family == "classification"
                else len(BASE_SEEDS) * 5 + 3
            )
            if (
                not isinstance(confirmation, Mapping)
                or not confirmation
                or _saved_record_count(confirmation) < expected_confirmations
            ):
                raise SealError(
                    f"{workload_id}: confirmation must contain every selected "
                    "frozen method record"
                )
            construction = _test_counter(result, "construction")
            if construction != 1:
                raise SealError(f"{workload_id}: official test construction must occur exactly once")
            forwards = _test_counter(result, "forward")
            evaluations = _test_counter(result, "evaluation")
            if forwards is None and evaluations is None:
                raise SealError(f"{workload_id}: declared test forward/evaluation counter is missing")
            if (forwards or evaluations or 0) <= 0:
                raise SealError(f"{workload_id}: test forward/evaluation counter must be positive")
            for key, value in confirmation.items():
                if any(token in str(key).lower() for token in ("repeat", "rerun", "tuning")) and value:
                    raise SealError(f"{workload_id}: confirmation contains repeat/tuning activity")
        declared = result["artifact_hashes"]
        if not isinstance(declared, Mapping):
            raise SealError(f"invalid artifact hash map: {workload_id}")
        for relative, expected in declared.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise SealError(f"invalid artifact hash entry: {workload_id}")
            artifact = (root / relative).resolve()
            if root.resolve() not in artifact.parents or not artifact.is_file() or fingerprint_file(artifact) != expected:
                raise SealError(f"declared artifact hash drift: {relative}")
        results[workload_id] = result
    return results


def _stable_matrix_artifacts(root: Path) -> list[str]:
    if not (root / "config.json").is_file():
        raise SealError("shared config.json is missing")
    artifacts = ["config.json"]
    if (root / "runtime_config.json").is_file():
        artifacts.append("runtime_config.json")
    for workload_id in DEFAULT_WORKLOAD_IDS:
        workload_root = root / "workloads" / workload_id
        if not workload_root.is_dir():
            raise SealError(f"missing workload artifact directory: {workload_id}")
        for path in sorted(workload_root.rglob("*")):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.name not in {"result.json", "frozen_manifest.json"}
            ):
                artifacts.append(str(path.relative_to(root)))
    return artifacts


def _run_adapter_phase(adapters: Sequence[Any], phase: str) -> list[Any]:
    outputs: list[Any] = []
    for adapter in adapters:
        runner = getattr(adapter, "run_phase", None)
        if not callable(runner):
            raise ProtocolError(f"adapter {type(adapter).__name__} lacks run_phase")
        outputs.append(runner(phase))
    return outputs

def _run_adapter_development(
    adapters: Sequence[Any],
    root: Path,
) -> list[Any]:
    outputs: list[Any] = []
    for adapter in adapters:
        workload_root = root / "workloads" / str(adapter.workload_id)
        marker_path = workload_root / "development_reuse.json"
        if not marker_path.is_file():
            outputs.extend(_run_adapter_phase((adapter,), "develop"))
            continue
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("protocol_version") != PROTOCOL_VERSION
            or not isinstance(marker.get("source_run"), str)
        ):
            raise SealError(
                f"invalid development reuse marker: {marker_path}"
            )
        result_path = workload_root / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            not isinstance(result, dict)
            or any(
                field not in result
                for field in _REQUIRED_RESULT_FIELDS
            )
            or result.get("workload_id") != adapter.workload_id
            or not _finite_record(result)
        ):
            raise SealError(
                f"reused workload result is invalid: "
                f"{workload_root.name}"
            )
        _validate_development_result(
            str(adapter.workload_id),
            result,
        )
        declared = result["artifact_hashes"]
        for relative, expected in declared.items():
            artifact = root / str(relative)
            if (
                not artifact.is_file()
                or fingerprint_file(artifact) != expected
            ):
                raise SealError(
                    f"reused artifact hash drift: {relative}"
                )
        outputs.append(result)
    return outputs


def _run_matrix_phase(
    phase: str,
    *,
    config: StudyConfig,
    root: Path,
    data_root: Path,
    allow_download: bool,
) -> Any:
    global _DEFER_MATRIX_FREEZE, _MATRIX_CONFIRMING, _DEFER_MATRIX_COMPLETION
    if phase == "prepare":
        if (root / "state.json").is_file():
            state = load_state(root)
            state.require(StudyState.PREPARED)
        else:
            state = prepare_run(root, config)
        outputs = _run_adapter_phase(_make_adapters(config, root, data_root, allow_download), "prepare")
        _validate_matrix_results(root)
        persist_state(root, state)
        return outputs
    state = load_state(root)
    if phase == "smoke":
        state.require(StudyState.PREPARED)
        _validate_matrix_results(root)
        outputs = _run_adapter_phase(_make_adapters(config, root, data_root, allow_download), "smoke")
        for result in _validate_matrix_results(root).values():
            leakage = result["leakage_counters"]
            if any("test" in str(key).lower() and isinstance(value, int) and value != 0 for key, value in leakage.items()):
                raise SealError("smoke opened official test data")
        return outputs
    if phase == "develop":
        if state.state == StudyState.PREPARED:
            state.transition(StudyState.DEVELOPING)
            persist_state(root, state)
        else:
            state.require(StudyState.DEVELOPING)
        _DEFER_MATRIX_FREEZE = True
        try:
            outputs = _run_adapter_development(
                _make_adapters(
                    config,
                    root,
                    data_root,
                    allow_download,
                ),
                root,
            )
        finally:
            _DEFER_MATRIX_FREEZE = False
        _validate_matrix_results(root, strict_development=True)
        for workload_id in DEFAULT_WORKLOAD_IDS:
            workload_root = root / "workloads" / workload_id
            atomic_write_bytes(
                workload_root / "development_result.json",
                (workload_root / "result.json").read_bytes(),
            )
        freeze_run(root, config, _stable_matrix_artifacts(root), state)
        persist_state(root, state)
        return outputs
    if phase == "confirm":
        state.require(StudyState.FROZEN)
        verify_frozen_manifest(root)
        state.transition(StudyState.CONFIRMING)
        persist_state(root, state)
        _MATRIX_CONFIRMING = True
        _DEFER_MATRIX_COMPLETION = True
        try:
            outputs = _run_adapter_phase(_make_adapters(config, root, data_root, allow_download), "confirm")
        finally:
            _MATRIX_CONFIRMING = False
            _DEFER_MATRIX_COMPLETION = False
        verify_frozen_manifest(root)
        _validate_matrix_results(root, require_confirmation=True, strict_development=True)
        state.transition(StudyState.COMPLETED)
        persist_state(root, state)
        return outputs
    if phase == "publish":
        state.require(StudyState.COMPLETED)
        verify_frozen_manifest(root)
        return _publish_saved(root)
    raise ProtocolError(f"unsupported matrix phase: {phase}")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _plot_saved_trajectories(root: Path, destination: Path, workload_ids: Sequence[str]) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ProtocolError("publish requires matplotlib to render saved trajectories") from exc
    figure, axis = plt.subplots(figsize=(10, 5))
    plotted = 0
    for workload_id in workload_ids:
        result = json.loads((root / "workloads" / workload_id / "result.json").read_text(encoding="utf-8"))
        for name, record in result.get("arms", {}).items():
            if not isinstance(record, Mapping) or not isinstance(record.get("trajectory"), list):
                continue
            points = [
                (item.get("generation"), item.get("objective_best"))
                for item in record["trajectory"]
                if isinstance(item, Mapping)
                and _finite_number(item.get("generation"))
                and _finite_number(item.get("objective_best"))
            ]
            if points:
                axis.plot([point[0] for point in points], [point[1] for point in points], alpha=0.7, label=f"{workload_id}:{name}")
                plotted += 1
    if not plotted:
        axis.text(0.5, 0.5, "No trajectory data recorded", ha="center", va="center")
        axis.set_title("Saved-data trajectory unavailable")
    else:
        axis.legend(fontsize=6, loc="best")
    axis.set_xlabel("evaluated generation")
    axis.set_ylabel("objective")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    temporary = tempfile.NamedTemporaryFile(prefix=".plot-", suffix=".png", delete=False)
    temporary.close()
    temporary_path = Path(temporary.name)
    try:
        figure.savefig(temporary_path, dpi=150)
        atomic_write_bytes(destination, temporary_path.read_bytes())
    finally:
        plt.close(figure)
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()
    return destination


def _publish_saved(root: Path) -> dict[str, str]:
    evaluator = importlib.import_module("test.evaluate_post_training_model_convergence")
    evaluate = getattr(evaluator, "evaluate_run", None)
    if not callable(evaluate):
        raise ProtocolError("evaluator does not expose evaluate_run")
    payload = evaluate(root)
    if not isinstance(payload, Mapping):
        raise ProtocolError("evaluator returned a non-object payload")
    results = _validate_matrix_results(root, require_confirmation=True)
    benchmark_root, plot_root = Path("benchmark_results"), Path("history_plt")
    benchmark_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)
    json_path = atomic_write_json(benchmark_root / "pso_v9_model_convergence.json", payload)
    evaluation_path = atomic_write_json(benchmark_root / "pso_v9_model_convergence_evaluation.json", payload)
    output = io.StringIO()
    writer = __import__("csv").DictWriter(output, fieldnames=["workload_id", "family", "valid", "issue_count"])
    writer.writeheader()
    issue_count = sum(len(items) for items in payload.get("issues", {}).values()) if isinstance(payload.get("issues"), Mapping) else 0
    for workload_id in DEFAULT_WORKLOAD_IDS:
        finding = payload.get("workloads", {}).get(workload_id, {}) if isinstance(payload.get("workloads"), Mapping) else {}
        writer.writerow({"workload_id": workload_id, "family": results[workload_id]["family"], "valid": finding.get("valid", False), "issue_count": issue_count})
    csv_path = atomic_write_bytes(benchmark_root / "pso_v9_model_convergence.csv", output.getvalue().encode("utf-8"))
    resnet_plot = _plot_saved_trajectories(root, plot_root / "pso_v9_resnet_convergence.png", DEFAULT_WORKLOAD_IDS[:2])
    yolo_plot = _plot_saved_trajectories(root, plot_root / "pso_v9_yolo_convergence.png", DEFAULT_WORKLOAD_IDS[2:])
    return {"json": str(json_path), "csv": str(csv_path), "evaluation": str(evaluation_path), "resnet_plot": str(resnet_plot), "yolo_plot": str(yolo_plot)}


def run_phase(
    phase: str,
    *,
    config: StudyConfig,
    run_root: str | os.PathLike[str],
    data_root: str | os.PathLike[str],
    allow_download: bool = False,
    adapter_runner: Callable[..., Any] | None = None,
) -> Any:
    root, data = Path(run_root), Path(data_root)
    if adapter_runner is not None:
        return adapter_runner(phase=phase, config=config, run_root=root, data_root=data, device=config.device, allow_download=allow_download)
    if phase == "all":
        return {current: _run_matrix_phase(current, config=config, root=root, data_root=data, allow_download=allow_download) for current in ("prepare", "smoke", "develop", "confirm", "publish")}
    return _run_matrix_phase(phase, config=config, root=root, data_root=data, allow_download=allow_download)


def _load_cli_config(args: argparse.Namespace) -> StudyConfig:
    path = args.run_root / "config.json"
    if args.phase not in {"prepare", "all"} and path.is_file():
        config = StudyConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if config.device != args.device and args.device != "cpu":
            raise ProtocolError("requested device differs from the prepared run")
        return config
    return StudyConfig(device=args.device)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    try:
        result = run_phase(args.phase, config=_load_cli_config(args), run_root=args.run_root, data_root=args.data_root, allow_download=args.allow_download)
        if isinstance(result, Mapping):
            print(json.dumps(_jsonable(result), sort_keys=True, separators=(",", ":")))
        return 0
    except (ProtocolError, SealError, StateTransitionError, ObjectiveEvaluationError, RuntimeError, OSError) as exc:
        try:
            state = load_state(args.run_root)
            if state.state not in {StudyState.COMPLETED, StudyState.FAILED}:
                state.fail(f"{args.phase} failed: {type(exc).__name__}: {exc}")
                persist_state(args.run_root, state)
        except (OSError, StateTransitionError, ProtocolError):
            pass
        try:
            atomic_write_json(args.run_root / "failure.json", {"phase": args.phase, "error": f"{type(exc).__name__}: {exc}"})
        except OSError:
            pass
        return 2


__all__ = [
    "AuditCallback", "AuditResult", "BASE_SEEDS", "BOOTSTRAP_SEED", "CandidateEndpoint",
    "DEFAULT_WORKLOAD_IDS", "FrozenManifest", "INITIAL_RADIUS", "ObjectiveCallback", "ObjectiveEvaluationError",
    "ObjectiveResult", "PARTICLE_COUNT", "PROJECTION_SEED", "ProtocolError", "PSO_GENERATIONS", "RANDOM_CANDIDATES",
    "RESIDUAL_BOUND", "RESIDUAL_DIMENSION", "ResourceCounters", "RunState", "SWARM_SEEDS", "SPLIT_SEED",
    "SealError", "SearchResult", "SelectedResidualCodec", "StateTransitionError", "StudyConfig", "StudyState",
    "StudyStateMachine", "WorkloadSpec", "atomic_write_bytes", "atomic_write_json", "begin_confirmation",
    "build_cli_parser", "canonical_json", "fingerprint_file", "fingerprint_module", "fingerprint_nonselected_state",
    "fingerprint_paths", "finish_confirmation", "freeze_run", "get_workload", "load_frozen_manifest", "load_state", "main",
    "prepare_run", "projection_salt", "publish_artifacts", "register_workload", "registered_workloads", "run_equal_budget_random",
    "run_phase", "run_random_search", "run_residual_pso", "run_state_neutral_audit", "select_endpoint",
    "sha256_bytes", "state_neutral_audit", "verify_frozen_manifest",
]


if __name__ == "__main__":
    import sys
    sys.modules.setdefault("test.post_training_model_convergence", sys.modules[__name__])
    raise SystemExit(main())
