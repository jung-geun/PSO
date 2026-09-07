"""
Equalized Signed-Hash Subspace Experiment Runner for Heavy Task Autoresearch.

Protocol Version: HEAVY-PSO-AUTORESEARCH 1.0.0

Runs matched equalized signed-hash subspace PSO experiments across four candidate
latent ratios (0.5, 0.25, 0.125, 0.03125) and four heavy workloads:
  - mnist_compact (Base geometry: G6)
  - mnist_wide (Base geometry: G5)
  - fashion_compact (Base geometry: G6)
  - fashion_wide (Base geometry: G5)

Configurations are matched to confirmation runs (12 particles, 80 epochs, fixed 10k, seeds 101-103).
Official test splits are NEVER loaded or evaluated (official_test_evaluations = 0).
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

# Ensure test directory and repo root are in Python path
REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name != "PSO" else Path(__file__).resolve().parent
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_suite import (
    calc_stats,
    compute_model_fingerprint,
    get_hardware_provenance,
    resolve_execution_device,
    save_json_atomic,
)
from deep_pso_v6 import (
    V6GeometryConfig,
    V6LatentTransform,
    compute_equalized_subspace_radius,
    get_v6_geometry_table,
    run_v6_pso,
    artifact_safe_run,
)
from heavy_task_feasibility import (
    WORKLOADS,
    create_model,
    prepare_heavy_task_data,
)

# Protocol Constant
AUTORESEARCH_PROTOCOL_VERSION = "HEAVY-PSO-AUTORESEARCH 1.0.0"

# Defaults
DEFAULT_RATIOS = (1.0, 0.5, 0.25, 0.125, 0.03125)
DEFAULT_PARTICLES = 12
DEFAULT_EPOCHS = 80
DEFAULT_SUBSET_SIZE = 10000
DEFAULT_SEEDS = (101, 102, 103)
DEFAULT_SPLIT_SEED = 20260902
DEFAULT_GEOMETRY_POLICY = "recovered"
DEFAULT_PROJECTION_SCOPE = "global"
PROJECTION_SCOPES = ("global", "tensor_local", "balanced_global", "two_hash_global", "largest_tensor_hash", "largest_tensor_row_hash", "adjacent_pair", "adjacent_difference")
DEFAULT_PROJECTION_SEED_MODE = "coupled"
PROJECTION_SEED_MODES = ("coupled", "fixed", "explicit")
DEFAULT_GEOMETRY_MULTIPLIER = 1.0


def validate_geometry_multiplier(geometry_multiplier: Any) -> None:
    """
    Validates that geometry_multiplier is a finite positive float (> 0).
    Raises ValueError on non-numeric, non-finite, or non-positive values.
    """
    if (
        not isinstance(geometry_multiplier, (int, float))
        or isinstance(geometry_multiplier, bool)
        or not math.isfinite(geometry_multiplier)
        or geometry_multiplier <= 0
    ):
        raise ValueError(
            f"geometry_multiplier must be a finite positive float (> 0), got {geometry_multiplier!r}"
        )

def parse_projection_seed_arg(val: Any) -> Optional[Union[int, Dict[str, int]]]:
    """
    Parses projection_seed parameter or CLI arg into None, int, or Dict[str, int].
    Accepts integer values, integer strings, JSON dict strings, or comma/colon key-value strings.
    """
    if val is None or val == "" or val == "None":
        return None
    if isinstance(val, (int, dict)):
        return val
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str:
            return None
        try:
            return int(val_str)
        except ValueError:
            pass
        if val_str.startswith("{") and val_str.endswith("}"):
            try:
                parsed = json.loads(val_str)
                if isinstance(parsed, dict):
                    return {str(k): int(v) for k, v in parsed.items()}
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"Failed to parse projection_seed JSON dict string '{val}': {exc}")
        if ":" in val_str or "=" in val_str:
            res = {}
            for item in val_str.replace(";", ",").split(","):
                item = item.strip()
                if not item:
                    continue
                if ":" in item:
                    k, v = item.split(":", 1)
                elif "=" in item:
                    k, v = item.split("=", 1)
                else:
                    raise ValueError(f"Invalid key-value projection_seed string item '{item}'")
                res[k.strip()] = int(v.strip())
            if res:
                return res
    raise ValueError(f"Cannot parse projection_seed value: {val!r}")


def validate_projection_seed_config(
    projection_seed_mode: str,
    projection_seed: Optional[Union[int, Dict[str, int]]],
) -> None:
    """
    Validates projection_seed_mode and projection_seed invariants.
    Raises ValueError on invalid modes, missing explicit seeds, out-of-range explicit seeds,
    or seeds supplied to non-explicit modes.
    """
    if projection_seed_mode not in PROJECTION_SEED_MODES:
        raise ValueError(
            f"Invalid projection_seed_mode '{projection_seed_mode}'. Must be one of {list(PROJECTION_SEED_MODES)}"
        )
    if projection_seed_mode == "explicit":
        if projection_seed is None:
            raise ValueError(
                "projection_seed must be provided when projection_seed_mode is 'explicit'"
            )
        if isinstance(projection_seed, int) and not isinstance(projection_seed, bool):
            if not (0 <= projection_seed < (2**31 - 1)):
                raise ValueError(
                    f"projection_seed must be a non-negative integer < 2**31-1, got {projection_seed!r}"
                )
        elif isinstance(projection_seed, dict):
            if not projection_seed:
                raise ValueError("projection_seed dictionary cannot be empty")
            expected_keys = set(WORKLOADS.keys())
            provided_keys = set(projection_seed.keys())
            missing_keys = expected_keys - provided_keys
            unknown_keys = provided_keys - expected_keys
            if missing_keys or unknown_keys:
                details = []
                if missing_keys:
                    details.append(f"missing required workload key(s) {sorted(missing_keys)}")
                if unknown_keys:
                    details.append(f"Unknown workload_id key(s) {sorted(unknown_keys)}")
                raise ValueError(
                    f"projection_seed dictionary must contain exactly WORKLOADS keys ({sorted(expected_keys)}); "
                    + ", ".join(details)
                )
            for k, v in projection_seed.items():
                if (
                    not isinstance(v, int)
                    or isinstance(v, bool)
                    or not (0 <= v < (2**31 - 1))
                ):
                    raise ValueError(
                        f"projection_seed for workload '{k}' must be a non-negative integer < 2**31-1, got {v!r}"
                    )
        else:
            raise ValueError(
                f"projection_seed must be a non-negative integer < 2**31-1 or a "
                f"Dict[str, int], got {projection_seed!r}"
            )
    else:
        if projection_seed is not None:
            raise ValueError(
                f"projection_seed can only be provided when projection_seed_mode is 'explicit', got mode='{projection_seed_mode}' and projection_seed={projection_seed!r}"
            )
def parse_projection_scope_arg(val: Any) -> Union[str, Dict[str, str]]:
    """
    Parses projection_scope parameter or CLI arg into a string or Dict[str, str].
    Accepts valid scope strings, JSON dict strings, or comma/colon key-value strings.
    """
    if val is None or val == "":
        return DEFAULT_PROJECTION_SCOPE
    if isinstance(val, dict):
        return {str(k): str(v) for k, v in val.items()}
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str:
            return DEFAULT_PROJECTION_SCOPE
        if val_str in PROJECTION_SCOPES:
            return val_str
        if val_str.startswith("{") and val_str.endswith("}"):
            try:
                parsed = json.loads(val_str)
                if isinstance(parsed, dict):
                    return {str(k): str(v) for k, v in parsed.items()}
            except Exception as exc:
                raise ValueError(f"Failed to parse projection_scope JSON dict string '{val}': {exc}")
        if ":" in val_str or "=" in val_str:
            res = {}
            for item in val_str.replace(";", ",").split(","):
                item = item.strip()
                if not item:
                    continue
                if ":" in item:
                    k, v = item.split(":", 1)
                elif "=" in item:
                    k, v = item.split("=", 1)
                else:
                    raise ValueError(f"Invalid key-value projection_scope string item '{item}'")
                res[k.strip()] = v.strip()
            if res:
                return res
        return val_str
    raise ValueError(f"Cannot parse projection_scope value: {val!r}")


def validate_projection_scope_config(projection_scope: Any) -> None:
    """
    Validates projection_scope string or dictionary config.
    Raises ValueError if string is not in PROJECTION_SCOPES, or if dict
    keys do not match WORKLOADS exactly or values are not in PROJECTION_SCOPES.
    """
    if isinstance(projection_scope, str):
        if projection_scope not in PROJECTION_SCOPES:
            raise ValueError(
                f"Invalid projection_scope '{projection_scope}'. Must be one of {list(PROJECTION_SCOPES)}"
            )
    elif isinstance(projection_scope, dict):
        if not projection_scope:
            raise ValueError("projection_scope dictionary cannot be empty")
        if set(projection_scope.keys()) != set(WORKLOADS.keys()):
            missing = sorted(list(set(WORKLOADS.keys()) - set(projection_scope.keys())))
            extra = sorted(list(set(projection_scope.keys()) - set(WORKLOADS.keys())))
            details = []
            if missing:
                details.append(f"missing keys {missing}")
            if extra:
                details.append(f"unknown keys {extra}")
            raise ValueError(
                f"projection_scope dictionary must contain exact workload keys {sorted(list(WORKLOADS.keys()))}, got {', '.join(details)}"
            )
        for k, v in projection_scope.items():
            if v not in PROJECTION_SCOPES:
                raise ValueError(
                    f"Invalid projection_scope '{v}' for workload '{k}'. Must be one of {list(PROJECTION_SCOPES)}"
                )
    else:
        raise ValueError(
            f"projection_scope must be a string or Dict[str, str], got {projection_scope!r}"
        )


def get_effective_projection_scope(
    projection_scope: Union[str, Dict[str, str]],
    workload_id: str,
) -> str:
    """
    Returns the effective projection scope string for a given workload.
    """
    if isinstance(projection_scope, dict):
        if workload_id not in projection_scope:
            raise ValueError(f"Missing workload_id '{workload_id}' in projection_scope dict")
        return str(projection_scope[workload_id])
    return str(projection_scope)


def allocate_tensor_latent_dims(
    param_numels: Sequence[int],
    aggregate_latent_dim: int,
) -> List[int]:
    """
    Allocates aggregate_latent_dim across parameter tensors deterministically,
    proportionally to tensor numel, with at least one coordinate per tensor,
    no tensor exceeding numel, and exact sum aggregate_latent_dim.
    """
    total_dim = sum(param_numels)
    if not (0 < aggregate_latent_dim <= total_dim):
        raise ValueError(
            f"aggregate_latent_dim must be in (0, {total_dim}], got {aggregate_latent_dim}"
        )
    num_tensors = len(param_numels)
    if aggregate_latent_dim < num_tensors:
        raise ValueError(
            f"aggregate_latent_dim ({aggregate_latent_dim}) must be at least number of parameter tensors ({num_tensors})"
        )

    if aggregate_latent_dim == total_dim:
        return list(param_numels)

    quotas = [aggregate_latent_dim * n / total_dim for n in param_numels]
    allocs = [max(1, min(n, int(math.floor(q)))) for n, q in zip(param_numels, quotas)]
    current_sum = sum(allocs)

    if current_sum < aggregate_latent_dim:
        deficit = aggregate_latent_dim - current_sum
        candidates = [i for i in range(num_tensors) if allocs[i] < param_numels[i]]
        candidates.sort(
            key=lambda i: (quotas[i] - math.floor(quotas[i]), param_numels[i], -i),
            reverse=True,
        )
        for i in candidates[:deficit]:
            allocs[i] += 1
    elif current_sum > aggregate_latent_dim:
        surplus = current_sum - aggregate_latent_dim
        candidates = [i for i in range(num_tensors) if allocs[i] > 1]
        candidates.sort(
            key=lambda i: (quotas[i] - math.floor(quotas[i]), param_numels[i], -i),
            reverse=False,
        )
        for i in candidates[:surplus]:
            allocs[i] -= 1

    assert sum(allocs) == aggregate_latent_dim, (
        f"Allocation sum {sum(allocs)} does not match aggregate_latent_dim {aggregate_latent_dim}"
    )
    assert all(1 <= a <= n for a, n in zip(allocs, param_numels)), (
        f"Allocation bounds violated: {allocs} vs numels {param_numels}"
    )
    return allocs


class TensorLocalLatentTransform(V6LatentTransform):
    """
    Subclass of V6LatentTransform that isolates signed-hash coordinates within each model
    parameter tensor, mapping each tensor's parameters only to its assigned contiguous latent slice.
    """

    def __init__(
        self,
        base_model: nn.Module,
        geom_config: V6GeometryConfig,
        device: torch.device,
    ):
        super().__init__(base_model, geom_config, device)
        if not self.is_full:
            self.tensor_latent_dims = allocate_tensor_latent_dims(
                self.param_numels, self.latent_dim
            )
            k_indices = np.zeros(self.total_dim, dtype=np.int64)
            h2_signs = np.zeros(self.total_dim, dtype=np.float32)
            seed_offset = (
                geom_config.projection_seed
                if geom_config.projection_seed is not None
                else 0
            )

            j_offset = 0
            l_offset = 0
            for numel, d_m in zip(self.param_numels, self.tensor_latent_dims):
                j_local = np.arange(numel, dtype=np.int64)
                j_global = j_offset + j_local
                h1 = ((j_global + 1 + seed_offset) * 2654435761) % (2**32)
                k_local = h1 % d_m
                k_indices[j_global] = l_offset + k_local
                h2 = ((j_global + 1 + seed_offset) * 1597334677) % (2**32)
                h2_signs[j_global] = np.where((h2 % 2) == 0, 1.0, -1.0)
                j_offset += numel
                l_offset += d_m

            bin_counts = np.bincount(k_indices, minlength=self.latent_dim)
            count_per_j = bin_counts[k_indices]
            scale_per_j = 1.0 / np.sqrt(np.maximum(count_per_j, 1))
            combined_weights = h2_signs * scale_per_j

            self.k_indices = torch.tensor(k_indices, dtype=torch.long, device=device)
            self.weights = torch.tensor(combined_weights, dtype=torch.float32, device=device)
        else:
            self.tensor_latent_dims = list(self.param_numels)

class BalancedGlobalLatentTransform(V6LatentTransform):
    """
    Subclass of V6LatentTransform that enforces balanced parameter occupancy
    across global latent coordinate buckets using a deterministic permutation and sign assignment.
    """

    def __init__(
        self,
        base_model: nn.Module,
        geom_config: V6GeometryConfig,
        device: torch.device,
    ):
        super().__init__(base_model, geom_config, device)
        if not self.is_full:
            seed_offset = (
                geom_config.projection_seed
                if geom_config.projection_seed is not None
                else 0
            )
            rng = np.random.RandomState(seed_offset)
            perm = rng.permutation(self.total_dim)
            k_indices = np.zeros(self.total_dim, dtype=np.int64)
            k_indices[perm] = np.arange(self.total_dim, dtype=np.int64) % self.latent_dim
            h2_signs = rng.choice(np.array([1.0, -1.0], dtype=np.float32), size=self.total_dim)

            bin_counts = np.bincount(k_indices, minlength=self.latent_dim)
            count_per_j = bin_counts[k_indices]
            scale_per_j = 1.0 / np.sqrt(np.maximum(count_per_j, 1))
            combined_weights = h2_signs * scale_per_j

            self.k_indices = torch.tensor(k_indices, dtype=torch.long, device=device)
            self.weights = torch.tensor(combined_weights, dtype=torch.float32, device=device)


class TwoHashGlobalLatentTransform(V6LatentTransform):
    """
    Subclass of V6LatentTransform that maps each parameter to two distinct global latent
    coordinate buckets using independent deterministic signed-hash mappings, normalized by
    per-bucket 1/sqrt(count) weights and decoded as (term1 + term2)/sqrt(2).
    """

    def __init__(
        self,
        base_model: nn.Module,
        geom_config: V6GeometryConfig,
        device: torch.device,
    ):
        super().__init__(base_model, geom_config, device)
        if not self.is_full:
            j_indices = np.arange(self.total_dim, dtype=np.int64)
            seed_offset = (
                geom_config.projection_seed
                if geom_config.projection_seed is not None
                else 0
            )

            # Hash Map 1 (Primary signed hash projection)
            h1_1 = ((j_indices + 1 + seed_offset) * 2654435761) % (2**32)
            k1_indices = h1_1 % self.latent_dim
            h1_2 = ((j_indices + 1 + seed_offset) * 1597334677) % (2**32)
            signs1 = np.where((h1_2 % 2) == 0, 1.0, -1.0)

            bin_counts1 = np.bincount(k1_indices, minlength=self.latent_dim)
            count_per_j1 = bin_counts1[k1_indices]
            scale_per_j1 = 1.0 / np.sqrt(np.maximum(count_per_j1, 1))
            combined_weights1 = signs1 * scale_per_j1

            # Hash Map 2 (Secondary independent signed hash projection)
            h2_1 = ((j_indices + 1 + seed_offset) * 2246822519) % (2**32)
            if self.latent_dim > 1:
                offset = 1 + (h2_1 % (self.latent_dim - 1))
                k2_indices = (k1_indices + offset) % self.latent_dim
            else:
                k2_indices = np.zeros(self.total_dim, dtype=np.int64)

            h2_2 = ((j_indices + 1 + seed_offset) * 3266489917) % (2**32)
            signs2 = np.where(((h2_2 >> 16) % 2) == 0, 1.0, -1.0)

            bin_counts2 = np.bincount(k2_indices, minlength=self.latent_dim)
            count_per_j2 = bin_counts2[k2_indices]
            scale_per_j2 = 1.0 / np.sqrt(np.maximum(count_per_j2, 1))
            combined_weights2 = signs2 * scale_per_j2

            self.k1_indices = torch.tensor(k1_indices, dtype=torch.long, device=device)
            self.weights1 = torch.tensor(combined_weights1, dtype=torch.float32, device=device)
            self.k2_indices = torch.tensor(k2_indices, dtype=torch.long, device=device)
            self.weights2 = torch.tensor(combined_weights2, dtype=torch.float32, device=device)

            # Backward compatibility aliases
            self.k_indices = self.k1_indices
            self.weights = self.weights1

    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Transforms latent batch Z (N, d) into full parameter batch (N, D).
        When not full, term1 = Z[:, k1] * weights1, term2 = Z[:, k2] * weights2,
        delta = (term1 + term2) / sqrt(2).
        theta = base_vec + scale_vec * delta
        """
        if self.is_full:
            delta = Z
        else:
            term1 = Z[:, self.k1_indices] * self.weights1
            term2 = Z[:, self.k2_indices] * self.weights2
            delta = (term1 + term2) / math.sqrt(2.0)
        return self.base_vec + self.scale_vec * delta


class LargestTensorHashLatentTransform(V6LatentTransform):
    """
    Subclass of V6LatentTransform that isolates the single largest parameter tensor by numel
    (with stable first-index tie break) for signed-hash projection, while assigning every parameter
    in all other tensors a unique direct latent coordinate with weight 1.0.
    """

    def __init__(
        self,
        base_model: nn.Module,
        geom_config: V6GeometryConfig,
        device: torch.device,
    ):
        super().__init__(base_model, geom_config, device)
        if not self.is_full:
            largest_idx = int(np.argmax(self.param_numels))
            protected_dim = sum(
                numel for i, numel in enumerate(self.param_numels) if i != largest_idx
            )
            residual_dim = self.latent_dim - protected_dim
            if residual_dim < 1:
                raise ValueError(
                    f"latent_dim ({self.latent_dim}) must be greater than protected non-largest parameter dimension ({protected_dim}) "
                    f"to provide at least 1 residual latent coordinate for the largest tensor (index {largest_idx}, numel {self.param_numels[largest_idx]})"
                )

            k_indices = np.zeros(self.total_dim, dtype=np.int64)
            weights = np.zeros(self.total_dim, dtype=np.float32)
            seed_offset = (
                geom_config.projection_seed
                if geom_config.projection_seed is not None
                else 0
            )

            j_offset = 0
            direct_coord = 0
            for i, numel in enumerate(self.param_numels):
                j_indices_tensor = np.arange(j_offset, j_offset + numel, dtype=np.int64)
                if i != largest_idx:
                    k_indices[j_indices_tensor] = direct_coord + np.arange(numel, dtype=np.int64)
                    weights[j_indices_tensor] = 1.0
                    direct_coord += numel
                else:
                    h1 = ((j_indices_tensor + 1 + seed_offset) * 2654435761) % (2**32)
                    k_rel = h1 % residual_dim
                    k_indices[j_indices_tensor] = protected_dim + k_rel
                    h2 = ((j_indices_tensor + 1 + seed_offset) * 1597334677) % (2**32)
                    h2_signs = np.where((h2 % 2) == 0, 1.0, -1.0)
                    weights[j_indices_tensor] = h2_signs
                j_offset += numel

            bin_counts = np.bincount(k_indices, minlength=self.latent_dim)
            count_per_j = bin_counts[k_indices]
            scale_per_j = 1.0 / np.sqrt(np.maximum(count_per_j, 1))
            combined_weights = weights * scale_per_j

            self.k_indices = torch.tensor(k_indices, dtype=torch.long, device=device)
            self.weights = torch.tensor(combined_weights, dtype=torch.float32, device=device)


class LargestTensorRowHashLatentTransform(V6LatentTransform):
    """
    Subclass of V6LatentTransform that isolates the single largest parameter tensor by numel
    (with stable first-index tie break) for row-partitioned signed-hash projection across its first dimension
    (rows), while assigning every parameter in all other tensors a unique direct latent coordinate with weight 1.0.
    """

    def __init__(
        self,
        base_model: nn.Module,
        geom_config: V6GeometryConfig,
        device: torch.device,
    ):
        super().__init__(base_model, geom_config, device)
        largest_idx = int(np.argmax(self.param_numels))
        shape = self.param_shapes[largest_idx]
        num_rows = shape[0] if len(shape) >= 2 else 1
        largest_numel = self.param_numels[largest_idx]
        if len(shape) >= 2:
            elements_per_row = largest_numel // num_rows
            row_numels = [elements_per_row] * num_rows
        else:
            row_numels = [largest_numel]

        if not self.is_full:
            protected_dim = sum(
                numel for i, numel in enumerate(self.param_numels) if i != largest_idx
            )
            residual_dim = self.latent_dim - protected_dim
            if residual_dim < num_rows:
                raise ValueError(
                    f"latent_dim ({self.latent_dim}) must be at least protected dimension ({protected_dim}) "
                    f"+ number of rows ({num_rows}) for largest tensor row hashing, got residual_dim {residual_dim}"
                )

            row_latent_dims = allocate_tensor_latent_dims(row_numels, residual_dim)

            k_indices = np.zeros(self.total_dim, dtype=np.int64)
            weights = np.zeros(self.total_dim, dtype=np.float32)
            seed_offset = (
                geom_config.projection_seed
                if geom_config.projection_seed is not None
                else 0
            )

            j_offset = 0
            direct_coord = 0
            for i, numel in enumerate(self.param_numels):
                if i != largest_idx:
                    j_indices_tensor = np.arange(j_offset, j_offset + numel, dtype=np.int64)
                    k_indices[j_indices_tensor] = direct_coord + np.arange(numel, dtype=np.int64)
                    weights[j_indices_tensor] = 1.0
                    direct_coord += numel
                    j_offset += numel
                else:
                    row_slice_start = protected_dim
                    for r, (r_numel, r_dim) in enumerate(zip(row_numels, row_latent_dims)):
                        j_indices_row = np.arange(j_offset, j_offset + r_numel, dtype=np.int64)
                        h1 = ((j_indices_row + 1 + seed_offset) * 2654435761) % (2**32)
                        k_rel = h1 % r_dim
                        k_indices[j_indices_row] = row_slice_start + k_rel
                        h2 = ((j_indices_row + 1 + seed_offset) * 1597334677) % (2**32)
                        h2_signs = np.where((h2 % 2) == 0, 1.0, -1.0)
                        weights[j_indices_row] = h2_signs
                        j_offset += r_numel
                        row_slice_start += r_dim

            bin_counts = np.bincount(k_indices, minlength=self.latent_dim)
            count_per_j = bin_counts[k_indices]
            scale_per_j = 1.0 / np.sqrt(np.maximum(count_per_j, 1))
            combined_weights = weights * scale_per_j

            self.k_indices = torch.tensor(k_indices, dtype=torch.long, device=device)
            self.weights = torch.tensor(combined_weights, dtype=torch.float32, device=device)
            self.row_latent_dims = row_latent_dims
        else:
            self.row_latent_dims = row_numels

class AdjacentPairLatentTransform(V6LatentTransform):
    """
    Subclass of V6LatentTransform that maps each parameter tensor independently
    by assigning consecutive pairs of parameters to one unique latent coordinate with weights 1/sqrt(2),
    and a final unpaired parameter (if any) to weight 1.0, concatenating tensor coordinate ranges without sharing.
    """

    def __init__(
        self,
        base_model: nn.Module,
        geom_config: V6GeometryConfig,
        device: torch.device,
    ):
        super().__init__(base_model, geom_config, device)
        if not self.is_full:
            required_dim = sum(math.ceil(n / 2) for n in self.param_numels)
            if self.latent_dim != required_dim:
                raise ValueError(
                    f"AdjacentPairLatentTransform requires latent_dim == sum(ceil(numel_i/2)) = {required_dim}, got {self.latent_dim}"
                )
            k_indices = np.zeros(self.total_dim, dtype=np.int64)
            weights = np.zeros(self.total_dim, dtype=np.float32)
            inv_sqrt2 = 1.0 / math.sqrt(2.0)

            j_offset = 0
            l_offset = 0
            self.tensor_latent_dims = []
            for numel in self.param_numels:
                d_m = math.ceil(numel / 2)
                self.tensor_latent_dims.append(d_m)
                for p in range(numel):
                    j = j_offset + p
                    k_local = p // 2
                    k_indices[j] = l_offset + k_local
                    if p % 2 == 0 and p == numel - 1:
                        weights[j] = 1.0
                    else:
                        weights[j] = inv_sqrt2
                j_offset += numel
                l_offset += d_m

            self.k_indices = torch.tensor(k_indices, dtype=torch.long, device=device)
            self.weights = torch.tensor(weights, dtype=torch.float32, device=device)
        else:
            self.tensor_latent_dims = list(self.param_numels)
class AdjacentDifferenceLatentTransform(AdjacentPairLatentTransform):
    """
    Subclass of AdjacentPairLatentTransform that maps each parameter tensor independently
    by assigning consecutive pairs of parameters to one unique latent coordinate with opposite weights
    (+1/sqrt(2), -1/sqrt(2)), and a final unpaired parameter (if any) to weight 1.0,
    concatenating tensor coordinate ranges without sharing.
    """

    def __init__(
        self,
        base_model: nn.Module,
        geom_config: V6GeometryConfig,
        device: torch.device,
    ):
        try:
            super().__init__(base_model, geom_config, device)
        except ValueError as e:
            raise ValueError(
                str(e).replace("AdjacentPairLatentTransform", "AdjacentDifferenceLatentTransform")
            ) from None

        if not self.is_full:
            weights = self.weights.clone()
            j_offset = 0
            for numel in self.param_numels:
                for p in range(1, numel, 2):
                    weights[j_offset + p] = -weights[j_offset + p]
                j_offset += numel
            self.weights = weights


def format_ratio_id(
    ratio: float,
    geometry_policy: str = DEFAULT_GEOMETRY_POLICY,
    projection_scope: Union[str, Dict[str, str]] = DEFAULT_PROJECTION_SCOPE,
    projection_seed_mode: str = DEFAULT_PROJECTION_SEED_MODE,
    projection_seed: Optional[Union[int, Dict[str, int]]] = None,
    geometry_multiplier: float = DEFAULT_GEOMETRY_MULTIPLIER,
) -> str:
    validate_geometry_multiplier(geometry_multiplier)
    validate_projection_seed_config(projection_seed_mode, projection_seed)
    validate_projection_scope_config(projection_scope)
    r_str = f"{ratio:g}"
    base_id = f"aligned_r{r_str}" if geometry_policy == "baseline_aligned" else f"r{r_str}"
    if isinstance(projection_scope, dict):
        unique_scopes = set(projection_scope.values())
        if len(unique_scopes) == 1:
            eff_scope = next(iter(unique_scopes))
            if eff_scope == "tensor_local":
                base_id = f"local_{base_id}"
            elif eff_scope == "balanced_global":
                base_id = f"balanced_{base_id}"
            elif eff_scope == "two_hash_global":
                base_id = f"two_hash_{base_id}"
            elif eff_scope == "largest_tensor_hash":
                base_id = f"largest_tensor_hash_{base_id}"
            elif eff_scope == "largest_tensor_row_hash":
                base_id = f"largest_tensor_row_hash_{base_id}"
            elif eff_scope == "adjacent_pair":
                base_id = f"adjacent_pair_{base_id}"
            elif eff_scope == "adjacent_difference":
                base_id = f"adjacent_difference_{base_id}"
        else:
            base_id = f"mixed_{base_id}"
    elif projection_scope == "tensor_local":
        base_id = f"local_{base_id}"
    elif projection_scope == "balanced_global":
        base_id = f"balanced_{base_id}"
    elif projection_scope == "two_hash_global":
        base_id = f"two_hash_{base_id}"
    elif projection_scope == "largest_tensor_hash":
        base_id = f"largest_tensor_hash_{base_id}"
    elif projection_scope == "largest_tensor_row_hash":
        base_id = f"largest_tensor_row_hash_{base_id}"
    elif projection_scope == "adjacent_pair":
        base_id = f"adjacent_pair_{base_id}"
    elif projection_scope == "adjacent_difference":
        base_id = f"adjacent_difference_{base_id}"
    if projection_seed_mode == "fixed":
        base_id = f"fixed_{base_id}"
    elif projection_seed_mode == "explicit":
        if isinstance(projection_seed, dict):
            base_id = f"pexplicit_{base_id}"
        else:
            base_id = f"p{projection_seed}_{base_id}"
    if float(geometry_multiplier) != 1.0:
        g_str = f"{geometry_multiplier:g}"
        base_id = f"g{g_str}_{base_id}"
    return base_id
# Geometry Policies Assignment
GEOMETRY_POLICIES: Dict[str, Dict[str, str]] = {
    "recovered": {
        "mnist_compact": "G6",
        "mnist_wide": "G5",
        "fashion_compact": "G6",
        "fashion_wide": "G5",
    },
    "baseline_aligned": {
        "mnist_compact": "G8",
        "mnist_wide": "G5",
        "fashion_compact": "G8",
        "fashion_wide": "G5",
    },
}

# Base Candidate Geometry Assignment (retained for backward compatibility)
BASE_GEOMETRIES: Dict[str, str] = GEOMETRY_POLICIES["recovered"]
BASELINE_METHODS: Dict[str, str] = {
    "mnist_compact": "G8",
    "mnist_wide": "G5",
    "fashion_compact": "G8",
    "fashion_wide": "G5",
}



def compute_latent_dim(total_dim: int, ratio: float) -> int:
    """
    Computes exact model-relative latent dimension from total parameter count and ratio.
    Validates 0 < ratio <= 1.0 and uses deterministic half-up rounding floor(D*ratio+0.5).
    Ensures dimension is at least 1 and capped at total_dim.
    """
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    dim = int(math.floor(total_dim * ratio + 0.5))
    return max(1, min(total_dim, dim))


def derive_projection_seed(
    workload_id: str,
    ratio: float,
    seed: int,
    projection_salt: str = "",
    mode: str = DEFAULT_PROJECTION_SEED_MODE,
    projection_seed_mode: Optional[str] = None,
    projection_seed: Optional[Union[int, Dict[str, int]]] = None,
) -> int:
    """
    Derives a deterministic 32-bit projection seed from workload_id, ratio, swarm seed,
    optional projection_salt, and projection seed mode ('coupled', 'fixed', or 'explicit').
    In 'explicit' mode, returns projection_seed directly (or projection_seed[workload_id] if a dict).
    """
    if projection_seed_mode is not None:
        mode = projection_seed_mode
    validate_projection_seed_config(mode, projection_seed)
    if mode == "explicit":
        if isinstance(projection_seed, dict):
            if workload_id not in projection_seed:
                raise ValueError(f"Missing explicit projection_seed for workload '{workload_id}'")
            return projection_seed[workload_id]
        assert projection_seed is not None
        return projection_seed
    parts = [workload_id, f"{ratio:.5f}"]
    if mode != "fixed":
        parts.append(str(seed))
    if projection_salt:
        parts.append(projection_salt)
    key = ":".join(parts).encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)

def construct_equalized_geometry(
    base_geom: V6GeometryConfig,
    total_dim: int,
    latent_dim: int,
    projection_seed: int,
    ratio_str: str = "",
    geometry_multiplier: float = DEFAULT_GEOMETRY_MULTIPLIER,
) -> V6GeometryConfig:
    """
    Constructs an equalized subspace geometry configuration from a base geometry (G6 or G5).
    Multiplies position radius, launch velocity radius, mutation reset radius, and reflective bound
    by sqrt(total_dim / latent_dim) * geometry_multiplier to hold decoded per-parameter variance constant.
    """
    validate_geometry_multiplier(geometry_multiplier)
    scale_factor = math.sqrt(total_dim / latent_dim) if latent_dim < total_dim else 1.0
    effective_mult = scale_factor * float(geometry_multiplier)
    return V6GeometryConfig(
        config_id=f"{base_geom.config_id}_eq_{ratio_str}",
        scale_type=base_geom.scale_type,
        init_position_mode=base_geom.init_position_mode,
        position_radius=float(base_geom.position_radius * effective_mult),
        initial_velocity_radius=float(base_geom.initial_velocity_radius * effective_mult),
        mutation_prob=base_geom.mutation_prob,
        reset_velocity_radius=float(base_geom.reset_velocity_radius * effective_mult),
        reflective_bound=float(base_geom.reflective_bound * effective_mult),
        projection_seed=projection_seed,
        latent_dim=latent_dim,
        description=f"Equalized subspace (ratio={ratio_str}, d={latent_dim}/{total_dim})",
    )

def compute_core_swarm_state_bytes(particles: int, latent_dim: int) -> int:
    """
    Computes exact core swarm state memory footprint in bytes.
    5 tensors (Z, V, M, V_sq, P) of size (particles, latent_dim) float32 (4 bytes/elem).
    """
    return 5 * particles * latent_dim * 4

def compute_baseline_core_swarm_state_bytes(
    workload_id: str,
    particles: int,
    total_dim: int,
) -> int:
    """Match the retained baseline method's persistent core-state accounting."""
    particle_states = 5 * particles
    if BASELINE_METHODS[workload_id] == "G8":
        particle_states += 1
    return particle_states * total_dim * 4




def run_heavy_pso_autoresearch(
    ratios: Sequence[float] = DEFAULT_RATIOS,
    particles: int = DEFAULT_PARTICLES,
    epochs: int = DEFAULT_EPOCHS,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    geometry_policy: str = DEFAULT_GEOMETRY_POLICY,
    device_str: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
    projection_salt: str = "",
    projection_scope: Union[str, Dict[str, str]] = DEFAULT_PROJECTION_SCOPE,
    projection_seed_mode: str = DEFAULT_PROJECTION_SEED_MODE,
    projection_seed: Optional[Union[int, Dict[str, int]]] = None,
    geometry_multiplier: float = DEFAULT_GEOMETRY_MULTIPLIER,
) -> Dict[str, Any]:
    """
    Executes the heavy task autoresearch experiment candidate runs across candidate ratios,
    workloads, and seeds. Aggregates metrics and atomically writes the candidate JSON artifact.
    """
    validate_geometry_multiplier(geometry_multiplier)
    validate_projection_seed_config(projection_seed_mode, projection_seed)
    validate_projection_scope_config(projection_scope)
    if geometry_policy not in GEOMETRY_POLICIES:
        raise ValueError(
            f"Invalid geometry_policy '{geometry_policy}'. Must be one of {list(GEOMETRY_POLICIES.keys())}"
        )
    policy_geometries = GEOMETRY_POLICIES[geometry_policy]

    start_time = time.time()
    device = resolve_execution_device(device_str)
    hardware_info = get_hardware_provenance(device)
    geom_table = get_v6_geometry_table()

    if cache_dir is None:
        cache_dir = REPO_ROOT / "result" / "cache"

    data_cache: Dict[str, Any] = {}
    workload_meta: Dict[str, Any] = {}

    # Pre-load datasets and models metadata
    for wl_id, wl_cfg in WORKLOADS.items():
        if wl_cfg.dataset_name not in data_cache:
            x_search, y_search, x_val, y_val, nested_subsets, data_fp, provenance = prepare_heavy_task_data(
                dataset_name=wl_cfg.dataset_name,
                split_seed=split_seed,
                cache_dir=cache_dir,
            )
            data_cache[wl_cfg.dataset_name] = {
                "x_search": x_search,
                "y_search": y_search,
                "x_val": x_val,
                "y_val": y_val,
                "nested_subsets": nested_subsets,
                "data_fp": data_fp,
                "provenance": provenance,
            }
        
        bm = create_model(wl_cfg.model_name, seed=41)
        param_count = sum(p.numel() for p in bm.parameters())
        model_fp = compute_model_fingerprint(bm)
        d_info = data_cache[wl_cfg.dataset_name]

        eff_proj_seed = (
            projection_seed[wl_id]
            if isinstance(projection_seed, dict)
            else projection_seed
        )

        workload_meta[wl_id] = {
            "workload_id": wl_id,
            "dataset_name": wl_cfg.dataset_name,
            "model_name": wl_cfg.model_name,
            "parameter_count": param_count,
            "total_dim": param_count,
            "geometry_policy": geometry_policy,
            "geometry_multiplier": float(geometry_multiplier),
            "projection_scope": get_effective_projection_scope(projection_scope, wl_id),
            "projection_seed_mode": str(projection_seed_mode),
            "projection_seed": eff_proj_seed,
            "base_geometry_id": policy_geometries[wl_id],
            "model_fingerprint": model_fp,
            "data_fingerprint": d_info["data_fp"],
            "split_fingerprint": d_info["provenance"]["split_fingerprint"],
            "description": wl_cfg.description,
        }

    candidate_runs: Dict[str, Dict[str, Any]] = {}
    total_runs_executed = 0
    total_queries_executed = 0
    total_samples_evaluated = 0

    for r in ratios:
        ratio_id = format_ratio_id(
            r,
            geometry_policy=geometry_policy,
            projection_scope=projection_scope,
            projection_seed_mode=projection_seed_mode,
            projection_seed=projection_seed,
            geometry_multiplier=geometry_multiplier,
        )
        wl_candidates: Dict[str, Any] = {}

        for wl_id, wl_cfg in WORKLOADS.items():
            d_info = data_cache[wl_cfg.dataset_name]
            x_search, y_search = d_info["x_search"], d_info["y_search"]
            x_val, y_val = d_info["x_val"], d_info["y_val"]
            nested_subsets = d_info["nested_subsets"]

            base_model = create_model(wl_cfg.model_name, seed=41)
            total_dim = sum(p.numel() for p in base_model.parameters())
            latent_dim = compute_latent_dim(total_dim, r)
            base_geom_id = policy_geometries[wl_id]
            base_geom = geom_table[base_geom_id]
            state_bytes = compute_core_swarm_state_bytes(particles, latent_dim)
            baseline_state_bytes = compute_baseline_core_swarm_state_bytes(
                workload_id=wl_id,
                particles=particles,
                total_dim=total_dim,
            )

            per_seed_runs: List[Dict[str, Any]] = []

            for s in seeds:
                proj_seed = derive_projection_seed(
                    wl_id,
                    r,
                    s,
                    projection_salt=projection_salt,
                    mode=projection_seed_mode,
                    projection_seed=projection_seed,
                )
                geom_cfg = construct_equalized_geometry(
                    base_geom=base_geom,
                    total_dim=total_dim,
                    latent_dim=latent_dim,
                    projection_seed=proj_seed,
                    ratio_str=ratio_id,
                    geometry_multiplier=geometry_multiplier,
                )

                eff_scope = get_effective_projection_scope(projection_scope, wl_id)
                if eff_scope == "tensor_local":
                    transform = TensorLocalLatentTransform(base_model, geom_cfg, device)
                elif eff_scope == "balanced_global":
                    transform = BalancedGlobalLatentTransform(base_model, geom_cfg, device)
                elif eff_scope == "two_hash_global":
                    transform = TwoHashGlobalLatentTransform(base_model, geom_cfg, device)
                elif eff_scope == "largest_tensor_hash":
                    transform = LargestTensorHashLatentTransform(base_model, geom_cfg, device)
                elif eff_scope == "largest_tensor_row_hash":
                    transform = LargestTensorRowHashLatentTransform(base_model, geom_cfg, device)
                elif eff_scope == "adjacent_pair":
                    transform = AdjacentPairLatentTransform(base_model, geom_cfg, device)
                elif eff_scope == "adjacent_difference":
                    transform = AdjacentDifferenceLatentTransform(base_model, geom_cfg, device)
                elif eff_scope == "global":
                    transform = V6LatentTransform(base_model, geom_cfg, device)
                else:
                    raise ValueError(f"Invalid effective projection_scope '{eff_scope}' for workload '{wl_id}'")
                res = run_v6_pso(
                    transform=transform,
                    base_model=base_model,
                    x_search=x_search,
                    y_search=y_search,
                    x_val=x_val,
                    y_val=y_val,
                    nested_subsets=nested_subsets,
                    schedule_str=f"{subset_size}:{epochs}",
                    epochs=epochs,
                    swarm_size=particles,
                    seed=s,
                    device=device,
                    geom_config=geom_cfg,
                    val_check_interval=10,
                )

                opt_time = max(float(res["optimization_wall_time_sec"]), 1e-6)
                sps = round(float(res["total_sample_evaluations"]) / opt_time, 2)
                val_metrics_raw = artifact_safe_run(res.get("val_metrics", {}))
                has_valid_val_metrics = (
                    isinstance(val_metrics_raw, dict)
                    and len(val_metrics_raw) > 0
                    and all(
                        isinstance(v, (int, float)) and math.isfinite(float(v))
                        for v in val_metrics_raw.values()
                    )
                )
                scalar_metrics = [
                    res.get("val_selected_loss"),
                    res.get("val_selected_acc"),
                    res.get("gbest_loss"),
                    res.get("gbest_acc"),
                    res.get("wall_time_sec"),
                    res.get("optimization_wall_time_sec"),
                    res.get("validation_wall_time_sec"),
                    sps,
                ]
                has_valid_scalars = all(
                    v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))
                    for v in scalar_metrics
                )
                is_finite = bool(has_valid_val_metrics and has_valid_scalars)

                seed_record = {
                    "seed": int(s),
                    "projection_seed": int(proj_seed),
                    "projection_salt": str(projection_salt),
                    "projection_scope": get_effective_projection_scope(projection_scope, wl_id),
                    "projection_seed_mode": str(projection_seed_mode),
                    "geometry_multiplier": float(geometry_multiplier),
                    "val_selected_loss": float(res["val_selected_loss"]),
                    "val_selected_acc": float(res["val_selected_acc"]),
                    "val_metrics": val_metrics_raw,
                    "gbest_loss": float(res["gbest_loss"]),
                    "gbest_acc": float(res["gbest_acc"]),
                    "wall_time_sec": float(res["wall_time_sec"]),
                    "optimization_wall_time_sec": float(res["optimization_wall_time_sec"]),
                    "validation_wall_time_sec": float(res["validation_wall_time_sec"]),
                    "total_queries": int(res["total_queries"]),
                    "total_sample_evaluations": int(res["total_sample_evaluations"]),
                    "validation_evaluations": int(res["validation_evaluations"]),
                    "official_test_evaluations": 0,
                    "core_swarm_state_bytes": int(state_bytes),
                    "throughput_samples_per_sec": float(sps),
                    "is_finite": is_finite,
                }
                per_seed_runs.append(seed_record)

                total_runs_executed += 1
                total_queries_executed += int(res["total_queries"])
                total_samples_evaluated += int(res["total_sample_evaluations"])

            # Statistics calculation across seeds
            acc_list = [r_entry["val_selected_acc"] for r_entry in per_seed_runs]
            nll_list = [r_entry["val_selected_loss"] for r_entry in per_seed_runs]
            brier_list = [r_entry["val_metrics"]["brier"] for r_entry in per_seed_runs if r_entry.get("val_metrics")]
            ece_list = [r_entry["val_metrics"]["ece"] for r_entry in per_seed_runs if r_entry.get("val_metrics")]
            g_loss_list = [r_entry["gbest_loss"] for r_entry in per_seed_runs]
            g_acc_list = [r_entry["gbest_acc"] for r_entry in per_seed_runs]
            wall_list = [r_entry["wall_time_sec"] for r_entry in per_seed_runs]
            opt_wall_list = [r_entry["optimization_wall_time_sec"] for r_entry in per_seed_runs]
            sps_list = [r_entry["throughput_samples_per_sec"] for r_entry in per_seed_runs]

            stats = {
                "val_acc": calc_stats(acc_list),
                "val_nll": calc_stats(nll_list),
                "val_brier": calc_stats(brier_list) if brier_list else None,
                "val_ece": calc_stats(ece_list) if ece_list else None,
                "gbest_loss": calc_stats(g_loss_list),
                "gbest_acc": calc_stats(g_acc_list),
                "wall_time_sec": calc_stats(wall_list),
                "optimization_wall_time_sec": calc_stats(opt_wall_list),
                "throughput_samples_per_sec": calc_stats(sps_list),
            }

            state_ratio = float(state_bytes / baseline_state_bytes)

            wl_candidates[wl_id] = {
                "candidate_id": ratio_id,
                "ratio": float(r),
                "workload_id": wl_id,
                "geometry_policy": geometry_policy,
                "geometry_multiplier": float(geometry_multiplier),
                "projection_scope": get_effective_projection_scope(projection_scope, wl_id),
                "projection_seed_mode": str(projection_seed_mode),
                "projection_seed": (
                    projection_seed[wl_id]
                    if isinstance(projection_seed, dict)
                    else projection_seed
                ),
                "base_geometry_id": base_geom_id,
                "total_dim": total_dim,
                "latent_dim": latent_dim,
                "state_ratio": state_ratio,
                "subset_size": subset_size,
                "particles": particles,
                "epochs": epochs,
                "seeds": list(seeds),
                "core_swarm_state_bytes": state_bytes,
                "baseline_core_swarm_state_bytes": baseline_state_bytes,
                "split_seed": int(split_seed),
                "data_fingerprint": d_info["data_fp"],
                "split_fingerprint": d_info["provenance"]["split_fingerprint"],
                "stats": stats,
                "per_seed_runs": per_seed_runs,
            }

        candidate_runs[ratio_id] = wl_candidates

    total_wall_time = round(time.time() - start_time, 4)

    payload = {
        "protocol_version": AUTORESEARCH_PROTOCOL_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hardware_info,
        "official_test_data_loaded": False,
        "official_test_evaluations": 0,
        "experiment_config": {
            "geometry_policy": geometry_policy,
            "geometry_multiplier": float(geometry_multiplier),
            "projection_scope": projection_scope,
            "projection_seed_mode": str(projection_seed_mode),
            "projection_seed": projection_seed,
            "projection_salt": str(projection_salt),
            "ratios": [float(r) for r in ratios],
            "particles": int(particles),
            "epochs": int(epochs),
            "subset_size": int(subset_size),
            "seeds": [int(s) for s in seeds],
            "split_seed": int(split_seed),
            "total_runs": total_runs_executed,
            "total_queries": total_queries_executed,
            "total_sample_evaluations": total_samples_evaluated,
            "total_wall_time_sec": total_wall_time,
        },
        "workloads": workload_meta,
        "candidate_runs": candidate_runs,
    }

    if output_path is not None:
        save_json_atomic(payload, Path(output_path))

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MNIST / FashionMNIST PSO Heavy Autoresearch Candidate Experiment Runner"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Dataset cache directory")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=f"Dataset train/validation split seed (default: {DEFAULT_SPLIT_SEED})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results/pso_v6_heavy_autoresearch_candidates.json",
        help="Output candidate JSON path",
    )
    parser.add_argument(
        "--geometry-policy",
        type=str,
        default=DEFAULT_GEOMETRY_POLICY,
        choices=list(GEOMETRY_POLICIES.keys()),
        help="Geometry policy ('recovered' or 'baseline_aligned')",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="1,0.5,0.25,0.125,0.03125",
        help="Comma-separated latent subspace ratios",
    )
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES, help="Swarm size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="PSO epochs")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=DEFAULT_SUBSET_SIZE,
        help="Training subset size",
    )
    parser.add_argument("--seeds", type=str, default="101,102,103", help="Comma-separated seeds")
    parser.add_argument(
        "--projection-salt",
        type=str,
        default="",
        help="Optional salt string for projection seed derivation (default: empty)",
    )
    parser.add_argument(
        "--projection-scope",
        type=parse_projection_scope_arg,
        default=DEFAULT_PROJECTION_SCOPE,
        help="Projection scope ('global', 'tensor_local', 'balanced_global', 'two_hash_global', 'largest_tensor_hash', 'largest_tensor_row_hash', 'adjacent_pair', 'adjacent_difference', or workload dict)",
    )
    parser.add_argument(
        "--projection-seed-mode",
        type=str,
        default=DEFAULT_PROJECTION_SEED_MODE,
        choices=list(PROJECTION_SEED_MODES),
        help="Projection seed mode ('coupled', 'fixed', or 'explicit')",
    )
    parser.add_argument(
        "--projection-seed",
        type=parse_projection_seed_arg,
        default=None,
        help="Exact nonnegative projection seed (int or dict) for explicit mode",
    )
    parser.add_argument(
        "--geometry-multiplier",
        type=float,
        default=DEFAULT_GEOMETRY_MULTIPLIER,
        help="Geometry radius/bound multiplier (default: 1.0)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    cache_path = Path(args.cache_dir) if args.cache_dir else None
    out_path = Path(args.output) if args.output else None

    run_heavy_pso_autoresearch(
        ratios=ratios,
        split_seed=args.split_seed,
        projection_seed_mode=args.projection_seed_mode,
        projection_seed=args.projection_seed,
        particles=args.particles,
        epochs=args.epochs,
        subset_size=args.subset_size,
        seeds=seeds,
        geometry_policy=args.geometry_policy,
        device_str=args.device,
        cache_dir=cache_path,
        output_path=out_path,
        projection_salt=args.projection_salt,
        projection_scope=args.projection_scope,
        geometry_multiplier=args.geometry_multiplier,
    )

if __name__ == "__main__":
    main()
