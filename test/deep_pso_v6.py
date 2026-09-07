"""
MNIST PSO V6 Study - Phase A & B: Geometry Ablation & Root-Cause Isolation.

Protocol Version: MNIST-PSO-RAW-V6 1.0.0
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, asdict
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    sync_device,
)
from deep_pso_methods import (
    make_compact_cnn,
    build_nested_stratified_subsets,
    evaluate_probabilistic_metrics,
    get_model_probabilities,
)
from pso import __version__ as pso_version

PROTOCOL_VERSION = "MNIST-PSO-RAW-V6 1.0.0"


# =====================================================================
# 1. Dataset Preparation: Train-Only Search/Validation (No Test Split)
# =====================================================================

def prepare_mnist_v6_data(
    split_seed: int = 20260902,
    cache_dir: Optional[Path] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    Dict[int, torch.Tensor],
    str, Dict[str, Any]
]:
    """
    Train-only MNIST data preparation using exclusively MNIST(train=True).
    Never constructs MNIST(train=False).
    Preserves exact V5 split seed (20260902), search-only normalization,
    and nested 2k/10k/50k index stratification.
    """
    from torchvision.datasets import MNIST

    if cache_dir is None:
        cache_dir = Path("result/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Strictly train=True. Never construct train=False.
    raw_train = MNIST(root=str(cache_dir), train=True, download=True)

    x_train_raw = raw_train.data.float() / 255.0  # (60000, 28, 28)
    y_train_raw = raw_train.targets.long()

    # Stratified split: 50,000 search set and 10,000 validation set
    indices = np.arange(len(y_train_raw))
    search_idx, val_idx = train_test_split(
        indices,
        train_size=50000,
        test_size=10000,
        stratify=y_train_raw.numpy(),
        random_state=split_seed,
    )

    x_search_raw = x_train_raw[search_idx]
    y_search = y_train_raw[search_idx]
    x_val_raw = x_train_raw[val_idx]
    y_val = y_train_raw[val_idx]

    # Fit mean and std on 50k search subset ONLY
    mean_val = float(x_search_raw.mean())
    std_val = float(x_search_raw.std())

    x_search_norm = ((x_search_raw - mean_val) / std_val).unsqueeze(1)  # (50000, 1, 28, 28)
    x_val_norm = ((x_val_raw - mean_val) / std_val).unsqueeze(1)        # (10000, 1, 28, 28)

    # Nested stratified subsets inside 50k search set: 2k inside 10k inside 50k
    nested_subsets = build_nested_stratified_subsets(
        y_search=y_search,
        subset_sizes=[2000, 10000, 50000],
        subset_seed=split_seed,
    )

    # Data fingerprint over search and validation splits (no test split)
    h = hashlib.sha256()
    for t in (x_search_norm, x_val_norm, y_search, y_val):
        h.update(t.detach().cpu().numpy().tobytes())
    data_fp = h.hexdigest()[:16]

    split_h = hashlib.sha256()
    split_h.update(search_idx.tobytes())
    split_h.update(val_idx.tobytes())
    split_fp = split_h.hexdigest()[:16]

    provenance = {
        "input_shape": [1, 28, 28],
        "pca": False,
        "raw_inputs": True,
        "normalization_scope": "search_train_50000_only",
        "train_mean": round(mean_val, 6),
        "train_std": round(std_val, 6),
        "search_samples": 50000,
        "val_samples": 10000,
        "test_samples": 0,
        "official_test_evaluations": 0,
        "split_seed": split_seed,
        "split_fingerprint": split_fp,
    }

    return (
        x_search_norm, y_search,
        x_val_norm, y_val,
        nested_subsets,
        data_fp, provenance
    )


# =====================================================================
# 2. Immutable Geometry Configuration & Protocol Table (G0 - G8)
# =====================================================================

@dataclass(frozen=True)
class V6GeometryConfig:
    """
    Immutable geometry configuration specifying coordinate scales,
    initial position / velocity distributions, mutation, and bounds.
    """
    config_id: str
    scale_type: str = "per_tensor_sd"       # "per_tensor_sd", "global_rms", "identity", "optimizer_default"
    init_position_mode: str = "antithetic"  # "antithetic", "independent"
    position_radius: float = 0.5            # initial position radius
    initial_velocity_radius: float = 0.0    # 0.0 for zero launch velocity, >0 for U(-r, r)
    mutation_prob: float = 0.0              # 0.0 or 0.02
    reset_velocity_radius: float = 0.02     # velocity radius sampled upon mutation
    reflective_bound: float = 3.0           # reflective box half-width (e.g. 3.0 or 6.0)
    projection_seed: Optional[int] = None   # seed for sparse signed-hash subspace projection
    latent_dim: Union[int, str] = "full"    # "full" or integer dimension (e.g. 290, 1024)
    description: str = ""


def get_v6_geometry_table() -> Dict[str, V6GeometryConfig]:
    """
    Returns the complete, approved Phase B geometry configuration table (G0 - G8).
    """
    return {
        "G0": V6GeometryConfig(
            config_id="G0",
            scale_type="per_tensor_sd",
            init_position_mode="antithetic",
            position_radius=0.5,
            initial_velocity_radius=0.0,
            mutation_prob=0.0,
            reflective_bound=3.0,
            description="exact V5 control",
        ),
        "G1": V6GeometryConfig(
            config_id="G1",
            scale_type="global_rms",
            init_position_mode="antithetic",
            position_radius=0.5,
            initial_velocity_radius=0.0,
            mutation_prob=0.0,
            reflective_bound=3.0,
            description="isolate anisotropic per-tensor scaling",
        ),
        "G2": V6GeometryConfig(
            config_id="G2",
            scale_type="per_tensor_sd",
            init_position_mode="antithetic",
            position_radius=0.5,
            initial_velocity_radius=0.5,
            mutation_prob=0.0,
            reflective_bound=3.0,
            description="isolate nonzero launch velocity",
        ),
        "G3": V6GeometryConfig(
            config_id="G3",
            scale_type="per_tensor_sd",
            init_position_mode="antithetic",
            position_radius=0.5,
            initial_velocity_radius=0.0,
            mutation_prob=0.02,
            reset_velocity_radius=0.02,
            reflective_bound=3.0,
            description="isolate mutation",
        ),
        "G4": V6GeometryConfig(
            config_id="G4",
            scale_type="per_tensor_sd",
            init_position_mode="antithetic",
            position_radius=0.5,
            initial_velocity_radius=0.5,
            mutation_prob=0.02,
            reset_velocity_radius=0.02,
            reflective_bound=3.0,
            description="velocity x mutation interaction",
        ),
        "G5": V6GeometryConfig(
            config_id="G5",
            scale_type="per_tensor_sd",
            init_position_mode="antithetic",
            position_radius=0.5,
            initial_velocity_radius=0.5,
            mutation_prob=0.02,
            reset_velocity_radius=0.02,
            reflective_bound=6.0,
            description="test sufficient bound expansion",
        ),
        "G6": V6GeometryConfig(
            config_id="G6",
            scale_type="per_tensor_sd",
            init_position_mode="antithetic",
            position_radius=1.5,
            initial_velocity_radius=0.5,
            mutation_prob=0.02,
            reset_velocity_radius=0.02,
            reflective_bound=6.0,
            description="test broader normalized initialization",
        ),
        "G7": V6GeometryConfig(
            config_id="G7",
            scale_type="per_tensor_sd",
            init_position_mode="independent",
            position_radius=0.5,
            initial_velocity_radius=0.5,
            mutation_prob=0.0,
            reflective_bound=3.0,
            description="isolate antithetic position coupling against G2",
        ),
        "G8": V6GeometryConfig(
            config_id="G8",
            scale_type="optimizer_default",
            init_position_mode="independent",
            position_radius=0.05,
            initial_velocity_radius=0.05,
            mutation_prob=0.02,
            reset_velocity_radius=0.02,
            reflective_bound=3.0,
            description="retained semantic control (public Optimizer)",
        ),
    }


def compute_equalized_subspace_radius(
    latent_dim: int,
    total_dim: int = 9098,
    base_radius: float = 0.5,
) -> float:
    """
    Computes equalized subspace initialization/bound radius for Phase C.
    Scales base_radius by sqrt(total_dim / latent_dim) to hold decoded per-parameter RMS constant.
    """
    if latent_dim >= total_dim:
        return base_radius
    return float(base_radius * math.sqrt(total_dim / latent_dim))


# =====================================================================
# 3. Deterministic Latent Transform & Swarm Construction
# =====================================================================

class V6LatentTransform:
    def __init__(
        self,
        base_model: nn.Module,
        geom_config: V6GeometryConfig,
        device: torch.device,
    ):
        self.device = device
        self.geom_config = geom_config
        self.base_params = [p.detach().clone().to(device) for p in base_model.parameters()]
        self.param_shapes = [p.shape for p in self.base_params]
        self.param_numels = [p.numel() for p in self.base_params]
        self.total_dim = sum(self.param_numels)
        self.base_vec = torch.cat([p.view(-1) for p in self.base_params]).to(device)

        scale_type = geom_config.scale_type.lower()
        if scale_type == "per_tensor_sd":
            tensor_scales = []
            for p in self.base_params:
                std_val = float(p.std())
                scale = max(std_val, 1e-4)
                scale_tensor = torch.full_like(p, scale)
                tensor_scales.append(scale_tensor.view(-1))
            self.scale_vec = torch.cat(tensor_scales).to(device)
        elif scale_type == "global_rms":
            rms_val = float(torch.sqrt(torch.mean(self.base_vec ** 2)))
            scale = max(rms_val, 1e-4)
            self.scale_vec = torch.full_like(self.base_vec, scale, device=device)
        elif scale_type == "identity":
            self.scale_vec = torch.ones_like(self.base_vec, device=device)
        else:
            # Default fallback for optimizer or custom
            self.scale_vec = torch.ones_like(self.base_vec, device=device)

        latent_dim = geom_config.latent_dim
        if isinstance(latent_dim, str) and latent_dim.lower() == "full":
            self.latent_dim = self.total_dim
            self.is_full = True
        else:
            self.latent_dim = int(latent_dim)
            self.is_full = (self.latent_dim == self.total_dim)

        if not self.is_full:
            j_indices = np.arange(self.total_dim, dtype=np.int64)
            seed_offset = geom_config.projection_seed if geom_config.projection_seed is not None else 0
            h1 = ((j_indices + 1 + seed_offset) * 2654435761) % (2**32)
            k_indices = h1 % self.latent_dim
            h2 = ((j_indices + 1 + seed_offset) * 1597334677) % (2**32)
            signs = np.where((h2 % 2) == 0, 1.0, -1.0)

            bin_counts = np.bincount(k_indices, minlength=self.latent_dim)
            count_per_j = bin_counts[k_indices]
            scale_per_j = 1.0 / np.sqrt(np.maximum(count_per_j, 1))
            combined_weights = signs * scale_per_j

            self.k_indices = torch.tensor(k_indices, dtype=torch.long, device=device)
            self.weights = torch.tensor(combined_weights, dtype=torch.float32, device=device)

    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Transforms latent batch Z (N, d) into full parameter batch (N, D).
        theta = base_vec + scale_vec * delta
        """
        if self.is_full:
            delta = Z
        else:
            delta = Z[:, self.k_indices] * self.weights
        return self.base_vec + self.scale_vec * delta

    def load_vector_to_model(self, theta_vec: torch.Tensor, model: nn.Module):
        """Loads a single parameter vector into model parameters in-place."""
        offset = 0
        with torch.no_grad():
            for p, shape, numel in zip(model.parameters(), self.param_shapes, self.param_numels):
                p.copy_(theta_vec[offset:offset + numel].view(shape))
                offset += numel

    def init_swarm(self, swarm_size: int, seed: int) -> torch.Tensor:
        """
        Initializes particle positions in latent space Z (N, d).
        Particle 0 is exact base vector (z = 0).
        Uses a separate CPU generator to preserve G0 parity with V5.
        """
        init_radius = self.geom_config.position_radius
        mode = self.geom_config.init_position_mode.lower()

        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)

        Z = torch.zeros((swarm_size, self.latent_dim), dtype=torch.float32)

        if mode == "antithetic":
            max_pair_idx = swarm_size - 1 if (swarm_size % 2 != 0) else swarm_size - 2
            idx = 1
            while idx < max_pair_idx:
                sample = (torch.rand(self.latent_dim, generator=rng) * 2.0 - 1.0) * init_radius
                Z[idx] = sample
                Z[idx + 1] = -sample
                idx += 2
        else:  # independent
            for idx in range(1, swarm_size):
                Z[idx] = (torch.rand(self.latent_dim, generator=rng) * 2.0 - 1.0) * init_radius

        return Z.to(self.device)


# =====================================================================
# 4. Device-Resident Configurable V6 Latent Adaptive-Moment PSO Engine
# =====================================================================

def evaluate_latent_batch(
    Z: torch.Tensor,
    transform: V6LatentTransform,
    model: nn.Module,
    x_sub_dev: torch.Tensor,
    y_sub_dev: torch.Tensor,
    batch_size: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates latent batch Z (N, d) on device-resident (x_sub_dev, y_sub_dev).
    Returns (losses, accuracies) tensors of shape (N,).
    """
    N = Z.shape[0]
    device = Z.device
    losses = torch.zeros(N, dtype=torch.float32, device=device)
    accuracies = torch.zeros(N, dtype=torch.float32, device=device)

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    num_samples = len(y_sub_dev)

    model.eval()
    with torch.inference_mode():
        for i in range(N):
            theta_vec = transform.decode(Z[i:i+1]).squeeze(0)
            transform.load_vector_to_model(theta_vec, model)

            total_loss = torch.tensor(0.0, device=device)
            correct = torch.tensor(0, dtype=torch.long, device=device)

            for b_start in range(0, num_samples, batch_size):
                xb = x_sub_dev[b_start:b_start + batch_size]
                yb = y_sub_dev[b_start:b_start + batch_size]
                logits = model(xb)
                batch_loss = loss_fn(logits, yb)
                total_loss += batch_loss
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum()

            losses[i] = total_loss / num_samples
            accuracies[i] = (correct.float() / num_samples) * 100.0

    return losses, accuracies


def run_v6_pso(
    transform: V6LatentTransform,
    base_model: nn.Module,
    x_search: torch.Tensor,
    y_search: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    nested_subsets: Dict[int, torch.Tensor],
    schedule_str: str,
    epochs: int,
    swarm_size: int,
    seed: int,
    device: torch.device,
    geom_config: V6GeometryConfig,
    val_check_interval: int = 10,
    transition_reset_policy: str = "none",
) -> Dict[str, Any]:
    """
    Device-resident Latent Adaptive-Moment PSO for V6 study.
    Supports G0-G7 configurations, nonzero launch velocity, mutation moment reset,
    reflective bounds, state-neutral validation checkpoints, and full telemetry.
    """
    sync_device(device)
    start_time = time.time()
    validation_wall_time = 0.0

    schedule_stages = []
    if schedule_str:
        parts = schedule_str.split(",")
        for p in parts:
            sz_str, ep_str = p.split(":")
            schedule_stages.append((int(sz_str), int(ep_str)))

    if not schedule_stages:
        schedule_stages = [(50000, epochs)]

    c0 = c1 = 1.49618
    w = 0.7298
    blend = 0.06
    step = 0.5
    beta1 = 0.9
    beta2 = 0.999
    reflective_bound = geom_config.reflective_bound

    latent_dim = transform.latent_dim
    Z = transform.init_swarm(swarm_size=swarm_size, seed=seed)

    # Initial launch velocity
    if geom_config.initial_velocity_radius > 0.0:
        vel_rng = torch.Generator(device="cpu")
        vel_rng.manual_seed(seed + 1000)
        r_v = geom_config.initial_velocity_radius
        V_cpu = (torch.rand((swarm_size, latent_dim), generator=vel_rng) * 2.0 - 1.0) * r_v
        V_cpu[0] = 0.0  # Particle 0 launch velocity remains zero
        V = V_cpu.to(device)
    else:
        V = torch.zeros((swarm_size, latent_dim), dtype=torch.float32, device=device)

    M = torch.zeros((swarm_size, latent_dim), dtype=torch.float32, device=device)
    V_sq = torch.zeros((swarm_size, latent_dim), dtype=torch.float32, device=device)
    moment_steps = torch.zeros(swarm_size, dtype=torch.int64, device=device)

    P = Z.clone()
    P_loss = torch.full((swarm_size,), float("inf"), dtype=torch.float32, device=device)
    P_acc = torch.zeros((swarm_size,), dtype=torch.float32, device=device)

    gbest_z = Z[0].clone()
    gbest_loss = float("inf")
    gbest_acc = 0.0

    # Counters & Telemetry
    total_queries = 0
    total_sample_evaluations = 0
    transition_reevaluation_counts = 0
    validation_evaluations = 0
    pbest_update_counts = 0
    boundary_hits = 0
    last_improvement_epoch = 0
    mutation_events = 0

    # Random generators: move_rng handles standard velocity draws; mut_rng handles mutation draws
    move_rng = torch.Generator(device=device)
    move_rng.manual_seed(seed)
    mut_rng = torch.Generator(device=device)
    mut_rng.manual_seed(seed + 2000)

    model = copy.deepcopy(base_model).to(device)
    x_val_dev = x_val.to(device)
    y_val_dev = y_val.to(device)

    stage_histories = []
    epoch_counter = 0

    for stage_idx, (size, stage_epochs) in enumerate(schedule_stages):
        subset_indices = nested_subsets[size]
        x_sub_dev = x_search[subset_indices].to(device)
        y_sub_dev = y_search[subset_indices].to(device)

        # Objective transition check
        if stage_idx > 0:
            re_losses, re_accs = evaluate_latent_batch(
                P, transform, model, x_sub_dev, y_sub_dev
            )
            P_loss = re_losses
            P_acc = re_accs

            total_queries += swarm_size
            total_sample_evaluations += swarm_size * size
            transition_reevaluation_counts += swarm_size

            min_loss_val = P_loss.min()
            candidates_mask = (P_loss <= min_loss_val + 1e-7)
            best_p_idx = int(torch.where(candidates_mask, P_acc, torch.tensor(-1.0, device=device)).argmax().item())

            gbest_z = P[best_p_idx].clone()
            gbest_loss = float(P_loss[best_p_idx].item())
            gbest_acc = float(P_acc[best_p_idx].item())

            # Transition reset policy
            if transition_reset_policy in ("reset_vm", "reset_all"):
                V.zero_()
                M.zero_()
                V_sq.zero_()
                moment_steps.zero_()

        for ep in range(1, stage_epochs + 1):
            epoch_counter += 1

            # 1. Evaluate current swarm positions
            curr_losses, curr_accs = evaluate_latent_batch(
                Z, transform, model, x_sub_dev, y_sub_dev
            )
            total_queries += swarm_size
            total_sample_evaluations += swarm_size * size

            # 2. Vectorized pbest updates
            better_loss = curr_losses < P_loss - 1e-7
            equal_loss = torch.abs(curr_losses - P_loss) <= 1e-7
            better_acc = curr_accs > P_acc
            update_mask = better_loss | (equal_loss & better_acc)

            n_updated = int(update_mask.sum().item())
            pbest_update_counts += n_updated
            if n_updated > 0:
                last_improvement_epoch = epoch_counter

            P[update_mask] = Z[update_mask]
            P_loss[update_mask] = curr_losses[update_mask]
            P_acc[update_mask] = curr_accs[update_mask]

            # Rebuild gbest from P each epoch
            min_loss_val = P_loss.min()
            candidates_mask = (P_loss <= min_loss_val + 1e-7)
            best_p_idx = int(torch.where(candidates_mask, P_acc, torch.tensor(-1.0, device=device)).argmax().item())

            gbest_z = P[best_p_idx].clone()
            gbest_loss = float(P_loss[best_p_idx].item())
            gbest_acc = float(P_acc[best_p_idx].item())

            # Validation Checkpoint (State-Neutral)
            val_loss = None
            val_acc = None
            if val_check_interval > 0 and (epoch_counter % val_check_interval == 0 or epoch_counter == epochs):
                sync_device(device)
                validation_start = time.time()
                with torch.inference_mode():
                    v_losses, v_accs = evaluate_latent_batch(
                        gbest_z.unsqueeze(0), transform, model, x_val_dev, y_val_dev
                    )
                    val_loss = float(v_losses[0].item())
                    val_acc = float(v_accs[0].item())
                    validation_evaluations += 1
                sync_device(device)
                validation_wall_time += time.time() - validation_start

            stage_histories.append({
                "epoch": epoch_counter,
                "stage": stage_idx,
                "subset_size": size,
                "gbest_loss": round(gbest_loss, 6),
                "gbest_acc": round(gbest_acc, 4),
                "val_loss": round(val_loss, 6) if val_loss is not None else None,
                "val_acc": round(val_acc, 4) if val_acc is not None else None,
            })

            # 3. Movement Step
            r1 = torch.rand((swarm_size, latent_dim), generator=move_rng, device=device)
            r2 = torch.rand((swarm_size, latent_dim), generator=move_rng, device=device)

            V_raw = w * V + c0 * r1 * (P - Z) + c1 * r2 * (gbest_z.unsqueeze(0) - Z)

            # Mutation check
            if geom_config.mutation_prob > 0.0:
                mut_draws = torch.rand(swarm_size, generator=mut_rng, device=device)
                mut_mask = mut_draws < geom_config.mutation_prob
                if mut_mask.any():
                    n_mut = int(mut_mask.sum().item())
                    r_mut = geom_config.reset_velocity_radius
                    mut_v = (torch.rand((n_mut, latent_dim), generator=mut_rng, device=device) * 2.0 - 1.0) * r_mut
                    V_raw[mut_mask] = mut_v
                    # Clear moment state for mutated particles
                    M[mut_mask] = 0.0
                    V_sq[mut_mask] = 0.0
                    moment_steps[mut_mask] = 0
                    mutation_events += n_mut

            moment_steps += 1
            M = beta1 * M + (1 - beta1) * V_raw
            V_sq = beta2 * V_sq + (1 - beta2) * (V_raw ** 2)

            step_values = moment_steps.to(dtype=M.dtype).unsqueeze(1)
            M_hat = M / (1.0 - torch.pow(beta1, step_values))
            V_sq_hat = V_sq / (1.0 - torch.pow(beta2, step_values))

            dir_moment = M_hat / (torch.sqrt(V_sq_hat) + 1e-8)
            historical_scale = torch.sqrt(torch.mean(V_sq_hat, dim=1, keepdim=True))
            V_moment = step * dir_moment * historical_scale
            V_new = (1.0 - blend) * V_raw + blend * V_moment

            Z_new = Z + V_new

            pos_mask = Z_new > reflective_bound
            neg_mask = Z_new < -reflective_bound
            hit_mask = pos_mask | neg_mask
            boundary_hits += int(hit_mask.sum().item())

            Z_new[pos_mask] = 2.0 * reflective_bound - Z_new[pos_mask]
            V_new[pos_mask] = -V_new[pos_mask]

            Z_new[neg_mask] = -2.0 * reflective_bound - Z_new[neg_mask]
            V_new[neg_mask] = -V_new[neg_mask]

            Z_new = torch.clamp(Z_new, -reflective_bound, reflective_bound)

            Z = Z_new
            V = V_new

            if epoch_counter >= epochs:
                break
        if epoch_counter >= epochs:
            break

    sync_device(device)
    validation_start = time.time()
    # Select the best final pbest on validation, matching the G8 control.
    pbest_val_losses, pbest_val_accs = evaluate_latent_batch(
        P, transform, model, x_val_dev, y_val_dev
    )
    validation_evaluations += swarm_size
    min_val_loss = pbest_val_losses.min()
    val_candidates = pbest_val_losses <= min_val_loss + 1e-7
    val_best_idx = int(
        torch.where(
            val_candidates,
            pbest_val_accs,
            torch.tensor(-1.0, device=device),
        ).argmax().item()
    )

    val_selected_model = copy.deepcopy(base_model).to(device)
    theta_selected = transform.decode(P[val_best_idx].unsqueeze(0)).squeeze(0)
    transform.load_vector_to_model(theta_selected, val_selected_model)
    val_probabilities = get_model_probabilities(val_selected_model, x_val_dev, device)
    val_metrics = evaluate_probabilistic_metrics(val_probabilities, y_val_dev)
    validation_evaluations += 1
    sync_device(device)
    validation_wall_time += time.time() - validation_start

    gbest_val_loss = float(pbest_val_losses[best_p_idx].item())
    gbest_val_acc = float(pbest_val_accs[best_p_idx].item())

    wall_time = time.time() - start_time
    optimization_wall_time = max(0.0, wall_time - validation_wall_time)

    velocity_rms = float(torch.sqrt(torch.mean(V ** 2)).item())
    position_radius = float(torch.norm(Z - Z.mean(dim=0), dim=1).mean().item())
    total_coords = swarm_size * latent_dim * max(epoch_counter, 1)
    boundary_occupancy = round(boundary_hits / max(total_coords, 1), 6)

    return {
        "config_id": geom_config.config_id,
        "gbest_z": gbest_z,
        "gbest_loss": round(gbest_loss, 6),
        "gbest_acc": round(gbest_acc, 4),
        "gbest_val_loss": round(gbest_val_loss, 6),
        "gbest_val_acc": round(gbest_val_acc, 4),
        "val_selected_particle_idx": val_best_idx,
        "val_selected_loss": round(val_metrics["nll"], 6),
        "val_selected_acc": round(val_metrics["accuracy"], 4),
        "val_metrics": val_metrics,
        "final_P": P,
        "wall_time_sec": round(wall_time, 4),
        "optimization_wall_time_sec": round(optimization_wall_time, 4),
        "validation_wall_time_sec": round(validation_wall_time, 4),
        "total_queries": total_queries,
        "total_sample_evaluations": total_sample_evaluations,
        "transition_reevaluation_counts": transition_reevaluation_counts,
        "validation_evaluations": validation_evaluations,
        "official_test_evaluations": 0,
        "mutation_events": mutation_events,
        "final_moment_steps": moment_steps.detach().cpu().tolist(),
        "pbest_update_counts": pbest_update_counts,
        "boundary_hits": boundary_hits,
        "boundary_occupancy": boundary_occupancy,
        "last_improvement_epoch": last_improvement_epoch,
        "velocity_rms": round(velocity_rms, 6),
        "position_radius": round(position_radius, 6),
        "stage_histories": stage_histories,
    }


# =====================================================================
# 5. Public Optimizer G8 Semantic Control Engine
# =====================================================================

def run_g8_optimizer(
    base_model: nn.Module,
    x_2k: torch.Tensor,
    y_2k: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    swarm_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Runs public Optimizer for G8 semantic control over the exact 2k search tensors.
    Evaluates every particle's pbest on validation to select the endpoint.
    Retains training gbest metrics separately.
    Official test evaluations count is explicitly zero.
    """
    from pso.optimizer import Optimizer
    sync_device(device)
    start_t = time.time()

    model = copy.deepcopy(base_model).to(device)
    loss_fn = nn.CrossEntropyLoss()
    validation_loss_fn = nn.CrossEntropyLoss(reduction="sum")

    opt = Optimizer(
        model=model,
        loss=loss_fn,
        task="multiclass",
        method="adaptive_moment",
        evaluation="full",
        n_particles=swarm_size,
        c0=1.49618,
        c1=1.49618,
        w_min=0.7298,
        w_max=0.7298,
        particle_min=-3.0,
        particle_max=3.0,
        boundary_strategy="reflect",
        velocity_limit_ratio=0.025,
        mutation_swarm=0.02,
        initialization="model_noise",
        initial_position_noise=0.05,
        moment_blend=0.06,
        moment_step_size=0.5,
        moment_beta1=0.9,
        seed=seed,
        device=device,
    )

    x_2k_dev = x_2k.to(device)
    y_2k_dev = y_2k.to(device)

    best_score = opt.fit(x_2k_dev, y_2k_dev, epochs=epochs, renewal="loss")
    sync_device(device)
    optimization_wall_time = time.time() - start_t
    validation_start = time.time()

    # Evaluate every particle pbest vector on the validation set
    val_losses = []
    val_accs = []
    pbest_models = []

    x_val_dev = x_val.to(device)
    y_val_dev = y_val.to(device)
    num_val = len(y_val_dev)

    for p in opt.particles:
        p_model = copy.deepcopy(base_model).to(device)
        offset = 0
        with torch.no_grad():
            for param in p_model.parameters():
                n = param.numel()
                param.copy_(p.personal_best_weights[offset:offset + n].view(param.shape))
                offset += n

        p_model.eval()
        with torch.inference_mode():
            total_loss = 0.0
            correct = 0
            for b_start in range(0, num_val, 1000):
                xb = x_val_dev[b_start:b_start + 1000]
                yb = y_val_dev[b_start:b_start + 1000]
                logits = p_model(xb)
                total_loss += float(validation_loss_fn(logits, yb).item())
                correct += int((logits.argmax(dim=1) == yb).sum().item())
            val_loss = total_loss / num_val
            val_acc = (correct / num_val) * 100.0

            val_losses.append(val_loss)
            val_accs.append(val_acc)
            pbest_models.append(p_model)

    # Rank particles on validation NLL ascending, val accuracy descending
    min_val_loss = min(val_losses)
    val_candidates = [
        index
        for index, loss in enumerate(val_losses)
        if loss <= min_val_loss + 1e-7
    ]
    best_idx = max(val_candidates, key=lambda index: val_accs[index])

    val_selected_model = pbest_models[best_idx]
    val_probabilities = get_model_probabilities(val_selected_model, x_val_dev, device)
    val_metrics = evaluate_probabilistic_metrics(val_probabilities, y_val_dev)
    sync_device(device)
    validation_wall_time = time.time() - validation_start
    wall_t = optimization_wall_time + validation_wall_time

    total_queries = swarm_size * epochs
    total_sample_evaluations = total_queries * len(y_2k)
    validation_evaluations = swarm_size + 1

    return {
        "config_id": "G8",
        "gbest_loss": round(float(best_score[0]), 6),
        "gbest_acc": round(float(best_score[1]) * 100.0, 4),
        "val_selected_loss": round(val_metrics["nll"], 6),
        "val_selected_acc": round(val_metrics["accuracy"], 4),
        "val_metrics": val_metrics,
        "wall_time_sec": round(wall_t, 4),
        "optimization_wall_time_sec": round(optimization_wall_time, 4),
        "validation_wall_time_sec": round(validation_wall_time, 4),
        "total_queries": total_queries,
        "total_sample_evaluations": total_sample_evaluations,
        "transition_reevaluation_counts": 0,
        "validation_evaluations": validation_evaluations,
        "official_test_evaluations": 0,
        "pbest_update_counts": 0,
        "boundary_hits": 0,
        "boundary_occupancy": 0.0,
        "last_improvement_epoch": epochs,
        "velocity_rms": 0.0,
        "position_radius": 0.0,
        "stage_histories": [],
    }


# =====================================================================
# 6. Selection & Paired Factor Analysis
# =====================================================================

def compute_paired_factor_deltas(
    screen_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Computes paired factor deltas for Phase B screen.
    A factor is provisionally material if NLL improves by >= 0.05 or Acc improves by >= 2.0pp.
    """
    pairs = [
        ("delta_scale_G1_vs_G0", "G1", "G0", "global RMS scale vs per-tensor SD"),
        ("delta_vel_G2_vs_G0", "G2", "G0", "launch velocity U(-0.5,0.5) vs 0"),
        ("delta_mut_G3_vs_G0", "G3", "G0", "mutation 0.02 vs 0"),
        ("delta_vel_mut_G4_vs_G2", "G4", "G2", "mutation interaction given velocity"),
        ("delta_bound_G5_vs_G4", "G5", "G4", "bound box 6 vs 3"),
        ("delta_radius_G6_vs_G5", "G6", "G5", "initial position radius 1.5 vs 0.5"),
        ("delta_init_G7_vs_G2", "G7", "G2", "independent vs antithetic init"),
    ]

    deltas = {}
    for key, c_test, c_ref, desc in pairs:
        if c_test in screen_results and c_ref in screen_results:
            r_test = screen_results[c_test]
            r_ref = screen_results[c_ref]
            nll_diff = round(r_test["val_selected_loss"] - r_ref["val_selected_loss"], 6)
            acc_diff = round(r_test["val_selected_acc"] - r_ref["val_selected_acc"], 4)
            is_material = (nll_diff <= -0.05) or (acc_diff >= 2.0)
            deltas[key] = {
                "test_config": c_test,
                "ref_config": c_ref,
                "nll_diff": nll_diff,
                "acc_diff": acc_diff,
                "material": is_material,
                "description": desc,
            }

    # Bundle comparison: G8 vs best normalized config
    norm_configs = [c for c in screen_results if c != "G8"]
    if norm_configs:
        best_norm_id = min(norm_configs, key=lambda k: (screen_results[k]["val_selected_loss"], -screen_results[k]["val_selected_acc"]))
        r_g8 = screen_results["G8"]
        r_best_norm = screen_results[best_norm_id]
        nll_diff = round(r_g8["val_selected_loss"] - r_best_norm["val_selected_loss"], 6)
        acc_diff = round(r_g8["val_selected_acc"] - r_best_norm["val_selected_acc"], 4)
        deltas["delta_bundle_G8_vs_best_norm"] = {
            "test_config": "G8",
            "ref_config": best_norm_id,
            "nll_diff": nll_diff,
            "acc_diff": acc_diff,
            "material": abs(nll_diff) >= 0.05 or abs(acc_diff) >= 2.0,
            "description": f"Optimizer G8 control vs best normalized ({best_norm_id})",
        }

    return deltas


def select_confirmation_configs(
    screen_results: Dict[str, Dict[str, Any]]
) -> List[str]:
    """
    Deterministically selects G0, G1, G8 plus top 2 eligible normalized configs from G2-G7.
    """
    mandatory = ["G0", "G1", "G8"]
    eligible = ["G2", "G3", "G4", "G5", "G6", "G7"]

    # Filter available eligible configs
    valid_eligible = [c for c in eligible if c in screen_results]
    valid_eligible.sort(
        key=lambda c: (screen_results[c]["val_selected_loss"], -screen_results[c]["val_selected_acc"])
    )

    top2_other = valid_eligible[:2]
    selected = mandatory + top2_other
    return selected


def evaluate_root_cause_statuses(
    confirm_aggregates: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Classify each predeclared geometry hypothesis from confirmed mean metrics."""
    statuses: Dict[str, Dict[str, Any]] = {}
    g8_stats = confirm_aggregates.get("G8")

    if g8_stats is None:
        statuses["regression_recovered"] = {
            "status": "unresolved",
            "recovered_configs": [],
            "description": "G8 control was not confirmed",
        }
    else:
        g8_acc = g8_stats["val_selected_acc"]["mean"]
        g8_nll = g8_stats["val_selected_nll"]["mean"]
        recovered = []
        for cid, stats in confirm_aggregates.items():
            if cid == "G8":
                continue
            acc = stats["val_selected_acc"]["mean"]
            nll = stats["val_selected_nll"]["mean"]
            if acc >= g8_acc - 1.0 and nll <= g8_nll + 0.05:
                recovered.append(cid)
        statuses["regression_recovered"] = {
            "status": "supported" if recovered else "rejected",
            "recovered_configs": recovered,
            "g8_mean_acc": g8_acc,
            "g8_mean_nll": g8_nll,
            "description": "Normalized geometry is within 1.0pp accuracy and 0.05 NLL of G8",
        }

    hypotheses = [
        ("anisotropic_per_tensor_scaling", "G1", "G0", "global RMS scale versus per-tensor SD"),
        ("nonzero_launch_velocity", "G2", "G0", "nonzero launch velocity versus zero"),
        ("mutation", "G3", "G0", "mutation 0.02 versus none"),
        ("velocity_mutation_interaction", "G4", "G2", "mutation given nonzero velocity"),
        ("bound_expansion", "G5", "G4", "normalized bound 6 versus 3"),
        ("broader_initialization", "G6", "G5", "initial radius 1.5 versus 0.5"),
        ("independent_initialization", "G7", "G2", "independent versus antithetic positions"),
    ]
    for name, test_id, ref_id, description in hypotheses:
        if test_id not in confirm_aggregates or ref_id not in confirm_aggregates:
            statuses[name] = {
                "status": "unresolved",
                "test_config": test_id,
                "ref_config": ref_id,
                "description": description,
            }
            continue
        test_stats = confirm_aggregates[test_id]
        ref_stats = confirm_aggregates[ref_id]
        nll_diff = (
            test_stats["val_selected_nll"]["mean"]
            - ref_stats["val_selected_nll"]["mean"]
        )
        acc_diff = (
            test_stats["val_selected_acc"]["mean"]
            - ref_stats["val_selected_acc"]["mean"]
        )
        supported = nll_diff <= -0.05 or acc_diff >= 2.0
        statuses[name] = {
            "status": "supported" if supported else "rejected",
            "test_config": test_id,
            "ref_config": ref_id,
            "mean_nll_diff": round(nll_diff, 6),
            "mean_acc_diff": round(acc_diff, 4),
            "description": description,
        }

    return statuses


def artifact_safe_run(result: Dict[str, Any]) -> Dict[str, Any]:
    """Drop engine-only state and convert remaining tensor values for JSON."""
    safe: Dict[str, Any] = {}
    for key, value in result.items():
        if key in {"gbest_z", "final_P"}:
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
        safe[key] = value
    return safe


# =====================================================================
# 7. Experiment Runner Pipeline (Screen, Confirm, All)
# =====================================================================

def run_phase_b_screen(
    x_search: torch.Tensor,
    y_search: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    nested_subsets: Dict[int, torch.Tensor],
    device: torch.device,
    seed: int = 91,
    swarm_size: int = 30,
    epochs: int = 160,
) -> Dict[str, Any]:
    """Runs Phase B screen (G0 - G8 at seed 91)."""
    table = get_v6_geometry_table()
    screen_results = {}

    for cid in ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"]:
        cfg = table[cid]
        base_model = make_compact_cnn(seed=41).to(device)
        transform = V6LatentTransform(base_model, cfg, device)
        res = run_v6_pso(
            transform=transform,
            base_model=base_model,
            x_search=x_search,
            y_search=y_search,
            x_val=x_val,
            y_val=y_val,
            nested_subsets=nested_subsets,
            schedule_str=f"2000:{epochs}",
            epochs=epochs,
            swarm_size=swarm_size,
            seed=seed,
            device=device,
            geom_config=cfg,
        )
        res["seed"] = seed
        res["geometry_config"] = asdict(cfg)
        screen_results[cid] = artifact_safe_run(res)

    # Run G8
    base_model_g8 = make_compact_cnn(seed=41).to(device)
    x_2k = x_search[nested_subsets[2000]]
    y_2k = y_search[nested_subsets[2000]]
    g8_res = run_g8_optimizer(
        base_model=base_model_g8,
        x_2k=x_2k,
        y_2k=y_2k,
        x_val=x_val,
        y_val=y_val,
        epochs=epochs,
        swarm_size=swarm_size,
        seed=seed,
        device=device,
    )
    g8_res["seed"] = seed
    g8_res["geometry_config"] = asdict(table["G8"])
    screen_results["G8"] = artifact_safe_run(g8_res)

    # Paired factor deltas & confirmation selection
    factor_deltas = compute_paired_factor_deltas(screen_results)
    selected_for_confirm = select_confirmation_configs(screen_results)

    return {
        "phase": "screen",
        "seed": seed,
        "swarm_size": swarm_size,
        "epochs": epochs,
        "screen_results": screen_results,
        "factor_deltas": factor_deltas,
        "selected_for_confirm": selected_for_confirm,
    }


def run_phase_b_confirm(
    selected_configs: List[str],
    x_search: torch.Tensor,
    y_search: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    nested_subsets: Dict[int, torch.Tensor],
    device: torch.device,
    seeds: List[int] = [101, 102, 103],
    swarm_size: int = 60,
    epochs: int = 420,
) -> Dict[str, Any]:
    """Runs Phase B confirmation over selected configurations across seeds 101-103."""
    table = get_v6_geometry_table()
    confirm_runs = {cid: [] for cid in selected_configs}

    x_2k = x_search[nested_subsets[2000]]
    y_2k = y_search[nested_subsets[2000]]

    for cid in selected_configs:
        for seed in seeds:
            if cid == "G8":
                base_model = make_compact_cnn(seed=41).to(device)
                res = run_g8_optimizer(
                    base_model=base_model,
                    x_2k=x_2k,
                    y_2k=y_2k,
                    x_val=x_val,
                    y_val=y_val,
                    epochs=epochs,
                    swarm_size=swarm_size,
                    seed=seed,
                    device=device,
                )
            else:
                cfg = table[cid]
                base_model = make_compact_cnn(seed=41).to(device)
                transform = V6LatentTransform(base_model, cfg, device)
                res = run_v6_pso(
                    transform=transform,
                    base_model=base_model,
                    x_search=x_search,
                    y_search=y_search,
                    x_val=x_val,
                    y_val=y_val,
                    nested_subsets=nested_subsets,
                    schedule_str=f"2000:{epochs}",
                    epochs=epochs,
                    swarm_size=swarm_size,
                    seed=seed,
                    device=device,
                    geom_config=cfg,
                )
            res["seed"] = seed
            res["geometry_config"] = asdict(table[cid])
            confirm_runs[cid].append(artifact_safe_run(res))

    # Compute aggregate stats per config across seeds
    confirm_aggregates = {}
    for cid, runs in confirm_runs.items():
        accs = [r["val_selected_acc"] for r in runs]
        nlls = [r["val_selected_loss"] for r in runs]
        briers = [r["val_metrics"]["brier"] for r in runs if "val_metrics" in r and "brier" in r["val_metrics"]]
        eces = [r["val_metrics"]["ece"] for r in runs if "val_metrics" in r and "ece" in r["val_metrics"]]

        confirm_aggregates[cid] = {
            "val_selected_acc": calc_stats(accs),
            "val_selected_nll": calc_stats(nlls),
            "brier": calc_stats(briers) if briers else {},
            "ece": calc_stats(eces) if eces else {},
            "num_seeds": len(runs),
        }

    root_cause_statuses = evaluate_root_cause_statuses(confirm_aggregates)

    return {
        "phase": "confirm",
        "seeds": seeds,
        "swarm_size": swarm_size,
        "epochs": epochs,
        "selected_configs": selected_configs,
        "confirm_runs": confirm_runs,
        "confirm_aggregates": confirm_aggregates,
        "root_cause_statuses": root_cause_statuses,
    }


# =====================================================================
# 8. CSV & Plot Artifact Writers
# =====================================================================

def save_csv_summary_v6(payload: Dict[str, Any], csv_path: Path):
    """Saves concise CSV summary of V6 study results."""
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["protocol_version", payload.get("protocol_version", PROTOCOL_VERSION)])
        writer.writerow([])

        if "screen_payload" in payload and "screen_results" in payload["screen_payload"]:
            writer.writerow(["--- Phase B Screen Results ---"])
            writer.writerow(["config_id", "description", "val_nll", "val_acc_%", "queries", "sample_evals", "wall_time_sec"])
            table = get_v6_geometry_table()
            s_results = payload["screen_payload"]["screen_results"]
            for cid in sorted(s_results.keys()):
                r = s_results[cid]
                desc = table[cid].description if cid in table else ""
                writer.writerow([
                    cid, desc,
                    r["val_selected_loss"],
                    r["val_selected_acc"],
                    r["total_queries"],
                    r["total_sample_evaluations"],
                    r["wall_time_sec"],
                ])
            writer.writerow([])

        if "confirm_payload" in payload and "confirm_aggregates" in payload["confirm_payload"]:
            writer.writerow(["--- Phase B Confirmation Aggregates ---"])
            writer.writerow(["config_id", "mean_val_acc_%", "std_val_acc", "mean_val_nll", "std_val_nll", "num_seeds"])
            c_aggs = payload["confirm_payload"]["confirm_aggregates"]
            for cid in sorted(c_aggs.keys()):
                agg = c_aggs[cid]
                writer.writerow([
                    cid,
                    agg["val_selected_acc"]["mean"],
                    agg["val_selected_acc"]["std"],
                    agg["val_selected_nll"]["mean"],
                    agg["val_selected_nll"]["std"],
                    agg["num_seeds"],
                ])


def generate_study_plots_v6(payload: Dict[str, Any], plot_path: Path):
    """Generates validation trajectory & comparison figures for V6 study."""
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: Validation Trajectories from Screen
    ax_traj = axes[0]
    if "screen_payload" in payload and "screen_results" in payload["screen_payload"]:
        s_results = payload["screen_payload"]["screen_results"]
        for cid, res in s_results.items():
            if "stage_histories" in res and res["stage_histories"]:
                epochs = [h["epoch"] for h in res["stage_histories"] if h.get("val_loss") is not None]
                nlls = [h["val_loss"] for h in res["stage_histories"] if h.get("val_loss") is not None]
                if epochs and nlls:
                    ax_traj.plot(epochs, nlls, label=cid, alpha=0.8)
            elif cid == "G8":
                ax_traj.scatter(
                    [payload["screen_payload"]["epochs"]],
                    [res["val_selected_loss"]],
                    marker="x",
                    s=60,
                    label="G8 final",
                )

    ax_traj.set_title("Screen Validation NLL Trajectory")
    ax_traj.set_xlabel("Epoch")
    ax_traj.set_ylabel("Validation NLL")
    ax_traj.grid(True, linestyle="--", alpha=0.5)
    ax_traj.legend(fontsize=8, loc="upper right")

    # Right subplot: Confirmation Mean Validation Accuracy Bar Chart
    ax_bar = axes[1]
    if "confirm_payload" in payload and "confirm_aggregates" in payload["confirm_payload"]:
        c_aggs = payload["confirm_payload"]["confirm_aggregates"]
        cids = sorted(c_aggs.keys())
        means = [c_aggs[c]["val_selected_acc"]["mean"] for c in cids]
        stds = [c_aggs[c]["val_selected_acc"]["std"] for c in cids]

        colors = ["skyblue" if c != "G8" else "coral" for c in cids]
        ax_bar.bar(cids, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
        ax_bar.set_ylabel("Mean Validation Accuracy (%)")
        ax_bar.set_title("Confirmation Accuracy (3 Seeds)")
        ax_bar.set_ylim(0, 100)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax_bar.text(i, m + s + 1.0, f"{m:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


# =====================================================================
# 9. Main CLI Entrypoint
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNIST Deep PSO V6 Root-Cause Study (Phase A & B)")
    parser.add_argument("--phase", type=str, choices=["screen", "confirm", "all"], default="all", help="Phase to execute")
    parser.add_argument("--device", type=str, default=None, help="Execution device (e.g. mps, cuda, cpu)")
    parser.add_argument("--screen-artifact", type=str, default="benchmark_results/pso_v6_phase_b_screen.json", help="Path to screen artifact for confirm phase")
    parser.add_argument("--out-dir", type=str, default="benchmark_results", help="Output directory for results")
    parser.add_argument("--plot-dir", type=str, default="history_plt", help="Output directory for plots")
    parser.add_argument("--override-particles", type=int, default=None, help="Override particle count for testing/smokes")
    parser.add_argument("--override-epochs", type=int, default=None, help="Override epoch count for testing/smokes")
    return parser


def run_deep_pso_v6_study(args: argparse.Namespace) -> Dict[str, Any]:
    device = resolve_execution_device(args.device)
    out_dir = Path(args.out_dir)
    plot_dir = Path(args.plot_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare data strictly without test set
    x_search, y_search, x_val, y_val, nested_subsets, data_fp, provenance = prepare_mnist_v6_data()
    hw_prov = get_hardware_provenance(device)
    geometry_table = get_v6_geometry_table()
    base_model = make_compact_cnn(seed=41)

    final_payload: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "pso_version": pso_version,
        "official_test_data_loaded": False,
        "official_test_evaluations": 0,
        "data_fingerprint": data_fp,
        "base_model_seed": 41,
        "base_model_fingerprint": compute_model_fingerprint(base_model),
        "geometry_configs": {
            config_id: asdict(config)
            for config_id, config in geometry_table.items()
        },
        "selection_rule": "lowest_validation_nll_then_highest_accuracy",
        "provenance": provenance,
        "hardware_provenance": hw_prov,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    screen_payload = None
    confirm_payload = None

    screen_particles = args.override_particles if args.override_particles is not None else 30
    screen_epochs = args.override_epochs if args.override_epochs is not None else 160

    confirm_particles = args.override_particles if args.override_particles is not None else 60
    confirm_epochs = args.override_epochs if args.override_epochs is not None else 420

    # 2. Execute Screen Phase if requested or in 'all'
    if args.phase in ("screen", "all"):
        screen_payload = run_phase_b_screen(
            x_search=x_search,
            y_search=y_search,
            x_val=x_val,
            y_val=y_val,
            nested_subsets=nested_subsets,
            device=device,
            seed=91,
            swarm_size=screen_particles,
            epochs=screen_epochs,
        )
        final_payload["screen_payload"] = screen_payload
        save_json_atomic(final_payload, out_dir / "pso_v6_phase_b_screen.json")

    # 3. Execute Confirm Phase if requested or in 'all'
    if args.phase in ("confirm", "all"):
        if screen_payload is not None:
            selected_configs = screen_payload["selected_for_confirm"]
        else:
            screen_art_path = Path(args.screen_artifact)
            if screen_art_path.exists():
                import json
                with open(screen_art_path, "r") as f:
                    art_data = json.load(f)
                selected_configs = art_data.get("screen_payload", {}).get("selected_for_confirm", ["G0", "G1", "G8"])
            else:
                selected_configs = ["G0", "G1", "G8", "G2", "G4"]

        confirm_payload = run_phase_b_confirm(
            selected_configs=selected_configs,
            x_search=x_search,
            y_search=y_search,
            x_val=x_val,
            y_val=y_val,
            nested_subsets=nested_subsets,
            device=device,
            seeds=[101, 102, 103],
            swarm_size=confirm_particles,
            epochs=confirm_epochs,
        )
        final_payload["confirm_payload"] = confirm_payload

    persisted_runs: List[Dict[str, Any]] = []
    if screen_payload is not None:
        persisted_runs.extend(screen_payload["screen_results"].values())
    if confirm_payload is not None:
        for runs in confirm_payload["confirm_runs"].values():
            persisted_runs.extend(runs)
    final_payload["resource_totals"] = {
        "candidate_objective_queries": sum(r["total_queries"] for r in persisted_runs),
        "candidate_sample_evaluations": sum(
            r["total_sample_evaluations"] for r in persisted_runs
        ),
        "validation_model_evaluations": sum(
            r["validation_evaluations"] for r in persisted_runs
        ),
        "summed_optimization_wall_time_sec": round(
            sum(r["optimization_wall_time_sec"] for r in persisted_runs), 4
        ),
        "summed_validation_wall_time_sec": round(
            sum(r["validation_wall_time_sec"] for r in persisted_runs), 4
        ),
        "official_test_evaluations": 0,
    }

    # Save final atomic JSON, CSV, and plots
    json_path = out_dir / "pso_v6_phase_b.json"
    csv_path = out_dir / "pso_v6_phase_b.csv"
    plot_path = plot_dir / "pso_v6_phase_b.png"

    save_json_atomic(final_payload, json_path)
    save_csv_summary_v6(final_payload, csv_path)
    generate_study_plots_v6(final_payload, plot_path)

    return final_payload


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_deep_pso_v6_study(args)
