#!/usr/bin/env python3
"""
Post-Training PSO Ensemble Study Runner

Determines whether PSO is useful and efficient after ordinary backpropagation by
optimizing prediction-space ensemble weights for independently Adam-trained CNNs.

Key Invariants:
1. Development source: only each dataset's official train=True 60,000 examples.
2. Stratified split: 50,000 search / 10,000 validation with split seed 20260904.
3. Normalization: mean and std fitted on search split only (unrounded).
4. Model: CompactCNN (9,098 parameters).
5. Pool seeds: 201, 202, 203, 204, 205.
6. Baseline single_50e: seed 201 captured at epoch 10 and continued to epoch 50.
7. Optimization methods: reference_single_10e, best_single_10e, single_50e,
   uniform_ensemble, uniform_temperature, slsqp_weights, pso_weights.
8. Official test dataset (train=False) loaded only after development gates pass.
9. Trained models retained in memory and reused for official test without retraining.
10. Atomic writes via unique same-directory temporary files + os.replace.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pso.optimizer import Optimizer, resolve_device


PROTOCOL_VERSION = "POST-TRAINING-PSO-ENSEMBLE 1.1.0"


def sync_device(device: Optional[Union[str, torch.device]] = None):
    dev = resolve_device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


# =====================================================================
# 1. Architecture: CompactCNN (9,098 Parameters)
# =====================================================================

class CompactCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(784, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2 and x.shape[1] == 784:
            x = x.view(-1, 1, 28, 28)
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = self.flatten(out)
        return self.fc(out)


def compute_model_fingerprint(model: nn.Module) -> str:
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


# =====================================================================
# 2. PyTorch Softmax Weight Wrapper over Cached Member Probabilities
# =====================================================================

class CachedProbabilityEnsemble(nn.Module):
    """
    PyTorch module parameterizing ensemble weights via a softmax vector over M member probabilities.
    Exposes raw_weights parameter optimized by public pso.Optimizer.
    Returns normalized log probabilities for nn.NLLLoss evaluation.
    Member probabilities input has shape (N, M, K).
    """
    def __init__(self, num_members: int = 5, init_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_members = num_members
        if init_weights is not None:
            if init_weights.shape != (num_members,):
                raise ValueError(f"init_weights must have shape ({num_members},)")
            self.raw_weights = nn.Parameter(init_weights.clone().float())
        else:
            self.raw_weights = nn.Parameter(torch.zeros(num_members, dtype=torch.float32))

    def weights(self) -> torch.Tensor:
        return F.softmax(self.raw_weights, dim=0)

    def forward(self, member_probabilities: torch.Tensor) -> torch.Tensor:
        """
        member_probabilities: canonical tensor of shape (N, M, K)
        Returns log probabilities of shape (N, K)
        """
        if member_probabilities.dim() != 3:
            raise ValueError("member_probabilities must be a 3D tensor")
        if member_probabilities.shape[1] != self.num_members:
            raise ValueError(
                "member_probabilities must use canonical (N, M, K) orientation "
                f"with M={self.num_members}, got {tuple(member_probabilities.shape)}"
            )

        w = self.weights().view(1, -1, 1)
        mix = torch.sum(w * member_probabilities, dim=1)
        return torch.log(torch.clamp(mix, min=1e-12))


# =====================================================================
# 3. Probability Cache & Metric Utilities
# =====================================================================

def validate_probability_cache(probabilities: Union[torch.Tensor, np.ndarray]) -> bool:
    if isinstance(probabilities, torch.Tensor):
        arr = probabilities.detach().cpu().numpy()
    else:
        arr = np.asarray(probabilities)

    if arr.ndim != 3:
        return False

    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return False

    if arr.shape[2] < 2:
        return False

    if not np.all(np.isfinite(arr)):
        return False

    if np.any(arr < -1e-6) or np.any(arr > 1.0 + 1e-6):
        return False

    row_sums = arr.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        return False

    return True


def mixture_probabilities(
    weights: Union[torch.Tensor, np.ndarray],
    member_probabilities: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(weights, torch.Tensor):
        w_np = weights.detach().cpu().numpy()
    else:
        w_np = np.asarray(weights, dtype=np.float64)

    if not np.all(np.isfinite(w_np)):
        raise ValueError("weights contain non-finite values")
    if np.any(w_np < -1e-6):
        raise ValueError("weights contain negative values")
    w_sum = float(w_np.sum())
    if w_sum <= 0:
        raise ValueError("weights sum to zero or negative")

    if not validate_probability_cache(member_probabilities):
        raise ValueError("member_probabilities fails validate_probability_cache")

    is_torch = isinstance(member_probabilities, torch.Tensor)
    if member_probabilities.shape[0] != len(w_np):
        raise ValueError(
            "member_probabilities must use canonical (M, N, K) orientation "
            f"with M={len(w_np)}, got {tuple(member_probabilities.shape)}"
        )

    if is_torch:
        w = torch.as_tensor(
            weights,
            dtype=member_probabilities.dtype,
            device=member_probabilities.device,
        )
        w = w / w.sum()
        return torch.sum(w.view(-1, 1, 1) * member_probabilities, dim=0)

    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    probs = np.asarray(member_probabilities, dtype=np.float64)
    return np.sum(w[:, None, None] * probs, axis=0)


def probabilistic_metrics(
    probabilities: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().cpu().numpy()
    else:
        probs = np.asarray(probabilities, dtype=np.float64)

    if isinstance(targets, torch.Tensor):
        labels = targets.detach().cpu().numpy()
    else:
        labels = np.asarray(targets, dtype=np.int64)

    if probs.ndim != 2:
        raise ValueError(f"probabilities must be 2D array, got shape {probs.shape}")

    N, K = probs.shape
    if labels.ndim != 1 or len(labels) != N:
        raise ValueError(f"targets must be 1D array of length N={N}, got shape {labels.shape}")

    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities contain non-finite values")

    if np.any(labels < 0) or np.any(labels >= K):
        raise ValueError(f"targets must contain integers in range [0, {K-1}]")

    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError("probabilities rows must sum to 1.0")

    preds = probs.argmax(axis=1)
    acc = float((preds == labels).mean()) * 100.0

    eps = 1e-12
    clipped = np.clip(probs, eps, 1.0 - eps)
    nll = -float(np.log(clipped[np.arange(N), labels]).mean())

    y_onehot = np.zeros((N, K), dtype=np.float64)
    y_onehot[np.arange(N), labels] = 1.0
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    n_bins = 15
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    confidences = probs.max(axis=1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop = in_bin.mean()
        if prop > 0:
            accuracy_in_bin = (preds[in_bin] == labels[in_bin]).mean()
            avg_conf = confidences[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_conf) * prop

    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
    margin_mean = float(margins.mean())

    return {
        "accuracy": round(acc, 4),
        "nll": round(nll, 6),
        "brier": round(brier, 6),
        "ece": round(float(ece), 6),
        "margin": round(margin_mean, 6),
    }


evaluate_probabilistic_metrics = probabilistic_metrics


def temp_scaled_probs(uniform_probs: np.ndarray, temp: float) -> np.ndarray:
    eps = 1e-12
    log_p = np.log(np.clip(uniform_probs, eps, 1.0))
    scaled_log_p = log_p / temp
    max_log_p = np.max(scaled_log_p, axis=1, keepdims=True)
    exp_p = np.exp(scaled_log_p - max_log_p)
    return exp_p / np.sum(exp_p, axis=1, keepdims=True)


def fit_uniform_temperature(uniform_probs: np.ndarray, targets: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    eval_count = 0
    def obj(t: float) -> float:
        nonlocal eval_count
        eval_count += 1
        p = temp_scaled_probs(uniform_probs, t)
        return probabilistic_metrics(p, targets)["nll"]

    t0 = time.perf_counter()
    res = scipy.optimize.minimize_scalar(obj, bounds=(0.01, 10.0), method="bounded")
    wall_t = time.perf_counter() - t0

    if not res.success or not math.isfinite(res.x) or res.x <= 0:
        raise RuntimeError(f"Temperature scaling optimization failed: success={res.success}, x={res.x}")

    best_t = float(res.x)
    best_probs = temp_scaled_probs(uniform_probs, best_t)
    metrics = probabilistic_metrics(best_probs, targets)

    temp_record = {
        "fitted_temperature": round(best_t, 6),
        "wall_time_seconds": float(wall_t),
        "evaluations": int(eval_count),
        "metrics": metrics,
    }
    return best_t, temp_record


# =====================================================================
# 4. SLSQP Solver with Analytical Simplex NLL Gradient
# =====================================================================

def simplex_nll_and_grad(
    weights: np.ndarray,
    member_probabilities: np.ndarray,
    targets: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Calculate validation NLL and its analytical gradient.

    `member_probabilities` has one canonical orientation: (M, N, K).
    """
    w = np.asarray(weights, dtype=np.float64)
    probs_mnk = np.asarray(member_probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if not np.all(np.isfinite(w)):
        raise ValueError("Weights contain non-finite values")
    if probs_mnk.ndim != 3:
        raise ValueError(
            f"member_probabilities must be 3D array, got {probs_mnk.ndim}D"
        )

    M, N, _ = probs_mnk.shape
    if len(w) != M:
        raise ValueError(f"Weights length {len(w)} does not match num_members M={M}")
    if labels.ndim != 1 or len(labels) != N:
        raise ValueError(
            "member_probabilities must use canonical (M, N, K) orientation "
            f"with target count N={len(labels)}, got {probs_mnk.shape}"
        )

    mix_p = np.sum(w[:, None, None] * probs_mnk, axis=0)
    p_true = mix_p[np.arange(N), labels]
    p_true_clamped = np.maximum(p_true, 1e-12)
    nll = -float(np.mean(np.log(p_true_clamped)))

    P_true = probs_mnk[
        np.arange(M)[:, None],
        np.arange(N)[None, :],
        labels[None, :],
    ]
    grad = -np.mean(P_true / p_true_clamped[None, :], axis=1)
    return nll, grad


def optimize_slsqp_weights(
    member_probabilities: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    probs_mnk = np.asarray(member_probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if probs_mnk.ndim != 3:
        raise ValueError(
            f"member_probabilities must be 3D array, got {probs_mnk.ndim}D"
        )

    M, N, _ = probs_mnk.shape
    if labels.ndim != 1 or len(labels) != N:
        raise ValueError(
            "member_probabilities must use canonical (M, N, K) orientation "
            f"with target count N={len(labels)}, got {probs_mnk.shape}"
        )

    w0 = np.full(M, 1.0 / M, dtype=np.float64)
    bounds = [(0.0, 1.0)] * M
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones_like(w)}

    eval_count = 0
    def obj_func(w):
        nonlocal eval_count
        eval_count += 1
        return simplex_nll_and_grad(w, probs_mnk, labels)

    start_t = time.perf_counter()
    res = scipy.optimize.minimize(
        fun=obj_func,
        x0=w0,
        method="SLSQP",
        jac=True,
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000},
    )
    wall_t = time.perf_counter() - start_t

    raw_w = np.maximum(res.x, 0.0)
    sum_w = raw_w.sum()
    norm_w = raw_w / sum_w if sum_w > 0 else np.full(M, 1.0 / M)

    mix_probs = mixture_probabilities(norm_w, probs_mnk)
    metrics = probabilistic_metrics(mix_probs, labels)

    return {
        "weights": norm_w.tolist(),
        "evaluations": int(eval_count),
        "wall_time_seconds": float(wall_t),
        "success": bool(res.success),
        "message": str(res.message),
        "metrics": metrics,
    }


# =====================================================================
# 5. Public PSO Optimizer Wrapper over Ensemble Weights
# =====================================================================

def run_pso_weights(
    member_probabilities: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    swarm_seeds: Optional[List[int]] = None,
    particles: int = 30,
    epochs: int = 30,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    if swarm_seeds is None:
        swarm_seeds = [301, 302, 303]

    if isinstance(member_probabilities, np.ndarray):
        probs_t = torch.from_numpy(member_probabilities).float()
    else:
        probs_t = member_probabilities.float()

    if isinstance(targets, np.ndarray):
        targets_t = torch.from_numpy(targets).long()
    else:
        targets_t = targets.long()

    if device is None:
        dev = resolve_device()
    else:
        dev = torch.device(device)

    if probs_t.dim() != 3:
        raise ValueError(
            f"member_probabilities must be 3D tensor, got {probs_t.dim()}D"
        )

    M, N, _ = probs_t.shape
    if targets_t.dim() != 1 or len(targets_t) != N:
        raise ValueError(
            "member_probabilities must use canonical (M, N, K) orientation "
            f"with target count N={len(targets_t)}, got {tuple(probs_t.shape)}"
        )
    probs_mnk = probs_t.contiguous()
    probs_nmk = probs_mnk.permute(1, 0, 2).contiguous()

    N, M, K = probs_nmk.shape
    probs_dev = probs_nmk.to(dev)
    targets_dev = targets_t.to(dev)

    queries_per_seed = particles * epochs
    sample_evaluations_per_seed = particles * epochs * N

    per_seed_runs = []
    best_nll = float('inf')
    selected_seed = swarm_seeds[0]
    selected_weights = [1.0 / M] * M
    selected_metrics: Dict[str, float] = {}

    for seed in swarm_seeds:
        model = CachedProbabilityEnsemble(num_members=M).to(dev)
        nn.init.zeros_(model.raw_weights)

        loss_fn = nn.NLLLoss()
        opt = Optimizer(
            model=model,
            loss=loss_fn,
            task="multiclass",
            method="constriction",
            evaluation="full",
            n_particles=particles,
            particle_min=-4.0,
            particle_max=4.0,
            boundary_strategy="reflect",
            velocity_limit_ratio=0.1,
            initialization="model_noise",
            initial_position_noise=0.0,
            seed=seed,
            device=dev,
        )

        sync_device(dev)
        start_t = time.perf_counter()
        opt.fit(probs_dev, targets_dev, epochs=epochs, renewal="loss")
        sync_device(dev)
        wall_t = time.perf_counter() - start_t

        # Retrieve optimized weights from opt.get_best_model().weights(), never opt.model
        best_model = opt.get_best_model()
        weights_tensor = best_model.weights().detach().cpu()
        weights_list = weights_tensor.numpy().tolist()

        mix_p = mixture_probabilities(weights_tensor, probs_mnk)
        metrics = probabilistic_metrics(mix_p, targets_t)

        run_rec = {
            "seed": seed,
            "queries": queries_per_seed,
            "sample_evaluations": sample_evaluations_per_seed,
            "wall_time_seconds": float(wall_t),
            "metrics": metrics,
            "weights": weights_list,
        }
        per_seed_runs.append(run_rec)

        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            selected_seed = seed
            selected_weights = weights_list
            selected_metrics = metrics

    wall_times = [r["wall_time_seconds"] for r in per_seed_runs]
    median_wall = float(np.median(wall_times))
    total_wall = float(np.sum(wall_times))

    return {
        "per_seed_runs": per_seed_runs,
        "selected_seed": selected_seed,
        "selected_weights": selected_weights,
        "metrics": selected_metrics,
        "queries_per_seed": queries_per_seed,
        "sample_evaluations_per_seed": sample_evaluations_per_seed,
        "total_queries": queries_per_seed * len(swarm_seeds),
        "total_sample_evaluations": sample_evaluations_per_seed * len(swarm_seeds),
        "median_one_seed_wall_time_seconds": median_wall,
        "total_wall_time_seconds": total_wall,
    }


# =====================================================================
# 6. Data Preparation (Strict Train=True Only During Development)
# =====================================================================

def prepare_dataset_splits(
    dataset_name: str,
    split_seed: int = 20260904,
    cache_dir: Optional[Path] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    Dict[str, Any]
]:
    """
    Constructs 50,000 search split and 10,000 validation split using exclusively train=True.
    Fits mean and std on search split only (unrounded). Never constructs train=False.
    """
    if cache_dir is None:
        cache_dir = Path("result/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds_lower = dataset_name.lower()
    if ds_lower == "mnist":
        from torchvision.datasets import MNIST
        raw_train = MNIST(root=str(cache_dir), train=True, download=True)
        canonical_name = "MNIST"
    elif ds_lower in ("fashion_mnist", "fashion"):
        from torchvision.datasets import FashionMNIST
        raw_train = FashionMNIST(root=str(cache_dir), train=True, download=True)
        canonical_name = "FashionMNIST"
    else:
        raise ValueError(f"Unsupported dataset name: '{dataset_name}'")

    x_train_raw = raw_train.data.float() / 255.0  # (60000, 28, 28)
    y_train_raw = raw_train.targets.long()

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

    mean_val = float(x_search_raw.mean())
    std_val = float(x_search_raw.std())

    x_search_norm = ((x_search_raw - mean_val) / std_val).unsqueeze(1)
    x_val_norm = ((x_val_raw - mean_val) / std_val).unsqueeze(1)

    h_data = hashlib.sha256()
    for t in (x_search_norm, x_val_norm, y_search, y_val):
        h_data.update(t.detach().cpu().numpy().tobytes())
    data_fp = h_data.hexdigest()[:16]

    h_split = hashlib.sha256()
    h_split.update(search_idx.tobytes())
    h_split.update(val_idx.tobytes())
    split_fp = h_split.hexdigest()[:16]

    provenance = {
        "dataset_name": canonical_name,
        "split_seed": split_seed,
        "search_samples": 50000,
        "validation_samples": 10000,
        "normalization": {"mean": mean_val, "std": std_val},
        "data_fingerprint": data_fp,
        "split_fingerprint": split_fp,
    }

    return x_search_norm, y_search, x_val_norm, y_val, provenance


def load_official_test_data(
    dataset_name: str,
    mean_val: float,
    std_val: float,
    cache_dir: Optional[Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deferred test loader called exclusively after all development gates pass.
    Uses exact unrounded normalization parameters fitted on search split.
    """
    if cache_dir is None:
        cache_dir = Path("result/cache")

    ds_lower = dataset_name.lower()
    if ds_lower == "mnist":
        from torchvision.datasets import MNIST
        raw_test = MNIST(root=str(cache_dir), train=False, download=True)
    elif ds_lower in ("fashion_mnist", "fashion"):
        from torchvision.datasets import FashionMNIST
        raw_test = FashionMNIST(root=str(cache_dir), train=False, download=True)
    else:
        raise ValueError(f"Unsupported dataset name: '{dataset_name}'")

    x_test_raw = raw_test.data.float() / 255.0
    y_test = raw_test.targets.long()
    x_test_norm = ((x_test_raw - mean_val) / std_val).unsqueeze(1)
    return x_test_norm, y_test


def get_model_probabilities(
    model: nn.Module,
    x_data: torch.Tensor,
    device: torch.device,
    batch_size: int = 1000,
) -> Tuple[torch.Tensor, float]:
    model.eval()
    model.to(device)
    probs_list = []
    sync_device(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(x_data), batch_size):
            batch_x = x_data[i:i+batch_size].to(device)
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs.cpu())
    sync_device(device)
    wall_t = time.perf_counter() - t0
    res_t = torch.cat(probs_list, dim=0)
    return res_t, wall_t


# =====================================================================
# 7. Development Gate Evaluator
# =====================================================================

def evaluate_development_gates(workloads_data: Dict[str, Any]) -> Dict[str, Any]:
    gate_results = {}
    issues = []

    # 1. All values finite & valid simplex weights
    finite_ok = True
    simplex_ok = True
    for wl_id, wl in workloads_data.items():
        methods = wl["validation"]["methods"]
        for m_name, m_val in methods.items():
            if m_name in ("pso_weights", "slsqp_weights", "uniform_temperature"):
                mets = m_val["metrics"]
            else:
                mets = m_val
            for k, v in mets.items():
                if not math.isfinite(v):
                    finite_ok = False
                    issues.append(f"{wl_id} {m_name} metric {k}={v} non-finite")

        slsqp_w = np.array(methods["slsqp_weights"]["weights"])
        pso_w = np.array(methods["pso_weights"]["selected_weights"])
        for name, w in [("slsqp", slsqp_w), ("pso", pso_w)]:
            if np.any(w < -1e-6) or not math.isclose(np.sum(w), 1.0, abs_tol=1e-6):
                simplex_ok = False
                issues.append(f"{wl_id} {name} weights {w} invalid simplex")

    gate_results["all_values_finite"] = finite_ok and simplex_ok

    # 2. Validation pool forward passes = 5
    fwd_ok = all(
        wl["validation_cache"]["pool_forward_passes"] == 5
        for wl in workloads_data.values()
    )
    gate_results["validation_pool_forward_passes_exact"] = fwd_ok

    # 3. Optimization base model forward passes = 0
    base_fwd_ok = all(
        wl["validation_cache"]["base_cnn_forward_passes_during_optimization"] == 0
        for wl in workloads_data.values()
    )
    gate_results["optimization_base_model_forward_passes"] = base_fwd_ok

    # 4. Official test data loaded before freeze = False and evals = 0
    test_leak_ok = all(
        not wl.get("official_test_data_loaded_before_freeze", False) and
        wl.get("official_test_evaluations_before_freeze", 0) == 0
        for wl in workloads_data.values()
    )
    gate_results["official_test_data_loaded_before_freeze"] = test_leak_ok

    # 5. SLSQP solver success
    slsqp_success_ok = all(
        wl["validation"]["methods"]["slsqp_weights"]["success"]
        for wl in workloads_data.values()
    )
    if not slsqp_success_ok:
        issues.append("SLSQP solver failed on one or more workloads")
    gate_results["slsqp_solver_success"] = slsqp_success_ok

    # 6. Exact query and sample accounting (900 queries, 9,000,000 samples per seed for Iteration 1)
    acct_ok = True
    for wl_id, wl in workloads_data.items():
        pso_rec = wl["validation"]["methods"]["pso_weights"]
        for r in pso_rec["per_seed_runs"]:
            if r["queries"] != 900 or r["sample_evaluations"] != 9000000:
                acct_ok = False
                issues.append(f"{wl_id} seed {r['seed']} queries={r['queries']} samples={r['sample_evaluations']}")
    gate_results["query_and_sample_accounting_exact"] = acct_ok

    # 7. PSO validation NLL <= uniform ensemble NLL + 1e-7
    pso_nll_vs_uniform_ok = True
    for wl_id, wl in workloads_data.items():
        pso_nll = wl["validation"]["methods"]["pso_weights"]["metrics"]["nll"]
        uni_nll = wl["validation"]["methods"]["uniform_ensemble"]["nll"]
        if pso_nll > uni_nll + 1e-7:
            pso_nll_vs_uniform_ok = False
            issues.append(f"{wl_id} PSO val NLL {pso_nll:.6f} > uniform {uni_nll:.6f}")
    gate_results["maximum_pso_nll_regression_vs_uniform"] = pso_nll_vs_uniform_ok

    # 8. PSO validation accuracy regression vs uniform <= 0.10 pp
    pso_acc_vs_uniform_ok = True
    for wl_id, wl in workloads_data.items():
        pso_acc = wl["validation"]["methods"]["pso_weights"]["metrics"]["accuracy"]
        uni_acc = wl["validation"]["methods"]["uniform_ensemble"]["accuracy"]
        if uni_acc - pso_acc > 0.10:
            pso_acc_vs_uniform_ok = False
            issues.append(f"{wl_id} PSO val acc {pso_acc:.4f}% regressed >0.10pp vs uniform {uni_acc:.4f}%")
    gate_results["maximum_pso_accuracy_regression_vs_uniform_pp"] = pso_acc_vs_uniform_ok

    # 9. PSO validation NLL < reference single 10e NLL
    pso_nll_vs_ref_ok = True
    for wl_id, wl in workloads_data.items():
        pso_nll = wl["validation"]["methods"]["pso_weights"]["metrics"]["nll"]
        ref_nll = wl["validation"]["methods"]["reference_single_10e"]["nll"]
        if pso_nll >= ref_nll:
            pso_nll_vs_ref_ok = False
            issues.append(f"{wl_id} PSO val NLL {pso_nll:.6f} >= ref single {ref_nll:.6f}")
    gate_results["pso_nll_below_reference_single"] = pso_nll_vs_ref_ok

    # 10. PSO validation NLL <= single_50e NLL + 1e-7
    pso_nll_vs_50e_ok = True
    for wl_id, wl in workloads_data.items():
        pso_nll = wl["validation"]["methods"]["pso_weights"]["metrics"]["nll"]
        s50_nll = wl["validation"]["methods"]["single_50e"]["nll"]
        if pso_nll > s50_nll + 1e-7:
            pso_nll_vs_50e_ok = False
            issues.append(f"{wl_id} PSO val NLL {pso_nll:.6f} > single_50e {s50_nll:.6f}")
    gate_results["maximum_pso_nll_regression_vs_equal_budget_single"] = pso_nll_vs_50e_ok

    # 11. PSO validation NLL within 0.5% of SLSQP validation NLL
    pso_gap_slsqp_ok = True
    for wl_id, wl in workloads_data.items():
        pso_nll = wl["validation"]["methods"]["pso_weights"]["metrics"]["nll"]
        slsqp_nll = wl["validation"]["methods"]["slsqp_weights"]["metrics"]["nll"]
        rel_gap = (pso_nll - slsqp_nll) / slsqp_nll
        if rel_gap > 0.005:
            pso_gap_slsqp_ok = False
            issues.append(f"{wl_id} PSO vs SLSQP relative NLL gap {rel_gap:.4f} > 0.005")
    gate_results["maximum_relative_pso_nll_gap_vs_slsqp"] = pso_gap_slsqp_ok

    # 12. Cross-dataset mean relative PSO NLL reduction vs uniform >= 0.0
    rel_reductions = []
    for wl_id, wl in workloads_data.items():
        pso_nll = wl["validation"]["methods"]["pso_weights"]["metrics"]["nll"]
        uni_nll = wl["validation"]["methods"]["uniform_ensemble"]["nll"]
        rel_red = (uni_nll - pso_nll) / uni_nll
        rel_reductions.append(rel_red)
    mean_rel_red = float(np.mean(rel_reductions)) if rel_reductions else -1.0
    mean_rel_red_ok = mean_rel_red >= 0.0
    if not mean_rel_red_ok:
        issues.append(f"Mean relative NLL reduction vs uniform {mean_rel_red:.6f} < 0.0")
    gate_results["cross_dataset_mean_relative_pso_nll_reduction_vs_uniform_minimum"] = mean_rel_red_ok

    # 13. Median 1-seed PSO wall time / pool training wall time <= 0.10
    wall_ratio_ok = True
    for wl_id, wl in workloads_data.items():
        med_pso_wall = wl["validation"]["methods"]["pso_weights"]["median_one_seed_wall_time_seconds"]
        pool_wall = wl["training"]["adam_pool_wall_time_seconds"]
        ratio = med_pso_wall / pool_wall if pool_wall > 0 else 1.0
        if ratio > 0.10:
            wall_ratio_ok = False
            issues.append(f"{wl_id} PSO median wall time ratio {ratio:.4f} > 0.10")
    gate_results["maximum_median_one_seed_pso_to_pool_training_wall_ratio"] = wall_ratio_ok

    all_pass = all(gate_results.values())
    failed_count = sum(1 for v in gate_results.values() if not v)

    return {
        "pass": all_pass,
        "failed_hard_gate_count": failed_count,
        "gate_results": gate_results,
        "issues": issues,
    }


# =====================================================================
# 8. Publication Output Writers (Atomic Write via os.replace)
# =====================================================================

def atomic_write_file(target_path: Path, content_str_or_bytes: Union[str, bytes], is_binary: bool = False):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target_path.parent), prefix=f".tmp_{target_path.name}_")
    try:
        with os.fdopen(fd, 'wb' if is_binary else 'w', encoding=None if is_binary else 'utf-8') as f:
            f.write(content_str_or_bytes)
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_csv_report(artifact: Dict[str, Any], output_path: Path):
    lines = [
        "Workload,Phase,Method,Accuracy,NLL,Brier,ECE,Margin,WallTimeSeconds,ParameterMultiplier,InferenceMultiplier"
    ]

    method_multipliers = {
        "reference_single_10e": (1.0, 1.0),
        "best_single_10e": (1.0, 1.0),
        "single_50e": (1.0, 1.0),
        "uniform_ensemble": (5.0, 5.0),
        "uniform_temperature": (5.0, 5.0),
        "slsqp_weights": (5.0, 5.0),
        "pso_weights": (5.0, 5.0),
    }

    for wl_id, wl in artifact["workloads"].items():
        # Validation Phase
        val_methods = wl["validation"]["methods"]
        for m_name, m_data in val_methods.items():
            if m_name == "pso_weights":
                mets = m_data["metrics"]
                wall_t = m_data["median_one_seed_wall_time_seconds"]
            elif m_name == "slsqp_weights":
                mets = m_data["metrics"]
                wall_t = m_data["wall_time_seconds"]
            elif m_name == "uniform_temperature":
                mets = m_data["metrics"]
                wall_t = m_data.get("wall_time_seconds", 0.0)
            else:
                mets = m_data
                wall_t = 0.0

            param_m, inf_m = method_multipliers.get(m_name, (1.0, 1.0))
            line = f"{wl_id},validation,{m_name},{mets['accuracy']:.4f},{mets['nll']:.6f},{mets['brier']:.6f},{mets['ece']:.6f},{mets['margin']:.6f},{wall_t:.4f},{param_m:.1f},{inf_m:.1f}"
            lines.append(line)

        # Confirmation / Test Phase
        if wl.get("confirmation") is not None:
            test_methods = wl["confirmation"]["methods"]
            for m_name, mets in test_methods.items():
                param_m, inf_m = method_multipliers.get(m_name, (1.0, 1.0))
                line = f"{wl_id},official_test,{m_name},{mets['accuracy']:.4f},{mets['nll']:.6f},{mets['brier']:.6f},{mets['ece']:.6f},{mets['margin']:.6f},0.0000,{param_m:.1f},{inf_m:.1f}"
                lines.append(line)

    csv_content = "\n".join(lines) + "\n"
    atomic_write_file(output_path, csv_content, is_binary=False)


def save_publication_plot(artifact: Dict[str, Any], output_path: Path):
    """Write the validation/test summary figure without assuming confirmation ran.

    The development artifact is the primary source for this figure.  Official-test
    panels are intentionally left empty when development gates fail, rather than
    silently reusing validation values or omitting methods from the comparison.
    """
    methods_order = [
        "reference_single_10e",
        "best_single_10e",
        "single_50e",
        "uniform_ensemble",
        "uniform_temperature",
        "slsqp_weights",
        "pso_weights",
    ]
    method_labels = [
        "Ref 10e",
        "Best 10e",
        "Single 50e",
        "Uniform",
        "Temp uniform",
        "SLSQP",
        "PSO",
    ]

    workloads_data = artifact.get("workloads", {})
    workloads = list(workloads_data.keys())

    def _method_record(wl: Dict[str, Any], phase: str, method: str) -> Dict[str, Any]:
        phase_record = wl.get(phase)
        if not isinstance(phase_record, dict):
            return {}
        methods = phase_record.get("methods")
        if not isinstance(methods, dict):
            return {}
        record = methods.get(method)
        return record if isinstance(record, dict) else {}

    def _metrics(wl: Dict[str, Any], phase: str, method: str) -> Dict[str, Any]:
        record = _method_record(wl, phase, method)
        nested = record.get("metrics")
        return nested if isinstance(nested, dict) else record

    def _number(value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return value if math.isfinite(value) else float("nan")

    def _count_text(value: Any) -> str:
        """Format optional sample counts without assuming a complete artifact."""
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return "—"

    def _metric(wl: Dict[str, Any], phase: str, method: str, name: str) -> float:
        return _number(_metrics(wl, phase, method).get(name))

    def _grouped_metric(
        ax: Any,
        phase: str,
        metric_name: str,
        title: str,
        ylabel: str,
        unavailable_text: Optional[str] = None,
    ) -> bool:
        """Draw one phase/metric panel and return whether any value was present."""
        x = np.arange(len(methods_order), dtype=float)
        n_workloads = max(len(workloads), 1)
        width = min(0.8 / n_workloads, 0.28)
        plotted = False
        observed: List[float] = []
        cmap = plt.get_cmap("tab10")

        for workload_idx, wl_id in enumerate(workloads):
            wl = workloads_data[wl_id]
            values = [
                _metric(wl, phase, method, metric_name)
                for method in methods_order
            ]
            observed.extend(value for value in values if math.isfinite(value))
            if any(math.isfinite(value) for value in values):
                plotted = True
            offset = (workload_idx - (n_workloads - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=wl_id.replace("_", " ").title(),
                color=cmap(workload_idx % 10),
                alpha=0.88,
                edgecolor="white",
                linewidth=0.4,
            )

        display_title = title
        if plotted and metric_name == "nll":
            positive_values = [value for value in observed if value > 0]
            if len(positive_values) == len(observed):
                ax.set_yscale("log")
                display_title = f"{title} (log scale)"
        elif plotted and metric_name == "accuracy":
            # Accuracy differences are sub-percentage-point on the official
            # test set; a zero-based axis makes the methods indistinguishable.
            low = max(0.0, min(observed) - 1.0)
            high = min(100.0, max(observed) + 0.5)
            if high <= low:
                high = min(100.0, low + 1.0)
            ax.set_ylim(low, high)

        ax.set_title(display_title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=32, ha="right", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        if not plotted and unavailable_text:
            ax.text(
                0.5,
                0.52,
                unavailable_text,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#555555",
                wrap=True,
            )
        return plotted

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    fig.suptitle(
        "Post-Training Prediction-Space Ensemble Study",
        fontsize=16,
        fontweight="bold",
    )

    # The first four panels keep validation context beside the sealed,
    # one-shot official-test confirmation results.
    _grouped_metric(
        axes[0, 0],
        "validation",
        "nll",
        "Validation NLL (lower is better)",
        "NLL",
    )
    has_test_nll = _grouped_metric(
        axes[0, 1],
        "confirmation",
        "nll",
        "Official-test NLL (lower is better)",
        "NLL",
        "Official test not run:\ndevelopment gates failed",
    )
    _grouped_metric(
        axes[1, 0],
        "validation",
        "accuracy",
        "Validation accuracy (higher is better)",
        "Accuracy (%)",
    )
    has_test_accuracy = _grouped_metric(
        axes[1, 1],
        "confirmation",
        "accuracy",
        "Official-test accuracy (higher is better)",
        "Accuracy (%)",
        "Official test not run:\ndevelopment gates failed",
    )

    # Efficiency is shown separately from accuracy/NLL so the plot does not
    # imply that a slower optimizer is a better ensemble method.
    ax_eff = axes[0, 2]
    efficiency_categories = [
        "Pool\n5x10e",
        "Single\n50e",
        "Temp\nuniform",
        "SLSQP",
        "PSO\n1 seed",
        "PSO\n3 seeds",
    ]
    n_workloads = max(len(workloads), 1)
    x_eff = np.arange(len(efficiency_categories), dtype=float)
    width_eff = min(0.8 / n_workloads, 0.28)
    efficiency_plotted = False
    cmap = plt.get_cmap("tab10")
    for workload_idx, wl_id in enumerate(workloads):
        wl = workloads_data[wl_id]
        training = wl.get("training", {})
        val_methods = wl.get("validation", {}).get("methods", {})
        temp_rec = val_methods.get("uniform_temperature", {})
        slsqp_rec = val_methods.get("slsqp_weights", {})
        pso_rec = val_methods.get("pso_weights", {})
        values = [
            _number(training.get("adam_pool_wall_time_seconds")),
            _number(training.get("single_50e_wall_time_seconds")),
            _number(temp_rec.get("wall_time_seconds")),
            _number(slsqp_rec.get("wall_time_seconds")),
            _number(pso_rec.get("median_one_seed_wall_time_seconds")),
            _number(pso_rec.get("total_wall_time_seconds")),
        ]
        if any(math.isfinite(value) and value > 0 for value in values):
            efficiency_plotted = True
        offset = (workload_idx - (n_workloads - 1) / 2.0) * width_eff
        ax_eff.bar(
            x_eff + offset,
            values,
            width=width_eff,
            label=wl_id.replace("_", " ").title(),
            color=cmap(workload_idx % 10),
            alpha=0.88,
            edgecolor="white",
            linewidth=0.4,
        )
    ax_eff.set_title(
        "Efficiency: wall time (log scale)" if efficiency_plotted
        else "Efficiency: wall time"
    )
    ax_eff.set_ylabel("Seconds")
    ax_eff.set_xticks(x_eff)
    ax_eff.set_xticklabels(efficiency_categories, fontsize=8)
    if efficiency_plotted:
        ax_eff.set_yscale("log")
    ax_eff.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_eff.set_axisbelow(True)
    if not efficiency_plotted:
        ax_eff.text(
            0.5,
            0.52,
            "Efficiency data unavailable",
            transform=ax_eff.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#555555",
        )

    # Context panel makes the data freeze and a development-only artifact
    # explicit in the publication figure.
    ax_context = axes[1, 2]
    ax_context.axis("off")
    config = artifact.get("config", {})
    resource_totals = artifact.get("resource_totals", {})
    dev_pass = artifact.get("development_pass")
    test_loaded = artifact.get("official_test_data_loaded")
    confirmation_available = has_test_nll or has_test_accuracy
    status = "PASS" if dev_pass is True else "FAIL / not confirmed"
    test_status = "available" if confirmation_available else "not run"
    pool_seeds = config.get("pool_seeds", [])
    lines = [
        "Study context",
        f"Workloads: {', '.join(w.replace('_', ' ').title() for w in workloads) or 'none'}",
        f"Validation: {_count_text(config.get('search_samples'))} search / "
        f"{_count_text(config.get('validation_samples'))} holdout",
        f"Pool: {len(pool_seeds) or 5} independently trained models",
        f"Development gates: {status}",
        f"Official test data: {'loaded' if test_loaded else 'sealed'}",
        f"Official confirmation: {test_status}",
        "",
        "Prediction-space weights; no model soup",
    ]
    pso_research = resource_totals.get(
        "pso_research_wall_time_seconds",
        resource_totals.get("total_pso_wall_time_seconds"),
    )
    pso_ratio = resource_totals.get("pso_to_pool_wall_ratio")
    slsqp_total = resource_totals.get("slsqp_total_wall_time_seconds")
    if pso_research is not None or slsqp_total is not None:
        lines.extend(
            [
                "",
                f"PSO research time: {_number(pso_research):.3f}s",
                f"SLSQP total time: {_number(slsqp_total):.3f}s",
            ]
        )
    if pso_ratio is not None:
        lines.append(f"Median workload PSO / pool ratio: {_number(pso_ratio):.2%}")
    ax_context.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax_context.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        linespacing=1.45,
        family="DejaVu Sans",
    )
    if not confirmation_available:
        ax_context.text(
            0.03,
            0.08,
            "Validation results are retained; official-test panels are\n"
            "intentionally unavailable because the policy was not confirmed.",
            transform=ax_context.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            color="#8a3b12",
            wrap=True,
        )
    # One shared workload legend keeps the data panels uncluttered while
    # preserving the dataset color mapping across all comparisons.
    if workloads:
        legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.925),
                ncol=min(4, len(legend_labels)),
                frameon=False,
                fontsize=9,
                title="Dataset",
            )

    fig.tight_layout(rect=(0, 0, 1, 0.88))


    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    buf_path = Path(buf.name)
    buf.close()
    try:
        fig.savefig(buf_path, format="png", dpi=180, bbox_inches="tight")
        with open(buf_path, "rb") as f:
            img_bytes = f.read()
        # Keep publication output atomic even when plotting or serialization
        # fails partway through.
        atomic_write_file(output_path, img_bytes, is_binary=True)
    finally:
        plt.close(fig)
        if buf_path.exists():
            buf_path.unlink()

# =====================================================================
# 9. Main Orchestration Function
# =====================================================================

def run_post_training_study(
    cache_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    output_json: Optional[Union[str, Path]] = None,
    output_csv: Optional[Union[str, Path]] = None,
    output_png: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    if cache_dir is None:
        cache_dir = Path("result/cache")
    else:
        cache_dir = Path(cache_dir)

    if output_json is None:
        output_json = Path("benchmark_results/pso_v8_post_training_ensemble.json")
    else:
        output_json = Path(output_json)

    if output_csv is None:
        output_csv = Path("benchmark_results/pso_v8_post_training_ensemble.csv")
    else:
        output_csv = Path(output_csv)

    if output_png is None:
        output_png = Path("history_plt/pso_v8_post_training_ensemble.png")
    else:
        output_png = Path(output_png)

    dev = resolve_device(device)
    print(f"[{PROTOCOL_VERSION}] Starting study on device={dev}...")

    pool_seeds = [201, 202, 203, 204, 205]
    swarm_seeds = [301, 302, 303]
    dataset_names = ["mnist", "fashion_mnist"]

    workloads: Dict[str, Any] = {}
    retained_trained_models: Dict[str, Dict[str, Any]] = {}

    for ds_name in dataset_names:
        print(f"\n--- Workload: {ds_name} ---")
        x_search, y_search, x_val, y_val, provenance = prepare_dataset_splits(
            dataset_name=ds_name,
            split_seed=20260904,
            cache_dir=cache_dir,
        )

        # -------------------------------------------------------------
        # Step A: Train Pool & 50e Single Model
        # -------------------------------------------------------------
        pool_models: Dict[int, nn.Module] = {}
        pool_train_times: List[float] = []
        model_fingerprints: Dict[str, str] = {}

        # 1) Seed 201: Train for 10 epochs (capture reference single), then continue to 50 epochs (single_50e)
        torch.manual_seed(201)
        model_201 = CompactCNN().to(dev)
        opt_201 = torch.optim.Adam(model_201.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        g_201 = torch.Generator()
        g_201.manual_seed(201)
        search_ds = TensorDataset(x_search, y_search)
        loader_201 = DataLoader(search_ds, batch_size=256, shuffle=True, generator=g_201)

        sync_device(dev)
        t_start_201 = time.perf_counter()
        model_201.train()
        for epoch in range(1, 11):
            for bx, by in loader_201:
                bx, by = bx.to(dev), by.to(dev)
                opt_201.zero_grad()
                out = model_201(bx)
                loss = criterion(out, by)
                loss.backward()
                opt_201.step()
        sync_device(dev)
        t_10e_201 = time.perf_counter() - t_start_201
        pool_train_times.append(t_10e_201)

        # Save 10e model snapshot for pool seed 201
        m_201_10e = CompactCNN().to(dev)
        m_201_10e.load_state_dict(copy.deepcopy(model_201.state_dict()))
        pool_models[201] = m_201_10e
        model_fingerprints["201"] = compute_model_fingerprint(m_201_10e)

        # Continue exact same model_201 and optimizer stream to epoch 50
        for epoch in range(11, 51):
            for bx, by in loader_201:
                bx, by = bx.to(dev), by.to(dev)
                opt_201.zero_grad()
                out = model_201(bx)
                loss = criterion(out, by)
                loss.backward()
                opt_201.step()
        sync_device(dev)
        t_50e_201 = time.perf_counter() - t_start_201
        single_50e_model = model_201
        model_fingerprints["single_50e"] = compute_model_fingerprint(single_50e_model)

        # 2) Seeds 202-205: Train 10 epochs each
        for seed in [202, 203, 204, 205]:
            torch.manual_seed(seed)
            m = CompactCNN().to(dev)
            opt_m = torch.optim.Adam(m.parameters(), lr=0.001)
            g_m = torch.Generator()
            g_m.manual_seed(seed)
            loader_m = DataLoader(search_ds, batch_size=256, shuffle=True, generator=g_m)

            sync_device(dev)
            t_start_m = time.perf_counter()
            m.train()
            for epoch in range(1, 11):
                for bx, by in loader_m:
                    bx, by = bx.to(dev), by.to(dev)
                    opt_m.zero_grad()
                    out = m(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    opt_m.step()
            sync_device(dev)
            t_m = time.perf_counter() - t_start_m
            pool_train_times.append(t_m)
            pool_models[seed] = m
            model_fingerprints[str(seed)] = compute_model_fingerprint(m)

        pool_training_wall_t = float(sum(pool_train_times))

        training_rec = {
            "architecture": "CompactCNN",
            "parameters": 9098,
            "pool_seeds": pool_seeds,
            "pool_epochs_each": 10,
            "adam_pool_epochs": 50,
            "adam_lr": 0.001,
            "adam_batch_size": 256,
            "adam_pool_wall_time_seconds": pool_training_wall_t,
            "single_50e_epochs": 50,
            "single_50e_wall_time_seconds": float(t_50e_201),
            "model_fingerprints": model_fingerprints,
        }

        # Retain trained models in memory for exact official test evaluation without retraining
        retained_trained_models[ds_name] = {
            "pool_models": pool_models,
            "single_50e_model": single_50e_model,
        }

        # -------------------------------------------------------------
        # Step B: Create Validation Probability Cache
        # -------------------------------------------------------------
        sync_device(dev)
        t0_val_cache = time.perf_counter()
        val_probs_list = []
        for seed in pool_seeds:
            p_m, _ = get_model_probabilities(pool_models[seed], x_val, dev)
            val_probs_list.append(p_m)

        val_pool_probs_t = torch.stack(val_probs_list, dim=0)  # (5, 10000, 10)
        cache_valid = validate_probability_cache(val_pool_probs_t)

        val_50e_probs_t, _ = get_model_probabilities(single_50e_model, x_val, dev)
        sync_device(dev)
        val_cache_wall_t = time.perf_counter() - t0_val_cache

        val_pool_bytes = int(val_pool_probs_t.element_size() * val_pool_probs_t.nelement()) + int(val_50e_probs_t.element_size() * val_50e_probs_t.nelement())

        validation_cache_rec = {
            "valid": cache_valid,
            "pool_forward_passes": 5,
            "long_single_forward_passes": 1,
            "base_cnn_forward_passes_during_optimization": 0,
            "shape": list(val_pool_probs_t.shape),
            "memory_bytes": val_pool_bytes,
            "wall_time_seconds": float(val_cache_wall_t),
        }

        # -------------------------------------------------------------
        # Step C: Evaluate Validation Baselines & Optimization Methods
        # -------------------------------------------------------------
        val_pool_probs_np = val_pool_probs_t.numpy()
        val_50e_probs_np = val_50e_probs_t.numpy()
        y_val_np = y_val.numpy()

        # 1) reference_single_10e (seed 201)
        ref_single_metrics = probabilistic_metrics(val_pool_probs_np[0], y_val_np)

        # 2) best_single_10e
        pool_nlls = [probabilistic_metrics(val_pool_probs_np[i], y_val_np)["nll"] for i in range(5)]
        best_single_idx = int(np.argmin(pool_nlls))
        best_single_seed = pool_seeds[best_single_idx]
        best_single_metrics = probabilistic_metrics(val_pool_probs_np[best_single_idx], y_val_np)
        best_single_metrics["selected_seed"] = best_single_seed

        # 3) single_50e
        s50e_metrics = probabilistic_metrics(val_50e_probs_np, y_val_np)

        # 4) uniform_ensemble
        uniform_probs = val_pool_probs_np.mean(axis=0)
        uniform_metrics = probabilistic_metrics(uniform_probs, y_val_np)

        # 5) uniform_temperature
        fitted_temp, uniform_temp_rec = fit_uniform_temperature(uniform_probs, y_val_np)

        # 6) slsqp_weights
        slsqp_rec = optimize_slsqp_weights(val_pool_probs_np, y_val_np)

        # 7) pso_weights (Iteration 1: 30 particles x 30 epochs = 900 queries per seed)
        pso_rec = run_pso_weights(val_pool_probs_t, y_val, swarm_seeds=swarm_seeds, particles=30, epochs=30, device=str(dev))

        validation_rec = {
            "methods": {
                "reference_single_10e": ref_single_metrics,
                "best_single_10e": best_single_metrics,
                "single_50e": s50e_metrics,
                "uniform_ensemble": uniform_metrics,
                "uniform_temperature": uniform_temp_rec,
                "slsqp_weights": slsqp_rec,
                "pso_weights": pso_rec,
            }
        }

        workloads[ds_name] = {
            "provenance": provenance,
            "training": training_rec,
            "validation_cache": validation_cache_rec,
            "validation": validation_rec,
            "official_test_data_loaded_before_freeze": False,
            "official_test_evaluations_before_freeze": 0,
            "confirmation": None,
        }

    # -----------------------------------------------------------------
    # Step D: Development Gates Evaluation & Policy Freeze
    # -----------------------------------------------------------------
    dev_gate_res = evaluate_development_gates(workloads)
    dev_pass = dev_gate_res["pass"]

    print(f"\n=== Development Phase Summary ===")
    print(f"Development Pass: {dev_pass} (Failed Gates: {dev_gate_res['failed_hard_gate_count']})")
    for g_name, g_status in dev_gate_res["gate_results"].items():
        print(f"  - {g_name}: {'PASS' if g_status else 'FAIL'}")
    if dev_gate_res["issues"]:
        print("Issues:")
        for iss in dev_gate_res["issues"]:
            print(f"  * {iss}")

    policy_frozen = True
    official_test_data_loaded = False

    # -----------------------------------------------------------------
    # Step E: Deferred Official Test Loading & Evaluation (If Dev Passed)
    # Reuses retained models directly; NEVER retrains after freeze.
    # -----------------------------------------------------------------
    if dev_pass:
        print("\n=== Official Test Confirmation Phase ===")
        official_test_data_loaded = True

        for ds_name in dataset_names:
            wl = workloads[ds_name]
            mean_v = wl["provenance"]["normalization"]["mean"]
            std_v = wl["provenance"]["normalization"]["std"]

            x_test, y_test = load_official_test_data(
                dataset_name=ds_name,
                mean_val=mean_v,
                std_val=std_v,
                cache_dir=cache_dir,
            )

            # Reuse retained trained models directly (NO RETRAINING)
            ret_pool = retained_trained_models[ds_name]["pool_models"]
            ret_50e = retained_trained_models[ds_name]["single_50e_model"]

            sync_device(dev)
            t0_test_cache = time.perf_counter()
            test_pool_probs_list = []
            for seed in pool_seeds:
                p_m, _ = get_model_probabilities(ret_pool[seed], x_test, dev)
                test_pool_probs_list.append(p_m)

            p_test_50e, _ = get_model_probabilities(ret_50e, x_test, dev)
            sync_device(dev)
            test_cache_wall_t = time.perf_counter() - t0_test_cache

            test_pool_probs_t = torch.stack(test_pool_probs_list, dim=0)  # (5, 10000, 10)
            test_bytes = int(test_pool_probs_t.element_size() * test_pool_probs_t.nelement()) + int(p_test_50e.element_size() * p_test_50e.nelement())

            test_pool_probs_np = test_pool_probs_t.numpy()
            test_50e_probs_np = p_test_50e.numpy()
            y_test_np = y_test.numpy()

            # Retrieve frozen parameters from validation phase
            frozen_pso_w = wl["validation"]["methods"]["pso_weights"]["selected_weights"]
            frozen_pso_seed = wl["validation"]["methods"]["pso_weights"]["selected_seed"]
            frozen_slsqp_w = wl["validation"]["methods"]["slsqp_weights"]["weights"]
            frozen_temp = wl["validation"]["methods"]["uniform_temperature"]["fitted_temperature"]
            val_best_seed_idx = pool_seeds.index(wl["validation"]["methods"]["best_single_10e"]["selected_seed"])

            # Evaluate frozen methods on official test cache
            test_ref_single = probabilistic_metrics(test_pool_probs_np[0], y_test_np)
            test_best_single = probabilistic_metrics(test_pool_probs_np[val_best_seed_idx], y_test_np)
            test_single_50e = probabilistic_metrics(test_50e_probs_np, y_test_np)

            test_uniform_probs = test_pool_probs_np.mean(axis=0)
            test_uniform_ensemble = probabilistic_metrics(test_uniform_probs, y_test_np)

            test_temp_probs = temp_scaled_probs(test_uniform_probs, frozen_temp)
            test_uniform_temp = probabilistic_metrics(test_temp_probs, y_test_np)

            test_slsqp_mix = mixture_probabilities(frozen_slsqp_w, test_pool_probs_np)
            test_slsqp = probabilistic_metrics(test_slsqp_mix, y_test_np)

            test_pso_mix = mixture_probabilities(frozen_pso_w, test_pool_probs_np)
            test_pso = probabilistic_metrics(test_pso_mix, y_test_np)

            # Confirmation Gates
            c_finite = all(
                math.isfinite(v)
                for m_dict in [test_ref_single, test_best_single, test_single_50e, test_uniform_ensemble, test_uniform_temp, test_slsqp, test_pso]
                for v in m_dict.values()
            )
            c_acc_reg = (test_uniform_ensemble["accuracy"] - test_pso["accuracy"]) <= 0.20
            c_nll_ref = test_pso["nll"] < test_ref_single["nll"]
            c_nll_50e = test_pso["nll"] <= test_single_50e["nll"] + 1e-7
            c_pass = c_finite and c_acc_reg and c_nll_ref and c_nll_50e

            confirmation_rec = {
                "official_test_data_loaded": True,
                "test_cache_counts": {
                    "dataset_loads": 1,
                    "pool_forward_passes": 5,
                    "long_single_forward_passes": 1,
                    "base_cnn_forward_passes_during_optimization": 0,
                    "memory_bytes": test_bytes,
                    "wall_time_seconds": float(test_cache_wall_t),
                },
                "frozen_methods": {
                    "selected_pso_seed": frozen_pso_seed,
                    "selected_pso_weights": frozen_pso_w,
                    "slsqp_weights": frozen_slsqp_w,
                    "fitted_temperature": frozen_temp,
                },
                "methods": {
                    "reference_single_10e": test_ref_single,
                    "best_single_10e": test_best_single,
                    "single_50e": test_single_50e,
                    "uniform_ensemble": test_uniform_ensemble,
                    "uniform_temperature": test_uniform_temp,
                    "slsqp_weights": test_slsqp,
                    "pso_weights": test_pso,
                },
                "confirmation_gates": {
                    "all_values_finite": c_finite,
                    "official_test_dataset_loads": 1,
                    "official_test_pool_forward_passes": 5,
                    "official_test_long_single_forward_passes": 1,
                    "maximum_pso_accuracy_regression_vs_uniform_pp": c_acc_reg,
                    "pso_nll_below_reference_single": c_nll_ref,
                    "maximum_pso_nll_regression_vs_equal_budget_single": c_nll_50e,
                    "pass": c_pass,
                }
            }
            wl["confirmation"] = confirmation_rec

    # -----------------------------------------------------------------
    # Step F: Assemble Global Artifact & Resource Totals
    # -----------------------------------------------------------------
    adam_pool_epochs = sum(wl["training"]["adam_pool_epochs"] for wl in workloads.values())
    adam_pool_wall_t = float(sum(wl["training"]["adam_pool_wall_time_seconds"] for wl in workloads.values()))
    single_50e_wall_t = float(sum(wl["training"]["single_50e_wall_time_seconds"] for wl in workloads.values()))

    val_fwd_passes = sum(
        wl["validation_cache"]["pool_forward_passes"] + wl["validation_cache"]["long_single_forward_passes"]
        for wl in workloads.values()
    )

    pso_tot_queries = sum(wl["validation"]["methods"]["pso_weights"]["total_queries"] for wl in workloads.values())
    pso_tot_samples = sum(wl["validation"]["methods"]["pso_weights"]["total_sample_evaluations"] for wl in workloads.values())
    pso_res_wall_t = float(sum(wl["validation"]["methods"]["pso_weights"]["total_wall_time_seconds"] for wl in workloads.values()))
    pso_prod_wall_t = float(sum(
        next(r["wall_time_seconds"] for r in wl["validation"]["methods"]["pso_weights"]["per_seed_runs"]
             if r["seed"] == wl["validation"]["methods"]["pso_weights"]["selected_seed"])
        for wl in workloads.values()
    ))

    slsqp_tot_evals = sum(wl["validation"]["methods"]["slsqp_weights"]["evaluations"] for wl in workloads.values())
    slsqp_tot_wall_t = float(sum(wl["validation"]["methods"]["slsqp_weights"]["wall_time_seconds"] for wl in workloads.values()))

    test_fwd_passes = sum(
        wl["confirmation"]["test_cache_counts"]["pool_forward_passes"] + wl["confirmation"]["test_cache_counts"]["long_single_forward_passes"]
        if wl.get("confirmation") is not None else 0
        for wl in workloads.values()
    )

    pso_to_pool_ratios = [
        wl["validation"]["methods"]["pso_weights"]["median_one_seed_wall_time_seconds"] / wl["training"]["adam_pool_wall_time_seconds"]
        for wl in workloads.values()
    ]
    pso_to_pool_wall_ratio = float(np.median(pso_to_pool_ratios)) if pso_to_pool_ratios else 0.0
    pso_max_workload_wall_ratio = float(max(pso_to_pool_ratios)) if pso_to_pool_ratios else 0.0
    pso_production_to_pool_wall_ratio = (
        pso_prod_wall_t / adam_pool_wall_t if adam_pool_wall_t > 0 else 0.0
    )

    resource_totals = {
        "adam_pool_epochs": adam_pool_epochs,
        "adam_pool_wall_time_seconds": adam_pool_wall_t,
        "single_50e_wall_time_seconds": single_50e_wall_t,
        "validation_cache_forward_passes": val_fwd_passes,
        "pso_total_queries": pso_tot_queries,
        "pso_total_sample_evaluations": pso_tot_samples,
        "pso_research_wall_time_seconds": pso_res_wall_t,
        "pso_production_wall_time_seconds": pso_prod_wall_t,
        "pso_to_pool_wall_ratio": pso_to_pool_wall_ratio,
        "pso_max_workload_wall_ratio": pso_max_workload_wall_ratio,
        "pso_production_to_pool_wall_ratio": pso_production_to_pool_wall_ratio,
        "slsqp_total_evaluations": slsqp_tot_evals,
        "slsqp_total_wall_time_seconds": slsqp_tot_wall_t,
        "official_test_cache_forward_passes": test_fwd_passes,
    }

    artifact = {
        "protocol_version": PROTOCOL_VERSION,
        "config": {
            "iteration": 1,
            "archived_iteration0_reference": {
                "epochs": 50,
                "queries_per_seed": 1500,
                "sample_evaluations_per_seed": 15000000,
                "reason": "wall_time_ratio_gate_exceeded",
            },
            "datasets": dataset_names,
            "split_seed": 20260904,
            "search_samples": 50000,
            "validation_samples": 10000,
            "pool_seeds": pool_seeds,
            "reference_single_seed": 201,
            "equal_budget_single_epochs": 50,
            "adam_lr": 0.001,
            "adam_batch_size": 256,
            "pso": {
                "method": "constriction",
                "evaluation": "full",
                "renewal": "loss",
                "particles": 30,
                "epochs": 30,
                "swarm_seeds": swarm_seeds,
                "queries_per_seed": 900,
                "sample_evaluations_per_seed": 9000000,
                "particle_bounds": [-4.0, 4.0],
                "boundary_strategy": "reflect",
                "velocity_limit_ratio": 0.1,
                "initial_position_noise": 0.0,
            },
            "device": str(dev),
        },
        "workloads": workloads,
        "development_pass": dev_pass,
        "development_gates": dev_gate_res,
        "policy_frozen": policy_frozen,
        "official_test_data_loaded": official_test_data_loaded,
        "official_test_evaluations_before_freeze": 0,
        "post_test_tuning_or_reruns": 0,
        "resource_totals": resource_totals,
    }

    # Write output artifacts atomically using temporary files + os.replace
    json_bytes = json.dumps(artifact, indent=2).encode("utf-8")
    atomic_write_file(output_json, json_bytes, is_binary=True)
    print(f"\nArtifact saved to: {output_json}")

    save_csv_report(artifact, output_csv)
    print(f"CSV report saved to: {output_csv}")

    save_publication_plot(artifact, output_png)
    print(f"PNG plot saved to: {output_png}")

    return artifact


def main():
    parser = argparse.ArgumentParser(description="Post-Training PSO Ensemble Study Runner")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, mps, cuda)")
    parser.add_argument("--cache-dir", type=str, default="result/cache", help="Dataset cache directory")
    parser.add_argument("--output-json", type=str, default="benchmark_results/pso_v8_post_training_ensemble.json", help="JSON output path")
    parser.add_argument("--output-csv", type=str, default="benchmark_results/pso_v8_post_training_ensemble.csv", help="CSV output path")
    parser.add_argument("--output-png", type=str, default="history_plt/pso_v8_post_training_ensemble.png", help="PNG output path")

    args = parser.parse_args()

    run_post_training_study(
        cache_dir=args.cache_dir,
        device=args.device,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_png=args.output_png,
    )


if __name__ == "__main__":
    main()
