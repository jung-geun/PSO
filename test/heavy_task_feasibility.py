"""
Heavy Task Feasibility Study: Parameter Dimension & Data Hardness Scaling.

Protocol Version: HEAVY-TASK-PSO-V6 1.0.0

Feasibility probe to test retained PSO methods (G0, G5, G6, G8) along two axes:
1. Larger Parameter Dimension (WideCNN ~55k params vs CompactCNN 9,098 params)
2. Harder Data (FashionMNIST vs MNIST)

Workload Matrix (4 workloads x 4 methods = 16 screen cells):
- mnist_compact (MNIST + CompactCNN 9,098 params, control)
- mnist_wide (MNIST + WideCNN ~55k params, parameter scaling axis)
- fashion_compact (FashionMNIST + CompactCNN 9,098 params, data hardness axis)
- fashion_wide (FashionMNIST + WideCNN ~55k params, combined axis)

Methods: G0, G5, G6, G8
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, asdict
import hashlib
import json
import csv
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
from matplotlib.lines import Line2D

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
    CompactCNN,
    make_compact_cnn,
    build_nested_stratified_subsets,
    evaluate_probabilistic_metrics,
    get_model_probabilities,
)
from deep_pso_v6 import (
    V6GeometryConfig,
    get_v6_geometry_table,
    V6LatentTransform,
    run_v6_pso,
    run_g8_optimizer,
)
from pso import __version__ as pso_version

PROTOCOL_VERSION = "HEAVY-TASK-PSO-V6 1.0.0"
FEASIBILITY_NLL_REDUCTION = 0.20
FEASIBILITY_ACCURACY_GAIN_PP = 20.0



# =====================================================================
# 1. Architecture Definitions & Deterministic Factories
# =====================================================================

class WideCNN(nn.Module):
    """
    WideCNN architecture (~55,338 parameters):
    Conv1 (1->16, 3x3, pad=1), ReLU, MaxPool2d(2,2)
    Conv2 (16->32, 3x3, pad=1), ReLU, MaxPool2d(2,2)
    Flatten -> 32x7x7 = 1568
    Linear (1568 -> 32), ReLU
    Linear (32 -> 10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1568, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2 and x.shape[1] == 784:
            x = x.view(-1, 1, 28, 28)
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = self.flatten(out)
        out = self.relu3(self.fc1(out))
        return self.fc2(out)


def make_wide_cnn(seed: int = 41) -> nn.Module:
    """Deterministic factory for WideCNN by seed."""
    torch.manual_seed(seed)
    return WideCNN()


def create_model(model_name: str, seed: int = 41) -> nn.Module:
    """Factory function creating a model instance by name and seed."""
    name = model_name.lower()
    if name in ("compact_cnn", "compactcnn"):
        return make_compact_cnn(seed)
    elif name in ("wide_cnn", "widecnn"):
        return make_wide_cnn(seed)
    else:
        raise ValueError(f"Unknown model architecture name: '{model_name}'. Expected 'compact_cnn' or 'wide_cnn'.")


# =====================================================================
# 2. Generic Train-Only Data Preparation (MNIST & FashionMNIST)
# =====================================================================

def prepare_heavy_task_data(
    dataset_name: str,
    split_seed: int = 20260902,
    cache_dir: Optional[Path] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    Dict[int, torch.Tensor],
    str, Dict[str, Any]
]:
    """
    Train-only data preparation for MNIST or FashionMNIST using exclusively train=True.
    Never constructs train=False.
    Preserves exact split seed (20260902), search-only normalization,
    and nested 2k/10k/50k index stratification.
    """
    if cache_dir is None:
        cache_dir = Path("result/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds_lower = dataset_name.lower()
    if ds_lower in ("mnist", "mnist_compact", "mnist_wide"):
        from torchvision.datasets import MNIST
        raw_train = MNIST(root=str(cache_dir), train=True, download=True)
        canonical_name = "MNIST"
    elif ds_lower in ("fashion_mnist", "fashion", "fashion_compact", "fashion_wide"):
        from torchvision.datasets import FashionMNIST
        raw_train = FashionMNIST(root=str(cache_dir), train=True, download=True)
        canonical_name = "FashionMNIST"
    else:
        raise ValueError(f"Unsupported dataset name: '{dataset_name}'. Must be 'mnist' or 'fashion_mnist'.")

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
        "dataset_name": canonical_name,
        "input_shape": [1, 28, 28],
        "normalization_scope": "search_train_50000_only",
        "train_mean": round(mean_val, 6),
        "train_std": round(std_val, 6),
        "search_samples": 50000,
        "val_samples": 10000,
        "test_samples": 0,
        "official_test_data_loaded": False,
        "official_test_evaluations": 0,
        "split_seed": split_seed,
        "split_fingerprint": split_fp,
        "data_fingerprint": data_fp,
    }

    return (
        x_search_norm, y_search,
        x_val_norm, y_val,
        nested_subsets,
        data_fp, provenance
    )


# =====================================================================
# 3. Workload Configurations & Method Specifications
# =====================================================================

@dataclass(frozen=True)
class WorkloadConfig:
    workload_id: str
    dataset_name: str
    model_name: str
    description: str


WORKLOADS: Dict[str, WorkloadConfig] = {
    "mnist_compact": WorkloadConfig(
        workload_id="mnist_compact",
        dataset_name="mnist",
        model_name="compact_cnn",
        description="MNIST dataset with CompactCNN (9,098 params control)",
    ),
    "mnist_wide": WorkloadConfig(
        workload_id="mnist_wide",
        dataset_name="mnist",
        model_name="wide_cnn",
        description="MNIST dataset with WideCNN (~55k params larger model axis)",
    ),
    "fashion_compact": WorkloadConfig(
        workload_id="fashion_compact",
        dataset_name="fashion_mnist",
        model_name="compact_cnn",
        description="FashionMNIST dataset with CompactCNN (harder data axis)",
    ),
    "fashion_wide": WorkloadConfig(
        workload_id="fashion_wide",
        dataset_name="fashion_mnist",
        model_name="wide_cnn",
        description="FashionMNIST dataset with WideCNN (harder data + larger model axis)",
    ),
}

HEAVY_METHODS = ["G0", "G5", "G6", "G8"]


def get_heavy_geometry_table() -> Dict[str, V6GeometryConfig]:
    """Returns the subset of V6 geometry configurations used for heavy task study (G0, G5, G6, G8)."""
    full_table = get_v6_geometry_table()
    return {m: full_table[m] for m in HEAVY_METHODS}


# =====================================================================
# 4. Untrained Baseline Evaluation
# =====================================================================

def evaluate_untrained_baseline(
    base_model: nn.Module,
    x_sub: torch.Tensor,
    y_sub: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates an untrained base model on validation set and objective subset
    to establish baseline performance for feasibility classification.
    """
    model = copy.deepcopy(base_model).to(device)
    val_probs = get_model_probabilities(model, x_val.to(device), device)
    val_metrics = evaluate_probabilistic_metrics(val_probs, y_val.to(device))

    # Evaluate objective subset (e.g. 2k or 10k)
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    x_sub_dev = x_sub.to(device)
    y_sub_dev = y_sub.to(device)
    num_sub = len(y_sub_dev)

    with torch.inference_mode():
        total_loss = 0.0
        correct = 0
        for b_start in range(0, num_sub, 1000):
            xb = x_sub_dev[b_start:b_start + 1000]
            yb = y_sub_dev[b_start:b_start + 1000]
            logits = model(xb)
            total_loss += float(loss_fn(logits, yb).item())
            correct += int((logits.argmax(dim=1) == yb).sum().item())

    obj_loss = total_loss / num_sub
    obj_acc = (correct / num_sub) * 100.0

    return {
        "val_nll": val_metrics["nll"],
        "val_accuracy": val_metrics["accuracy"],
        "val_brier": val_metrics["brier"],
        "val_ece": val_metrics["ece"],
        "val_margin": val_metrics["margin"],
        "objective_loss": round(obj_loss, 6),
        "objective_accuracy": round(obj_acc, 4),
    }


# =====================================================================
# 5. Method Selection & Feasibility Classification Logic
# =====================================================================

def select_best_normalized_method(
    screen_results_for_workload: List[Dict[str, Any]]
) -> str:
    """
    Selects the best normalized custom method among G0, G5, G6 for a workload
    based on screen validation NLL ascending, with validation accuracy descending tiebreak.
    """
    candidates = []
    for result in screen_results_for_workload:
        if result["method_id"] not in ("G0", "G5", "G6"):
            continue
        val_loss = result.get("val_selected_loss")
        val_acc = result.get("val_selected_acc")
        if (
            val_loss is not None
            and val_acc is not None
            and math.isfinite(val_loss)
            and math.isfinite(val_acc)
        ):
            candidates.append(result)
    if not candidates:
        raise ValueError("No finite normalized method (G0, G5, G6) result found for selection.")

    return min(
        candidates,
        key=lambda result: (result["val_selected_loss"], -result["val_selected_acc"]),
    )["method_id"]


def evaluate_feasibility(
    confirmed_runs: List[Dict[str, Any]],
    baseline_val_nll: float,
    baseline_val_acc: float,
) -> Dict[str, Any]:
    """
    Classifies feasibility per workload and selected method:
    - execution_feasible: True iff every run completed with finite metrics.
    - optimization_feasible: True iff mean validation NLL is at least 20% below baseline
      and mean validation accuracy is at least 20 percentage points above baseline.
    """
    is_execution_feasible = True
    nll_vals = []
    acc_vals = []

    for run in confirmed_runs:
        val_nll = run.get("val_selected_loss")
        val_acc = run.get("val_selected_acc")
        if val_nll is None or val_acc is None:
            is_execution_feasible = False
            break
        if not (math.isfinite(val_nll) and math.isfinite(val_acc)):
            is_execution_feasible = False
            break
        nll_vals.append(val_nll)
        acc_vals.append(val_acc)

    target_nll_thresh = baseline_val_nll * (1.0 - FEASIBILITY_NLL_REDUCTION)
    target_acc_thresh = baseline_val_acc + FEASIBILITY_ACCURACY_GAIN_PP

    if not is_execution_feasible or len(nll_vals) == 0:
        return {
            "execution_feasible": False,
            "optimization_feasible": False,
            "baseline_val_nll": baseline_val_nll,
            "target_val_nll_threshold": round(target_nll_thresh, 6),
            "baseline_val_acc": baseline_val_acc,
            "target_val_acc_threshold": round(target_acc_thresh, 4),
            "mean_val_nll": None,
            "mean_val_acc": None,
        }

    mean_nll = float(np.mean(nll_vals))
    mean_acc = float(np.mean(acc_vals))

    is_optimization_feasible = (mean_nll <= target_nll_thresh) and (mean_acc >= target_acc_thresh)

    return {
        "execution_feasible": True,
        "optimization_feasible": is_optimization_feasible,
        "baseline_val_nll": baseline_val_nll,
        "target_val_nll_threshold": round(target_nll_thresh, 6),
        "baseline_val_acc": baseline_val_acc,
        "target_val_acc_threshold": round(target_acc_thresh, 4),
        "mean_val_nll": round(mean_nll, 6),
        "mean_val_acc": round(mean_acc, 4),
    }


# =====================================================================
# 6. Screening Stage Runner (16 Workload-Method Cells)
# =====================================================================

def run_heavy_task_screen(
    workloads: Dict[str, WorkloadConfig],
    methods: List[str],
    particles: int = 12,
    epochs: int = 40,
    seed: int = 91,
    device: torch.device = torch.device("cpu"),
    cache_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Screens all 16 workload-method cells at fixed 2k subset, seed 91, 12p x 40e.
    """
    screen_results = []
    untrained_baselines = {}
    data_cache = {}

    geom_table = get_heavy_geometry_table()

    for wl_id, wl_cfg in workloads.items():
        if wl_cfg.dataset_name not in data_cache:
            x_search, y_search, x_val, y_val, nested_subsets, data_fp, provenance = prepare_heavy_task_data(
                dataset_name=wl_cfg.dataset_name,
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
        else:
            d = data_cache[wl_cfg.dataset_name]
            x_search, y_search = d["x_search"], d["y_search"]
            x_val, y_val = d["x_val"], d["y_val"]
            nested_subsets = d["nested_subsets"]

        base_model = create_model(wl_cfg.model_name, seed=41)
        param_count = sum(p.numel() for p in base_model.parameters())
        x_2k = x_search[nested_subsets[2000]]
        y_2k = y_search[nested_subsets[2000]]

        if wl_id not in untrained_baselines:
            untrained_baselines[wl_id] = evaluate_untrained_baseline(
                base_model, x_2k, y_2k, x_val, y_val, device
            )

        for m_id in methods:
            geom_config = geom_table[m_id]

            if m_id in ("G0", "G5", "G6"):
                transform = V6LatentTransform(base_model, geom_config, device)
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
                    swarm_size=particles,
                    seed=seed,
                    device=device,
                    geom_config=geom_config,
                    val_check_interval=10,
                )
                # Analytical core swarm-state bytes for custom methods (Z, V, P, M, V_sq)
                core_swarm_state_bytes = 5 * particles * param_count * 4
            elif m_id == "G8":
                res = run_g8_optimizer(
                    base_model=base_model,
                    x_2k=x_2k,
                    y_2k=y_2k,
                    x_val=x_val,
                    y_val=y_val,
                    epochs=epochs,
                    swarm_size=particles,
                    seed=seed,
                    device=device,
                )
                # Analytical core swarm-state bytes for public Optimizer (5*particles + 1 global best)
                core_swarm_state_bytes = (5 * particles + 1) * param_count * 4
            else:
                raise ValueError(f"Unknown method ID: {m_id}")

            opt_time = max(res["optimization_wall_time_sec"], 1e-6)
            total_samples = res["total_sample_evaluations"]
            throughput_sps = round(total_samples / opt_time, 2)

            val_nll = res["val_selected_loss"]
            val_acc = res["val_selected_acc"]
            g_loss = res["gbest_loss"]
            g_acc = res["gbest_acc"]

            finite_metrics = (val_nll, val_acc, g_loss, g_acc)
            is_finite = all(
                val is not None and math.isfinite(val)
                for val in finite_metrics
            )

            cell_record = {
                "workload_id": wl_id,
                "method_id": m_id,
                "dataset_name": wl_cfg.dataset_name,
                "model_name": wl_cfg.model_name,
                "parameter_count": param_count,
                "subset_size": 2000,
                "particles": particles,
                "epochs": epochs,
                "seed": seed,
                "gbest_loss": g_loss,
                "gbest_acc": g_acc,
                "gbest_val_loss": res.get("gbest_val_loss"),
                "gbest_val_acc": res.get("gbest_val_acc"),
                "val_selected_particle_idx": res.get("val_selected_particle_idx"),
                "val_selected_loss": val_nll,
                "val_selected_acc": val_acc,
                "val_metrics": res.get("val_metrics"),
                "wall_time_sec": res["wall_time_sec"],
                "optimization_wall_time_sec": res["optimization_wall_time_sec"],
                "validation_wall_time_sec": res["validation_wall_time_sec"],
                "total_queries": res["total_queries"],
                "total_sample_evaluations": res["total_sample_evaluations"],
                "validation_evaluations": res["validation_evaluations"],
                "official_test_evaluations": 0,
                "core_swarm_state_bytes": core_swarm_state_bytes,
                "throughput_samples_per_sec": throughput_sps,
                "is_finite": is_finite,
            }
            screen_results.append(cell_record)

    # Workload summary metadata
    workload_metadata = {}
    for wl_id, wl_cfg in workloads.items():
        bm = create_model(wl_cfg.model_name, seed=41)
        param_count = sum(p.numel() for p in bm.parameters())
        model_fp = compute_model_fingerprint(bm)
        d = data_cache[wl_cfg.dataset_name]
        workload_metadata[wl_id] = {
            "workload_id": wl_id,
            "dataset_name": wl_cfg.dataset_name,
            "model_name": wl_cfg.model_name,
            "parameter_count": param_count,
            "model_fingerprint": model_fp,
            "data_fingerprint": d["data_fp"],
            "split_fingerprint": d["provenance"]["split_fingerprint"],
            "description": wl_cfg.description,
        }

    return screen_results, untrained_baselines, workload_metadata


# =====================================================================
# 7. Confirmation Stage Runner (Seeds 101-103, Fixed 10k, 12p x 80e)
# =====================================================================

def run_heavy_task_confirm(
    workloads: Dict[str, WorkloadConfig],
    selected_methods: Dict[str, List[str]],  # workload_id -> [G8, best_normalized]
    particles: int = 12,
    epochs: int = 80,
    seeds: List[int] = (101, 102, 103),
    split_seed: int = 20260902,
    device: torch.device = torch.device("cpu"),
    cache_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Confirms selected methods (G8 + best normalized) for each workload at fixed 10k,
    12p x 80e, seeds 101-103. Aggregates mean and sample SD across seeds.
    """
    confirmation_results = {}
    geom_table = get_heavy_geometry_table()
    data_cache = {}

    for wl_id, wl_cfg in workloads.items():
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
        else:
            d = data_cache[wl_cfg.dataset_name]
            x_search, y_search = d["x_search"], d["y_search"]
            x_val, y_val = d["x_val"], d["y_val"]
            nested_subsets = d["nested_subsets"]
            data_fp = d["data_fp"]
            provenance = d["provenance"]

        base_model = create_model(wl_cfg.model_name, seed=41)
        param_count = sum(p.numel() for p in base_model.parameters())
        x_10k = x_search[nested_subsets[10000]]
        y_10k = y_search[nested_subsets[10000]]

        wl_confirmations = {}
        methods_to_confirm = selected_methods[wl_id]

        for m_id in methods_to_confirm:
            geom_config = geom_table[m_id]
            per_seed_runs = []

            for s in seeds:
                if m_id in ("G0", "G5", "G6"):
                    transform = V6LatentTransform(base_model, geom_config, device)
                    res = run_v6_pso(
                        transform=transform,
                        base_model=base_model,
                        x_search=x_search,
                        y_search=y_search,
                        x_val=x_val,
                        y_val=y_val,
                        nested_subsets=nested_subsets,
                        schedule_str=f"10000:{epochs}",
                        epochs=epochs,
                        swarm_size=particles,
                        seed=s,
                        device=device,
                        geom_config=geom_config,
                        val_check_interval=10,
                    )
                    core_bytes = 5 * particles * param_count * 4
                elif m_id == "G8":
                    res = run_g8_optimizer(
                        base_model=base_model,
                        x_2k=x_10k,  # Passes 10k search tensor for 10k fit
                        y_2k=y_10k,
                        x_val=x_val,
                        y_val=y_val,
                        epochs=epochs,
                        swarm_size=particles,
                        seed=s,
                        device=device,
                    )
                    core_bytes = (5 * particles + 1) * param_count * 4
                else:
                    raise ValueError(f"Unknown method ID: {m_id}")

                opt_time = max(res["optimization_wall_time_sec"], 1e-6)
                sps = round(res["total_sample_evaluations"] / opt_time, 2)

                seed_record = {
                    "seed": s,
                    "val_selected_loss": res["val_selected_loss"],
                    "val_selected_acc": res["val_selected_acc"],
                    "val_metrics": res.get("val_metrics"),
                    "gbest_loss": res["gbest_loss"],
                    "gbest_acc": res["gbest_acc"],
                    "wall_time_sec": res["wall_time_sec"],
                    "optimization_wall_time_sec": res["optimization_wall_time_sec"],
                    "validation_wall_time_sec": res["validation_wall_time_sec"],
                    "total_queries": res["total_queries"],
                    "total_sample_evaluations": res["total_sample_evaluations"],
                    "validation_evaluations": res["validation_evaluations"],
                    "official_test_evaluations": 0,
                    "core_swarm_state_bytes": core_bytes,
                    "throughput_samples_per_sec": sps,
                }
                per_seed_runs.append(seed_record)

            # Compute aggregated mean & sample SD stats
            acc_list = [r["val_selected_acc"] for r in per_seed_runs]
            nll_list = [r["val_selected_loss"] for r in per_seed_runs]
            brier_list = [r["val_metrics"]["brier"] for r in per_seed_runs if r.get("val_metrics")]
            ece_list = [r["val_metrics"]["ece"] for r in per_seed_runs if r.get("val_metrics")]
            g_loss_list = [r["gbest_loss"] for r in per_seed_runs]
            g_acc_list = [r["gbest_acc"] for r in per_seed_runs]
            wall_list = [r["wall_time_sec"] for r in per_seed_runs]
            opt_wall_list = [r["optimization_wall_time_sec"] for r in per_seed_runs]
            sps_list = [r["throughput_samples_per_sec"] for r in per_seed_runs]

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

            wl_confirmations[m_id] = {
                "workload_id": wl_id,
                "method_id": m_id,
                "subset_size": 10000,
                "particles": particles,
                "epochs": epochs,
                "seeds": list(seeds),
                "split_seed": split_seed,
                "data_fingerprint": data_fp,
                "split_fingerprint": provenance["split_fingerprint"],
                "provenance": provenance,
                "stats": stats,
                "per_seed_runs": per_seed_runs,
            }

        confirmation_results[wl_id] = wl_confirmations

    return confirmation_results


# =====================================================================
# 8. CSV & Plot Artifact Generators
# =====================================================================

def save_csv_summary(payload: Dict[str, Any], csv_path: Path):
    """Saves a clean, concise CSV summary of screen, confirmation, and feasibility results."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "workload", "method", "metric", "value"])
        writer.writerow(["protocol", "global", "all", "version", payload["protocol_version"]])
        writer.writerow(["protocol", "global", "all", "official_test_data_loaded", payload["official_test_data_loaded"]])
        writer.writerow(["protocol", "global", "all", "official_test_evaluations", payload["official_test_evaluations"]])

        # Untrained Baselines
        baselines = payload.get("untrained_baselines", {})
        for wl_id, base in baselines.items():
            writer.writerow(["baseline", wl_id, "untrained", "val_nll", base["val_nll"]])
            writer.writerow(["baseline", wl_id, "untrained", "val_accuracy", base["val_accuracy"]])

        # Screen Results
        screen_res = payload.get("screen_results", [])
        for r in screen_res:
            wl = r["workload_id"]
            m = r["method_id"]
            writer.writerow(["screen", wl, m, "val_nll", r["val_selected_loss"]])
            writer.writerow(["screen", wl, m, "val_acc", r["val_selected_acc"]])
            writer.writerow(["screen", wl, m, "gbest_loss", r["gbest_loss"]])
            writer.writerow(["screen", wl, m, "gbest_acc", r["gbest_acc"]])
            writer.writerow(["screen", wl, m, "throughput_sps", r["throughput_samples_per_sec"]])

        # Confirmation Results
        confirm_res = payload.get("confirmation_results", {})
        for wl_id, m_dict in confirm_res.items():
            for m_id, conf in m_dict.items():
                st = conf["stats"]
                writer.writerow(["confirm", wl_id, m_id, "val_acc_mean", st["val_acc"]["mean"]])
                writer.writerow(["confirm", wl_id, m_id, "val_acc_std", st["val_acc"]["std"]])
                writer.writerow(["confirm", wl_id, m_id, "val_nll_mean", st["val_nll"]["mean"]])
                writer.writerow(["confirm", wl_id, m_id, "val_nll_std", st["val_nll"]["std"]])
                writer.writerow(["confirm", wl_id, m_id, "wall_time_sec_mean", st["wall_time_sec"]["mean"]])

        # Feasibility Evaluations
        feas_evals = payload.get("feasibility_evaluations", {})
        for wl_id, m_feas in feas_evals.items():
            for m_id, fe in m_feas.items():
                writer.writerow(["feasibility", wl_id, m_id, "execution_feasible", fe["execution_feasible"]])
                writer.writerow(["feasibility", wl_id, m_id, "optimization_feasible", fe["optimization_feasible"]])
                writer.writerow(["feasibility", wl_id, m_id, "mean_val_nll", fe["mean_val_nll"]])
                writer.writerow(["feasibility", wl_id, m_id, "mean_val_acc", fe["mean_val_acc"]])


def generate_feasibility_plot(payload: Dict[str, Any], plot_path: Path):
    """
    Generates a clear two-panel visualization:
    Panel 1: Validation NLL screen comparison across all 16 workload-method cells.
    Panel 2: Confirmation Accuracy vs Parameter Count for each workload,
             distinguishing dataset (MNIST vs FashionMNIST), model size, and method.
    """
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6))

    # Panel 1: Screen Validation NLL
    screen_results = payload.get("screen_results", [])
    workload_order = ["mnist_compact", "mnist_wide", "fashion_compact", "fashion_wide"]
    method_order = ["G0", "G5", "G6", "G8"]

    cell_dict = {(r["workload_id"], r["method_id"]): r["val_selected_loss"] for r in screen_results}

    x_indices = np.arange(len(workload_order))
    width = 0.18
    colors = {"G0": "#1f77b4", "G5": "#ff7f0e", "G6": "#2ca02c", "G8": "#d62728"}

    for idx, m_id in enumerate(method_order):
        vals = [cell_dict.get((wl_id, m_id), 0.0) for wl_id in workload_order]
        ax1.bar(x_indices + (idx - 1.5) * width, vals, width, label=m_id, color=colors[m_id])

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(["MNIST\nCompact", "MNIST\nWide", "Fashion\nCompact", "Fashion\nWide"], fontsize=9)
    ax1.set_ylabel("Validation NLL (Screen fixed2k, lower is better)")
    ax1.set_title("Screening Stage: Validation NLL (16 Cells)")
    ax1.legend(title="Method")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Confirmation Accuracy vs Parameter Count
    confirm_results = payload.get("confirmation_results", {})
    workload_info = payload.get("workloads", {})

    markers = {"mnist": "o", "fashion_mnist": "s"}
    dataset_x_factors = {"mnist": 0.94, "fashion_mnist": 1.06}
    method_x_factors = {"G8": 0.985, "G0": 1.015, "G5": 1.015, "G6": 1.015}

    parameter_counts = sorted({
        metadata["parameter_count"]
        for metadata in workload_info.values()
    })
    for parameter_count in parameter_counts:
        ax2.axvline(parameter_count, color="#bbbbbb", linewidth=0.8, linestyle=":")

    for wl_id, m_dict in confirm_results.items():
        wl_meta = workload_info.get(wl_id, {})
        ds_name = wl_meta.get("dataset_name", "mnist").lower()
        param_count = wl_meta.get("parameter_count", 9098)
        marker = markers.get(ds_name, "o")

        for m_id, conf in m_dict.items():
            acc_mean = conf["stats"]["val_acc"]["mean"]
            acc_std = conf["stats"]["val_acc"]["std"]
            color = colors.get(m_id, "#333333")
            display_x = (
                param_count
                * dataset_x_factors.get(ds_name, 1.0)
                * method_x_factors.get(m_id, 1.0)
            )

            ax2.errorbar(
                [display_x],
                [acc_mean],
                yerr=[acc_std],
                fmt=marker,
                color=color,
                linestyle="none",
                capsize=5,
                markersize=8,
                label="_nolegend_",
            )

    ax2.set_xlabel("Model Parameter Count D (log scale; points horizontally offset)")
    ax2.set_ylabel("Validation Accuracy % (Confirmation fixed10k, mean ± SD)")
    ax2.set_title("Confirmation: G8 + Screen-Selected Normalized Method")
    ax2.set_xticks(parameter_counts)
    ax2.set_xticklabels([f"{count:,}" for count in parameter_counts])
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.set_xscale("log")
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="#333333",
            linestyle="none",
            markersize=8,
            label=label,
        )
        for label, marker in (("MNIST", "o"), ("FashionMNIST", "s"))
    ]
    confirmed_method_ids = sorted({
        method_id
        for methods in confirm_results.values()
        for method_id in methods
    })
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors[method_id],
            linestyle="none",
            markersize=8,
            label=method_id,
        )
        for method_id in confirmed_method_ids
    ]
    dataset_legend = ax2.legend(
        handles=dataset_handles,
        title="Dataset marker",
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax2.add_artist(dataset_legend)
    ax2.legend(
        handles=method_handles,
        title="Method color",
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.68),
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)


# =====================================================================
# 9. Main Orchestration Runner & CLI Entrypoint
# =====================================================================

def run_heavy_task_study(args: argparse.Namespace) -> Dict[str, Any]:
    """Orchestrates screen, confirmation, feasibility evaluation, and artifact generation."""
    start_time_all = time.time()
    device = resolve_execution_device(args.device)

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    plot_dir = Path(args.plot_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    screen_particles = args.screen_particles
    screen_epochs = args.screen_epochs
    screen_seed = args.screen_seeds[0] if args.screen_seeds else 91

    confirm_particles = args.confirm_particles
    confirm_epochs = args.confirm_epochs
    confirm_seeds = args.confirm_seeds
    screen_design = {
        "methods": list(HEAVY_METHODS),
        "subset_size": 2000,
        "particles": screen_particles,
        "epochs": screen_epochs,
        "seed": screen_seed,
        "cells": len(WORKLOADS) * len(HEAVY_METHODS),
    }
    method_configs = {
        method_id: asdict(config)
        for method_id, config in get_heavy_geometry_table().items()
    }


    screen_results = []
    untrained_baselines = {}
    workload_metadata = {}
    normalized_selections = {}

    if args.stage in ("screen", "all"):
        print(f"[{PROTOCOL_VERSION}] Starting Screening Stage (16 Cells at fixed2k, {screen_particles}p x {screen_epochs}e, seed {screen_seed})...")
        screen_results, untrained_baselines, workload_metadata = run_heavy_task_screen(
            workloads=WORKLOADS,
            methods=HEAVY_METHODS,
            particles=screen_particles,
            epochs=screen_epochs,
            seed=screen_seed,
            device=device,
            cache_dir=cache_dir,
        )

        # Select best normalized method for each workload
        for wl_id in WORKLOADS:
            wl_screen = [r for r in screen_results if r["workload_id"] == wl_id]
            best_norm = select_best_normalized_method(wl_screen)
            normalized_selections[wl_id] = best_norm
            print(f"  Workload '{wl_id}': G8 + selected best normalized method '{best_norm}'")

        if args.stage == "screen":
            # Save screen-only artifact
            screen_payload = {
                "protocol_version": PROTOCOL_VERSION,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "hardware": get_hardware_provenance(device),
                "pso_version": pso_version,
                "official_test_data_loaded": False,
                "official_test_evaluations": 0,
                "study_design": {
                    "screen": screen_design,
                    "selection": {
                        "candidates": ["G0", "G5", "G6"],
                        "ranking": ["validation_nll_ascending", "validation_accuracy_descending"],
                        "confirmation_reference": "G8",
                    },
                    "validation": {
                        "source": "official_training_split_only",
                        "split": "50000_search_10000_validation_stratified",
                        "split_seed": 20260902,
                    },
                },
                "method_configs": method_configs,
                "workloads": workload_metadata,
                "untrained_baselines": untrained_baselines,
                "screen_results": screen_results,
                "normalized_method_selections": normalized_selections,
            }
            save_json_atomic(screen_payload, out_dir / "pso_v6_heavy_tasks_screen.json")
            print(f"Screening complete. Saved to {out_dir / 'pso_v6_heavy_tasks_screen.json'}")
            return screen_payload

    elif args.stage == "confirm":
        screen_artifact_path = args.screen_artifact or (out_dir / "pso_v6_heavy_tasks_screen.json")
        if not screen_artifact_path.exists():
            screen_artifact_path = out_dir / "pso_v6_heavy_tasks.json"
        if not screen_artifact_path.exists():
            raise FileNotFoundError(f"Screen artifact not found at {screen_artifact_path}. Run --stage screen or --stage all first.")

        with open(screen_artifact_path, "r", encoding="utf-8") as f:
            screen_payload = json.load(f)

        screen_results = screen_payload["screen_results"]
        untrained_baselines = screen_payload["untrained_baselines"]
        workload_metadata = screen_payload["workloads"]
        normalized_selections = screen_payload["normalized_method_selections"]
        screen_design = screen_payload["study_design"]["screen"]


    # Confirmation Stage
    selected_methods_to_confirm = {}
    for wl_id in WORKLOADS:
        best_norm = normalized_selections[wl_id]
        selected_methods_to_confirm[wl_id] = ["G8", best_norm]

    print(f"[{PROTOCOL_VERSION}] Starting Confirmation Stage (Seeds {confirm_seeds}, fixed10k, {confirm_particles}p x {confirm_epochs}e)...")
    confirm_results = run_heavy_task_confirm(
        workloads=WORKLOADS,
        selected_methods=selected_methods_to_confirm,
        particles=confirm_particles,
        epochs=confirm_epochs,
        seeds=confirm_seeds,
        device=device,
        cache_dir=cache_dir,
    )

    # Feasibility Evaluations
    feasibility_evals = {}
    for wl_id, m_dict in confirm_results.items():
        base = untrained_baselines[wl_id]
        wl_feas = {}
        for m_id, conf in m_dict.items():
            feas = evaluate_feasibility(
                confirmed_runs=conf["per_seed_runs"],
                baseline_val_nll=base["val_nll"],
                baseline_val_acc=base["val_accuracy"],
            )
            wl_feas[m_id] = feas
        feasibility_evals[wl_id] = wl_feas

    # Aggregate Resource Totals
    tot_queries = 0
    tot_samples = 0
    tot_opt_wall = 0.0
    tot_val_wall = 0.0
    tot_run_wall = 0.0

    for r in screen_results:
        tot_queries += r["total_queries"]
        tot_samples += r["total_sample_evaluations"]
        tot_opt_wall += r["optimization_wall_time_sec"]
        tot_val_wall += r["validation_wall_time_sec"]
        tot_run_wall += r["wall_time_sec"]

    for wl_id, m_dict in confirm_results.items():
        for m_id, conf in m_dict.items():
            for run in conf["per_seed_runs"]:
                tot_queries += run["total_queries"]
                tot_samples += run["total_sample_evaluations"]
                tot_opt_wall += run["optimization_wall_time_sec"]
                tot_val_wall += run["validation_wall_time_sec"]
                tot_run_wall += run["wall_time_sec"]

    resource_totals = {
        "total_queries": tot_queries,
        "total_sample_evaluations": tot_samples,
        "summed_optimization_wall_time_sec": round(tot_opt_wall, 4),
        "summed_validation_wall_time_sec": round(tot_val_wall, 4),
        "summed_recorded_run_wall_time_sec": round(tot_run_wall, 4),
        "elapsed_current_process_wall_time_sec": round(time.time() - start_time_all, 4),
    }

    final_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": get_hardware_provenance(device),
        "pso_version": pso_version,
        "official_test_data_loaded": False,
        "official_test_evaluations": 0,
        "study_design": {
            "screen": screen_design,
            "confirmation": {
                "methods_per_workload": selected_methods_to_confirm,
                "subset_size": 10000,
                "particles": confirm_particles,
                "epochs": confirm_epochs,
                "seeds": list(confirm_seeds),
            },
            "selection": {
                "candidates": ["G0", "G5", "G6"],
                "ranking": ["validation_nll_ascending", "validation_accuracy_descending"],
                "confirmation_reference": "G8",
            },
            "feasibility_contract": {
                "execution": "all_confirmed_runs_complete_with_finite_metrics",
                "minimum_validation_nll_reduction_fraction": FEASIBILITY_NLL_REDUCTION,
                "minimum_validation_accuracy_gain_percentage_points": FEASIBILITY_ACCURACY_GAIN_PP,
                "requires_both_optimization_thresholds": True,
            },
            "validation": {
                "source": "official_training_split_only",
                "split": "50000_search_10000_validation_stratified",
                "split_seed": 20260902,
                "official_test_split_used": False,
            },
        },
        "method_configs": method_configs,
        "workloads": workload_metadata,
        "untrained_baselines": untrained_baselines,
        "screen_results": screen_results,
        "normalized_method_selections": normalized_selections,
        "confirmation_results": confirm_results,
        "feasibility_evaluations": feasibility_evals,
        "resource_totals": resource_totals,
    }

    # Persist JSON, CSV, and plot
    json_path = out_dir / "pso_v6_heavy_tasks.json"
    csv_path = out_dir / "pso_v6_heavy_tasks.csv"
    plot_path = plot_dir / "pso_v6_heavy_tasks.png"

    save_json_atomic(final_payload, json_path)
    save_csv_summary(final_payload, csv_path)
    generate_feasibility_plot(final_payload, plot_path)

    print(f"[{PROTOCOL_VERSION}] Study complete!")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"  Plot: {plot_path}")

    return final_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNIST / FashionMNIST PSO Heavy Task Feasibility Study")
    parser.add_argument("--stage", choices=["screen", "confirm", "all"], default="all", help="Study stage to run")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--screen-particles", type=int, default=12, help="Screening particle count")
    parser.add_argument("--screen-epochs", type=int, default=40, help="Screening epoch count")
    parser.add_argument("--confirm-particles", type=int, default=12, help="Confirmation particle count")
    parser.add_argument("--confirm-epochs", type=int, default=80, help="Confirmation epoch count")
    parser.add_argument("--screen-seeds", type=int, nargs="+", default=[91], help="Screening seed")
    parser.add_argument("--confirm-seeds", type=int, nargs="+", default=[101, 102, 103], help="Confirmation seeds")
    parser.add_argument("--cache-dir", type=str, default="result/cache", help="Data cache directory")
    parser.add_argument("--out-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--plot-dir", type=str, default="history_plt", help="Plot directory")
    parser.add_argument("--screen-artifact", type=Path, default=None, help="Screen JSON artifact for confirm-only stage")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_heavy_task_study(args)


if __name__ == "__main__":
    main()
