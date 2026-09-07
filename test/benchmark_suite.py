import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pso import Optimizer, __version__ as pso_version

BENCHMARK_PROTOCOL_VERSION = "2.0.0"

METHOD_STYLE: Dict[str, Dict[str, str]] = {
    "original": {"color": "#E69F00", "hatch": ""},
    "inertia": {"color": "#56B4E9", "hatch": "//"},
    "constriction": {"color": "#009E73", "hatch": "\\\\"},
    "fips": {"color": "#F0E442", "hatch": "xx"},
    "clpso": {"color": "#0072B2", "hatch": ".."},
    "bare_bones": {"color": "#D55E00", "hatch": "++"},
    "adaptive_moment": {"color": "#CC79A7", "hatch": "||"},
}

FALLBACK_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
FALLBACK_HATCHES = ["", "//", "\\\\", "xx", "..", "++", "||"]


def get_method_style(method_name: str, idx: int = 0) -> Tuple[str, str]:
    if method_name in METHOD_STYLE:
        return METHOD_STYLE[method_name]["color"], METHOD_STYLE[method_name]["hatch"]
    c = FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]
    h = FALLBACK_HATCHES[idx % len(FALLBACK_HATCHES)]
    return c, h
# Student's t critical values for 95% 2-tailed confidence intervals (df -> t_crit)
T_TABLE = {
    1: 12.7062,
    2: 4.3027,
    3: 3.1824,
    4: 2.7764,
    5: 2.5706,
    6: 2.4469,
    7: 2.3646,
    8: 2.3060,
    9: 2.2622,
    10: 2.2281,
    15: 2.1314,
    20: 2.0860,
    30: 2.0423,
    60: 2.0003,
    120: 1.9799,
}


def get_t_crit(df: int) -> float:
    if df <= 0:
        return 0.0
    if df in T_TABLE:
        return T_TABLE[df]
    keys = sorted(T_TABLE.keys())
    if df < keys[0]:
        return T_TABLE[keys[0]]
    if df > keys[-1]:
        return 1.96
    for i in range(len(keys) - 1):
        if keys[i] <= df <= keys[i + 1]:
            k0, k1 = keys[i], keys[i + 1]
            v0, v1 = T_TABLE[k0], T_TABLE[k1]
            return v0 + (v1 - v0) * (df - k0) / (k1 - k0)
    return 1.96


def calc_stats(vals: List[float]) -> Dict[str, float]:
    arr = np.array(vals, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "iqr": 0.0, "ci95_t": 0.0}
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    med_val = float(np.median(arr))
    if n > 1:
        q75, q25 = np.percentile(arr, [75, 25])
        iqr_val = float(q75 - q25)
    else:
        iqr_val = 0.0
    t_crit = get_t_crit(n - 1)
    ci95 = float(t_crit * std_val / math.sqrt(n)) if n > 0 else 0.0
    return {
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "median": round(med_val, 6),
        "iqr": round(iqr_val, 6),
        "ci95_t": round(ci95, 6),
    }


# ==========================================
# Data Loaders & Model Factories
# ==========================================

def get_xor_data(seed: int = 41) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    return x, x, y, y


def make_xor_model(seed: int = 41) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
    )


def get_iris_data(seed: int = 41) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    iris = load_iris()
    x = iris.data.astype("float32")
    y = iris.target.astype("int64")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.int64),
        torch.tensor(y_test, dtype=torch.int64),
    )


def make_iris_model(seed: int = 41) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 3),
    )


def get_seeds_data(seed: int = 41) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seeds_path = Path("data/seeds/seeds_dataset.txt")
    if not seeds_path.exists():
        raise FileNotFoundError(f"Seeds dataset not found at {seeds_path}")

    with open(seeds_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            rows.append([float(p) for p in parts])

    data = np.array(rows, dtype=np.float32)
    x = data[:, :-1]
    y = (data[:, -1] - 1).astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=seed
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.int64),
        torch.tensor(y_test, dtype=torch.int64),
    )


def make_seeds_model(seed: int = 41) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(7, 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
    )


def get_digits_data(seed: int = 41) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    digits = load_digits()
    x = digits.data.astype("float32")
    y = digits.target.astype("int64")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=seed
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.int64),
        torch.tensor(y_test, dtype=torch.int64),
    )


def make_digits_model(seed: int = 41) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(64, 12),
        nn.ReLU(),
        nn.Linear(12, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
    )


def get_mnist_data(seed: int = 41) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from torchvision.datasets import MNIST

    cache_dir = Path("result/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = MNIST(root=str(cache_dir), train=True, download=True)
    test_dataset = MNIST(root=str(cache_dir), train=False, download=True)

    x_train_raw = (train_dataset.data[:3000].float() / 255.0).reshape(3000, -1).numpy()
    y_train = train_dataset.targets[:3000].long()

    x_test_raw = (test_dataset.data[:1000].float() / 255.0).reshape(1000, -1).numpy()
    y_test = test_dataset.targets[:1000].long()

    pca = PCA(n_components=32, whiten=True, random_state=seed)
    x_train_pca = pca.fit_transform(x_train_raw)
    x_test_pca = pca.transform(x_test_raw)

    return (
        torch.tensor(x_train_pca, dtype=torch.float32),
        torch.tensor(x_test_pca, dtype=torch.float32),
        y_train,
        y_test,
    )


def make_mnist_model(seed: int = 41) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Linear(32, 10)


DATASET_CACHE: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def get_cached_dataset(
    ds_name: str, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (ds_name, seed)
    if key not in DATASET_CACHE:
        loader = WORKLOADS[ds_name]["data_loader"]
        DATASET_CACHE[key] = loader(seed=seed)
    return DATASET_CACHE[key]


def compute_data_fingerprint(
    x_train: torch.Tensor, x_test: torch.Tensor, y_train: torch.Tensor, y_test: torch.Tensor
) -> str:
    h = hashlib.sha256()
    for t in (x_train, x_test, y_train, y_test):
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def compute_model_fingerprint(model: nn.Module) -> str:
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def get_hardware_provenance(device: torch.device) -> Dict[str, Any]:
    prov: Dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "pso_version": pso_version,
        "device_type": device.type,
        "device_str": str(device),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        prov["cuda_device_name"] = torch.cuda.get_device_name(device)
    elif device.type == "mps" and hasattr(torch.backends, "mps"):
        prov["mps_available"] = torch.backends.mps.is_available()
    return prov


def extract_plugin_metadata(opt: Optimizer) -> Dict[str, Any]:
    return {
        "movement": {
            "title": opt.movement_plugin.metadata.title,
            "source": opt.movement_plugin.metadata.source,
            "fidelity": opt.movement_plugin.metadata.fidelity,
            "gradient_required": opt.movement_plugin.metadata.gradient_required,
            "options": opt.movement_plugin.get_options(),
        },
        "initialization": {
            "title": opt.initialization_plugin.metadata.title,
            "source": opt.initialization_plugin.metadata.source,
            "fidelity": opt.initialization_plugin.metadata.fidelity,
            "gradient_required": opt.initialization_plugin.metadata.gradient_required,
            "options": opt.initialization_plugin.get_options(),
        },
        "evaluation": {
            "title": opt.evaluation_plugin.metadata.title,
            "source": opt.evaluation_plugin.metadata.source,
            "fidelity": opt.evaluation_plugin.metadata.fidelity,
            "gradient_required": opt.evaluation_plugin.metadata.gradient_required,
            "options": opt.evaluation_plugin.get_options(),
        },
        "convergence": {
            "title": opt.convergence_plugin.metadata.title,
            "source": opt.convergence_plugin.metadata.source,
            "fidelity": opt.convergence_plugin.metadata.fidelity,
            "gradient_required": opt.convergence_plugin.metadata.gradient_required,
            "options": opt.convergence_plugin.get_options(),
        },
        "refinement": {
            "title": opt.refinement_plugin.metadata.title,
            "source": opt.refinement_plugin.metadata.source,
            "fidelity": opt.refinement_plugin.metadata.fidelity,
            "gradient_required": opt.refinement_plugin.metadata.gradient_required,
            "options": opt.refinement_plugin.get_options(),
        },
    }


WORKLOADS = {
    "XOR": {
        "task": "binary",
        "loss_fn": lambda: nn.BCEWithLogitsLoss(),
        "model_factory": make_xor_model,
        "data_loader": get_xor_data,
        "n_particles": 24,
        "epochs": 80,
        "evaluation": "full",
        "fitness_size": None,
        "batch_size": None,
        "particle_min": -5.0,
        "particle_max": 5.0,
        "initial_position_noise": 1.0,
        "renewal": "loss",
        "held_out": False,
        "pca_config": None,
    },
    "Iris": {
        "task": "multiclass",
        "loss_fn": lambda: nn.CrossEntropyLoss(),
        "model_factory": make_iris_model,
        "data_loader": get_iris_data,
        "n_particles": 24,
        "epochs": 60,
        "evaluation": "full",
        "fitness_size": None,
        "batch_size": None,
        "particle_min": -3.0,
        "particle_max": 3.0,
        "initial_position_noise": 0.5,
        "renewal": "loss",
        "held_out": True,
        "pca_config": None,
    },
    "Seeds": {
        "task": "multiclass",
        "loss_fn": lambda: nn.CrossEntropyLoss(),
        "model_factory": make_seeds_model,
        "data_loader": get_seeds_data,
        "n_particles": 24,
        "epochs": 60,
        "evaluation": "full",
        "fitness_size": None,
        "batch_size": None,
        "particle_min": -3.0,
        "particle_max": 3.0,
        "initial_position_noise": 0.5,
        "renewal": "loss",
        "held_out": True,
        "pca_config": None,
    },
    "Digits": {
        "task": "multiclass",
        "loss_fn": lambda: nn.CrossEntropyLoss(),
        "model_factory": make_digits_model,
        "data_loader": get_digits_data,
        "n_particles": 24,
        "epochs": 50,
        "evaluation": "fixed_subset",
        "fitness_size": 1000,
        "batch_size": 250,
        "particle_min": -3.0,
        "particle_max": 3.0,
        "initial_position_noise": 0.25,
        "renewal": "loss",
        "held_out": True,
        "pca_config": None,
    },
    "MNIST": {
        "task": "multiclass",
        "loss_fn": lambda: nn.CrossEntropyLoss(),
        "model_factory": make_mnist_model,
        "data_loader": get_mnist_data,
        "n_particles": 30,
        "epochs": 80,
        "evaluation": "fixed_subset",
        "fitness_size": 2000,
        "batch_size": 1000,
        "particle_min": -3.0,
        "particle_max": 3.0,
        "initial_position_noise": 0.05,
        "renewal": "loss",
        "held_out": True,
        "pca_config": {"n_components": 32, "whiten": True},
    },
}

MAIN_METHODS = ["original", "inertia", "constriction", "fips", "clpso", "bare_bones", "adaptive_moment"]

ABLATION_PROFILES = {
    "inertia_canonical": {
        "method": "inertia",
        "c0": 2.0,
        "c1": 2.0,
        "w_min": 0.4,
        "w_max": 0.9,
        "velocity_limit_ratio": 0.1,
        "mutation_swarm": 0.0,
        "evaluation": "fixed_subset",
        "fitness_size": 2000,
        "batch_size": 1000,
        "initialization": "model_noise",
        "convergence": "none",
        "refinement": "none",
    },
    "inertia_tuned": {
        "method": "inertia",
        "c0": 1.49618,
        "c1": 1.49618,
        "w_min": 0.7298,
        "w_max": 0.7298,
        "velocity_limit_ratio": 0.025,
        "mutation_swarm": 0.02,
        "evaluation": "fixed_subset",
        "fitness_size": 2000,
        "batch_size": 1000,
        "initialization": "model_noise",
        "convergence": "none",
        "refinement": "none",
    },
    "tuned_no_mutation": {
        "method": "inertia",
        "c0": 1.49618,
        "c1": 1.49618,
        "w_min": 0.7298,
        "w_max": 0.7298,
        "velocity_limit_ratio": 0.025,
        "mutation_swarm": 0.0,
        "evaluation": "fixed_subset",
        "fitness_size": 2000,
        "batch_size": 1000,
        "initialization": "model_noise",
        "convergence": "none",
        "refinement": "none",
    },
    "tuned_full_evaluation": {
        "method": "inertia",
        "c0": 1.49618,
        "c1": 1.49618,
        "w_min": 0.7298,
        "w_max": 0.7298,
        "velocity_limit_ratio": 0.025,
        "mutation_swarm": 0.02,
        "evaluation": "full",
        "fitness_size": None,
        "batch_size": None,
        "initialization": "model_noise",
        "convergence": "none",
        "refinement": "none",
    },
    "tuned_uniform_initialization": {
        "method": "inertia",
        "c0": 1.49618,
        "c1": 1.49618,
        "w_min": 0.7298,
        "w_max": 0.7298,
        "velocity_limit_ratio": 0.025,
        "mutation_swarm": 0.02,
        "evaluation": "fixed_subset",
        "fitness_size": 2000,
        "batch_size": 1000,
        "initialization": "uniform",
        "convergence": "none",
        "refinement": "none",
    },
    "tuned_particle_reset": {
        "method": "inertia",
        "c0": 1.49618,
        "c1": 1.49618,
        "w_min": 0.7298,
        "w_max": 0.7298,
        "velocity_limit_ratio": 0.025,
        "mutation_swarm": 0.02,
        "evaluation": "fixed_subset",
        "fitness_size": 2000,
        "batch_size": 1000,
        "initialization": "model_noise",
        "convergence": "particle_reset",
        "convergence_patience": 10,
        "convergence_min_delta": 0.0001,
        "refinement": "none",
    },
    "tuned_adam_100_lr.01": {
        "method": "inertia",
        "c0": 1.49618,
        "c1": 1.49618,
        "w_min": 0.7298,
        "w_max": 0.7298,
        "velocity_limit_ratio": 0.025,
        "mutation_swarm": 0.02,
        "evaluation": "fixed_subset",
        "fitness_size": 2000,
        "batch_size": 1000,
        "initialization": "model_noise",
        "convergence": "none",
        "refinement": "adam",
        "refinement_epochs": 100,
        "refinement_lr": 0.01,
    },
    "adaptive_moment_.10": {
        "method": "adaptive_moment",
        "c0": 1.49618,
        "c1": 1.49618,
        "w_min": 0.7298,
        "w_max": 0.7298,
        "velocity_limit_ratio": 0.025,
        "mutation_swarm": 0.02,
        "moment_blend": 0.10,
        "evaluation": "fixed_subset",
        "fitness_size": 2000,
        "batch_size": 1000,
        "initialization": "model_noise",
        "convergence": "none",
        "refinement": "none",
    },
    "adaptive_moment_.25": {
        "method": "adaptive_moment",
        "c0": 1.49618,
        "c1": 1.49618,
        "w_min": 0.7298,
        "w_max": 0.7298,
        "velocity_limit_ratio": 0.025,
        "mutation_swarm": 0.02,
        "moment_blend": 0.25,
        "evaluation": "fixed_subset",
        "fitness_size": 2000,
        "batch_size": 1000,
        "initialization": "model_noise",
        "convergence": "none",
        "refinement": "none",
    },
    "adaptive_moment_.50": {
        "method": "adaptive_moment",
        "c0": 1.49618,
        "c1": 1.49618,
        "w_min": 0.7298,
        "w_max": 0.7298,
        "velocity_limit_ratio": 0.025,
        "mutation_swarm": 0.02,
        "moment_blend": 0.50,
        "evaluation": "fixed_subset",
        "fitness_size": 2000,
        "batch_size": 1000,
        "initialization": "model_noise",
        "convergence": "none",
        "refinement": "none",
    },
}


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def resolve_execution_device(user_device: Optional[str] = None) -> torch.device:
    if user_device:
        dev = torch.device(user_device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if dev.type == "mps":
            built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
            avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if not (built and avail):
                raise RuntimeError("MPS requested but not available.")
        return dev
    else:
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


def compute_summaries_and_ranks(runs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    main_runs = [r for r in runs if r.get("type") == "main" and r.get("completed")]
    ablation_runs = [r for r in runs if r.get("type") == "ablation" and r.get("completed")]

    def process_group(group_runs: List[Dict[str, Any]], is_ablation: bool = False) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for r in group_runs:
            ds = r["dataset"]
            method_key = r["profile"] if is_ablation else r["method"]
            key = (ds, method_key)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)

        summaries = []
        for (ds, m_key), r_list in grouped.items():
            eval_accs = [r["eval_metrics"]["accuracy"] for r in r_list]
            eval_losses = [r["eval_metrics"]["loss"] for r in r_list]
            eval_mses = [r["eval_metrics"]["mse"] for r in r_list]
            train_accs = [r["train_metrics"]["accuracy"] for r in r_list]
            train_losses = [r["train_metrics"]["loss"] for r in r_list]
            runtimes = [r["runtime_seconds"] for r in r_list]

            first_run = r_list[0]
            summary_entry = {
                "dataset": ds,
                "method" if not is_ablation else "profile": m_key,
                "method_name": first_run["method"],
                "n_particles": first_run["n_particles"],
                "epochs": first_run["epochs"],
                "n_runs": len(r_list),
                "eval_acc": calc_stats(eval_accs),
                "eval_loss": calc_stats(eval_losses),
                "eval_mse": calc_stats(eval_mses),
                "train_acc": calc_stats(train_accs),
                "train_loss": calc_stats(train_losses),
                "runtime_seconds": calc_stats(runtimes),
            }
            summaries.append(summary_entry)

        # Compute per-dataset ranks
        datasets = sorted(list(set(s["dataset"] for s in summaries)))
        for ds in datasets:
            ds_items = [s for s in summaries if s["dataset"] == ds]
            ds_items.sort(
                key=lambda s: (
                    -s["eval_acc"]["mean"],
                    s["eval_loss"]["mean"],
                    s["eval_mse"]["mean"],
                )
            )
            for rank_idx, item in enumerate(ds_items, start=1):
                item["rank_acc"] = rank_idx

            ds_items.sort(
                key=lambda s: (
                    s["eval_loss"]["mean"],
                    -s["eval_acc"]["mean"],
                    s["eval_mse"]["mean"],
                )
            )
            for rank_idx, item in enumerate(ds_items, start=1):
                item["rank_loss"] = rank_idx

        return summaries

    return {
        "main": process_group(main_runs, is_ablation=False),
        "ablation": process_group(ablation_runs, is_ablation=True),
    }


def save_json_atomic(data: Dict[str, Any], json_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = json_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(json_path)


def write_csv_reports(summaries: Dict[str, List[Dict[str, Any]]], main_csv_path: Path, ablation_csv_path: Path):
    main_csv_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames_main = [
        "dataset",
        "method",
        "n_particles",
        "epochs",
        "n_seeds",
        "eval_acc_mean",
        "eval_acc_std",
        "eval_acc_median",
        "eval_acc_iqr",
        "eval_acc_ci95",
        "eval_loss_mean",
        "eval_loss_std",
        "eval_loss_median",
        "eval_loss_iqr",
        "eval_loss_ci95",
        "eval_mse_mean",
        "eval_mse_std",
        "train_acc_mean",
        "train_loss_mean",
        "runtime_seconds_mean",
        "runtime_seconds_std",
        "rank_acc",
        "rank_loss",
    ]

    with open(main_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_main)
        writer.writeheader()
        for s in sorted(summaries["main"], key=lambda x: (x["dataset"], x["method"])):
            writer.writerow(
                {
                    "dataset": s["dataset"],
                    "method": s["method"],
                    "n_particles": s["n_particles"],
                    "epochs": s["epochs"],
                    "n_seeds": s["n_runs"],
                    "eval_acc_mean": s["eval_acc"]["mean"],
                    "eval_acc_std": s["eval_acc"]["std"],
                    "eval_acc_median": s["eval_acc"]["median"],
                    "eval_acc_iqr": s["eval_acc"]["iqr"],
                    "eval_acc_ci95": s["eval_acc"]["ci95_t"],
                    "eval_loss_mean": s["eval_loss"]["mean"],
                    "eval_loss_std": s["eval_loss"]["std"],
                    "eval_loss_median": s["eval_loss"]["median"],
                    "eval_loss_iqr": s["eval_loss"]["iqr"],
                    "eval_loss_ci95": s["eval_loss"]["ci95_t"],
                    "eval_mse_mean": s["eval_mse"]["mean"],
                    "eval_mse_std": s["eval_mse"]["std"],
                    "train_acc_mean": s["train_acc"]["mean"],
                    "train_loss_mean": s["train_loss"]["mean"],
                    "runtime_seconds_mean": s["runtime_seconds"]["mean"],
                    "runtime_seconds_std": s["runtime_seconds"]["std"],
                    "rank_acc": s.get("rank_acc", 0),
                    "rank_loss": s.get("rank_loss", 0),
                }
            )

    fieldnames_ablation = [
        "profile",
        "dataset",
        "method",
        "n_particles",
        "epochs",
        "n_seeds",
        "eval_acc_mean",
        "eval_acc_std",
        "eval_acc_median",
        "eval_acc_iqr",
        "eval_acc_ci95",
        "eval_loss_mean",
        "eval_loss_std",
        "eval_loss_median",
        "eval_loss_iqr",
        "eval_loss_ci95",
        "eval_mse_mean",
        "eval_mse_std",
        "train_acc_mean",
        "train_loss_mean",
        "runtime_seconds_mean",
        "runtime_seconds_std",
        "rank_acc",
        "rank_loss",
    ]

    with open(ablation_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_ablation)
        writer.writeheader()
        for s in sorted(summaries["ablation"], key=lambda x: (x["dataset"], x["profile"])):
            writer.writerow(
                {
                    "profile": s["profile"],
                    "dataset": s["dataset"],
                    "method": s["method_name"],
                    "n_particles": s["n_particles"],
                    "epochs": s["epochs"],
                    "n_seeds": s["n_runs"],
                    "eval_acc_mean": s["eval_acc"]["mean"],
                    "eval_acc_std": s["eval_acc"]["std"],
                    "eval_acc_median": s["eval_acc"]["median"],
                    "eval_acc_iqr": s["eval_acc"]["iqr"],
                    "eval_acc_ci95": s["eval_acc"]["ci95_t"],
                    "eval_loss_mean": s["eval_loss"]["mean"],
                    "eval_loss_std": s["eval_loss"]["std"],
                    "eval_loss_median": s["eval_loss"]["median"],
                    "eval_loss_iqr": s["eval_loss"]["iqr"],
                    "eval_loss_ci95": s["eval_loss"]["ci95_t"],
                    "eval_mse_mean": s["eval_mse"]["mean"],
                    "eval_mse_std": s["eval_mse"]["std"],
                    "train_acc_mean": s["train_acc"]["mean"],
                    "train_loss_mean": s["train_loss"]["mean"],
                    "runtime_seconds_mean": s["runtime_seconds"]["mean"],
                    "runtime_seconds_std": s["runtime_seconds"]["std"],
                    "rank_acc": s.get("rank_acc", 0),
                    "rank_loss": s.get("rank_loss", 0),
                }
            )


def render_plots(summaries: Dict[str, List[Dict[str, Any]]], figure_dir: Path):
    from matplotlib.patches import Patch

    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    main_sums = summaries.get("main", [])
    n_seeds = max([s.get("n_runs", 5) for s in main_sums]) if main_sums else 5
    present_datasets = {s["dataset"] for s in main_sums}
    datasets = [name for name in WORKLOADS if name in present_datasets]
    methods = [m for m in MAIN_METHODS if any(s["method"] == m for s in main_sums)] if main_sums else []
    if main_sums and not methods:
        methods = sorted(list(set(s["method"] for s in main_sums)))

    # 1. Accuracy Plot (pso_v4_accuracy.png)
    if main_sums and datasets and methods:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(datasets))
        width = 0.8 / max(1, len(methods))

        for i, m in enumerate(methods):
            means = []
            yerrs = []
            for ds in datasets:
                match = [s for s in main_sums if s["dataset"] == ds and s["method"] == m]
                if match:
                    means.append(match[0]["eval_acc"]["mean"])
                    yerrs.append(match[0]["eval_acc"]["std"])
                else:
                    means.append(np.nan)
                    yerrs.append(np.nan)

            offset = x - 0.4 + width * i + width / 2
            col, hatch = get_method_style(m, i)
            ax.bar(
                offset,
                means,
                width,
                yerr=yerrs,
                label=m,
                color=col,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.7,
                capsize=3,
            )

        ax.set_ylabel("Evaluation Accuracy")
        ax.set_title(f"PSO Benchmark Evaluation Accuracy by Workload\n(mean ± 1 SD, n={n_seeds}; XOR=train, others=held-out)")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 1.05)
        ax.legend(title="Method", bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.savefig(figure_dir / "pso_v4_accuracy.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 2. Loss Plot (pso_v4_loss.png)
    if main_sums and datasets and methods:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(datasets))
        width = 0.8 / max(1, len(methods))

        for i, m in enumerate(methods):
            means = []
            lower_errs = []
            upper_errs = []
            for ds in datasets:
                match = [s for s in main_sums if s["dataset"] == ds and s["method"] == m]
                if match:
                    m_val = match[0]["eval_loss"]["mean"]
                    s_val = match[0]["eval_loss"]["std"]
                    means.append(m_val)
                    lower_errs.append(min(s_val, max(0.0, m_val - 1e-6)))
                    upper_errs.append(s_val)
                else:
                    means.append(np.nan)
                    lower_errs.append(np.nan)
                    upper_errs.append(np.nan)

            offset = x - 0.4 + width * i + width / 2
            col, hatch = get_method_style(m, i)
            yerr = [lower_errs, upper_errs]
            ax.bar(
                offset,
                means,
                width,
                yerr=yerr,
                label=m,
                color=col,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.7,
                capsize=3,
            )

        ax.set_yscale("log")
        ax.set_ylabel("Evaluation Loss (log scale)")
        ax.set_title(f"PSO Benchmark Evaluation Loss by Workload\n(mean ± 1 SD, n={n_seeds}; XOR=train, others=held-out)")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend(title="Method", bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.savefig(figure_dir / "pso_v4_loss.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 3. Runtime Plot (pso_v4_runtime.png)
    if main_sums and datasets and methods:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(datasets))
        width = 0.8 / max(1, len(methods))

        for i, m in enumerate(methods):
            means = []
            yerrs = []
            for ds in datasets:
                match = [s for s in main_sums if s["dataset"] == ds and s["method"] == m]
                if match:
                    means.append(match[0]["runtime_seconds"]["mean"])
                    yerrs.append(match[0]["runtime_seconds"]["std"])
                else:
                    means.append(np.nan)
                    yerrs.append(np.nan)

            offset = x - 0.4 + width * i + width / 2
            col, hatch = get_method_style(m, i)
            ax.bar(
                offset,
                means,
                width,
                yerr=yerrs,
                label=m,
                color=col,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.7,
                capsize=3,
            )

        ax.set_ylabel("Fit Runtime (seconds)")
        ax.set_title(f"PSO Benchmark Fit Runtime by Workload\n(mean ± 1 SD, n={n_seeds}; fit-only after warmup)")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend(title="Method", bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.savefig(figure_dir / "pso_v4_runtime.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 4. Rank Heatmap Plot (pso_v4_rank_heatmap.png)
    if main_sums and datasets and methods:
        fig, ax = plt.subplots(figsize=(8, 6))
        rank_matrix = np.full((len(methods), len(datasets)), np.nan)

        for i, m in enumerate(methods):
            for j, ds in enumerate(datasets):
                match = [s for s in main_sums if s["dataset"] == ds and s["method"] == m]
                if match and "rank_acc" in match[0]:
                    rank_matrix[i, j] = match[0]["rank_acc"]

        masked_matrix = np.ma.masked_invalid(rank_matrix)
        n_methods = len(methods)
        cmap = plt.get_cmap("YlGnBu_r", n_methods)
        norm = matplotlib.colors.BoundaryNorm(np.arange(0.5, n_methods + 1.5, 1.0), n_methods)

        cax = ax.matshow(
            masked_matrix,
            cmap=cmap,
            norm=norm,
        )
        cb = fig.colorbar(cax, ticks=np.arange(1, n_methods + 1), label="Rank (1 = Best Evaluation Accuracy)")
        cb.ax.set_yticklabels([str(r) for r in range(1, n_methods + 1)])

        ax.set_xticks(np.arange(len(datasets)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(datasets)
        ax.set_yticklabels(methods)

        ax.set_xticks(np.arange(len(datasets)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(methods)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", size=0)

        for i in range(len(methods)):
            for j in range(len(datasets)):
                val = rank_matrix[i, j]
                if not np.isnan(val):
                    rgba = cmap(norm(val))
                    luminance = (
                        0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    )
                    ax.text(
                        j,
                        i,
                        str(int(val)),
                        ha="center",
                        va="center",
                        color="black" if luminance > 0.55 else "white",
                        fontweight="bold",
                    )
        ax.tick_params(
            bottom=False,
            labelbottom=False,
            top=True,
            labeltop=True,
        )
        ax.set_title(f"PSO Method Accuracy Ranks Across Workloads\n(mean ± 1 SD, n={n_seeds}; XOR=train, others=held-out)", pad=20)
        fig.savefig(figure_dir / "pso_v4_rank_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 5. MNIST Ablation Plot (pso_v4_mnist_ablation.png)
    ablation_sums = summaries.get("ablation", [])
    if ablation_sums:
        n_abl_seeds = max([s.get("n_runs", 5) for s in ablation_sums]) if ablation_sums else 5

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

        by_profile = {s["profile"]: s for s in ablation_sums}
        sorted_ablation = [
            by_profile[name] for name in ABLATION_PROFILES if name in by_profile
        ]
        profiles = [s["profile"] for s in sorted_ablation]
        acc_means = [s["eval_acc"]["mean"] for s in sorted_ablation]
        acc_errs = [s["eval_acc"]["std"] for s in sorted_ablation]
        loss_means = [s["eval_loss"]["mean"] for s in sorted_ablation]
        loss_errs = [s["eval_loss"]["std"] for s in sorted_ablation]

        colors = []
        hatches = []
        display_labels = []

        for p_name in profiles:
            p_cfg = ABLATION_PROFILES.get(p_name, {})
            if p_cfg.get("refinement") == "adam" or "adam" in p_name:
                colors.append("#D55E00")
                hatches.append("xx")
                display_labels.append(f"{p_name}\n(gradient-based)")
            elif p_cfg.get("method") == "adaptive_moment" or "adaptive_moment" in p_name:
                colors.append("#CC79A7")
                hatches.append("||")
                display_labels.append(f"{p_name}\n(derivative-free)")
            else:
                colors.append("#0072B2")
                hatches.append("//")
                display_labels.append(f"{p_name}\n(derivative-free)")

        x = np.arange(len(profiles))
        for i in range(len(profiles)):
            ax1.bar(
                x[i],
                acc_means[i],
                yerr=acc_errs[i],
                color=colors[i],
                hatch=hatches[i],
                edgecolor="black",
                linewidth=0.8,
                capsize=4,
            )

        ax1.set_ylabel("Held-Out Accuracy")
        ax1.set_title(f"MNIST PCA32 Linear Ablation Study Profiles\n(mean ± 1 SD, n={n_abl_seeds}; Held-Out Evaluation)")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        legend_patches = [
            Patch(facecolor="#0072B2", hatch="//", edgecolor="black", label="Derivative-Free (Standard PSO)"),
            Patch(facecolor="#CC79A7", hatch="||", edgecolor="black", label="Derivative-Free (Adaptive Moment)"),
            Patch(facecolor="#D55E00", hatch="xx", edgecolor="black", label="Gradient-Based (Adam Refinement)"),
        ]
        ax1.legend(handles=legend_patches, loc="upper left")

        lower_loss_errs = [min(s, max(0.0, m - 1e-6)) for m, s in zip(loss_means, loss_errs)]
        upper_loss_errs = loss_errs

        for i in range(len(profiles)):
            yerr_single = [[lower_loss_errs[i]], [upper_loss_errs[i]]]
            ax2.bar(
                x[i],
                loss_means[i],
                yerr=yerr_single,
                color=colors[i],
                hatch=hatches[i],
                edgecolor="black",
                linewidth=0.8,
                capsize=4,
            )

        ax2.set_yscale("log")
        ax2.set_ylabel("Held-Out Loss (log scale)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_labels, rotation=45, ha="right")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        fig.savefig(figure_dir / "pso_v4_mnist_ablation.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

def run_benchmark(
    quick: bool = False,
    overwrite: bool = False,
    skip_main: bool = False,
    skip_ablation: bool = False,
    dataset_filter: Optional[List[str]] = None,
    method_filter: Optional[List[str]] = None,
    seed_filter: Optional[List[int]] = None,
    device_name: Optional[str] = None,
    output_json: Path = Path("benchmark_results/pso_v4_benchmark.json"),
    main_csv: Path = Path("benchmark_results/pso_v4_main_benchmark.csv"),
    ablation_csv: Path = Path("benchmark_results/pso_v4_ablation_benchmark.csv"),
    figure_dir: Path = Path("history_plt"),
):
    device = resolve_execution_device(device_name)
    print(f"Executing benchmark suite on device: {device}")

    hw_provenance = get_hardware_provenance(device)

    # Load existing JSON if available and not overwrite
    existing_data: Dict[str, Any] = {}
    completed_runs: Dict[str, Dict[str, Any]] = {}
    if output_json.exists() and not overwrite:
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if existing_data.get("benchmark_protocol_version") == BENCHMARK_PROTOCOL_VERSION:
                for r in existing_data.get("runs", []):
                    if r.get("completed") and "run_id" in r:
                        completed_runs[r["run_id"]] = r
                print(f"Loaded {len(completed_runs)} existing completed runs from {output_json}")
            else:
                print(
                    f"Existing JSON protocol version ({existing_data.get('benchmark_protocol_version')}) "
                    f"differs from {BENCHMARK_PROTOCOL_VERSION}. Starting fresh."
                )
        except Exception as e:
            print(f"Warning: Failed to load existing JSON ({e}). Starting fresh.")

    all_runs: List[Dict[str, Any]] = list(completed_runs.values())

    # Build targets
    main_datasets = list(WORKLOADS.keys())
    if dataset_filter:
        ds_filter_upper = [d.upper() for d in dataset_filter]
        main_datasets = [d for d in main_datasets if d.upper() in ds_filter_upper]

    main_methods = list(MAIN_METHODS)
    if method_filter:
        m_filter_lower = [m.lower() for m in method_filter]
        main_methods = [m for m in main_methods if m.lower() in m_filter_lower]

    main_seeds = [41, 42, 43, 44, 45]
    if seed_filter:
        main_seeds = list(seed_filter)
    if quick:
        main_seeds = main_seeds[:1]

    ablation_profiles = dict(ABLATION_PROFILES)
    if method_filter:
        p_filter_lower = [m.lower() for m in method_filter]
        ablation_profiles = {
            k: v for k, v in ablation_profiles.items() if k.lower() in p_filter_lower or v["method"].lower() in p_filter_lower
        }

    ablation_seeds = [46, 47, 48, 49, 50]
    if seed_filter:
        ablation_seeds = list(seed_filter)
    if quick:
        ablation_seeds = ablation_seeds[:1]

    # --- 1. Main Benchmark Runs ---
    if not skip_main:
        print("\n=== Running Main Benchmark Suite ===")
        for ds_name in main_datasets:
            wl = WORKLOADS[ds_name]
            task = wl["task"]
            n_particles = 2 if quick else wl["n_particles"]
            epochs = 2 if quick else wl["epochs"]
            evaluation = wl["evaluation"]
            fitness_size = (50 if quick else wl["fitness_size"]) if evaluation == "fixed_subset" else None
            batch_size = (25 if quick else wl["batch_size"]) if evaluation == "fixed_subset" else None

            for method in main_methods:
                vel_limit = None if method == "bare_bones" else 0.1
                for seed in main_seeds:
                    config_payload = {
                        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
                        "quick": quick,
                        "device": str(device),
                        "pso_version": pso_version,
                        "dataset": ds_name,
                        "type": "main",
                        "method": method,
                        "profile": method,
                        "seed": seed,
                        "n_particles": n_particles,
                        "epochs": epochs,
                        "evaluation": evaluation,
                        "fitness_size": fitness_size,
                        "batch_size": batch_size,
                        "particle_min": wl["particle_min"],
                        "particle_max": wl["particle_max"],
                        "initial_position_noise": wl["initial_position_noise"],
                        "renewal": wl["renewal"],
                        "warmup_pso_epochs": 2,
                        "warmup_refinement_epochs": 0,
                    }
                    fp_bytes = json.dumps(config_payload, sort_keys=True, default=str).encode("utf-8")
                    config_fp = hashlib.sha256(fp_bytes).hexdigest()[:12]
                    run_id = f"main_{ds_name}_{method}_seed{seed}_{config_fp}"

                    if run_id in completed_runs and not overwrite:
                        print(f"Skipping completed run: {run_id}")
                        continue

                    print(f"Running {run_id}...")

                    # Data loading cached once per dataset+seed
                    x_train, x_test, y_train, y_test = get_cached_dataset(ds_name, seed=seed)
                    data_fp = compute_data_fingerprint(x_train, x_test, y_train, y_test)

                    # --- Untimed Warmup Phase ---
                    warmup_model = wl["model_factory"](seed=seed)
                    warmup_loss = wl["loss_fn"]()
                    warmup_opt = Optimizer(
                        model=warmup_model,
                        loss=warmup_loss,
                        task=task,
                        method=method,
                        initialization="model_noise",
                        evaluation=evaluation,
                        convergence="none",
                        refinement="none",
                        n_particles=n_particles,
                        velocity_limit_ratio=vel_limit,
                        boundary_strategy="reflect",
                        particle_min=wl["particle_min"],
                        particle_max=wl["particle_max"],
                        initial_position_noise=wl["initial_position_noise"],
                        seed=seed,
                        device=device,
                        fitness_size=fitness_size,
                    )
                    warmup_opt.fit(
                        x_train,
                        y_train,
                        epochs=2,
                        batch_size=batch_size,
                        fitness_size=fitness_size,
                        renewal=wl["renewal"],
                    )
                    sync_device(device)
                    del warmup_opt, warmup_model, warmup_loss

                    # --- Timed Model & Optimizer Construction ---
                    model = wl["model_factory"](seed=seed)
                    model_fp = compute_model_fingerprint(model)
                    loss_inst = wl["loss_fn"]()
                    model_params = sum(p.numel() for p in model.parameters())

                    opt = Optimizer(
                        model=model,
                        loss=loss_inst,
                        task=task,
                        method=method,
                        initialization="model_noise",
                        evaluation=evaluation,
                        convergence="none",
                        refinement="none",
                        n_particles=n_particles,
                        velocity_limit_ratio=vel_limit,
                        boundary_strategy="reflect",
                        particle_min=wl["particle_min"],
                        particle_max=wl["particle_max"],
                        initial_position_noise=wl["initial_position_noise"],
                        seed=seed,
                        device=device,
                        fitness_size=fitness_size,
                    )

                    resolved_plugins = extract_plugin_metadata(opt)

                    # Fit-only timing scope
                    sync_device(device)
                    t0 = time.perf_counter()

                    train_score = opt.fit(
                        x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        fitness_size=fitness_size,
                        renewal=wl["renewal"],
                    )

                    sync_device(device)
                    t1 = time.perf_counter()
                    runtime_sec = t1 - t0

                    # Score evaluation
                    score_source = "train" if not wl["held_out"] else "held_out"
                    if wl["held_out"]:
                        eval_score = opt.evaluate(x_test, y_test, batch_size=batch_size)
                    else:
                        eval_score = opt.evaluate(x_train, y_train, batch_size=batch_size)

                    run_record = {
                        "run_id": run_id,
                        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
                        "config_fingerprint": config_fp,
                        "data_fingerprint": data_fp,
                        "initial_model_fingerprint": model_fp,
                        "type": "main",
                        "dataset": ds_name,
                        "method": method,
                        "profile": method,
                        "seed": seed,
                        "n_particles": n_particles,
                        "epochs": epochs,
                        "score_source": score_source,
                        "model_param_count": model_params,
                        "train_data_size": len(x_train),
                        "eval_data_size": len(x_test) if wl["held_out"] else len(x_train),
                        "configured_pca_choice": wl["pca_config"],
                        "timing_scope": "fit_only_after_method_specific_warmup",
                        "warmup_protocol": {
                            "pso_epochs": 2,
                            "refinement_epochs": 0,
                            "untimed": True,
                        },
                        "config": config_payload,
                        "resolved_plugins": resolved_plugins,
                        "hardware_provenance": hw_provenance,
                        "device": str(device),
                        "pso_version": pso_version,
                        "torch_version": torch.__version__,
                        "train_metrics": {
                            "loss": float(train_score[0]),
                            "accuracy": float(train_score[1]),
                            "mse": float(train_score[2]),
                        },
                        "eval_metrics": {
                            "loss": float(eval_score[0]),
                            "accuracy": float(eval_score[1]),
                            "mse": float(eval_score[2]),
                        },
                        "runtime_seconds": float(runtime_sec),
                        "completed": True,
                        "error": None,
                    }

                    all_runs.append(run_record)
                    completed_runs[run_id] = run_record

                    sums = compute_summaries_and_ranks(all_runs)
                    save_json_atomic(
                        {
                            "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
                            "version": pso_version,
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "environment": hw_provenance,
                            "runs": all_runs,
                            "summaries": sums,
                        },
                        output_json,
                    )

    # --- 2. Ablation Suite Runs ---
    if not skip_ablation and (dataset_filter is None or any(d.upper() == "MNIST" for d in dataset_filter)):
        print("\n=== Running MNIST Ablation Benchmark Suite ===")
        wl = WORKLOADS["MNIST"]
        task = wl["task"]

        for p_name, p_cfg in ablation_profiles.items():
            method = p_cfg["method"]
            evaluation = p_cfg["evaluation"]
            n_particles = 2 if quick else wl["n_particles"]
            epochs = 2 if quick else wl["epochs"]
            fitness_size = (50 if quick else p_cfg["fitness_size"]) if evaluation == "fixed_subset" else None
            batch_size = (25 if quick else p_cfg["batch_size"]) if evaluation == "fixed_subset" else None
            refinement_epochs = min(2, p_cfg.get("refinement_epochs", 0)) if quick else p_cfg.get("refinement_epochs", 0)
            refinement_lr = p_cfg.get("refinement_lr", 0.001)
            is_adam = (p_cfg.get("refinement") == "adam")
            warmup_refinement_epochs = 1 if is_adam else 0

            for seed in ablation_seeds:
                config_payload = {
                    "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
                    "quick": quick,
                    "device": str(device),
                    "pso_version": pso_version,
                    "dataset": "MNIST",
                    "type": "ablation",
                    "method": method,
                    "profile": p_name,
                    "seed": seed,
                    "n_particles": n_particles,
                    "epochs": epochs,
                    "evaluation": evaluation,
                    "fitness_size": fitness_size,
                    "batch_size": batch_size,
                    "particle_min": wl["particle_min"],
                    "particle_max": wl["particle_max"],
                    "initial_position_noise": wl["initial_position_noise"],
                    "renewal": wl["renewal"],
                    "c0": p_cfg.get("c0"),
                    "c1": p_cfg.get("c1"),
                    "w_min": p_cfg.get("w_min"),
                    "w_max": p_cfg.get("w_max"),
                    "velocity_limit_ratio": p_cfg.get("velocity_limit_ratio"),
                    "mutation_swarm": p_cfg.get("mutation_swarm", 0.0),
                    "initialization": p_cfg["initialization"],
                    "convergence": p_cfg["convergence"],
                    "refinement": p_cfg["refinement"],
                    "refinement_epochs": refinement_epochs,
                    "refinement_lr": refinement_lr,
                    "moment_blend": p_cfg.get("moment_blend"),
                    "warmup_pso_epochs": 2,
                    "warmup_refinement_epochs": warmup_refinement_epochs,
                }
                fp_bytes = json.dumps(config_payload, sort_keys=True, default=str).encode("utf-8")
                config_fp = hashlib.sha256(fp_bytes).hexdigest()[:12]
                run_id = f"ablation_MNIST_{p_name}_seed{seed}_{config_fp}"

                if run_id in completed_runs and not overwrite:
                    print(f"Skipping completed run: {run_id}")
                    continue

                print(f"Running {run_id}...")

                x_train, x_test, y_train, y_test = get_cached_dataset("MNIST", seed=seed)
                data_fp = compute_data_fingerprint(x_train, x_test, y_train, y_test)

                # --- Untimed Warmup Phase ---
                warmup_model = wl["model_factory"](seed=seed)
                warmup_loss = wl["loss_fn"]()
                warmup_opt_kwargs = {
                    "model": warmup_model,
                    "loss": warmup_loss,
                    "task": task,
                    "method": method,
                    "initialization": p_cfg["initialization"],
                    "evaluation": evaluation,
                    "convergence": p_cfg["convergence"],
                    "refinement": p_cfg["refinement"],
                    "n_particles": n_particles,
                    "c0": p_cfg.get("c0"),
                    "c1": p_cfg.get("c1"),
                    "w_min": p_cfg.get("w_min"),
                    "w_max": p_cfg.get("w_max"),
                    "velocity_limit_ratio": p_cfg.get("velocity_limit_ratio"),
                    "mutation_swarm": p_cfg.get("mutation_swarm", 0.0),
                    "boundary_strategy": "reflect",
                    "particle_min": wl["particle_min"],
                    "particle_max": wl["particle_max"],
                    "initial_position_noise": wl["initial_position_noise"],
                    "seed": seed,
                    "device": device,
                    "fitness_size": fitness_size,
                    "convergence_patience": p_cfg.get("convergence_patience", 10),
                    "convergence_min_delta": p_cfg.get("convergence_min_delta", 0.0001),
                    "refinement_epochs": warmup_refinement_epochs,
                    "refinement_lr": refinement_lr,
                    "moment_blend": p_cfg.get("moment_blend"),
                }
                warmup_opt = Optimizer(**warmup_opt_kwargs)
                warmup_opt.fit(
                    x_train,
                    y_train,
                    epochs=2,
                    batch_size=batch_size,
                    fitness_size=fitness_size,
                    renewal=wl["renewal"],
                    refinement_epochs=warmup_refinement_epochs,
                    refinement_lr=refinement_lr,
                )
                sync_device(device)
                del warmup_opt, warmup_model, warmup_loss

                # --- Timed Model & Optimizer Construction ---
                model = wl["model_factory"](seed=seed)
                model_fp = compute_model_fingerprint(model)
                loss_inst = wl["loss_fn"]()
                model_params = sum(p.numel() for p in model.parameters())

                opt_kwargs = {
                    "model": model,
                    "loss": loss_inst,
                    "task": task,
                    "method": method,
                    "initialization": p_cfg["initialization"],
                    "evaluation": evaluation,
                    "convergence": p_cfg["convergence"],
                    "refinement": p_cfg["refinement"],
                    "n_particles": n_particles,
                    "c0": p_cfg.get("c0"),
                    "c1": p_cfg.get("c1"),
                    "w_min": p_cfg.get("w_min"),
                    "w_max": p_cfg.get("w_max"),
                    "velocity_limit_ratio": p_cfg.get("velocity_limit_ratio"),
                    "mutation_swarm": p_cfg.get("mutation_swarm", 0.0),
                    "boundary_strategy": "reflect",
                    "particle_min": wl["particle_min"],
                    "particle_max": wl["particle_max"],
                    "initial_position_noise": wl["initial_position_noise"],
                    "seed": seed,
                    "device": device,
                    "fitness_size": fitness_size,
                    "convergence_patience": p_cfg.get("convergence_patience", 10),
                    "convergence_min_delta": p_cfg.get("convergence_min_delta", 0.0001),
                    "refinement_epochs": refinement_epochs,
                    "refinement_lr": refinement_lr,
                    "moment_blend": p_cfg.get("moment_blend"),
                }

                opt = Optimizer(**opt_kwargs)
                resolved_plugins = extract_plugin_metadata(opt)

                sync_device(device)
                t0 = time.perf_counter()

                train_score = opt.fit(
                    x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    fitness_size=fitness_size,
                    renewal=wl["renewal"],
                    refinement_epochs=refinement_epochs,
                    refinement_lr=refinement_lr,
                )

                sync_device(device)
                t1 = time.perf_counter()
                runtime_sec = t1 - t0

                eval_score = opt.evaluate(x_test, y_test, batch_size=batch_size)

                run_record = {
                    "run_id": run_id,
                    "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
                    "config_fingerprint": config_fp,
                    "data_fingerprint": data_fp,
                    "initial_model_fingerprint": model_fp,
                    "type": "ablation",
                    "dataset": "MNIST",
                    "method": method,
                    "profile": p_name,
                    "seed": seed,
                    "n_particles": n_particles,
                    "epochs": epochs,
                    "score_source": "held_out",
                    "model_param_count": model_params,
                    "train_data_size": len(x_train),
                    "eval_data_size": len(x_test),
                    "configured_pca_choice": wl["pca_config"],
                    "timing_scope": "fit_only_after_method_specific_warmup",
                    "warmup_protocol": {
                        "pso_epochs": 2,
                        "refinement_epochs": warmup_refinement_epochs,
                        "untimed": True,
                    },
                    "config": config_payload,
                    "resolved_plugins": resolved_plugins,
                    "hardware_provenance": hw_provenance,
                    "device": str(device),
                    "pso_version": pso_version,
                    "torch_version": torch.__version__,
                    "train_metrics": {
                        "loss": float(train_score[0]),
                        "accuracy": float(train_score[1]),
                        "mse": float(train_score[2]),
                    },
                    "eval_metrics": {
                        "loss": float(eval_score[0]),
                        "accuracy": float(eval_score[1]),
                        "mse": float(eval_score[2]),
                    },
                    "runtime_seconds": float(runtime_sec),
                    "completed": True,
                    "error": None,
                }

                all_runs.append(run_record)
                completed_runs[run_id] = run_record

                sums = compute_summaries_and_ranks(all_runs)
                save_json_atomic(
                    {
                        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
                        "version": pso_version,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "environment": hw_provenance,
                        "runs": all_runs,
                        "summaries": sums,
                    },
                    output_json,
                )

    # Compute final summaries, write CSVs, render PNG charts
    final_summaries = compute_summaries_and_ranks(all_runs)
    save_json_atomic(
        {
            "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
            "version": pso_version,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "environment": hw_provenance,
            "runs": all_runs,
            "summaries": final_summaries,
        },
        output_json,
    )

    print("\nWriting CSV reports...")
    write_csv_reports(final_summaries, main_csv, ablation_csv)

    print("Rendering Matplotlib figure charts...")
    render_plots(final_summaries, figure_dir)

    print(f"\nBenchmark suite completed successfully!")
    print(f"- JSON: {output_json}")
    print(f"- Main CSV: {main_csv}")
    print(f"- Ablation CSV: {ablation_csv}")
    print(f"- Figures in: {figure_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="PSO v4 Deterministic Multi-Seed Benchmark Runner & Analysis Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true", help="Run tiny subset for rapid testing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached JSON run results")
    parser.add_argument("--skip-main", action="store_true", help="Skip main benchmark execution")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation benchmark execution")
    parser.add_argument("--datasets", type=str, help="Comma-separated dataset filter (e.g. XOR,Iris,MNIST)")
    parser.add_argument("--methods", type=str, help="Comma-separated method/profile filter (e.g. original,inertia)")
    parser.add_argument("--seeds", type=str, help="Comma-separated seeds or range (e.g. 41,42 or 41-45)")
    parser.add_argument("--device", type=str, help="Target execution device (cpu, cuda, mps)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/pso_v4_benchmark.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--main-csv",
        type=Path,
        default=Path("benchmark_results/pso_v4_main_benchmark.csv"),
        help="Main benchmark CSV output path",
    )
    parser.add_argument(
        "--ablation-csv",
        type=Path,
        default=Path("benchmark_results/pso_v4_ablation_benchmark.csv"),
        help="Ablation CSV output path",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("history_plt"),
        help="Figure export directory for PNGs",
    )

    args = parser.parse_args()

    ds_filter = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    m_filter = [m.strip() for m in args.methods.split(",")] if args.methods else None

    seed_filter = None
    if args.seeds:
        seed_filter = []
        for s_part in args.seeds.split(","):
            s_part = s_part.strip()
            if "-" in s_part:
                start_s, end_s = s_part.split("-", 1)
                seed_filter.extend(list(range(int(start_s), int(end_s) + 1)))
            else:
                seed_filter.append(int(s_part))

    run_benchmark(
        quick=args.quick,
        overwrite=args.overwrite,
        skip_main=args.skip_main,
        skip_ablation=args.skip_ablation,
        dataset_filter=ds_filter,
        method_filter=m_filter,
        seed_filter=seed_filter,
        device_name=args.device,
        output_json=args.output,
        main_csv=args.main_csv,
        ablation_csv=args.ablation_csv,
        figure_dir=args.figure_dir,
    )


if __name__ == "__main__":
    main()
