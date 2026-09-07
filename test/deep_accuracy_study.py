"""
MNIST Deep Accuracy Study: Architecture vs Optimizer Profiles

Evaluates official MNIST (60,000 train / 10,000 test) across:
1. Architecture Lane: Raw Linear, Raw MLP, Compact CNN under standard full-data Adam.
2. Optimizer Lane: Adam-Only, PSO-Only (adaptive_moment on 2k subset), and Hybrid (PSO warm start + Adam fine-tuning) on Compact CNN.

Contract:
- Official 60k train / 10k test split with train-only statistics normalization (no PCA).
- Avoid BatchNorm/Dropout so PSO and eval semantics match.
- Seed model construction identically per seed to preserve explicit initial state fingerprint across lanes.
- Fixed no-scheduler contract for Adam with CrossEntropyLoss and lr=1e-3.
"""

import argparse
import csv
import datetime
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Ensure test/ directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_suite import (
    calc_stats,
    compute_data_fingerprint,
    compute_model_fingerprint,
    extract_plugin_metadata,
    get_hardware_provenance,
    resolve_execution_device,
    save_json_atomic,
    sync_device,
)
from pso import Optimizer, __version__ as pso_version

DEEP_ACCURACY_PROTOCOL_VERSION = "1.0.0"


# ==========================================
# Data Preparation (No PCA, Raw 1x28x28)
# ==========================================

def prepare_deep_accuracy_mnist_data() -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    str,
    Dict[str, Any],
]:
    """
    Loads official torchvision MNIST dataset (60,000 train / 10,000 test).
    Normalizes images using train-only mean and std (no PCA).
    Validates sample counts and label range [0, 9].
    Returns (x_train, y_train, x_test, y_test, data_fingerprint, provenance_dict).
    """
    from torchvision.datasets import MNIST

    cache_dir = Path("result/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = MNIST(root=str(cache_dir), train=True, download=True)
    test_dataset = MNIST(root=str(cache_dir), train=False, download=True)

    n_train = len(train_dataset.data)
    n_test = len(test_dataset.data)
    if n_train != 60000:
        raise ValueError(f"Expected 60,000 training samples; got {n_train}")
    if n_test != 10000:
        raise ValueError(f"Expected 10,000 test samples; got {n_test}")

    y_train = train_dataset.targets.long()
    y_test = test_dataset.targets.long()

    min_tr, max_tr = int(y_train.min()), int(y_train.max())
    min_te, max_te = int(y_test.min()), int(y_test.max())
    if min_tr != 0 or max_tr != 9:
        raise ValueError(f"Train label range must be [0, 9]; got [{min_tr}, {max_tr}]")
    if min_te != 0 or max_te != 9:
        raise ValueError(f"Test label range must be [0, 9]; got [{min_te}, {max_te}]")

    x_train_raw = train_dataset.data.float() / 255.0  # (60000, 28, 28)
    x_test_raw = test_dataset.data.float() / 255.0    # (10000, 28, 28)

    # Compute normalization statistics from TRAIN split only
    mean_val = float(x_train_raw.mean())
    std_val = float(x_train_raw.std())

    x_train_norm = ((x_train_raw - mean_val) / std_val).unsqueeze(1)  # (60000, 1, 28, 28)
    x_test_norm = ((x_test_raw - mean_val) / std_val).unsqueeze(1)    # (10000, 1, 28, 28)

    data_fp = compute_data_fingerprint(x_train_norm, x_test_norm, y_train, y_test)
    normalization_provenance = {
        "input_shape": [1, 28, 28],
        "pca": False,
        "raw_inputs": True,
        "normalization_scope": "official_train_split_60000_only",
        "train_mean": round(mean_val, 6),
        "train_std": round(std_val, 6),
        "train_samples": 60000,
        "test_samples": 10000,
    }
    return x_train_norm, y_train, x_test_norm, y_test, data_fp, normalization_provenance


# ==========================================
# Architectures (No BatchNorm / No Dropout)
# ==========================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_raw_linear(seed: int = 41) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 10),
    )


def make_raw_mlp(seed: int = 41) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


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


def make_compact_cnn(seed: int = 41) -> nn.Module:
    torch.manual_seed(seed)
    return CompactCNN()


# ==========================================
# Evaluation & Training Routines
# ==========================================

def evaluate_model_on_test(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[float, float]:
    """
    Evaluates model on test data without gradients.
    Returns (test_loss, test_accuracy).
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        n_test = x_test.shape[0]
        for i in range(0, n_test, batch_size):
            bx = x_test[i : i + batch_size].to(device)
            by = y_test[i : i + batch_size].to(device)
            outputs = model(bx)
            loss = criterion(outputs, by)
            total_loss += loss.item() * bx.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return float(avg_loss), float(accuracy)


def train_adam_routine(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], float, float, float]:
    """
    Standard full-data Adam training routine with CrossEntropyLoss, Adam lr, fixed no-scheduler contract.
    Returns (history, final_test_loss, final_test_acc, elapsed_sec).
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    generator = torch.Generator().manual_seed(seed)
    n_train = x_train.shape[0]

    history = []
    init_loss, init_acc = evaluate_model_on_test(model, x_test, y_test, device)
    history.append({"epoch": 0, "test_loss": round(init_loss, 6), "test_acc": round(init_acc, 6)})

    sync_device(device)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=generator)
        for i in range(0, n_train, batch_size):
            indices = perm[i : i + batch_size]
            bx = x_train[indices].to(device)
            by = y_train[indices].to(device)
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()

        test_loss, test_acc = evaluate_model_on_test(model, x_test, y_test, device)
        history.append({
            "epoch": epoch,
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 6),
        })

    sync_device(device)
    elapsed = time.time() - t0

    final_loss = history[-1]["test_loss"]
    final_acc = history[-1]["test_acc"]
    return history, final_loss, final_acc, elapsed


def run_pso_routine(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    pso_epochs: int,
    n_particles: int,
    fitness_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, float], float, float, Dict[str, Any]]:
    """
    PSO-only adaptive_moment on fixed train subset without gradient refinement.
    Returns (best_model, fitness_score_dict, test_loss, test_acc, pso_metadata).
    """
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    opt = Optimizer(
        model=model,
        loss=loss_fn,
        task="multiclass",
        method="adaptive_moment",
        evaluation="fixed_subset",
        fitness_size=fitness_size,
        n_particles=n_particles,
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

    t0 = time.time()
    opt.fit(x_train, y_train, epochs=pso_epochs)
    sync_device(device)
    elapsed = time.time() - t0

    best_model = opt.get_best_model()
    best_score_tuple = opt.get_best_score()
    fitness_score = {
        "subset_loss": round(float(best_score_tuple[0]), 6),
        "subset_acc": round(float(best_score_tuple[1]), 6),
        "subset_mse": round(float(best_score_tuple[2]), 6),
    }

    test_loss, test_acc = evaluate_model_on_test(best_model, x_test, y_test, device)
    plugin_meta = extract_plugin_metadata(opt)

    pso_meta = {
        "elapsed_sec": round(elapsed, 4),
        "particles": n_particles,
        "pso_epochs": pso_epochs,
        "fitness_size": fitness_size,
        "plugins": plugin_meta,
    }
    return best_model, fitness_score, float(test_loss), float(test_acc), pso_meta


# ==========================================
# Output Generation (CSV, Plot)
# ==========================================

def save_csv_records(records: List[Dict[str, Any]], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lane",
        "profile_or_arch",
        "seed",
        "model_name",
        "param_count",
        "initial_test_acc",
        "final_test_acc",
        "final_test_loss",
        "subset_fitness_acc",
        "subset_fitness_loss",
        "pso_epochs",
        "adam_epochs",
        "elapsed_sec",
        "model_fingerprint",
        "data_fingerprint",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "lane": r.get("lane"),
                "profile_or_arch": r.get("profile_or_arch"),
                "seed": r.get("seed"),
                "model_name": r.get("model_name"),
                "param_count": r.get("param_count"),
                "initial_test_acc": r.get("initial_test_acc"),
                "final_test_acc": r.get("final_test_acc"),
                "final_test_loss": r.get("final_test_loss"),
                "subset_fitness_acc": r.get("subset_fitness_acc"),
                "subset_fitness_loss": r.get("subset_fitness_loss"),
                "pso_epochs": r.get("pso_epochs"),
                "adam_epochs": r.get("adam_epochs"),
                "elapsed_sec": r.get("elapsed_sec"),
                "model_fingerprint": r.get("model_fingerprint"),
                "data_fingerprint": r.get("data_fingerprint"),
            })


def render_plots(
    arch_summary: Dict[str, Dict[str, float]],
    opt_summary: Dict[str, Dict[str, float]],
    figure_path: Path,
):
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if arch_summary:
        arch_labels = list(arch_summary.keys())
        arch_means = [arch_summary[k]["mean"] * 100 for k in arch_labels]
        arch_sds = [arch_summary[k]["std"] * 100 for k in arch_labels]
        arch_display = {
            "raw_linear": "Raw Linear",
            "raw_mlp": "Raw MLP",
            "compact_cnn": "Compact CNN",
        }

        x_arch = np.arange(len(arch_labels))
        ax1.bar(x_arch, arch_means, yerr=arch_sds, capsize=5, color="#56B4E9", edgecolor="black", alpha=0.85)
        ax1.axhline(98.0, color="red", linestyle="--", linewidth=1.5, label="98% Target")
        ax1.set_xticks(x_arch)
        ax1.set_xticklabels([arch_display[k] for k in arch_labels], rotation=15)
        ax1.set_ylabel("Final Test Accuracy (%)")
        ax1.set_title("Architecture Lane (Full-Data Adam)")
        ax1.set_ylim(0, 110)
        ax1.set_axisbelow(True)
        ax1.grid(axis="y", linestyle=":", alpha=0.6)
        ax1.legend(loc="lower left")

        for i, (m, sd) in enumerate(zip(arch_means, arch_sds)):
            ax1.text(
                i,
                105.0 if m >= 90.0 else min(m + sd + 1.5, 102.5),
                f"{m:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
            )

    if opt_summary:
        opt_labels = list(opt_summary.keys())
        opt_means = [opt_summary[k]["mean"] * 100 for k in opt_labels]
        opt_sds = [opt_summary[k]["std"] * 100 for k in opt_labels]
        opt_display = {
            "adam_only": "Adam Only",
            "pso_only": "PSO Only",
            "hybrid": "PSO → Adam",
        }

        x_opt = np.arange(len(opt_labels))
        colors = ["#009E73", "#E69F00", "#CC79A7"]
        ax2.bar(x_opt, opt_means, yerr=opt_sds, capsize=5, color=colors[:len(opt_labels)], edgecolor="black", alpha=0.85)
        ax2.axhline(98.0, color="red", linestyle="--", linewidth=1.5, label="98% Target")
        ax2.set_xticks(x_opt)
        ax2.set_xticklabels([opt_display[k] for k in opt_labels], rotation=15)
        ax2.set_ylabel("Final Test Accuracy (%)")
        ax2.set_title("Optimizer Lane (Compact CNN)")
        ax2.set_ylim(0, 110)
        ax2.set_axisbelow(True)
        ax2.grid(axis="y", linestyle=":", alpha=0.6)
        ax2.legend(loc="lower left")

        for i, (m, sd) in enumerate(zip(opt_means, opt_sds)):
            ax2.text(
                i,
                105.0 if m >= 90.0 else min(m + sd + 1.5, 102.5),
                f"{m:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
            )

    plt.suptitle("MNIST Deep Accuracy Study: Architectures & Optimizer Profiles", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


# ==========================================
# Main CLI & Runner
# ==========================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNIST Deep Accuracy Study: Architecture vs Optimizer Profiles"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="101,102,103",
        help="Comma-separated random seeds (default: 101,102,103)",
    )
    parser.add_argument(
        "--adam-epochs",
        type=int,
        default=10,
        help="Full-data Adam training epochs (default: 10)",
    )
    parser.add_argument(
        "--pso-epochs",
        type=int,
        default=40,
        help="PSO swarm optimization epochs (default: 40)",
    )
    parser.add_argument(
        "--particles",
        "--n-particles",
        type=int,
        default=30,
        dest="particles",
        help="Number of PSO particles (default: 30)",
    )
    parser.add_argument(
        "--fitness-size",
        type=int,
        default=2000,
        help="Fixed train subset fitness size for PSO (default: 2000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for Adam DataLoader (default: 256)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=1e-3,
        dest="lr",
        help="Learning rate for Adam optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target PyTorch device ('cpu', 'cuda', 'mps'; default: auto-detect)",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="benchmark_results/pso_v4_deep_accuracy.json",
        help="JSON result output path (default: benchmark_results/pso_v4_deep_accuracy.json)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="benchmark_results/pso_v4_deep_accuracy.csv",
        help="CSV result output path (default: benchmark_results/pso_v4_deep_accuracy.csv)",
    )
    parser.add_argument(
        "--figure-path",
        "--plot-path",
        type=str,
        default="history_plt/pso_v4_deep_accuracy.png",
        dest="figure_path",
        help="Figure output path (default: history_plt/pso_v4_deep_accuracy.png)",
    )
    parser.add_argument(
        "--lanes",
        "--profiles",
        type=str,
        choices=["all", "architectures", "optimizers"],
        default="all",
        help="Lanes/profiles to evaluate (choices: all, architectures, optimizers; default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seed_list:
        raise ValueError("--seeds must contain at least one integer")
    if len(seed_list) != len(set(seed_list)):
        raise ValueError("--seeds must not contain duplicates")
    if any(seed < 0 for seed in seed_list):
        raise ValueError("--seeds values must be non-negative")
    for name, value in (
        ("--adam-epochs", args.adam_epochs),
        ("--pso-epochs", args.pso_epochs),
        ("--particles", args.particles),
        ("--fitness-size", args.fitness_size),
        ("--batch-size", args.batch_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        raise ValueError("--lr must be a positive finite number")

    device = resolve_execution_device(args.device)

    json_path = Path(args.json_path)
    csv_path = Path(args.csv_path)
    figure_path = Path(args.figure_path)

    print(f"=== Starting MNIST Deep Accuracy Study (Protocol v{DEEP_ACCURACY_PROTOCOL_VERSION}) ===")
    print(f"Device: {device} | Seeds: {seed_list} | Adam Epochs: {args.adam_epochs} | PSO Epochs: {args.pso_epochs}")
    print(f"Particles: {args.particles} | Fitness Size: {args.fitness_size} | Batch Size: {args.batch_size} | LR: {args.lr}")

    # Prepare data
    x_train, y_train, x_test, y_test, data_fp, norm_provenance = prepare_deep_accuracy_mnist_data()
    print(f"Data Loaded: Train {x_train.shape[0]} / Test {x_test.shape[0]} | Fingerprint: {data_fp[:12]}...")

    hardware_prov = get_hardware_provenance(device)
    all_csv_records: List[Dict[str, Any]] = []
    architecture_lane_runs: List[Dict[str, Any]] = []
    optimizer_lane_runs: List[Dict[str, Any]] = []

    # Map architecture factories
    arch_factories = {
        "raw_linear": ("Raw Linear (784->10)", make_raw_linear),
        "raw_mlp": ("Raw MLP (784->128->64->10)", make_raw_mlp),
        "compact_cnn": ("Compact CNN (9,098 params)", make_compact_cnn),
    }

    try:
        # ----------------------------------------------------
        # 1. Architecture Lane
        # ----------------------------------------------------
        compact_cnn_adam_cache: Dict[int, Dict[str, Any]] = {}

        if args.lanes in ("all", "architectures"):
            print("\n--- Running Architecture Lane (Full-Data Adam) ---")
            for arch_key, (arch_name, factory) in arch_factories.items():
                for s in seed_list:
                    # Construct base model and record initial fingerprint
                    model = factory(seed=s)
                    p_count = count_parameters(model)
                    init_fp = compute_model_fingerprint(model)

                    init_loss, init_acc = evaluate_model_on_test(model, x_test, y_test, device)

                    print(f"[Arch: {arch_key} | Seed: {s}] Params: {p_count} | Init Acc: {init_acc*100:.2f}% | Training Adam...")

                    history, final_loss, final_acc, elapsed = train_adam_routine(
                        model=model,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        epochs=args.adam_epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        seed=s,
                        device=device,
                    )

                    run_record = {
                        "lane": "architecture",
                        "profile_or_arch": arch_key,
                        "seed": s,
                        "model_name": arch_name,
                        "param_count": p_count,
                        "initial_test_acc": round(init_acc, 6),
                        "final_test_acc": round(final_acc, 6),
                        "final_test_loss": round(final_loss, 6),
                        "subset_fitness_acc": None,
                        "subset_fitness_loss": None,
                        "pso_epochs": 0,
                        "adam_epochs": args.adam_epochs,
                        "elapsed_sec": round(elapsed, 4),
                        "model_fingerprint": init_fp,
                        "data_fingerprint": data_fp,
                        "epoch_history": history,
                    }
                    architecture_lane_runs.append(run_record)
                    all_csv_records.append(run_record)

                    if arch_key == "compact_cnn":
                        compact_cnn_adam_cache[s] = run_record

                    print(f"  -> Final Test Acc: {final_acc*100:.2f}% | Loss: {final_loss:.4f} | Time: {elapsed:.2f}s")

        # ----------------------------------------------------
        # 2. Optimizer Lane (Compact CNN)
        # ----------------------------------------------------
        if args.lanes in ("all", "optimizers"):
            print("\n--- Running Optimizer Lane (Compact CNN) ---")
            profiles = ["adam_only", "pso_only", "hybrid"]

            for prof in profiles:
                for s in seed_list:
                    # Construct Compact CNN with seed s to ensure same base initial state
                    model_base = make_compact_cnn(seed=s)
                    p_count = count_parameters(model_base)
                    init_fp = compute_model_fingerprint(model_base)
                    init_loss, init_acc = evaluate_model_on_test(model_base, x_test, y_test, device)

                    if prof == "adam_only":
                        if s in compact_cnn_adam_cache:
                            # Explicitly reuse record from Architecture Lane
                            cached = compact_cnn_adam_cache[s]
                            run_record = {
                                "lane": "optimizer",
                                "profile_or_arch": "adam_only",
                                "seed": s,
                                "model_name": "Compact CNN (Adam-Only)",
                                "param_count": p_count,
                                "initial_test_acc": cached["initial_test_acc"],
                                "final_test_acc": cached["final_test_acc"],
                                "final_test_loss": cached["final_test_loss"],
                                "subset_fitness_acc": None,
                                "subset_fitness_loss": None,
                                "pso_epochs": 0,
                                "adam_epochs": args.adam_epochs,
                                "elapsed_sec": cached["elapsed_sec"],
                                "model_fingerprint": init_fp,
                                "data_fingerprint": data_fp,
                                "reused_from_architecture_lane": True,
                                "epoch_history": cached["epoch_history"],
                            }
                            print(f"[Opt: adam_only | Seed: {s}] Reused from Architecture Lane | Final Acc: {cached['final_test_acc']*100:.2f}%")
                        else:
                            print(f"[Opt: adam_only | Seed: {s}] Training Adam...")
                            history, final_loss, final_acc, elapsed = train_adam_routine(
                                model=model_base,
                                x_train=x_train,
                                y_train=y_train,
                                x_test=x_test,
                                y_test=y_test,
                                epochs=args.adam_epochs,
                                batch_size=args.batch_size,
                                lr=args.lr,
                                seed=s,
                                device=device,
                            )
                            run_record = {
                                "lane": "optimizer",
                                "profile_or_arch": "adam_only",
                                "seed": s,
                                "model_name": "Compact CNN (Adam-Only)",
                                "param_count": p_count,
                                "initial_test_acc": round(init_acc, 6),
                                "final_test_acc": round(final_acc, 6),
                                "final_test_loss": round(final_loss, 6),
                                "subset_fitness_acc": None,
                                "subset_fitness_loss": None,
                                "pso_epochs": 0,
                                "adam_epochs": args.adam_epochs,
                                "elapsed_sec": round(elapsed, 4),
                                "model_fingerprint": init_fp,
                                "data_fingerprint": data_fp,
                                "reused_from_architecture_lane": False,
                                "epoch_history": history,
                            }
                            print(f"  -> Final Test Acc: {final_acc*100:.2f}% | Loss: {final_loss:.4f} | Time: {elapsed:.2f}s")

                        optimizer_lane_runs.append(run_record)
                        all_csv_records.append(run_record)

                    elif prof == "pso_only":
                        print(
                            f"[Opt: pso_only | Seed: {s}] Running PSO Adaptive Moment "
                            f"on {args.fitness_size:,} fixed-subset samples..."
                        )
                        best_model, fitness_score, pso_test_loss, pso_test_acc, pso_meta = run_pso_routine(
                            model=model_base,
                            x_train=x_train,
                            y_train=y_train,
                            x_test=x_test,
                            y_test=y_test,
                            pso_epochs=args.pso_epochs,
                            n_particles=args.particles,
                            fitness_size=args.fitness_size,
                            seed=s,
                            device=device,
                        )

                        run_record = {
                            "lane": "optimizer",
                            "profile_or_arch": "pso_only",
                            "seed": s,
                            "model_name": "Compact CNN (PSO-Only)",
                            "param_count": p_count,
                            "initial_test_acc": round(init_acc, 6),
                            "final_test_acc": round(pso_test_acc, 6),
                            "final_test_loss": round(pso_test_loss, 6),
                            "subset_fitness_acc": fitness_score["subset_acc"],
                            "subset_fitness_loss": fitness_score["subset_loss"],
                            "pso_epochs": args.pso_epochs,
                            "adam_epochs": 0,
                            "elapsed_sec": pso_meta["elapsed_sec"],
                            "model_fingerprint": init_fp,
                            "data_fingerprint": data_fp,
                            "pso_metadata": pso_meta,
                        }
                        optimizer_lane_runs.append(run_record)
                        all_csv_records.append(run_record)

                        print(f"  -> Fitness Subset Acc: {fitness_score['subset_acc']*100:.2f}% | Full Test Acc: {pso_test_acc*100:.2f}% | Time: {pso_meta['elapsed_sec']:.2f}s")

                    elif prof == "hybrid":
                        print(f"[Opt: hybrid | Seed: {s}] Running Hybrid (PSO Warm Start + Adam Fine-Tuning)...")
                        # 1. PSO Warm Start
                        t_hyb_start = time.time()
                        pso_best_model, fitness_score, pso_test_loss, pso_test_acc, pso_meta = run_pso_routine(
                            model=model_base,
                            x_train=x_train,
                            y_train=y_train,
                            x_test=x_test,
                            y_test=y_test,
                            pso_epochs=args.pso_epochs,
                            n_particles=args.particles,
                            fitness_size=args.fitness_size,
                            seed=s,
                            device=device,
                        )

                        # 2. Continue with full-data Adam
                        post_adam_history, final_test_loss, final_test_acc, adam_elapsed = train_adam_routine(
                            model=pso_best_model,
                            x_train=x_train,
                            y_train=y_train,
                            x_test=x_test,
                            y_test=y_test,
                            epochs=args.adam_epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            seed=s,
                            device=device,
                        )
                        hyb_total_elapsed = time.time() - t_hyb_start

                        run_record = {
                            "lane": "optimizer",
                            "profile_or_arch": "hybrid",
                            "seed": s,
                            "model_name": "Compact CNN (Hybrid)",
                            "param_count": p_count,
                            "initial_test_acc": round(init_acc, 6),
                            "final_test_acc": round(final_test_acc, 6),
                            "final_test_loss": round(final_test_loss, 6),
                            "subset_fitness_acc": fitness_score["subset_acc"],
                            "subset_fitness_loss": fitness_score["subset_loss"],
                            "pso_epochs": args.pso_epochs,
                            "adam_epochs": args.adam_epochs,
                            "elapsed_sec": round(hyb_total_elapsed, 4),
                            "model_fingerprint": init_fp,
                            "data_fingerprint": data_fp,
                            "post_pso_test_acc": round(pso_test_acc, 6),
                            "post_pso_test_loss": round(pso_test_loss, 6),
                            "post_adam_history": post_adam_history,
                            "efficiency_label": "HYBRID_GETS_EXTRA_WORK_UNFAIR_EFFICIENCY_COMPARISON",
                        }
                        optimizer_lane_runs.append(run_record)
                        all_csv_records.append(run_record)

                        print(f"  -> Post-PSO Test Acc: {pso_test_acc*100:.2f}% | Final Hybrid Test Acc: {final_test_acc*100:.2f}% | Time: {hyb_total_elapsed:.2f}s")

        # Compute summary statistics
        arch_summary: Dict[str, Dict[str, float]] = {}
        for arch_key in arch_factories.keys():
            vals = [r["final_test_acc"] for r in architecture_lane_runs if r["profile_or_arch"] == arch_key]
            if vals:
                arch_summary[arch_key] = calc_stats(vals)

        opt_summary: Dict[str, Dict[str, float]] = {}
        for prof in ["adam_only", "pso_only", "hybrid"]:
            vals = [r["final_test_acc"] for r in optimizer_lane_runs if r["profile_or_arch"] == prof]
            if vals:
                opt_summary[prof] = calc_stats(vals)

        # Structure complete JSON output
        result_payload = {
            "protocol_version": DEEP_ACCURACY_PROTOCOL_VERSION,
            "pso_version": pso_version,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "completed": True,
            "error": None,
            "hardware_provenance": hardware_prov,
            "data_provenance": norm_provenance,
            "data_fingerprint": data_fp,
            "configuration": {
                "seeds": seed_list,
                "adam_epochs": args.adam_epochs,
                "pso_epochs": args.pso_epochs,
                "particles": args.particles,
                "fitness_size": args.fitness_size,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "device": str(device),
            },
            "summaries": {
                "architecture_lane": arch_summary,
                "optimizer_lane": opt_summary,
            },
            "architecture_lane_runs": architecture_lane_runs,
            "optimizer_lane_runs": optimizer_lane_runs,
        }

        # Save JSON output atomically
        save_json_atomic(result_payload, json_path)
        print(f"\nSaved JSON results to {json_path}")

        # Save CSV records and render plot only after all runs succeed
        save_csv_records(all_csv_records, csv_path)
        print(f"Saved CSV records to {csv_path}")

        render_plots(arch_summary, opt_summary, figure_path)
        print(f"Saved summary figure to {figure_path}")

        # Print final summary table
        print("\n========================================================")
        print("                 FINAL SUMMARY TABLE                    ")
        print("========================================================")
        if arch_summary:
            print("Architecture Lane (Full-Data Adam):")
            for arch_key, stats in arch_summary.items():
                print(f"  - {arch_key:15s}: Mean Acc = {stats['mean']*100:6.2f}% ± {stats['std']*100:5.2f}% (Median: {stats['median']*100:.2f}%)")
        if opt_summary:
            print("\nOptimizer Lane (Compact CNN):")
            for prof, stats in opt_summary.items():
                print(f"  - {prof:15s}: Mean Acc = {stats['mean']*100:6.2f}% ± {stats['std']*100:5.2f}% (Median: {stats['median']*100:.2f}%)")
        print("========================================================\n")

    except Exception as e:
        error_payload = {
            "protocol_version": DEEP_ACCURACY_PROTOCOL_VERSION,
            "pso_version": pso_version,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "completed": False,
            "error": str(e),
            "hardware_provenance": hardware_prov,
            "configuration": {
                "seeds": seed_list,
                "adam_epochs": args.adam_epochs,
                "pso_epochs": args.pso_epochs,
                "particles": args.particles,
                "fitness_size": args.fitness_size,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "device": str(device),
            },
            "architecture_lane_runs": architecture_lane_runs,
            "optimizer_lane_runs": optimizer_lane_runs,
        }
        save_json_atomic(error_payload, json_path)
        print(f"\n[ERROR] Study failed: {e}")
        print(f"Saved failure audit record to {json_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
