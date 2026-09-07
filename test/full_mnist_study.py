"""
Full-MNIST 60k/10k Particle Swarm Optimization Trajectory Analysis

Evaluates the selected Adaptive Moment PSO configuration on full official MNIST (60,000 train / 10,000 test)
across 120 particles for 240 continuous epochs, scoring all 60,000 training examples for every particle
at every epoch.

Predeclared Diagnostic Criteria:
1. Post-80 Training Convergence: Mean training loss falls >= 1% from epoch 80 to epoch 240.
2. Epoch 240 Test Gain: Mean test accuracy at epoch 240 rises >= 1 percentage point vs epoch 80.
3. Late Plateau (200->240): Training loss improvement < 1% AND absolute test accuracy change < 0.5 percentage points.
"""

import argparse
import csv
import datetime
import json
import math
import sys
import tempfile
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
    extract_plugin_metadata,
    compute_data_fingerprint,
    compute_model_fingerprint,
    get_hardware_provenance,
    make_mnist_model,
    resolve_execution_device,
    save_json_atomic,
    sync_device,
)
from pso import Optimizer, __version__ as pso_version
from reproduce_scaling import validate_and_load_baseline

FULL_MNIST_PROTOCOL_VERSION = "1.0.0"


def prepare_full_mnist_data() -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    str,
    Dict[str, Any],
]:
    """
    Load official torchvision MNIST full train (60,000) and test (10,000).
    Normalize pixels, flatten to 784, fit PCA(n_components=32, whiten=True, random_state=42)
    on train only, transform test. Validate 60,000/10,000 sample counts and label range [0, 9].
    """
    from torchvision.datasets import MNIST
    from sklearn.decomposition import PCA

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

    y_train_60000 = train_dataset.targets.long()
    y_test_10000 = test_dataset.targets.long()

    min_tr_lbl, max_tr_lbl = int(y_train_60000.min()), int(y_train_60000.max())
    min_te_lbl, max_te_lbl = int(y_test_10000.min()), int(y_test_10000.max())

    if min_tr_lbl != 0 or max_tr_lbl != 9:
        raise ValueError(f"Train label range must be [0, 9]; got [{min_tr_lbl}, {max_tr_lbl}]")
    if min_te_lbl != 0 or max_te_lbl != 9:
        raise ValueError(f"Test label range must be [0, 9]; got [{min_te_lbl}, {max_te_lbl}]")

    x_train_raw = (train_dataset.data.float() / 255.0).reshape(60000, -1).numpy()
    x_test_raw = (test_dataset.data.float() / 255.0).reshape(10000, -1).numpy()

    pca = PCA(n_components=32, whiten=True, random_state=42)
    x_full_tr = torch.tensor(pca.fit_transform(x_train_raw), dtype=torch.float32)
    x_full_test = torch.tensor(pca.transform(x_test_raw), dtype=torch.float32)

    data_fp = compute_data_fingerprint(x_full_tr, x_full_test, y_train_60000, y_test_10000)
    pca_provenance = {
        "n_components": 32,
        "whiten": True,
        "random_state": 42,
        "fit_scope": "official_train_split_60000_only",
        "train_samples_fit": 60000,
        "test_samples_transformed": 10000,
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return (
        x_full_tr,
        y_train_60000,
        x_full_test,
        y_test_10000,
        data_fp,
        pca_provenance,
    )


def render_trajectory_plot(
    checkpoint_stats: Dict[int, Dict[str, Any]],
    subset_comparison: Dict[str, Any],
    figure_path: Path,
):
    """
    Render a readable two-panel mean +/- SD trajectory plot for training loss and test accuracy,
    with optional dashed subset-study mean comparison and epoch 80 marker.
    """
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    eps = sorted(list(checkpoint_stats.keys()))
    tr_loss_mean = [checkpoint_stats[ep]["train_loss"]["mean"] for ep in eps]
    tr_loss_std = [checkpoint_stats[ep]["train_loss"]["std"] for ep in eps]
    te_acc_mean = [checkpoint_stats[ep]["test_acc"]["mean"] for ep in eps]
    te_acc_std = [checkpoint_stats[ep]["test_acc"]["std"] for ep in eps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: Training Loss
    ax1.plot(eps, tr_loss_mean, "o-", color="tab:blue", linewidth=2, label="Full MNIST Train Loss (Mean)")
    ax1.fill_between(
        eps,
        np.array(tr_loss_mean) - np.array(tr_loss_std),
        np.array(tr_loss_mean) + np.array(tr_loss_std),
        color="tab:blue",
        alpha=0.2,
        label="±1 SD",
    )

    if "matching_epoch_deltas" in subset_comparison:
        sub_deltas = subset_comparison["matching_epoch_deltas"]
        sub_eps = sorted([ep for ep in eps if str(ep) in sub_deltas or ep in sub_deltas])
        if sub_eps:
            sub_tr_loss = [sub_deltas.get(str(ep), sub_deltas.get(ep, {}))["subset_study_train_loss_mean"] for ep in sub_eps]
            ax1.plot(sub_eps, sub_tr_loss, "--", color="gray", alpha=0.8, label="Subset Study (2k sample) Train Loss")

    if 80 in eps:
        ax1.axvline(80, color="red", linestyle=":", label="Epoch 80 Marker")

    ax1.set_ylabel("Training Loss")
    ax1.set_title("Full MNIST (60,000 Train / 10,000 Test) PSO Trajectory (120 Particles, AM)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Panel 2: Test Accuracy
    ax2.plot(eps, te_acc_mean, "s-", color="tab:green", linewidth=2, label="Full MNIST Test Accuracy (Mean)")
    ax2.fill_between(
        eps,
        np.array(te_acc_mean) - np.array(te_acc_std),
        np.array(te_acc_mean) + np.array(te_acc_std),
        color="tab:green",
        alpha=0.2,
        label="±1 SD",
    )

    if "matching_epoch_deltas" in subset_comparison:
        sub_deltas = subset_comparison["matching_epoch_deltas"]
        sub_eps = sorted([ep for ep in eps if str(ep) in sub_deltas or ep in sub_deltas])
        if sub_eps:
            sub_te_acc = [sub_deltas.get(str(ep), sub_deltas.get(ep, {}))["subset_study_test_acc_mean"] for ep in sub_eps]
            ax2.plot(sub_eps, sub_te_acc, "--", color="gray", alpha=0.8, label="Subset Study (2k sample) Test Acc")

    if 80 in eps:
        ax2.axvline(80, color="red", linestyle=":", label="Epoch 80 Marker")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def run_full_mnist_study(
    baseline_path: Path,
    subset_study_path: Path,
    output_json_path: Path,
    output_csv_path: Path,
    figure_path: Path,
    device_str: str = "auto",
    epochs: int = 240,
    seeds: List[int] = None,
) -> bool:
    if seeds is None:
        seeds = [71, 72, 73, 74, 75]
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if epochs % 20 != 0:
        raise ValueError("epochs must be a multiple of 20 so every final checkpoint exists")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a non-empty list of unique integers")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("every seed must be a non-negative integer")


    dev_input = None if device_str == "auto" else device_str
    device = resolve_execution_device(dev_input)
    hw_provenance = get_hardware_provenance(device)

    # 1. Validate baseline JSON & load winner config
    baseline_data, _baseline_records, winner_cfg, _expected_fp = validate_and_load_baseline(
        baseline_path
    )

    if baseline_data.get("device") != device.type:
        raise ValueError(
            f"Baseline device mismatch: baseline requires {baseline_data.get('device')!r}, got {device.type!r}"
        )
    if baseline_data.get("pso_version") != pso_version:
        raise ValueError(
            f"PSO version mismatch: baseline requires {baseline_data.get('pso_version')!r}, got {pso_version!r}"
        )
    if baseline_data.get("torch_version") != torch.__version__:
        raise ValueError(
            f"Torch version mismatch: baseline requires {baseline_data.get('torch_version')!r}, got {torch.__version__!r}"
        )

    # 2. Prepare Full MNIST PCA Data (60k train / 10k test)
    (
        x_full_tr,
        y_train_60000,
        x_full_test,
        y_test_10000,
        data_fp,
        pca_provenance,
    ) = prepare_full_mnist_data()

    opt_kwargs = winner_cfg.to_optimizer_kwargs(quick=False)
    opt_kwargs["evaluation"] = "full"
    opt_kwargs.pop("fitness_size", None)

    n_particles = 120
    target_epochs = epochs
    batch_size = 60000
    checkpoint_interval = 20

    ckpt_epochs = [ep for ep in range(checkpoint_interval, target_epochs + 1, checkpoint_interval)]
    if not ckpt_epochs or ckpt_epochs[-1] != target_epochs:
        if target_epochs not in ckpt_epochs:
            ckpt_epochs.append(target_epochs)
    ckpt_epochs = sorted(list(set(ckpt_epochs)))

    runs: List[Dict[str, Any]] = []
    flat_csv_rows: List[Dict[str, Any]] = []
    plugin_meta: Dict[str, Any] | None = None

    # 3. Seed Runs
    for seed in sorted(seeds):
        # Warmup Phase (2 epochs full eval)
        warmup_model = make_mnist_model(seed=seed)
        warmup_loss = nn.CrossEntropyLoss()
        warmup_opt = Optimizer(
            model=warmup_model,
            loss=warmup_loss,
            task="multiclass",
            n_particles=n_particles,
            seed=seed,
            device=device,
            **opt_kwargs,
        )
        warmup_opt.fit(
            x_full_tr,
            y_train_60000,
            epochs=2,
            batch_size=batch_size,
            renewal="loss",
        )
        sync_device(device)
        del warmup_opt, warmup_model, warmup_loss

        # Timed Continuous Trajectory
        model = make_mnist_model(seed=seed)
        model_fp = compute_model_fingerprint(model)
        loss_inst = nn.CrossEntropyLoss()
        opt = Optimizer(
            model=model,
            loss=loss_inst,
            task="multiclass",
            n_particles=n_particles,
            seed=seed,
            device=device,
            **opt_kwargs,
        )
        current_plugin_meta = extract_plugin_metadata(opt)
        if plugin_meta is None:
            plugin_meta = current_plugin_meta
        elif current_plugin_meta != plugin_meta:
            raise RuntimeError("Resolved plugin metadata changed across seeds")


        with tempfile.TemporaryDirectory() as temp_dir_str:
            output_dir = Path(temp_dir_str)
            sync_device(device)
            t0 = time.perf_counter()
            train_loss_final, train_acc_final, train_mse_final = opt.fit(
                x_full_tr,
                y_train_60000,
                epochs=target_epochs,
                batch_size=batch_size,
                renewal="loss",
                output_dir=output_dir,
                log_format="csv",
                checkpoint_interval=checkpoint_interval,
            )
            sync_device(device)
            t1 = time.perf_counter()
            fit_time_sec = t1 - t0

            if not (math.isfinite(train_loss_final) and math.isfinite(train_acc_final) and math.isfinite(train_mse_final)):
                raise RuntimeError(f"Seed {seed} final metrics non-finite: loss={train_loss_final}, acc={train_acc_final}")

            history_csv_path = output_dir / "history.csv"
            epoch_history = []
            if history_csv_path.exists():
                with open(history_csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        epoch_history.append({
                            "epoch": int(row["epoch"]),
                            "loss": float(row["loss"]),
                            "accuracy": float(row["accuracy"]),
                            "mse": float(row["mse"]),
                        })

            prev_best_loss = float("inf")
            improvement_count = 0
            last_improvement_epoch = 1
            for row in epoch_history:
                ep_num = row["epoch"]
                l_val = row["loss"]
                if l_val < prev_best_loss:
                    improvement_count += 1
                    last_improvement_epoch = ep_num
                    prev_best_loss = l_val

            checkpoints: List[Dict[str, Any]] = []
            ckpt_dir = output_dir / "checkpoints"
            for ep in ckpt_epochs:
                ckpt_path = ckpt_dir / f"epoch-{ep}.pt"
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Missing checkpoint file: {ckpt_path}")
                payload = torch.load(ckpt_path, map_location=device, weights_only=True)
                ckpt_tr_loss, ckpt_tr_acc, ckpt_tr_mse = payload["score"]
                if not (math.isfinite(ckpt_tr_loss) and math.isfinite(ckpt_tr_acc) and math.isfinite(ckpt_tr_mse)):
                    raise RuntimeError(f"Seed {seed} epoch {ep} score non-finite: {payload['score']}")

                opt.eval_model.load_state_dict(payload["model_state_dict"])
                opt._global_best_weights = opt.codec.encode(opt.eval_model)
                test_loss, test_acc, test_mse = opt.evaluate(x_full_test, y_test_10000)

                if not (math.isfinite(test_loss) and math.isfinite(test_acc) and math.isfinite(test_mse)):
                    raise RuntimeError(f"Seed {seed} epoch {ep} test metrics non-finite: loss={test_loss}, acc={test_acc}")

                ckpt_record = {
                    "epoch": ep,
                    "train_loss": float(ckpt_tr_loss),
                    "train_acc": float(ckpt_tr_acc),
                    "train_mse": float(ckpt_tr_mse),
                    "test_loss": float(test_loss),
                    "test_acc": float(test_acc),
                    "test_mse": float(test_mse),
                }
                checkpoints.append(ckpt_record)

                flat_csv_rows.append({
                    "seed": seed,
                    "epoch": ep,
                    "train_loss": float(ckpt_tr_loss),
                    "train_acc": float(ckpt_tr_acc),
                    "train_mse": float(ckpt_tr_mse),
                    "test_loss": float(test_loss),
                    "test_acc": float(test_acc),
                    "test_mse": float(test_mse),
                    "fit_time_sec": round(fit_time_sec, 4),
                })

            runs.append({
                "seed": seed,
                "model_fingerprint": model_fp,
                "fit_time_sec": round(fit_time_sec, 4),
                "improvement_count": improvement_count,
                "last_improvement_epoch": last_improvement_epoch,
                "checkpoints": checkpoints,
                "completed": True,
                "error": None,
                "plugins": current_plugin_meta,
                "epoch_history": epoch_history,
            })

    # 4. Aggregations & Statistics
    checkpoint_stats: Dict[int, Dict[str, Any]] = {}
    for ep in ckpt_epochs:
        ep_train_losses = [next(c["train_loss"] for c in r["checkpoints"] if c["epoch"] == ep) for r in runs]
        ep_train_accs = [next(c["train_acc"] for c in r["checkpoints"] if c["epoch"] == ep) for r in runs]
        ep_train_mses = [next(c["train_mse"] for c in r["checkpoints"] if c["epoch"] == ep) for r in runs]

        ep_test_losses = [next(c["test_loss"] for c in r["checkpoints"] if c["epoch"] == ep) for r in runs]
        ep_test_accs = [next(c["test_acc"] for c in r["checkpoints"] if c["epoch"] == ep) for r in runs]
        ep_test_mses = [next(c["test_mse"] for c in r["checkpoints"] if c["epoch"] == ep) for r in runs]

        checkpoint_stats[ep] = {
            "train_loss": calc_stats(ep_train_losses),
            "train_acc": calc_stats(ep_train_accs),
            "train_mse": calc_stats(ep_train_mses),
            "test_loss": calc_stats(ep_test_losses),
            "test_acc": calc_stats(ep_test_accs),
            "test_mse": calc_stats(ep_test_mses),
        }

    # Endpoint paired deltas (80->240 and 200->240 if available)
    paired_deltas: Dict[str, Any] = {}
    paired_endpoint_deltas_by_seed: List[Dict[str, Any]] = []

    has_80 = 80 in checkpoint_stats
    has_200 = 200 in checkpoint_stats
    has_240 = 240 in checkpoint_stats

    if has_80 and has_240:
        deltas_80_to_240_train_rel = []
        deltas_80_to_240_test_acc = []
        for r in runs:
            c80 = next(c for c in r["checkpoints"] if c["epoch"] == 80)
            c240 = next(c for c in r["checkpoints"] if c["epoch"] == 240)
            tl80, tl240 = c80["train_loss"], c240["train_loss"]
            ta80, ta240 = c80["test_acc"], c240["test_acc"]

            rel_red = (tl80 - tl240) / tl80 if tl80 > 0 else 0.0
            acc_delta = ta240 - ta80
            deltas_80_to_240_train_rel.append(rel_red)
            deltas_80_to_240_test_acc.append(acc_delta)

            paired_rec = {
                "seed": r["seed"],
                "train_loss_relative_reduction_80_to_240": rel_red,
                "test_accuracy_delta_80_to_240": acc_delta,
            }
            if has_200:
                c200 = next(c for c in r["checkpoints"] if c["epoch"] == 200)
                tl200, ta200 = c200["train_loss"], c200["test_acc"]
                rel_red_200 = (tl200 - tl240) / tl200 if tl200 > 0 else 0.0
                acc_delta_200 = ta240 - ta200
                paired_rec["train_loss_relative_reduction_200_to_240"] = rel_red_200
                paired_rec["test_accuracy_delta_200_to_240"] = acc_delta_200
            paired_endpoint_deltas_by_seed.append(paired_rec)

        paired_deltas["80_to_240"] = {
            "train_loss_rel_reduction": calc_stats(deltas_80_to_240_train_rel),
            "test_acc_delta": calc_stats(deltas_80_to_240_test_acc),
        }

    if has_200 and has_240:
        deltas_200_to_240_train_rel = []
        deltas_200_to_240_test_acc = []
        deltas_200_to_240_test_acc_abs = []
        for r in runs:
            c200 = next(c for c in r["checkpoints"] if c["epoch"] == 200)
            c240 = next(c for c in r["checkpoints"] if c["epoch"] == 240)
            tl200, tl240 = c200["train_loss"], c240["train_loss"]
            ta200, ta240 = c200["test_acc"], c240["test_acc"]
            rel_red_200 = (tl200 - tl240) / tl200 if tl200 > 0 else 0.0
            acc_delta_200 = ta240 - ta200
            acc_abs_200 = abs(ta240 - ta200)
            deltas_200_to_240_train_rel.append(rel_red_200)
            deltas_200_to_240_test_acc.append(acc_delta_200)
            deltas_200_to_240_test_acc_abs.append(acc_abs_200)

        paired_deltas["200_to_240"] = {
            "train_loss_rel_reduction": calc_stats(deltas_200_to_240_train_rel),
            "test_acc_delta": calc_stats(deltas_200_to_240_test_acc),
            "test_acc_abs_change": calc_stats(deltas_200_to_240_test_acc_abs),
        }

    # Predeclared Diagnostics
    diagnostics = {}
    if has_80 and has_240:
        mean_tl80 = checkpoint_stats[80]["train_loss"]["mean"]
        mean_tl240 = checkpoint_stats[240]["train_loss"]["mean"]
        mean_ta80 = checkpoint_stats[80]["test_acc"]["mean"]
        mean_ta240 = checkpoint_stats[240]["test_acc"]["mean"]

        post80_train_improvement_pct = (mean_tl80 - mean_tl240) / mean_tl80 if mean_tl80 > 0 else 0.0
        epoch240_test_gain = mean_ta240 - mean_ta80

        diagnostics["post80_train_improvement_pct"] = round(post80_train_improvement_pct, 6)
        diagnostics["post80_train_improvement_passed"] = bool(post80_train_improvement_pct >= 0.01)

        diagnostics["epoch240_test_gain"] = round(epoch240_test_gain, 6)
        diagnostics["epoch240_test_gain_passed"] = bool(epoch240_test_gain >= 0.01)

        if has_200:
            mean_tl200 = checkpoint_stats[200]["train_loss"]["mean"]
            mean_ta200 = checkpoint_stats[200]["test_acc"]["mean"]
            late_rel_red = (mean_tl200 - mean_tl240) / mean_tl200 if mean_tl200 > 0 else 0.0
            late_acc_abs = abs(mean_ta240 - mean_ta200)

            is_late_plateau = bool(late_rel_red < 0.01 and late_acc_abs < 0.005)
            diagnostics["late_plateau_200_240"] = is_late_plateau
            diagnostics["late_plateau_train_loss_rel_reduction_200_240"] = round(late_rel_red, 6)
            diagnostics["late_plateau_test_acc_abs_change_200_240"] = round(late_acc_abs, 6)

    # 5. Descriptive Comparison vs Subset Study
    subset_comparison: Dict[str, Any] = {}
    if subset_study_path.exists():
        try:
            with open(subset_study_path, "r", encoding="utf-8") as f:
                sub_json_data = json.load(f)
            sub_ckpt_stats = sub_json_data.get("summary", {}).get("checkpoint_stats", {})
            matching_stats = {}
            for ep in ckpt_epochs:
                ep_str = str(ep)
                if ep_str in sub_ckpt_stats:
                    sub_e = sub_ckpt_stats[ep_str]
                    full_tr_l = checkpoint_stats[ep]["train_loss"]["mean"]
                    sub_tr_l = sub_e["train_loss"]["mean"]
                    full_te_a = checkpoint_stats[ep]["test_acc"]["mean"]
                    sub_te_a = sub_e["test_acc"]["mean"]

                    matching_stats[ep_str] = {
                        "full_mnist_train_loss_mean": full_tr_l,
                        "subset_study_train_loss_mean": sub_tr_l,
                        "train_loss_delta_full_minus_subset": round(full_tr_l - sub_tr_l, 6),
                        "full_mnist_test_acc_mean": full_te_a,
                        "subset_study_test_acc_mean": sub_te_a,
                        "test_acc_delta_full_minus_subset": round(full_te_a - sub_te_a, 6),
                    }

            subset_comparison = {
                "subset_study_path": str(subset_study_path),
                "disclaimer": (
                    "The full and subset studies optimize different training objectives and the evaluation "
                    "plugin changes RNG consumption (all 60,000 train samples versus a fixed 2,000-sample "
                    "fitness subset). Matching-seed and matching-epoch comparisons are descriptive, not an "
                    "exact paired causal isolation."
                ),
                "matching_epoch_deltas": matching_stats,
            }
        except Exception as err:
            subset_comparison = {"error": f"Failed to parse subset study JSON: {err}"}

    # 6. Save Artifacts
    # JSON output
    out_dict = {
        "full_mnist_protocol_version": FULL_MNIST_PROTOCOL_VERSION,
        "pso_version": pso_version,
        "torch_version": torch.__version__,
        "hardware": hw_provenance,
        "device": device.type,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_path": str(baseline_path),
        "subset_study_path": str(subset_study_path),
        "sample_counts": {
            "train_samples": 60000,
            "test_samples": 10000,
        },
        "pca_provenance": pca_provenance,
        "data_fingerprint": data_fp,
        "fitness_evaluation_contract": {
            "selector": "full",
            "fitness_size": None,
            "train_samples_per_particle_per_epoch": 60000,
            "particle_evaluations": n_particles * target_epochs * len(seeds),
            "particle_sample_evaluations": (
                n_particles * target_epochs * len(seeds) * 60000
            ),
            "test_samples_per_checkpoint": 10000,
        },
        "candidate_label": winner_cfg.candidate_label,
        "config": {
            **opt_kwargs,
            "n_particles": n_particles,
            "epochs": target_epochs,
            "batch_size": batch_size,
            "renewal": "loss",
            "checkpoint_interval": checkpoint_interval,
        },
        "plugins": plugin_meta,
        "timing_scope": "fit_only_after_full-evaluation_two-epoch_warmup",
        "diagnostics": diagnostics,
        "descriptive_subset_comparison": subset_comparison,
        "summary": {
            "epochs": ckpt_epochs,
            "checkpoint_stats": {str(k): v for k, v in checkpoint_stats.items()},
            "paired_deltas": paired_deltas,
            "paired_endpoint_deltas_by_seed": paired_endpoint_deltas_by_seed,
        },
        "runs": runs,
        "completed": True,
        "valid": True,
        "error": None,
    }
    save_json_atomic(out_dict, output_json_path)

    # CSV output
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "epoch",
        "train_loss",
        "train_acc",
        "train_mse",
        "test_loss",
        "test_acc",
        "test_mse",
        "fit_time_sec",
    ]
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in flat_csv_rows:
            writer.writerow(r)

    # Plot output
    render_trajectory_plot(checkpoint_stats, subset_comparison, figure_path)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Full-MNIST 60k/10k Particle Swarm Optimization Trajectory Analysis"
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=Path("benchmark_results/pso_v4_tuning.json"),
        help="Path to baseline tuning JSON (default: benchmark_results/pso_v4_tuning.json)",
    )
    parser.add_argument(
        "--subset-study-json",
        type=Path,
        default=Path("benchmark_results/pso_v4_epoch_convergence.json"),
        help="Path to subset epoch convergence JSON (default: benchmark_results/pso_v4_epoch_convergence.json)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmark_results/pso_v4_full_mnist.json"),
        help="Path to output JSON (default: benchmark_results/pso_v4_full_mnist.json)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_results/pso_v4_full_mnist.csv"),
        help="Path to output CSV (default: benchmark_results/pso_v4_full_mnist.csv)",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("history_plt/pso_v4_full_mnist.png"),
        help="Path to output plot figure PNG (default: history_plt/pso_v4_full_mnist.png)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device: auto, mps, cpu, cuda (default: auto)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=240,
        help="Number of PSO training epochs (default: 240)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="71,72,73,74,75",
        help="Comma-separated random seeds (default: 71,72,73,74,75)",
    )

    args = parser.parse_args()
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    run_full_mnist_study(
        baseline_path=args.baseline_json,
        subset_study_path=args.subset_study_json,
        output_json_path=args.output_json,
        output_csv_path=args.output_csv,
        figure_path=args.figure,
        device_str=args.device,
        epochs=args.epochs,
        seeds=seed_list,
    )


if __name__ == "__main__":
    main()
