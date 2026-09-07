"""
Adaptive Moment 120-Particle x 240-Epoch MNIST Convergence & Trajectory Analysis

Evaluates whether the published 120-particle x 80-epoch MNIST PCA32 Adaptive Moment run
benefits from continued optimization up to 240 epochs or enters a generalization plateau.

Predeclared Contract Criteria:
1. Exact Replay Verification at Epoch 80 (seeds 71-75):
   Per-seed test accuracy absolute delta vs baseline <= 0.005 (0.5%p).
2. Primary Diagnostic Endpoints: Epochs 80, 120, 160, 200, 240.
3. Classifications:
   - Training Still Improving: Mean global-best fitness loss falls >= 1% from epoch 80 to 240.
   - Meaningful Held-Out Gain: Mean test accuracy at epoch 240 rises >= 1 percentage point vs epoch 80.
   - Overfitting Signal: Training loss improves but epoch 240 test accuracy falls >= 1 point.
   - Early Stagnation: Training loss improves < 1% and absolute test change remains < 1 point.
   - Generalization Plateau: Training loss improves >= 1% while absolute test gain remains < 1 point.
   - Late Plateau Diagnostic: 200->240 training-loss improvement < 1% AND absolute test-accuracy change < 0.5 points.
"""

import argparse
import csv
import datetime
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

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
    compute_model_fingerprint,
    get_hardware_provenance,
    make_mnist_model,
    resolve_execution_device,
    save_json_atomic,
    sync_device,
)
from pso import Optimizer, __version__ as pso_version
from reproduce_scaling import (
    REPLAY_SEEDS,
    REPLAY_TOLERANCE,
    prepare_full_pca_data,
    validate_and_load_baseline,
)
from tuning_suite import TUNING_PROTOCOL_VERSION

EPOCH_CONVERGENCE_PROTOCOL_VERSION = "1.0.0"
CHECKPOINT_EPOCHS = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
POST80_LOSS_REDUCTION_THRESHOLD = 0.01
TEST_ACCURACY_GAIN_THRESHOLD = 0.01
OVERFITTING_ACCURACY_DROP_THRESHOLD = -0.01
LATE_LOSS_REDUCTION_THRESHOLD = 0.01
LATE_ACCURACY_CHANGE_THRESHOLD = 0.005


def run_epoch_convergence_analysis(
    baseline_path: Path,
    output_json_path: Path,
    output_csv_path: Path,
    figure_path: Path,
    device_str: str | None = None,
) -> bool:
    device = resolve_execution_device(device_str)
    hw_provenance = get_hardware_provenance(device)

    # 1. Validate and load baseline
    baseline_data, baseline_records, winner_cfg, expected_fp = validate_and_load_baseline(
        baseline_path
    )
    if baseline_data.get("device") != device.type:
        raise ValueError(
            f"Exact trajectory extension requires baseline device "
            f"{baseline_data.get('device')!r}; got {device.type!r}."
        )
    if baseline_data.get("pso_version") != pso_version:
        raise ValueError(
            f"Exact trajectory extension requires pso version "
            f"{baseline_data.get('pso_version')!r}; got {pso_version!r}."
        )
    if baseline_data.get("torch_version") != torch.__version__:
        raise ValueError(
            f"Exact trajectory extension requires torch version "
            f"{baseline_data.get('torch_version')!r}; got {torch.__version__!r}."
        )

    base_rec_by_seed = {r["seed"]: r for r in baseline_records}

    # 2. Prepare full PCA dataset
    x_full_tr, y_train_3000, x_full_test, y_test_1000, data_fp = prepare_full_pca_data()
    if data_fp != expected_fp:
        raise ValueError(
            f"Data fingerprint mismatch: computed {data_fp}, baseline expected {expected_fp}"
        )

    opt_kwargs = winner_cfg.to_optimizer_kwargs(quick=False)
    n_particles = 120
    target_epochs = 240
    batch_size = 1000

    runs: List[Dict[str, Any]] = []
    flat_csv_rows: List[Dict[str, Any]] = []
    fidelity_passed = True
    seed_fidelity_deltas: Dict[int, float] = {}

    # 3. Process seeds 71-75
    for seed in sorted(REPLAY_SEEDS):
        base_rec = base_rec_by_seed[seed]
        expected_model_fp = base_rec["model_fingerprint"]
        baseline_test_acc = float(base_rec["test_acc"])

        # --- Untimed 2-Epoch Warmup Phase ---
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
            y_train_3000,
            epochs=2,
            batch_size=batch_size,
            renewal="loss",
        )
        sync_device(device)
        del warmup_opt, warmup_model, warmup_loss

        # --- Timed 240-Epoch Continuous Trajectory ---
        model = make_mnist_model(seed=seed)
        model_fp = compute_model_fingerprint(model)
        fp_match = (model_fp == expected_model_fp)
        if not fp_match:
            print(
                f"[WARNING] Seed {seed} model fingerprint mismatch: "
                f"got {model_fp}, expected {expected_model_fp}"
            )
            fidelity_passed = False

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

        with tempfile.TemporaryDirectory() as temp_dir_str:
            output_dir = Path(temp_dir_str)

            sync_device(device)
            t0 = time.perf_counter()
            train_loss_final, train_acc_final, train_mse_final = opt.fit(
                x_full_tr,
                y_train_3000,
                epochs=target_epochs,
                batch_size=batch_size,
                renewal="loss",
                output_dir=output_dir,
                log_format="csv",
                checkpoint_interval=20,
            )
            sync_device(device)
            t1 = time.perf_counter()
            fit_time_sec = t1 - t0

            # Read full epoch history from history.csv
            history_csv_path = output_dir / "history.csv"
            epoch_history = []
            with open(history_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epoch_history.append({
                        "epoch": int(row["epoch"]),
                        "loss": float(row["loss"]),
                        "accuracy": float(row["accuracy"]),
                        "mse": float(row["mse"]),
                    })

            # Track training global-best improvement epochs
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

            # Read and evaluate checkpoints
            checkpoints: List[Dict[str, Any]] = []
            ckpt_dir = output_dir / "checkpoints"
            epoch80_test_acc = None

            for ep in CHECKPOINT_EPOCHS:
                ckpt_path = ckpt_dir / f"epoch-{ep}.pt"
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Missing checkpoint file: {ckpt_path}")

                payload = torch.load(ckpt_path, map_location=device, weights_only=True)
                ckpt_train_loss, ckpt_train_acc, ckpt_train_mse = payload["score"]

                # Load checkpoint state_dict into opt.eval_model and evaluate on held-out test tensor
                opt.eval_model.load_state_dict(payload["model_state_dict"])
                opt._global_best_weights = opt.codec.encode(opt.eval_model)
                test_loss, test_acc, test_mse = opt.evaluate(x_full_test, y_test_1000)

                ckpt_record = {
                    "epoch": ep,
                    "train_loss": float(ckpt_train_loss),
                    "train_acc": float(ckpt_train_acc),
                    "train_mse": float(ckpt_train_mse),
                    "test_loss": float(test_loss),
                    "test_acc": float(test_acc),
                    "test_mse": float(test_mse),
                }
                checkpoints.append(ckpt_record)

                if ep == 80:
                    epoch80_test_acc = float(test_acc)

                flat_csv_rows.append({
                    "seed": seed,
                    "epoch": ep,
                    "train_loss": float(ckpt_train_loss),
                    "train_acc": float(ckpt_train_acc),
                    "train_mse": float(ckpt_train_mse),
                    "test_loss": float(test_loss),
                    "test_acc": float(test_acc),
                    "test_mse": float(test_mse),
                    "fit_time_sec": round(fit_time_sec, 4),
                })

        # Fidelity check at epoch 80 vs baseline
        assert epoch80_test_acc is not None
        delta_ep80 = abs(epoch80_test_acc - baseline_test_acc)
        seed_fidelity_deltas[seed] = delta_ep80

        if delta_ep80 > REPLAY_TOLERANCE:
            print(
                f"[WARNING] Seed {seed} epoch 80 test accuracy delta {delta_ep80:.6f} "
                f"exceeds tolerance {REPLAY_TOLERANCE} (actual={epoch80_test_acc:.4f}, baseline={baseline_test_acc:.4f})"
            )
            fidelity_passed = False

        runs.append({
            "seed": seed,
            "model_fingerprint": model_fp,
            "expected_model_fingerprint": expected_model_fp,
            "fingerprint_matched": fp_match,
            "baseline_epoch80_test_acc": baseline_test_acc,
            "epoch80_test_acc": epoch80_test_acc,
            "epoch80_abs_delta": delta_ep80,
            "fit_time_sec": round(fit_time_sec, 4),
            "improvement_count": improvement_count,
            "last_improvement_epoch": last_improvement_epoch,
            "checkpoints": checkpoints,
        })

    max_ep80_delta = max(seed_fidelity_deltas.values()) if seed_fidelity_deltas else 0.0

    # 4. Aggregations and Statistical Summaries
    checkpoint_stats: Dict[int, Dict[str, Any]] = {}
    for ep in CHECKPOINT_EPOCHS:
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

    # Endpoint paired deltas (80->240 and 200->240)
    deltas_80_to_240_train_rel = []
    deltas_80_to_240_test_acc = []

    deltas_200_to_240_train_rel = []
    deltas_200_to_240_test_acc = []
    deltas_200_to_240_test_acc_abs = []

    for r in runs:
        ckpt_map = {c["epoch"]: c for c in r["checkpoints"]}
        
        # 80 to 240
        tl80 = ckpt_map[80]["train_loss"]
        tl240 = ckpt_map[240]["train_loss"]
        rel_red_80_240 = (tl80 - tl240) / tl80 if tl80 > 0 else 0.0
        deltas_80_to_240_train_rel.append(rel_red_80_240)

        ta80 = ckpt_map[80]["test_acc"]
        ta240 = ckpt_map[240]["test_acc"]
        acc_delta_80_240 = ta240 - ta80
        deltas_80_to_240_test_acc.append(acc_delta_80_240)

        # 200 to 240
        tl200 = ckpt_map[200]["train_loss"]
        rel_red_200_240 = (tl200 - tl240) / tl200 if tl200 > 0 else 0.0
        deltas_200_to_240_train_rel.append(rel_red_200_240)

        ta200 = ckpt_map[200]["test_acc"]
        acc_delta_200_240 = ta240 - ta200
        acc_abs_change_200_240 = abs(ta240 - ta200)
        deltas_200_to_240_test_acc.append(acc_delta_200_240)
        deltas_200_to_240_test_acc_abs.append(acc_abs_change_200_240)

    # Calculate overall mean metrics for classifications
    mean_train_loss_80 = checkpoint_stats[80]["train_loss"]["mean"]
    mean_train_loss_200 = checkpoint_stats[200]["train_loss"]["mean"]
    mean_train_loss_240 = checkpoint_stats[240]["train_loss"]["mean"]

    mean_test_acc_80 = checkpoint_stats[80]["test_acc"]["mean"]
    mean_test_acc_200 = checkpoint_stats[200]["test_acc"]["mean"]
    mean_test_acc_240 = checkpoint_stats[240]["test_acc"]["mean"]

    rel_train_loss_reduction_80_240 = (
        (mean_train_loss_80 - mean_train_loss_240) / mean_train_loss_80
    )
    test_acc_gain_80_240 = mean_test_acc_240 - mean_test_acc_80

    rel_train_loss_reduction_200_240 = (
        (mean_train_loss_200 - mean_train_loss_240) / mean_train_loss_200
    )
    abs_test_acc_change_200_240 = abs(mean_test_acc_240 - mean_test_acc_200)

    # 5. Shared Contract Predeclared Classifications
    post_80_training_converging = bool(
        rel_train_loss_reduction_80_240 >= POST80_LOSS_REDUCTION_THRESHOLD
    )
    meaningful_held_out_gain = bool(
        test_acc_gain_80_240 >= TEST_ACCURACY_GAIN_THRESHOLD
    )
    overfitting_signal = bool(
        post_80_training_converging
        and test_acc_gain_80_240 <= OVERFITTING_ACCURACY_DROP_THRESHOLD
    )
    early_stagnation = bool(
        not post_80_training_converging
        and abs(test_acc_gain_80_240) < TEST_ACCURACY_GAIN_THRESHOLD
    )
    generalization_plateau = bool(
        post_80_training_converging
        and not meaningful_held_out_gain
        and not overfitting_signal
    )
    late_plateau_diagnostic = bool(
        rel_train_loss_reduction_200_240 < LATE_LOSS_REDUCTION_THRESHOLD
        and abs_test_acc_change_200_240 < LATE_ACCURACY_CHANGE_THRESHOLD
    )

    if meaningful_held_out_gain:
        summary_verdict = (
            f"Training beyond epoch 80 continues to improve held-out test accuracy by "
            f"{test_acc_gain_80_240 * 100:.2f} percentage points "
            f"(from {mean_test_acc_80 * 100:.2f}% to {mean_test_acc_240 * 100:.2f}%)."
        )
    elif overfitting_signal:
        summary_verdict = (
            f"Training beyond epoch 80 exhibits overfitting: training loss falls by "
            f"{rel_train_loss_reduction_80_240 * 100:.2f}% while held-out test accuracy drops by "
            f"{abs(test_acc_gain_80_240) * 100:.2f} percentage points."
        )
    elif early_stagnation:
        summary_verdict = (
            f"Optimization has effectively stagnated after epoch 80: training loss falls by only "
            f"{rel_train_loss_reduction_80_240 * 100:.2f}% and held-out accuracy changes by "
            f"{test_acc_gain_80_240 * 100:+.2f} percentage points through epoch 240."
        )
    elif generalization_plateau:
        summary_verdict = (
            f"Training loss continues to improve after epoch 80, but held-out accuracy plateaus: "
            f"{test_acc_gain_80_240 * 100:+.2f} percentage points "
            f"(from {mean_test_acc_80 * 100:.2f}% to {mean_test_acc_240 * 100:.2f}%)."
        )
    else:
        summary_verdict = (
            "The fixed endpoint criteria are inconclusive; inspect the paired checkpoint "
            "trajectory before extending the epoch horizon."
        )

    last_improvement_epochs_dict = {r["seed"]: r["last_improvement_epoch"] for r in runs}
    paired_endpoint_deltas = []
    for r in runs:
        ckpt_map = {c["epoch"]: c for c in r["checkpoints"]}
        paired_endpoint_deltas.append(
            {
                "seed": r["seed"],
                "train_loss_relative_reduction_80_to_240": (
                    ckpt_map[80]["train_loss"] - ckpt_map[240]["train_loss"]
                )
                / ckpt_map[80]["train_loss"],
                "test_accuracy_delta_80_to_240": (
                    ckpt_map[240]["test_acc"] - ckpt_map[80]["test_acc"]
                ),
                "train_loss_relative_reduction_200_to_240": (
                    ckpt_map[200]["train_loss"] - ckpt_map[240]["train_loss"]
                )
                / ckpt_map[200]["train_loss"],
                "test_accuracy_delta_200_to_240": (
                    ckpt_map[240]["test_acc"] - ckpt_map[200]["test_acc"]
                ),
            }
        )


    # Build final payload
    payload = {
        "epoch_convergence_protocol_version": EPOCH_CONVERGENCE_PROTOCOL_VERSION,
        "tuning_protocol_version": TUNING_PROTOCOL_VERSION,
        "pso_version": pso_version,
        "torch_version": torch.__version__,
        "hardware": hw_provenance,
        "device": device.type,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_path": str(baseline_path),
        "data_fingerprint": data_fp,
        "candidate_label": winner_cfg.candidate_label,
        "config": {
            **opt_kwargs,
            "n_particles": n_particles,
            "epochs": target_epochs,
            "batch_size": batch_size,
            "renewal": "loss",
            "checkpoint_interval": 20,
        },
        "contract_criteria": {
            "primary_endpoints": [80, 120, 160, 200, 240],
            "post_80_convergence_threshold_loss_reduction": POST80_LOSS_REDUCTION_THRESHOLD,
            "meaningful_gain_threshold_test_acc": TEST_ACCURACY_GAIN_THRESHOLD,
            "overfitting_threshold_test_acc": OVERFITTING_ACCURACY_DROP_THRESHOLD,
            "late_plateau_200_240_loss_threshold": LATE_LOSS_REDUCTION_THRESHOLD,
            "late_plateau_200_240_acc_threshold": LATE_ACCURACY_CHANGE_THRESHOLD,
            "replay_tolerance": REPLAY_TOLERANCE,
        },
        "fidelity_validation": {
            "replay_seeds": sorted(REPLAY_SEEDS),
            "max_epoch80_test_acc_delta": round(max_ep80_delta, 6),
            "tolerance": REPLAY_TOLERANCE,
            "passed": fidelity_passed,
        },
        "predeclared_classifications": {
            "post_80_training_converging": post_80_training_converging,
            "meaningful_held_out_gain": meaningful_held_out_gain,
            "overfitting_signal": overfitting_signal,
            "early_stagnation": early_stagnation,
            "generalization_plateau": generalization_plateau,
            "late_plateau_diagnostic": late_plateau_diagnostic,
            "summary_verdict": summary_verdict,
        },
        "summary": {
            "epochs": CHECKPOINT_EPOCHS,
            "checkpoint_stats": {str(ep): stats for ep, stats in checkpoint_stats.items()},
            "paired_deltas": {
                "80_to_240": {
                    "train_loss_rel_reduction": calc_stats(deltas_80_to_240_train_rel),
                    "test_acc_delta": calc_stats(deltas_80_to_240_test_acc),
                },
                "200_to_240": {
                    "train_loss_rel_reduction": calc_stats(deltas_200_to_240_train_rel),
                    "test_acc_delta": calc_stats(deltas_200_to_240_test_acc),
                    "test_acc_abs_change": calc_stats(deltas_200_to_240_test_acc_abs),
                },
            },
            "paired_endpoint_deltas_by_seed": paired_endpoint_deltas,
            "last_training_best_improvement_epochs": last_improvement_epochs_dict,
        },
        "runs": runs,
        "completed": True,
        "valid": fidelity_passed,
        "error": None,
    }

    # Write output JSON
    save_json_atomic(payload, output_json_path)
    print(f"Saved analysis JSON to {output_json_path}")

    # Write output CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_csv_rows)
    print(f"Saved flat checkpoint CSV to {output_csv_path}")

    # 6. Render Figure
    render_convergence_figure(CHECKPOINT_EPOCHS, runs, checkpoint_stats, figure_path)
    print(f"Rendered plot to {figure_path}")

    # Print summary block
    print("\n" + "=" * 70)
    print("EPOCH CONVERGENCE & TRAJECTORY ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Device: {device.type} | Seeds: {sorted(REPLAY_SEEDS)}")
    print(f"Fidelity Replay Check (Epoch 80 <= {REPLAY_TOLERANCE}): Max Delta = {max_ep80_delta:.6f} -> Passed: {fidelity_passed}")
    print("-" * 70)
    print("Predeclared Classifications (Epoch 80 -> 240):")
    print(f"  Post-80 Training Converging (Loss Drop >= 1%): {post_80_training_converging} ({rel_train_loss_reduction_80_240 * 100:.2f}%)")
    print(f"  Meaningful Held-Out Gain (Acc Rise >= 1%p):   {meaningful_held_out_gain} ({test_acc_gain_80_240 * 100:+.2f}%p)")
    print(f"  Overfitting Signal:                            {overfitting_signal}")
    print(f"  Early Stagnation:                              {early_stagnation}")
    print(f"  Generalization Plateau:                       {generalization_plateau}")
    print(f"  Late Plateau Diagnostic (200 -> 240):         {late_plateau_diagnostic}")
    print("-" * 70)
    print(f"Verdict: {summary_verdict}")
    print("=" * 70 + "\n")

    return fidelity_passed


def render_convergence_figure(
    epochs: List[int],
    runs: List[Dict[str, Any]],
    checkpoint_stats: Dict[int, Dict[str, Any]],
    figure_path: Path,
):
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    mean_train_loss = [checkpoint_stats[ep]["train_loss"]["mean"] for ep in epochs]
    std_train_loss = [checkpoint_stats[ep]["train_loss"]["std"] for ep in epochs]

    mean_test_acc = [checkpoint_stats[ep]["test_acc"]["mean"] for ep in epochs]
    std_test_acc = [checkpoint_stats[ep]["test_acc"]["std"] for ep in epochs]

    epochs_arr = np.array(epochs)
    mean_tl_arr = np.array(mean_train_loss)
    std_tl_arr = np.array(std_train_loss)

    mean_ta_arr = np.array(mean_test_acc)
    std_ta_arr = np.array(std_test_acc)

    # Subplot 1: Training Loss
    for r in runs:
        r_epochs = [c["epoch"] for c in r["checkpoints"]]
        r_losses = [c["train_loss"] for c in r["checkpoints"]]
        ax1.plot(r_epochs, r_losses, color="#1f77b4", alpha=0.25, linestyle=":", linewidth=1.2)

    ax1.plot(epochs_arr, mean_tl_arr, marker="o", color="#1f77b4", linewidth=2.2, label="Mean Training Loss")
    ax1.fill_between(
        epochs_arr,
        mean_tl_arr - std_tl_arr,
        mean_tl_arr + std_tl_arr,
        color="#1f77b4",
        alpha=0.15,
        label="±1 Std Dev",
    )
    ax1.axvline(80, color="#d62728", linestyle="--", linewidth=1.5, label="Baseline Horizon (Epoch 80)")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Global-Best Training Loss", fontsize=11)
    ax1.set_title("Global-Best Training Loss Trajectory", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", frameon=True)

    # Subplot 2: Held-Out Test Accuracy
    for r in runs:
        r_epochs = [c["epoch"] for c in r["checkpoints"]]
        r_accs = [c["test_acc"] for c in r["checkpoints"]]
        ax2.plot(r_epochs, r_accs, color="#2ca02c", alpha=0.25, linestyle=":", linewidth=1.2)

    ax2.plot(epochs_arr, mean_ta_arr, marker="s", color="#2ca02c", linewidth=2.2, label="Mean Test Accuracy")
    ax2.fill_between(
        epochs_arr,
        mean_ta_arr - std_ta_arr,
        mean_ta_arr + std_ta_arr,
        color="#2ca02c",
        alpha=0.15,
        label="±1 Std Dev",
    )
    ax2.axvline(80, color="#d62728", linestyle="--", linewidth=1.5, label="Baseline Horizon (Epoch 80)")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Held-Out Test Accuracy", fontsize=11)
    ax2.set_title("Held-Out Test Accuracy Trajectory", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", frameon=True)

    fig.suptitle(
        "Adaptive Moment 120-Particle MNIST 240-Epoch Convergence & Trajectory Analysis",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Moment 120-Particle x 240-Epoch MNIST Convergence Analysis"
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=Path("benchmark_results/pso_v4_tuning.json"),
        help="Path to baseline tuning JSON file",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmark_results/pso_v4_epoch_convergence.json"),
        help="Path to output analysis JSON file",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_results/pso_v4_epoch_convergence.csv"),
        help="Path to output checkpoint CSV file",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("history_plt/pso_v4_epoch_convergence.png"),
        help="Path to output convergence PNG figure",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device (mps, cuda, cpu; default: auto-detect)",
    )

    args = parser.parse_args()
    passed = run_epoch_convergence_analysis(
        baseline_path=args.baseline_json,
        output_json_path=args.output_json,
        output_csv_path=args.output_csv,
        figure_path=args.figure,
        device_str=args.device,
    )

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
