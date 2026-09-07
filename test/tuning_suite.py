"""
MNIST Tuning & Particle Scaling Study Suite (v4.0 Protocol 1.0.0)

Implements a reproducible three-phase study:
1. Search Phase: Inner validation split (2400 train / 600 val, stratified) from first 3000 MNIST training examples.
   Fit PCA32 whitening on inner train only; transform inner val.
   Evaluates candidate space across 5 movement methods (adaptive_moment, inertia, constriction, local_best, quantum) over 3 seeds (51-53).
   Ranks validation accuracy descending, then validation loss ascending.
2. Confirmation Phase: Full 3000 training examples and 1000 untouched test examples.
   Fit PCA32 whitening on full 3000 train only; transform 1000 test.
   Evaluates top candidate per method over 5 seeds (61-65).
3. Scaling Phase: Full 3000 training / 1000 test set.
   Evaluates selected adaptive_moment winner across particle counts 30, 60, 90, 120
   under fixed-epoch (80) and fixed-budget (~2400 particle-epochs) regimens over 5 seeds (71-75).
"""

import argparse
import dataclasses
import hashlib
import json
import csv
import math
import os
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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_suite import (
    METHOD_STYLE,
    calc_stats,
    compute_data_fingerprint,
    compute_model_fingerprint,
    extract_plugin_metadata,
    get_hardware_provenance,
    get_method_style,
    get_t_crit,
    make_mnist_model,
    resolve_execution_device,
    save_json_atomic,
    sync_device,
)
from pso import Optimizer, __version__ as pso_version

TUNING_PROTOCOL_VERSION = "1.0.0"


@dataclasses.dataclass
class CandidateConfig:
    method: str
    candidate_label: str
    description: str
    c0: float = 1.49618
    c1: float = 1.49618
    w_min: float = 0.7298
    w_max: float = 0.7298
    velocity_limit_ratio: Optional[float] = 0.025
    mutation_swarm: float = 0.02
    negative_swarm: float = 0.0
    method_options: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_optimizer_kwargs(self, quick: bool = False) -> Dict[str, Any]:
        fitness_sz = 50 if quick else 2000
        kwargs: Dict[str, Any] = {
            "method": self.method,
            "velocity_limit_ratio": self.velocity_limit_ratio,
            "mutation_swarm": self.mutation_swarm,
            "negative_swarm": self.negative_swarm,
            "particle_min": -3.0,
            "particle_max": 3.0,
            "boundary_strategy": "reflect",
            "initialization": "model_noise",
            "initial_position_noise": 0.05,
            "evaluation": "fixed_subset",
            "fitness_size": fitness_sz,
            "convergence": "none",
            "refinement": "none",
        }
        if self.method in ("adaptive_moment", "inertia", "local_best"):
            kwargs["c0"] = self.c0
            kwargs["c1"] = self.c1
            kwargs["w_min"] = self.w_min
            kwargs["w_max"] = self.w_max
        elif self.method == "constriction":
            kwargs["c0"] = self.c0
            kwargs["c1"] = self.c1

        if self.method_options:
            kwargs["method_options"] = dict(self.method_options)
        return kwargs


def get_mnist_raw_data() -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    from torchvision.datasets import MNIST
    cache_dir = Path("result/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = MNIST(root=str(cache_dir), train=True, download=True)
    test_dataset = MNIST(root=str(cache_dir), train=False, download=True)

    x_train_raw = (train_dataset.data[:3000].float() / 255.0).reshape(3000, -1).numpy()
    y_train = train_dataset.targets[:3000].long()

    x_test_raw = (test_dataset.data[:1000].float() / 255.0).reshape(1000, -1).numpy()
    y_test = test_dataset.targets[:1000].long()

    return x_train_raw, x_test_raw, y_train, y_test


def get_search_candidates() -> Dict[str, List[CandidateConfig]]:
    candidates: Dict[str, List[CandidateConfig]] = {
        "adaptive_moment": [],
        "inertia": [],
        "constriction": [],
        "local_best": [],
        "quantum": [],
    }

    # --- 1. Adaptive Moment Candidates ---
    blends = [0.03, 0.06, 0.10, 0.15]
    steps = [0.5, 1.0, 1.5]
    for b in blends:
        for s in steps:
            label = f"am_b{b}_s{s}"
            candidates["adaptive_moment"].append(
                CandidateConfig(
                    method="adaptive_moment",
                    candidate_label=label,
                    description=f"Adaptive Moment blend={b}, step={s}, beta1=0.9",
                    c0=1.49618,
                    c1=1.49618,
                    w_min=0.7298,
                    w_max=0.7298,
                    velocity_limit_ratio=0.025,
                    mutation_swarm=0.02,
                    method_options={"moment_blend": b, "moment_step_size": s, "moment_beta1": 0.9},
                )
            )
    for beta1 in [0.8, 0.95]:
        label = f"am_b0.06_s1.0_beta{beta1}"
        candidates["adaptive_moment"].append(
            CandidateConfig(
                method="adaptive_moment",
                candidate_label=label,
                description=f"Adaptive Moment blend=0.06, step=1.0, beta1={beta1}",
                c0=1.49618,
                c1=1.49618,
                w_min=0.7298,
                w_max=0.7298,
                velocity_limit_ratio=0.025,
                mutation_swarm=0.02,
                method_options={"moment_blend": 0.06, "moment_step_size": 1.0, "moment_beta1": beta1},
            )
        )

    # --- 2. Inertia Candidates ---
    candidates["inertia"] = [
        CandidateConfig(
            method="inertia",
            candidate_label="inertia_canonical",
            description="Canonical Inertia (c0=c1=2.0, w=0.9->0.4, vel=0.1, mut=0)",
            c0=2.0,
            c1=2.0,
            w_min=0.4,
            w_max=0.9,
            velocity_limit_ratio=0.1,
            mutation_swarm=0.0,
        ),
        CandidateConfig(
            method="inertia",
            candidate_label="inertia_tuned",
            description="Tuned Inertia (c0=c1=1.49618, w=0.7298, vel=0.025, mut=0.02)",
            c0=1.49618,
            c1=1.49618,
            w_min=0.7298,
            w_max=0.7298,
            velocity_limit_ratio=0.025,
            mutation_swarm=0.02,
        ),
        CandidateConfig(
            method="inertia",
            candidate_label="inertia_low_w",
            description="Low Inertia w (c0=c1=1.49618, w=0.55, vel=0.025, mut=0.02)",
            c0=1.49618,
            c1=1.49618,
            w_min=0.55,
            w_max=0.55,
            velocity_limit_ratio=0.025,
            mutation_swarm=0.02,
        ),
        CandidateConfig(
            method="inertia",
            candidate_label="inertia_w_decay",
            description="Decaying Inertia (c0=c1=1.49618, w=0.9->0.4, vel=0.025, mut=0.02)",
            c0=1.49618,
            c1=1.49618,
            w_min=0.4,
            w_max=0.9,
            velocity_limit_ratio=0.025,
            mutation_swarm=0.02,
        ),
        CandidateConfig(
            method="inertia",
            candidate_label="inertia_asymmetric",
            description="Asymmetric Inertia (c0=1.8, c1=1.2, w=0.7298, vel=0.025, mut=0.02)",
            c0=1.8,
            c1=1.2,
            w_min=0.7298,
            w_max=0.7298,
            velocity_limit_ratio=0.025,
            mutation_swarm=0.02,
        ),
    ]

    # --- 3. Constriction Candidates ---
    candidates["constriction"] = [
        CandidateConfig(
            method="constriction",
            candidate_label="constriction_c201",
            description="Constriction c0=c1=2.01 (vel=0.025, mut=0)",
            c0=2.01,
            c1=2.01,
            w_min=0.7298,
            w_max=0.7298,
            velocity_limit_ratio=0.025,
            mutation_swarm=0.0,
        ),
        CandidateConfig(
            method="constriction",
            candidate_label="constriction_c205_canonical",
            description="Constriction c0=c1=2.05 canonical-ish (vel=0.05, mut=0.02)",
            c0=2.05,
            c1=2.05,
            w_min=0.7298,
            w_max=0.7298,
            velocity_limit_ratio=0.05,
            mutation_swarm=0.02,
        ),
        CandidateConfig(
            method="constriction",
            candidate_label="constriction_c205_tuned",
            description="Constriction c0=c1=2.05 tuned (vel=0.025, mut=0)",
            c0=2.05,
            c1=2.05,
            w_min=0.7298,
            w_max=0.7298,
            velocity_limit_ratio=0.025,
            mutation_swarm=0.0,
        ),
        CandidateConfig(
            method="constriction",
            candidate_label="constriction_c250",
            description="Constriction c0=c1=2.50 (vel=0.025, mut=0)",
            c0=2.50,
            c1=2.50,
            w_min=0.7298,
            w_max=0.7298,
            velocity_limit_ratio=0.025,
            mutation_swarm=0.0,
        ),
        CandidateConfig(
            method="constriction",
            candidate_label="constriction_asymmetric",
            description="Asymmetric Constriction c0=2.8, c1=1.3 (vel=0.025, mut=0)",
            c0=2.8,
            c1=1.3,
            w_min=0.7298,
            w_max=0.7298,
            velocity_limit_ratio=0.025,
            mutation_swarm=0.0,
        ),
    ]

    # --- 4. Local-Best Candidates ---
    candidates["local_best"] = [
        CandidateConfig(
            method="local_best",
            candidate_label="local_best_r1_constant",
            description="Local Best Ring Radius 1 (constant w=0.7298)",
            c0=1.49618,
            c1=1.49618,
            w_min=0.7298,
            w_max=0.7298,
            velocity_limit_ratio=0.025,
            mutation_swarm=0.02,
            method_options={"neighborhood_radius": 1, "c0": 1.49618, "c1": 1.49618, "w_min": 0.7298, "w_max": 0.7298},
        ),
        CandidateConfig(
            method="local_best",
            candidate_label="local_best_r2_constant",
            description="Local Best Ring Radius 2 (constant w=0.7298)",
            c0=1.49618,
            c1=1.49618,
            w_min=0.7298,
            w_max=0.7298,
            velocity_limit_ratio=0.025,
            mutation_swarm=0.02,
            method_options={"neighborhood_radius": 2, "c0": 1.49618, "c1": 1.49618, "w_min": 0.7298, "w_max": 0.7298},
        ),
        CandidateConfig(
            method="local_best",
            candidate_label="local_best_r4_constant",
            description="Local Best Ring Radius 4 (constant w=0.7298)",
            c0=1.49618,
            c1=1.49618,
            w_min=0.7298,
            w_max=0.7298,
            velocity_limit_ratio=0.025,
            mutation_swarm=0.02,
            method_options={"neighborhood_radius": 4, "c0": 1.49618, "c1": 1.49618, "w_min": 0.7298, "w_max": 0.7298},
        ),
        CandidateConfig(
            method="local_best",
            candidate_label="local_best_r1_decay",
            description="Local Best Ring Radius 1 (decaying w=0.9->0.4)",
            c0=1.49618,
            c1=1.49618,
            w_min=0.4,
            w_max=0.9,
            velocity_limit_ratio=0.025,
            mutation_swarm=0.02,
            method_options={"neighborhood_radius": 1, "c0": 1.49618, "c1": 1.49618, "w_min": 0.4, "w_max": 0.9},
        ),
    ]

    # --- 5. Quantum Candidates ---
    candidates["quantum"] = [
        CandidateConfig(
            method="quantum",
            candidate_label="quantum_beta_0.5_1.0",
            description="Quantum PSO (beta=1.0->0.5)",
            c0=1.49618,
            c1=1.49618,
            w_min=0.7298,
            w_max=0.7298,
            velocity_limit_ratio=None,
            mutation_swarm=0.0,
            negative_swarm=0.0,
            method_options={"beta_min": 0.5, "beta_max": 1.0},
        ),
        CandidateConfig(
            method="quantum",
            candidate_label="quantum_beta_0.6_1.0",
            description="Quantum PSO (beta=1.0->0.6)",
            c0=1.49618,
            c1=1.49618,
            w_min=0.7298,
            w_max=0.7298,
            velocity_limit_ratio=None,
            mutation_swarm=0.0,
            negative_swarm=0.0,
            method_options={"beta_min": 0.6, "beta_max": 1.0},
        ),
        CandidateConfig(
            method="quantum",
            candidate_label="quantum_beta_0.5_1.2",
            description="Quantum PSO (beta=1.2->0.5)",
            c0=1.49618,
            c1=1.49618,
            w_min=0.7298,
            w_max=0.7298,
            velocity_limit_ratio=None,
            mutation_swarm=0.0,
            negative_swarm=0.0,
            method_options={"beta_min": 0.5, "beta_max": 1.2},
        ),
        CandidateConfig(
            method="quantum",
            candidate_label="quantum_beta_0.4_0.9",
            description="Quantum PSO (beta=0.9->0.4)",
            c0=1.49618,
            c1=1.49618,
            w_min=0.7298,
            w_max=0.7298,
            velocity_limit_ratio=None,
            mutation_swarm=0.0,
            negative_swarm=0.0,
            method_options={"beta_min": 0.4, "beta_max": 0.9},
        ),
    ]

    return candidates


def run_single_experiment(
    cfg: CandidateConfig,
    seed: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    n_particles: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    quick: bool = False,
    eval_metric_name: str = "val",
    data_fp: str = "",
    run_type: str = "search",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    opt_kwargs = cfg.to_optimizer_kwargs(quick=quick)
    hw_provenance = get_hardware_provenance(device)
    warmup_ep = min(2, epochs)

    # --- Untimed Warmup Phase ---
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
        x_train,
        y_train,
        epochs=warmup_ep,
        batch_size=batch_size,
        renewal="loss",
    )
    sync_device(device)
    del warmup_opt, warmup_model, warmup_loss

    # --- Timed Fit Phase ---
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

    plugin_meta = extract_plugin_metadata(opt)

    sync_device(device)
    t0 = time.perf_counter()

    train_loss, train_acc, train_mse = opt.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        renewal="loss",
    )
    sync_device(device)
    t1 = time.perf_counter()
    fit_time = t1 - t0

    # Separate Evaluation on Validation or Test set
    eval_loss, eval_acc, eval_mse = opt.evaluate(x_eval, y_eval)

    full_resolved_config = dict(opt_kwargs)
    full_resolved_config.update({
        "n_particles": n_particles,
        "epochs": epochs,
        "batch_size": batch_size,
        "renewal": "loss",
    })

    res = {
        "protocol_version": TUNING_PROTOCOL_VERSION,
        "pso_version": pso_version,
        "torch_version": torch.__version__,
        "hardware": hw_provenance,
        "timing_scope": "fit_only_after_method_specific_warmup",
        "warmup_epochs": warmup_ep,
        "error": None,
        "phase": run_type,
        "method": cfg.method,
        "candidate_label": cfg.candidate_label,
        "seed": seed,
        "n_particles": n_particles,
        "epochs": epochs,
        "particle_epochs": n_particles * epochs,
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "train_mse": float(train_mse),
        f"{eval_metric_name}_loss": float(eval_loss),
        f"{eval_metric_name}_acc": float(eval_acc),
        f"{eval_metric_name}_mse": float(eval_mse),
        "fit_time_sec": float(fit_time),
        "data_fingerprint": data_fp,
        "model_fingerprint": model_fp,
        "device": str(device),
        "completed": True,
        "plugins": plugin_meta,
        "config": full_resolved_config,
    }
    if extra_meta:
        res.update(extra_meta)
    return res


def write_tuning_csvs(
    search_runs: List[Dict[str, Any]],
    confirmation_runs: List[Dict[str, Any]],
    scaling_runs: List[Dict[str, Any]],
    result_dir: Path,
):
    result_dir.mkdir(parents=True, exist_ok=True)

    # 1. Search CSV
    search_csv = result_dir / "pso_v4_tuning_search.csv"
    search_fields = [
        "method", "candidate_label", "seed", "n_particles", "epochs", "particle_epochs",
        "train_loss", "train_acc", "val_loss", "val_acc", "val_mse",
        "fit_time_sec", "data_fingerprint", "model_fingerprint", "device"
    ]
    with open(search_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=search_fields, extrasaction="ignore")
        writer.writeheader()
        for r in search_runs:
            if r.get("completed"):
                writer.writerow(r)

    # 2. Confirmation CSV
    confirm_csv = result_dir / "pso_v4_tuning_confirmation.csv"
    confirm_fields = [
        "method", "candidate_label", "seed", "n_particles", "epochs", "particle_epochs",
        "train_loss", "train_acc", "test_loss", "test_acc", "test_mse",
        "fit_time_sec", "data_fingerprint", "model_fingerprint", "device"
    ]
    with open(confirm_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=confirm_fields, extrasaction="ignore")
        writer.writeheader()
        for r in confirmation_runs:
            if r.get("completed"):
                writer.writerow(r)

    # 3. Particle Scaling CSV
    scaling_csv = result_dir / "pso_v4_particle_scaling.csv"
    scaling_fields = [
        "method", "candidate_label", "regimen", "n_particles", "epochs", "particle_epochs", "seed",
        "train_loss", "train_acc", "test_loss", "test_acc", "test_mse",
        "fit_time_sec", "data_fingerprint", "model_fingerprint", "device"
    ]
    with open(scaling_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scaling_fields, extrasaction="ignore")
        writer.writeheader()
        for r in scaling_runs:
            if r.get("completed"):
                writer.writerow(r)


def _persist_json_state(
    output_json: Path,
    device: torch.device,
    quick: bool,
    hw_provenance: Dict[str, Any],
    split_fingerprints: Dict[str, str],
    pca_provenance: Dict[str, Any],
    all_search_candidates: Dict[str, List[CandidateConfig]],
    search_runs: List[Dict[str, Any]],
    confirmation_runs: List[Dict[str, Any]],
    scaling_runs: List[Dict[str, Any]],
):
    search_summaries: Dict[str, Any] = {}
    for r in search_runs:
        if not r.get("completed"):
            continue
        lbl = r["candidate_label"]
        if lbl not in search_summaries:
            search_summaries[lbl] = {
                "method": r["method"],
                "candidate_label": lbl,
                "val_accs": [],
                "val_losses": [],
                "fit_times": [],
            }
        search_summaries[lbl]["val_accs"].append(r["val_acc"])
        search_summaries[lbl]["val_losses"].append(r["val_loss"])
        search_summaries[lbl]["fit_times"].append(r["fit_time_sec"])

    for lbl, s in search_summaries.items():
        s["val_acc_stats"] = calc_stats(s["val_accs"])
        s["val_loss_stats"] = calc_stats(s["val_losses"])
        s["fit_time_stats"] = calc_stats(s["fit_times"])

    confirm_summaries: Dict[str, Any] = {}
    for r in confirmation_runs:
        if not r.get("completed"):
            continue
        m = r["method"]
        if m not in confirm_summaries:
            confirm_summaries[m] = {
                "method": m,
                "candidate_label": r["candidate_label"],
                "test_accs": [],
                "test_losses": [],
                "fit_times": [],
            }
        confirm_summaries[m]["test_accs"].append(r["test_acc"])
        confirm_summaries[m]["test_losses"].append(r["test_loss"])
        confirm_summaries[m]["fit_times"].append(r["fit_time_sec"])

    for m, s in confirm_summaries.items():
        s["test_acc_stats"] = calc_stats(s["test_accs"])
        s["test_loss_stats"] = calc_stats(s["test_losses"])
        s["fit_time_stats"] = calc_stats(s["fit_times"])

    scaling_summaries: Dict[str, Any] = {}
    for r in scaling_runs:
        if not r.get("completed"):
            continue
        key = f"{r['n_particles']}p_{r['epochs']}e_{r.get('regimen', 'unknown')}"
        if key not in scaling_summaries:
            scaling_summaries[key] = {
                "n_particles": r["n_particles"],
                "epochs": r["epochs"],
                "particle_epochs": r["particle_epochs"],
                "regimen": r.get("regimen", "unknown"),
                "test_accs": [],
                "test_losses": [],
                "fit_times": [],
            }
        scaling_summaries[key]["test_accs"].append(r["test_acc"])
        scaling_summaries[key]["test_losses"].append(r["test_loss"])
        scaling_summaries[key]["fit_times"].append(r["fit_time_sec"])

    for key, s in scaling_summaries.items():
        s["test_acc_stats"] = calc_stats(s["test_accs"])
        s["test_loss_stats"] = calc_stats(s["test_losses"])
        s["fit_time_stats"] = calc_stats(s["fit_times"])

    required_seeds = 1 if quick else 3
    winners = {}
    for method, cand_list in all_search_candidates.items():
        cand_scores = []
        for cfg in cand_list:
            lbl = cfg.candidate_label
            matching_runs = [r for r in search_runs if r.get("candidate_label") == lbl and r.get("completed")]
            if len(matching_runs) < required_seeds:
                continue
            val_accs = [r["val_acc"] for r in matching_runs]
            val_losses = [r["val_loss"] for r in matching_runs]
            cand_scores.append({
                "candidate_label": lbl,
                "mean_val_loss": float(np.mean(val_losses)),
                "mean_val_acc": float(np.mean(val_accs)),
                "config": cfg.to_optimizer_kwargs(quick=quick),
            })
        if cand_scores:
            # Rank validation accuracy descending, then validation loss ascending
            cand_scores.sort(key=lambda c: (-c["mean_val_acc"], c["mean_val_loss"]))
            winners[method] = cand_scores[0]

    payload = {
        "tuning_protocol_version": TUNING_PROTOCOL_VERSION,
        "pso_version": pso_version,
        "torch_version": torch.__version__,
        "quick": quick,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw_provenance,
        "split_fingerprints": split_fingerprints,
        "pca_provenance": pca_provenance,
        "selection_criteria": "Validation accuracy descending, then validation loss ascending across required search seeds",
        "winners": winners,
        "summaries": {
            "search": search_summaries,
            "confirmation": confirm_summaries,
            "scaling": scaling_summaries,
        },
        "search_runs": search_runs,
        "confirmation_runs": confirmation_runs,
        "scaling_runs": scaling_runs,
    }
    save_json_atomic(payload, output_json)


def render_tuning_plots(
    search_runs: List[Dict[str, Any]],
    confirmation_runs: List[Dict[str, Any]],
    scaling_runs: List[Dict[str, Any]],
    winners: Dict[str, Dict[str, Any]],
    figure_dir: Path,
):
    figure_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Figure 1: pso_v4_extended_tuning.png
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    methods = ["adaptive_moment", "inertia", "constriction", "local_best", "quantum"]
    method_labels = {
        "adaptive_moment": "Adaptive Moment",
        "inertia": "Inertia Weight",
        "constriction": "Constriction",
        "local_best": "Local Best",
        "quantum": "Quantum PSO",
    }

    positions = []
    box_data = []
    winner_x = []
    winner_y = []
    x_ticks = []
    x_tick_labels = []

    for idx, m in enumerate(methods):
        m_runs = [r for r in search_runs if r.get("method") == m and r.get("completed")]
        if not m_runs:
            continue
        cand_means = {}
        for r in m_runs:
            lbl = r["candidate_label"]
            if lbl not in cand_means:
                cand_means[lbl] = []
            cand_means[lbl].append(r["val_loss"])

        c_means = [float(np.mean(vals)) for vals in cand_means.values()]
        box_data.append(c_means)
        pos = idx + 1
        positions.append(pos)
        x_ticks.append(pos)
        x_tick_labels.append(method_labels.get(m, m))

        win_info = winners.get(m)
        if win_info and win_info["candidate_label"] in cand_means:
            win_loss = float(np.mean(cand_means[win_info["candidate_label"]]))
            winner_x.append(pos)
            winner_y.append(win_loss)

    if box_data:
        bp = ax1.boxplot(
            box_data,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            showmeans=False,
        )
        for box, m in zip(bp["boxes"], methods[:len(box_data)]):
            c, _ = get_method_style(m)
            box.set_facecolor(c)
            box.set_alpha(0.6)
            box.set_edgecolor("#333333")

        if winner_x:
            ax1.scatter(
                winner_x,
                winner_y,
                color="#D55E00",
                marker="*",
                s=180,
                zorder=5,
                label="Selected Winner Candidate",
            )

    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels, rotation=15, ha="right", fontsize=10)
    ax1.set_ylabel("Validation Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Phase 1: Inner Validation Search (Candidates per Method)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    if winner_x:
        ax1.legend(loc="upper right")

    conf_x = []
    conf_y = []
    conf_ci = []
    conf_colors = []

    for idx, m in enumerate(methods):
        m_runs = [r for r in confirmation_runs if r.get("method") == m and r.get("completed")]
        if not m_runs:
            continue
        accs = [r["test_acc"] * 100.0 for r in m_runs]
        stats = calc_stats(accs)
        conf_x.append(idx + 1)
        conf_y.append(stats["mean"])
        conf_ci.append(stats["ci95_t"])
        c, _ = get_method_style(m)
        conf_colors.append(c)

    if conf_x:
        bars = ax2.bar(
            conf_x,
            conf_y,
            yerr=conf_ci,
            capsize=5,
            color=conf_colors,
            edgecolor="#333333",
            alpha=0.85,
            width=0.5,
        )
        ax2.set_xticks(conf_x)
        ax2.set_xticklabels([method_labels.get(m, m) for m in methods[:len(conf_x)]], rotation=15, ha="right", fontsize=10)
        ax2.set_ylabel("Held-Out Test Accuracy (%)", fontsize=11)
        ax2.set_title("Phase 2: Held-Out Test Confirmation (Method Winners)", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)

        for bar, y_val, ci_val in zip(bars, conf_y, conf_ci):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_val + ci_val + 0.5,
                f"{y_val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    fig_path1 = figure_dir / "pso_v4_extended_tuning.png"
    plt.savefig(fig_path1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered plot: {fig_path1}")

    # -------------------------------------------------------------
    # Figure 2: pso_v4_particle_scaling.png
    # -------------------------------------------------------------
    fig, (ax_acc, ax_loss, ax_time) = plt.subplots(1, 3, figsize=(16, 4.8))

    regimens = ["fixed_epoch", "fixed_budget"]
    regimen_names = {
        "fixed_epoch": "Fixed epochs: 80",
        "fixed_budget": "Fixed budget: ~2,400 particle-epochs",
    }
    regimen_colors = {
        "fixed_epoch": "#0072B2",
        "fixed_budget": "#D55E00",
    }
    regimen_markers = {
        "fixed_epoch": "o",
        "fixed_budget": "X",
    }

    for reg in regimens:
        reg_runs = [r for r in scaling_runs if r.get("regimen") == reg and r.get("completed")]
        if not reg_runs:
            continue
        
        by_p: Dict[int, List[Dict[str, Any]]] = {}
        for r in reg_runs:
            p = r["n_particles"]
            if p not in by_p:
                by_p[p] = []
            by_p[p].append(r)

        p_sorted = sorted(by_p.keys())
        x_offset = -1.2 if reg == "fixed_epoch" else 1.2
        plot_x = [p + x_offset for p in p_sorted]
        acc_means, acc_cis = [], []
        loss_means, loss_cis = [], []
        time_means, time_cis = [], []

        for p in p_sorted:
            p_runs = by_p[p]
            acc_st = calc_stats([r["test_acc"] * 100.0 for r in p_runs])
            loss_st = calc_stats([r["test_loss"] for r in p_runs])
            time_st = calc_stats([r["fit_time_sec"] for r in p_runs])

            acc_means.append(acc_st["mean"])
            acc_cis.append(acc_st["ci95_t"])
            loss_means.append(loss_st["mean"])
            loss_cis.append(loss_st["ci95_t"])
            time_means.append(time_st["mean"])
            time_cis.append(time_st["ci95_t"])

        color = regimen_colors[reg]
        marker = regimen_markers[reg]
        label = regimen_names[reg]

        ax_acc.errorbar(
            plot_x,
            acc_means,
            yerr=acc_cis,
            fmt=f"-{marker}",
            color=color,
            linewidth=2,
            markersize=6,
            capsize=4,
            label=label,
        )

        ax_loss.errorbar(
            plot_x,
            loss_means,
            yerr=loss_cis,
            fmt=f"-{marker}",
            color=color,
            linewidth=2,
            markersize=6,
            capsize=4,
            label=label,
        )

        ax_time.errorbar(
            plot_x,
            time_means,
            yerr=time_cis,
            fmt=f"-{marker}",
            color=color,
            linewidth=2,
            markersize=6,
            capsize=4,
            label=label,
        )

    ax_acc.set_title("Test Accuracy vs Particle Count", fontsize=11, fontweight="bold")
    ax_acc.set_xlabel("Particle Count", fontsize=10)
    ax_acc.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax_acc.set_xticks([30, 60, 90, 120])
    ax_acc.grid(True, linestyle="--", alpha=0.5)

    ax_loss.set_title("Test Cross-Entropy Loss vs Particle Count", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Particle Count", fontsize=10)
    ax_loss.set_ylabel("Test Cross-Entropy Loss", fontsize=10)
    ax_loss.set_xticks([30, 60, 90, 120])
    ax_loss.grid(True, linestyle="--", alpha=0.5)

    ax_time.set_title("Fit Runtime vs Particle Count", fontsize=11, fontweight="bold")
    ax_time.set_xlabel("Particle Count", fontsize=10)
    ax_time.set_ylabel("Fit Runtime (seconds)", fontsize=10)
    ax_time.set_xticks([30, 60, 90, 120])
    ax_time.grid(True, linestyle="--", alpha=0.5)
    handles, labels = ax_acc.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=True,
        )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig_path2 = figure_dir / "pso_v4_particle_scaling.png"
    plt.savefig(fig_path2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered plot: {fig_path2}")


def run_tuning_study(
    phase: str = "all",
    device_name: Optional[str] = None,
    quick: bool = False,
    method_filter: Optional[List[str]] = None,
    overwrite: bool = False,
    output_json: Path = Path("benchmark_results/pso_v4_tuning.json"),
    result_dir: Path = Path("benchmark_results"),
    figure_dir: Path = Path("history_plt"),
):
    device = resolve_execution_device(device_name)
    print(f"Executing MNIST Tuning Study on device: {device}")
    hw_provenance = get_hardware_provenance(device)

    existing_data: Dict[str, Any] = {}
    completed_search_runs: Dict[str, Dict[str, Any]] = {}
    completed_confirm_runs: Dict[str, Dict[str, Any]] = {}
    completed_scaling_runs: Dict[str, Dict[str, Any]] = {}

    if output_json.exists() and not overwrite:
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if (
                existing_data.get("tuning_protocol_version") == TUNING_PROTOCOL_VERSION
                and existing_data.get("quick") == quick
                and existing_data.get("device") == str(device)
            ):
                for r in existing_data.get("search_runs", []):
                    if r.get("completed") and "run_id" in r:
                        completed_search_runs[r["run_id"]] = r
                for r in existing_data.get("confirmation_runs", []):
                    if r.get("completed") and "run_id" in r:
                        completed_confirm_runs[r["run_id"]] = r
                for r in existing_data.get("scaling_runs", []):
                    if r.get("completed") and "run_id" in r:
                        completed_scaling_runs[r["run_id"]] = r
                print(
                    f"Loaded existing runs from {output_json}: "
                    f"{len(completed_search_runs)} search, "
                    f"{len(completed_confirm_runs)} confirmation, "
                    f"{len(completed_scaling_runs)} scaling."
                )
            else:
                print("Existing JSON protocol version, quick mode, or device differs. Starting fresh.")
        except Exception as e:
            print(f"Warning: Failed to load existing JSON ({e}). Starting fresh.")

    # Data loading and PCA preprocessing
    x_train_raw, x_test_raw, y_train_3000, y_test_1000 = get_mnist_raw_data()

    # Search inner split: Stratified 2400 train / 600 validation
    x_inner_tr_raw, x_inner_val_raw, y_inner_tr_np, y_inner_val_np = train_test_split(
        x_train_raw,
        y_train_3000.numpy(),
        train_size=2400,
        test_size=600,
        stratify=y_train_3000.numpy(),
        random_state=42,
    )
    y_inner_tr = torch.tensor(y_inner_tr_np, dtype=torch.long)
    y_inner_val = torch.tensor(y_inner_val_np, dtype=torch.long)

    # Fit PCA32 whitening on inner train ONLY for Search phase
    pca_search = PCA(n_components=32, whiten=True, random_state=42)
    x_inner_tr = torch.tensor(pca_search.fit_transform(x_inner_tr_raw), dtype=torch.float32)
    x_inner_val = torch.tensor(pca_search.transform(x_inner_val_raw), dtype=torch.float32)

    # Fit PCA32 whitening on full 3000 train ONLY for Confirmation & Scaling phases
    pca_full = PCA(n_components=32, whiten=True, random_state=42)
    x_full_tr = torch.tensor(pca_full.fit_transform(x_train_raw), dtype=torch.float32)
    x_full_test = torch.tensor(pca_full.transform(x_test_raw), dtype=torch.float32)

    search_data_fp = compute_data_fingerprint(x_inner_tr, x_inner_val, y_inner_tr, y_inner_val)
    full_data_fp = compute_data_fingerprint(x_full_tr, x_full_test, y_train_3000, y_test_1000)

    split_fingerprints = {
        "search_inner": search_data_fp,
        "full": full_data_fp,
    }
    pca_provenance = {
        "search": {
            "n_samples_fit": 2400,
            "n_samples_val": 600,
            "n_components": 32,
            "whiten": True,
            "random_state": 42,
            "explained_variance_ratio_sum": float(np.sum(pca_search.explained_variance_ratio_)),
        },
        "full": {
            "n_samples_fit": 3000,
            "n_samples_test": 1000,
            "n_components": 32,
            "whiten": True,
            "random_state": 42,
            "explained_variance_ratio_sum": float(np.sum(pca_full.explained_variance_ratio_)),
        },
    }

    all_search_candidates = get_search_candidates()
    if method_filter:
        unknown_methods = sorted(set(method_filter) - set(all_search_candidates))
        if unknown_methods:
            raise ValueError(
                f"Unknown tuning method(s) {unknown_methods}. "
                f"Available: {sorted(all_search_candidates)}"
            )

    search_seeds = [51, 52, 53] if not quick else [51]
    search_particles = 30 if not quick else 5
    search_epochs = 80 if not quick else 5
    batch_size = 1000 if not quick else 25

    search_runs: List[Dict[str, Any]] = list(completed_search_runs.values())

    # --- Phase 1: Search ---
    if phase in ("all", "search"):
        print("\n=== Phase 1: Search (Validation Tuning) ===")
        for method, cand_list in all_search_candidates.items():
            if method_filter and method not in method_filter:
                continue
            run_cands = cand_list[:1] if quick else cand_list
            for cfg in run_cands:
                for seed in search_seeds:
                    cfg_payload = {
                        "phase": "search",
                        "quick": quick,
                        "device": str(device),
                        "candidate_label": cfg.candidate_label,
                        "seed": seed,
                        "n_particles": search_particles,
                        "epochs": search_epochs,
                        "kwargs": cfg.to_optimizer_kwargs(quick=quick),
                    }
                    fp_bytes = json.dumps(cfg_payload, sort_keys=True, default=str).encode("utf-8")
                    cfg_fp = hashlib.sha256(fp_bytes).hexdigest()[:12]
                    run_id = f"search_{cfg.candidate_label}_seed{seed}_{cfg_fp}"

                    if run_id in completed_search_runs and not overwrite:
                        print(f"Skipping completed search run: {run_id}")
                        continue

                    print(f"Running {run_id} ({cfg.description})...")
                    run_res = run_single_experiment(
                        cfg=cfg,
                        seed=seed,
                        x_train=x_inner_tr,
                        y_train=y_inner_tr,
                        x_eval=x_inner_val,
                        y_eval=y_inner_val,
                        n_particles=search_particles,
                        epochs=search_epochs,
                        batch_size=batch_size,
                        device=device,
                        quick=quick,
                        eval_metric_name="val",
                        data_fp=search_data_fp,
                        run_type="search",
                        extra_meta={"run_id": run_id},
                    )
                    completed_search_runs[run_id] = run_res
                    search_runs = list(completed_search_runs.values())

                    _persist_json_state(
                        output_json=output_json,
                        device=device,
                        quick=quick,
                        hw_provenance=hw_provenance,
                        split_fingerprints=split_fingerprints,
                        pca_provenance=pca_provenance,
                        all_search_candidates=all_search_candidates,
                        search_runs=search_runs,
                        confirmation_runs=list(completed_confirm_runs.values()),
                        scaling_runs=list(completed_scaling_runs.values()),
                    )

    # Winner Selection Logic (Requires expected completed search seeds per candidate)
    required_seeds = 1 if quick else 3
    winners: Dict[str, Dict[str, Any]] = {}
    candidate_summaries: Dict[str, Dict[str, Any]] = {}
    incomplete_methods = []

    for method, cand_list in all_search_candidates.items():
        cand_scores = []
        for cfg in cand_list:
            lbl = cfg.candidate_label
            matching_runs = [r for r in search_runs if r.get("candidate_label") == lbl and r.get("completed")]
            if len(matching_runs) < required_seeds:
                continue
            val_accs = [r["val_acc"] for r in matching_runs]
            val_losses = [r["val_loss"] for r in matching_runs]
            mean_acc = float(np.mean(val_accs))
            mean_loss = float(np.mean(val_losses))
            cand_scores.append({
                "candidate_label": lbl,
                "cfg": cfg,
                "mean_val_loss": mean_loss,
                "mean_val_acc": mean_acc,
                "n_runs": len(matching_runs),
                "stats_acc": calc_stats(val_accs),
                "stats_loss": calc_stats(val_losses),
            })
            candidate_summaries[lbl] = cand_scores[-1]

        if cand_scores:
            # Rank validation accuracy descending, then validation loss ascending
            cand_scores.sort(key=lambda c: (-c["mean_val_acc"], c["mean_val_loss"]))
            top = cand_scores[0]
            winners[method] = {
                "method": method,
                "candidate_label": top["candidate_label"],
                "description": top["cfg"].description,
                "mean_val_loss": top["mean_val_loss"],
                "mean_val_acc": top["mean_val_acc"],
                "config": top["cfg"].to_optimizer_kwargs(quick=quick),
                "cfg": top["cfg"],
            }
        else:
            incomplete_methods.append(method)

    if phase in ("all", "confirmation", "scaling") and incomplete_methods:
        raise RuntimeError(
            f"Cannot proceed to {phase}: Search phase incomplete for method(s): {incomplete_methods}. "
            f"Expected {required_seeds} completed search seeds per candidate."
        )

    if winners:
        print("\n--- Search Winners Selected ---")
        for m, w in winners.items():
            print(f"  {m:15s} -> Winner: {w['candidate_label']} (Val Acc: {w['mean_val_acc']*100:.2f}%, Val Loss: {w['mean_val_loss']:.4f})")

    # --- Phase 2: Confirmation ---
    confirm_seeds = [61, 62, 63, 64, 65] if not quick else [61]
    confirm_particles = 30 if not quick else 5
    confirm_epochs = 80 if not quick else 5
    confirm_runs: List[Dict[str, Any]] = list(completed_confirm_runs.values())

    if phase in ("all", "confirmation"):
        print("\n=== Phase 2: Confirmation (Held-Out Test Confirmation) ===")
        for method, win_info in winners.items():
            if method_filter and method not in method_filter:
                continue
            cfg = win_info["cfg"]
            for seed in confirm_seeds:
                cfg_payload = {
                    "phase": "confirmation",
                    "quick": quick,
                    "device": str(device),
                    "candidate_label": cfg.candidate_label,
                    "seed": seed,
                    "n_particles": confirm_particles,
                    "epochs": confirm_epochs,
                    "kwargs": cfg.to_optimizer_kwargs(quick=quick),
                }
                fp_bytes = json.dumps(cfg_payload, sort_keys=True, default=str).encode("utf-8")
                cfg_fp = hashlib.sha256(fp_bytes).hexdigest()[:12]
                run_id = f"confirm_{method}_{cfg.candidate_label}_seed{seed}_{cfg_fp}"

                if run_id in completed_confirm_runs and not overwrite:
                    print(f"Skipping completed confirmation run: {run_id}")
                    continue

                print(f"Running confirmation {run_id} ({method} / {cfg.candidate_label})...")
                run_res = run_single_experiment(
                    cfg=cfg,
                    seed=seed,
                    x_train=x_full_tr,
                    y_train=y_train_3000,
                    x_eval=x_full_test,
                    y_eval=y_test_1000,
                    n_particles=confirm_particles,
                    epochs=confirm_epochs,
                    batch_size=batch_size,
                    device=device,
                    quick=quick,
                    eval_metric_name="test",
                    data_fp=full_data_fp,
                    run_type="confirmation",
                    extra_meta={"run_id": run_id},
                )
                completed_confirm_runs[run_id] = run_res
                confirm_runs = list(completed_confirm_runs.values())

                _persist_json_state(
                    output_json=output_json,
                    device=device,
                    quick=quick,
                    hw_provenance=hw_provenance,
                    split_fingerprints=split_fingerprints,
                    pca_provenance=pca_provenance,
                    all_search_candidates=all_search_candidates,
                    search_runs=search_runs,
                    confirmation_runs=confirm_runs,
                    scaling_runs=list(completed_scaling_runs.values()),
                )

    # --- Phase 3: Particle Scaling ---
    scaling_seeds = [71, 72, 73, 74, 75] if not quick else [71]
    scaling_runs: List[Dict[str, Any]] = list(completed_scaling_runs.values())

    if phase in ("all", "scaling"):
        print("\n=== Phase 3: Particle Scaling Study (Adaptive Moment Winner) ===")
        am_winner = winners.get("adaptive_moment")
        if not am_winner:
            raise RuntimeError("Error: No adaptive_moment search winner available for scaling study.")
        
        cfg = am_winner["cfg"]
        if quick:
            scaling_configs = [
                (5, 5, "fixed_epoch"),
                (10, 5, "fixed_epoch"),
                (5, 5, "fixed_budget"),
                (10, 3, "fixed_budget"),
            ]
        else:
            scaling_configs = [
                (30, 80, "fixed_epoch"),
                (60, 80, "fixed_epoch"),
                (90, 80, "fixed_epoch"),
                (120, 80, "fixed_epoch"),
                (30, 80, "fixed_budget"),
                (60, 40, "fixed_budget"),
                (90, 27, "fixed_budget"),
                (120, 20, "fixed_budget"),
            ]

        exec_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for r in scaling_runs:
            key = (r["n_particles"], r["epochs"], r["seed"])
            exec_cache[key] = r

        for (p_count, ep_count, regimen) in scaling_configs:
            for seed in scaling_seeds:
                exec_key = (p_count, ep_count, seed)
                cfg_payload = {
                    "phase": "scaling",
                    "quick": quick,
                    "device": str(device),
                    "candidate_label": cfg.candidate_label,
                    "regimen": regimen,
                    "seed": seed,
                    "n_particles": p_count,
                    "epochs": ep_count,
                    "kwargs": cfg.to_optimizer_kwargs(quick=quick),
                }
                fp_bytes = json.dumps(cfg_payload, sort_keys=True, default=str).encode("utf-8")
                cfg_fp = hashlib.sha256(fp_bytes).hexdigest()[:12]
                run_id = f"scaling_{p_count}p_{ep_count}e_{regimen}_seed{seed}_{cfg_fp}"

                if run_id in completed_scaling_runs and not overwrite:
                    print(f"Skipping completed scaling run: {run_id}")
                    continue

                if exec_key in exec_cache:
                    print(f"Reusing deduplicated run for {run_id} ({p_count} particles, {ep_count} epochs, seed {seed})...")
                    existing_res = dict(exec_cache[exec_key])
                    existing_res["run_id"] = run_id
                    existing_res["regimen"] = regimen
                    run_res = existing_res
                else:
                    print(f"Running scaling {run_id} ({p_count} particles, {ep_count} epochs, {regimen}, seed {seed})...")
                    run_res = run_single_experiment(
                        cfg=cfg,
                        seed=seed,
                        x_train=x_full_tr,
                        y_train=y_train_3000,
                        x_eval=x_full_test,
                        y_eval=y_test_1000,
                        n_particles=p_count,
                        epochs=ep_count,
                        batch_size=batch_size,
                        device=device,
                        quick=quick,
                        eval_metric_name="test",
                        data_fp=full_data_fp,
                        run_type="scaling",
                        extra_meta={"run_id": run_id, "regimen": regimen},
                    )
                    exec_cache[exec_key] = run_res

                completed_scaling_runs[run_id] = run_res
                scaling_runs = list(completed_scaling_runs.values())

                _persist_json_state(
                    output_json=output_json,
                    device=device,
                    quick=quick,
                    hw_provenance=hw_provenance,
                    split_fingerprints=split_fingerprints,
                    pca_provenance=pca_provenance,
                    all_search_candidates=all_search_candidates,
                    search_runs=search_runs,
                    confirmation_runs=confirm_runs,
                    scaling_runs=scaling_runs,
                )

    write_tuning_csvs(search_runs, confirm_runs, scaling_runs, result_dir)

    if phase in ("all", "plots"):
        render_tuning_plots(search_runs, confirm_runs, scaling_runs, winners, figure_dir)

    print("\nMNIST Tuning Study Complete!")
    print(f"- Primary JSON: {output_json}")
    print(f"- Search CSV: {result_dir / 'pso_v4_tuning_search.csv'}")
    print(f"- Confirmation CSV: {result_dir / 'pso_v4_tuning_confirmation.csv'}")
    print(f"- Scaling CSV: {result_dir / 'pso_v4_particle_scaling.csv'}")
    print(f"- Figures in: {figure_dir}")


def main():
    parser = argparse.ArgumentParser(description="MNIST PSO Tuning & Particle Scaling Study Suite")
    parser.add_argument(
        "--phase",
        choices=["all", "search", "confirmation", "scaling", "plots"],
        default="all",
        help="Study phase to execute (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('mps', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run reduced quick smoke test across all phases",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing completed run checkpoints and JSON results",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated method filter for search/confirmation reruns",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmark_results/pso_v4_tuning.json"),
        help="Path to output JSON result file",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory to save CSV report artifacts",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("history_plt"),
        help="Directory to save PNG figure artifacts",
    )

    args = parser.parse_args()
    method_filter = (
        [method.strip() for method in args.methods.split(",") if method.strip()]
        if args.methods
        else None
    )

    run_tuning_study(
        phase=args.phase,
        device_name=args.device,
        quick=args.quick,
        method_filter=method_filter,
        overwrite=args.overwrite,
        output_json=args.output_json,
        result_dir=args.result_dir,
        figure_dir=args.figure_dir,
    )


if __name__ == "__main__":
    main()
