"""
MNIST Deep PSO Methods Study: Latent Space Subspaces, Adaptive-Moment PSO, and Ensembling

Protocol: MNIST-PSO-RAW-V5 1.0.0
- Official raw MNIST (60,000 train / 10,000 test).
- Split 60k train first into 50k search and 10k validation using deterministic stratified sampling (seed 20260902).
- Mean and std fit on 50k search subset ONLY; applied to search, validation, and test.
- Nested stratified ordering: 2k subset inside 10k subset inside 50k search set.
- Base CompactCNN (9,098 parameters).
- Latent subspace transform via deterministic sparse signed hash mapping for d in [290, 1024, 4096, full].
- Exact base model at particle 0; remaining particles in antithetic pairs.
- Device-resident latent adaptive-moment PSO (c0=c1=1.49618, w=0.7298, blend=0.06, step=0.5, beta1=0.9, beta2=0.999).
- Lexicographical CE loss primary, accuracy tie-break selection.
- Objective transition: complete pbest re-evaluation and gbest rebuild.
- Validation-only pilot selection and elite selection.
- Official 10k test set evaluated exactly once per final reported endpoint.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_suite import (
    calc_stats,
    compute_data_fingerprint,
    compute_model_fingerprint,
    get_hardware_provenance,
    resolve_execution_device,
    save_json_atomic,
    sync_device,
)
from pso import __version__ as pso_version

PROTOCOL_VERSION = "MNIST-PSO-RAW-V5 1.0.0"


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


def make_compact_cnn(seed: int = 41) -> nn.Module:
    torch.manual_seed(seed)
    return CompactCNN()


# =====================================================================
# 2. Data Split, Normalization, & Stratified Subsets
# =====================================================================

def prepare_mnist_v5_data(
    split_seed: int = 20260902,
    cache_dir: Optional[Path] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    Dict[int, torch.Tensor],
    str, Dict[str, Any]
]:
    from torchvision.datasets import MNIST

    if cache_dir is None:
        cache_dir = Path("result/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_train = MNIST(root=str(cache_dir), train=True, download=True)
    raw_test = MNIST(root=str(cache_dir), train=False, download=True)

    x_train_raw = raw_train.data.float() / 255.0  # (60000, 28, 28)
    y_train_raw = raw_train.targets.long()
    x_test_raw = raw_test.data.float() / 255.0    # (10000, 28, 28)
    y_test_raw = raw_test.targets.long()

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
    x_test_norm = ((x_test_raw - mean_val) / std_val).unsqueeze(1)      # (10000, 1, 28, 28)

    # Nested stratified subsets inside 50k search set: 2k inside 10k inside 50k
    nested_subsets = build_nested_stratified_subsets(
        y_search=y_search,
        subset_sizes=[2000, 10000, 50000],
        subset_seed=split_seed,
    )

    # Data fingerprint over search and test
    data_fp = compute_data_fingerprint(x_search_norm, x_test_norm, y_search, y_test_raw)

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
        "test_samples": 10000,
        "split_seed": split_seed,
        "split_fingerprint": split_fp,
    }

    return (
        x_search_norm, y_search,
        x_val_norm, y_val,
        x_test_norm, y_test_raw,
        nested_subsets,
        data_fp, provenance
    )


def build_nested_stratified_subsets(
    y_search: torch.Tensor,
    subset_sizes: List[int],
    subset_seed: int = 20260902,
) -> Dict[int, torch.Tensor]:
    """
    Builds nested stratified index tensors: I_2k subset of I_10k subset of I_50k.
    Uses Hamilton / Largest-Remainder Method for exact subset sizes and nesting.
    """
    rng = np.random.RandomState(subset_seed)
    y_np = y_search.numpy()
    total_samples = len(y_np)
    unique_classes, counts = np.unique(y_np, return_counts=True)

    class_indices = {}
    for c in unique_classes:
        c_idxs = np.where(y_np == c)[0]
        rng.shuffle(c_idxs)
        class_indices[c] = c_idxs

    ordered_subset_sizes = sorted(subset_sizes)
    nested_subsets: Dict[int, torch.Tensor] = {}
    selected_per_class: Dict[int, List[int]] = {c: [] for c in unique_classes}

    for size in ordered_subset_sizes:
        if size == total_samples:
            nested_subsets[size] = torch.arange(total_samples, dtype=torch.long)
            continue

        exact_quotas = [size * (counts[i] / total_samples) for i in range(len(unique_classes))]
        floor_quotas = [int(np.floor(q)) for q in exact_quotas]
        remainders = [exact_quotas[i] - floor_quotas[i] for i in range(len(unique_classes))]

        needed_extra = size - sum(floor_quotas)
        ranked_indices = np.argsort(remainders)[::-1]
        target_counts = list(floor_quotas)
        for i in range(needed_extra):
            target_counts[ranked_indices[i]] += 1

        target_subset_idxs = []
        for i, c in enumerate(unique_classes):
            target_c_count = target_counts[i]
            current_list = selected_per_class[c]
            needed = target_c_count - len(current_list)
            if needed > 0:
                available = class_indices[c]
                added = list(available[len(current_list):len(current_list) + needed])
                current_list.extend(added)
            target_subset_idxs.extend(current_list[:target_c_count])

        subset_tensor = torch.tensor(sorted(target_subset_idxs), dtype=torch.long)
        nested_subsets[size] = subset_tensor

    return nested_subsets


# =====================================================================
# 3. Latent Subspace Transform & Antithetic Swarm Construction
# =====================================================================

class LatentTransform:
    def __init__(
        self,
        base_model: nn.Module,
        latent_dim: Union[int, str],
        device: torch.device,
    ):
        self.device = device
        self.base_params = [p.detach().clone().to(device) for p in base_model.parameters()]
        self.param_shapes = [p.shape for p in self.base_params]
        self.param_numels = [p.numel() for p in self.base_params]
        self.total_dim = sum(self.param_numels)

        # Per-tensor scale calculation: positive scale per parameter tensor
        tensor_scales = []
        for p in self.base_params:
            std_val = float(p.std())
            scale = max(std_val, 1e-4)
            scale_tensor = torch.full_like(p, scale)
            tensor_scales.append(scale_tensor.view(-1))
        self.scale_vec = torch.cat(tensor_scales).to(device)
        self.base_vec = torch.cat([p.view(-1) for p in self.base_params]).to(device)

        if isinstance(latent_dim, str) and latent_dim.lower() == "full":
            self.latent_dim = self.total_dim
            self.is_full = True
        else:
            self.latent_dim = int(latent_dim)
            self.is_full = (self.latent_dim == self.total_dim)

        if not self.is_full:
            # Deterministic sparse signed hash mapping
            j_indices = np.arange(self.total_dim, dtype=np.int64)
            h1 = ((j_indices + 1) * 2654435761) % (2**32)
            k_indices = h1 % self.latent_dim
            h2 = ((j_indices + 1) * 1597334677) % (2**32)
            signs = np.where((h2 % 2) == 0, 1.0, -1.0)

            # Count normalization to maintain unit variance
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

    def init_swarm(self, swarm_size: int, seed: int, init_radius: float = 0.5) -> torch.Tensor:
        """
        Initializes particle positions in latent space Z (N, d).
        Particle 0 is exact base vector (z = 0).
        For even swarm sizes N, particles 1..N-2 form (N-2)//2 exact pairs, and particle N-1 is zero.
        """
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)

        Z = torch.zeros((swarm_size, self.latent_dim), dtype=torch.float32)
        # Particle 0 stays exact 0

        max_pair_idx = swarm_size - 1 if (swarm_size % 2 != 0) else swarm_size - 2

        idx = 1
        while idx < max_pair_idx:
            sample = (torch.rand(self.latent_dim, generator=rng) * 2.0 - 1.0) * init_radius
            Z[idx] = sample
            Z[idx + 1] = -sample
            idx += 2

        return Z.to(self.device)


# =====================================================================
# 4. Device-Resident Latent Adaptive-Moment PSO Engine
# =====================================================================

def evaluate_latent_batch(
    Z: torch.Tensor,
    transform: LatentTransform,
    model: nn.Module,
    x_sub_dev: torch.Tensor,
    y_sub_dev: torch.Tensor,
    batch_size: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates latent batch Z (N, d) on device-resident (x_sub_dev, y_sub_dev) without host roundtrips.
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


def run_latent_pso(
    transform: LatentTransform,
    base_model: nn.Module,
    x_search: torch.Tensor,
    y_search: torch.Tensor,
    nested_subsets: Dict[int, torch.Tensor],
    schedule_str: str,
    epochs: int,
    swarm_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Runs device-resident Latent Adaptive-Moment PSO with progressive schedule.
    Performs full pbest re-evaluation and gbest rebuild ONLY on objective transitions (stages > 0).
    """
    sync_device(device)
    start_time = time.time()

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
    reflective_bound = 3.0

    latent_dim = transform.latent_dim
    Z = transform.init_swarm(swarm_size=swarm_size, seed=seed)
    V = torch.zeros((swarm_size, latent_dim), dtype=torch.float32, device=device)
    M = torch.zeros((swarm_size, latent_dim), dtype=torch.float32, device=device)
    V_sq = torch.zeros((swarm_size, latent_dim), dtype=torch.float32, device=device)

    # State tracking: P scores start at inf / 0. DO NOT evaluate before stage 0!
    P = Z.clone()
    P_loss = torch.full((swarm_size,), float("inf"), dtype=torch.float32, device=device)
    P_acc = torch.zeros((swarm_size,), dtype=torch.float32, device=device)

    gbest_z = Z[0].clone()
    gbest_loss = float("inf")
    gbest_acc = 0.0

    total_queries = 0
    total_sample_evaluations = 0
    transition_reevaluation_counts = 0

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    model = make_compact_cnn(seed=41).to(device)

    stage_histories = []
    t_step = 0
    epoch_counter = 0

    for stage_idx, (size, stage_epochs) in enumerate(schedule_stages):
        subset_indices = nested_subsets[size]
        # Move stage subset to device ONCE per stage
        x_sub_dev = x_search[subset_indices].to(device)
        y_sub_dev = y_search[subset_indices].to(device)

        # Objective transition check: ONLY for stages >= 1!
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

        for ep in range(1, stage_epochs + 1):
            epoch_counter += 1

            # 1. Evaluate current swarm positions
            curr_losses, curr_accs = evaluate_latent_batch(
                Z, transform, model, x_sub_dev, y_sub_dev
            )
            total_queries += swarm_size
            total_sample_evaluations += swarm_size * size

            # 2. Vectorized on-device pbest updates (lexicographical loss primary, acc tie-break)
            better_loss = curr_losses < P_loss - 1e-7
            equal_loss = torch.abs(curr_losses - P_loss) <= 1e-7
            better_acc = curr_accs > P_acc
            update_mask = better_loss | (equal_loss & better_acc)

            P[update_mask] = Z[update_mask]
            P_loss[update_mask] = curr_losses[update_mask]
            P_acc[update_mask] = curr_accs[update_mask]

            # Rebuild gbest from P each epoch on-device
            min_loss_val = P_loss.min()
            candidates_mask = (P_loss <= min_loss_val + 1e-7)
            best_p_idx = int(torch.where(candidates_mask, P_acc, torch.tensor(-1.0, device=device)).argmax().item())

            gbest_z = P[best_p_idx].clone()
            gbest_loss = float(P_loss[best_p_idx].item())
            gbest_acc = float(P_acc[best_p_idx].item())

            stage_histories.append({
                "epoch": epoch_counter,
                "stage": stage_idx,
                "subset_size": size,
                "gbest_loss": round(gbest_loss, 6),
                "gbest_acc": round(gbest_acc, 4),
            })

            # 3. Adaptive Moment Movement Step
            r1 = torch.rand((swarm_size, latent_dim), generator=rng, device=device)
            r2 = torch.rand((swarm_size, latent_dim), generator=rng, device=device)

            V_raw = w * V + c0 * r1 * (P - Z) + c1 * r2 * (gbest_z.unsqueeze(0) - Z)

            t_step += 1
            M = beta1 * M + (1 - beta1) * V_raw
            V_sq = beta2 * V_sq + (1 - beta2) * (V_raw ** 2)

            M_hat = M / (1.0 - beta1 ** t_step)
            V_sq_hat = V_sq / (1.0 - beta2 ** t_step)

            dir_moment = M_hat / (torch.sqrt(V_sq_hat) + 1e-8)
            historical_scale = torch.sqrt(torch.mean(V_sq_hat, dim=1, keepdim=True))
            V_moment = step * dir_moment * historical_scale
            V_new = (1.0 - blend) * V_raw + blend * V_moment

            Z_new = Z + V_new

            pos_mask = Z_new > reflective_bound
            neg_mask = Z_new < -reflective_bound

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
    wall_time = time.time() - start_time

    return {
        "gbest_z": gbest_z,
        "gbest_loss": gbest_loss,
        "gbest_acc": gbest_acc,
        "final_P": P,
        "wall_time_sec": round(wall_time, 4),
        "total_queries": total_queries,
        "total_sample_evaluations": total_sample_evaluations,
        "transition_reevaluation_counts": transition_reevaluation_counts,
        "stage_histories": stage_histories,
    }


# =====================================================================
# 5. Evaluation Metrics (NLL, Brier, ECE, Margin, Disagreement)
# =====================================================================

def evaluate_probabilistic_metrics(
    prob_matrix: torch.Tensor,
    y_true: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes accuracy, NLL, Brier score, 15-bin ECE, and probability margin.
    """
    N, C = prob_matrix.shape
    probs = prob_matrix.cpu().numpy()
    labels = y_true.cpu().numpy()

    preds = probs.argmax(axis=1)
    acc = float((preds == labels).mean()) * 100.0

    eps = 1e-12
    clipped_probs = np.clip(probs, eps, 1.0 - eps)
    nll = -float(np.log(clipped_probs[np.arange(N), labels]).mean())

    y_onehot = np.zeros((N, C), dtype=np.float32)
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
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = (preds[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

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


def compute_pairwise_disagreement(model_preds_list: List[np.ndarray]) -> float:
    num_models = len(model_preds_list)
    if num_models < 2:
        return 0.0

    disagreements = []
    for i in range(num_models):
        for j in range(i + 1, num_models):
            dis = float((model_preds_list[i] != model_preds_list[j]).mean())
            disagreements.append(dis)

    return round(float(np.mean(disagreements)), 6)


def select_diverse_candidates(
    candidates: List[Dict[str, Any]],
    *,
    max_size: int = 5,
    accuracy_window: float = 2.0,
) -> List[Dict[str, Any]]:
    if not candidates:
        raise ValueError("candidates must not be empty")
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    if accuracy_window < 0.0:
        raise ValueError("accuracy_window must be nonnegative")

    ranked = sorted(candidates, key=lambda c: (c["val_loss"], -c["val_acc"]))
    accuracy_threshold = ranked[0]["val_acc"] - accuracy_window
    eligible = [c for c in ranked if c["val_acc"] >= accuracy_threshold]

    selected = [eligible[0]]
    selected_keys = {(eligible[0]["seed"], eligible[0]["particle_idx"])}
    predictions = {
        (candidate["seed"], candidate["particle_idx"]):
        candidate["val_probs"].argmax(dim=1).cpu().numpy()
        for candidate in eligible
    }

    while len(selected) < min(max_size, len(eligible)):
        best_next = None
        best_key = None
        max_disagreement = -1.0
        best_val_loss = float("inf")

        for candidate in eligible:
            candidate_key = (candidate["seed"], candidate["particle_idx"])
            if candidate_key in selected_keys:
                continue
            candidate_predictions = predictions[candidate_key]
            mean_disagreement = float(np.mean([
                (candidate_predictions - predictions[
                    (member["seed"], member["particle_idx"])
                ] != 0).mean()
                for member in selected
            ]))
            if (
                mean_disagreement > max_disagreement + 1e-7
                or (
                    abs(mean_disagreement - max_disagreement) <= 1e-7
                    and candidate["val_loss"] < best_val_loss
                )
            ):
                best_next = candidate
                best_key = candidate_key
                max_disagreement = mean_disagreement
                best_val_loss = candidate["val_loss"]

        if best_next is None or best_key is None:
            break
        selected.append(best_next)
        selected_keys.add(best_key)

    return selected


# =====================================================================
# 6. Full Experiment Pipeline & CLI Runner
# =====================================================================

def validate_cli_args(args: argparse.Namespace):
    if args.pilot_epochs <= 0 or args.confirmation_epochs <= 0:
        raise ValueError("Pilot and confirmation epochs must be positive integers.")
    if args.pilot_particles <= 0 or args.confirmation_particles <= 0:
        raise ValueError("Pilot and confirmation particles must be positive integers.")
    if any(s < 0 for s in args.seeds):
        raise ValueError("Seeds must be non-negative integers.")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Confirmation seeds must be unique.")

    max_d = 9098
    for d_str in args.dimensions:
        if d_str.lower() != "full":
            try:
                d_val = int(d_str)
                if d_val <= 0 or d_val > max_d:
                    raise ValueError(f"Latent dimension {d_val} must be in range [1, {max_d}].")
            except ValueError:
                raise ValueError(f"Invalid dimension specifier: {d_str}")

    if args.confirmation_schedule:
        stages = args.confirmation_schedule.split(",")
        total_sched_epochs = 0
        known_sizes = {2000, 10000, 50000}
        for s in stages:
            sz_str, ep_str = s.split(":")
            sz, ep = int(sz_str), int(ep_str)
            if sz not in known_sizes:
                raise ValueError(f"Unknown schedule subset size {sz}; known sizes are {known_sizes}.")
            if ep <= 0:
                raise ValueError(f"Schedule epochs must be positive; got {ep}.")
            total_sched_epochs += ep
        if total_sched_epochs != args.confirmation_epochs:
            raise ValueError(
                f"Schedule epoch sum ({total_sched_epochs}) must equal confirmation_epochs ({args.confirmation_epochs})."
            )


def run_deep_pso_study(args: argparse.Namespace) -> Dict[str, Any]:
    validate_cli_args(args)

    device = resolve_execution_device(args.device)
    print(f"=== MNIST Deep PSO Methods Study (Protocol {PROTOCOL_VERSION}) ===")
    print(f"Device: {device}")

    # Data preparation
    (
        x_search, y_search,
        x_val, y_val,
        x_test, y_test,
        nested_subsets,
        data_fp, provenance
    ) = prepare_mnist_v5_data(split_seed=args.split_seed)

    base_model = make_compact_cnn(seed=41).to(device)
    base_fp = compute_model_fingerprint(base_model)
    hardware_prov = get_hardware_provenance(device)

    failure_record: Optional[str] = None

    # Pilot Phase: Validation-only selection of latent dimension
    print("\n--- Pilot Phase (Dimension Selection on 10k Validation Set) ---")
    pilot_results = []
    best_pilot_dim = None
    best_pilot_val_loss = float("inf")
    best_pilot_val_acc = 0.0

    dimensions = args.dimensions

    for dim_str in dimensions:
        print(f"Running Pilot: dim={dim_str}, seed={args.pilot_seed}, particles={args.pilot_particles}, epochs={args.pilot_epochs}")
        transform = LatentTransform(base_model, latent_dim=dim_str, device=device)
        res = run_latent_pso(
            transform=transform,
            base_model=base_model,
            x_search=x_search,
            y_search=y_search,
            nested_subsets=nested_subsets,
            schedule_str=f"2000:{args.pilot_epochs}",
            epochs=args.pilot_epochs,
            swarm_size=args.pilot_particles,
            seed=args.pilot_seed,
            device=device,
        )

        model = make_compact_cnn(seed=41).to(device)
        transform.load_vector_to_model(transform.decode(res["gbest_z"].unsqueeze(0)).squeeze(0), model)

        probs_val = get_model_probabilities(model, x_val, device)
        val_metrics = evaluate_probabilistic_metrics(probs_val, y_val)

        pilot_record = {
            "dimension": str(dim_str),
            "val_loss": val_metrics["nll"],
            "val_acc": val_metrics["accuracy"],
            "wall_time_sec": res["wall_time_sec"],
            "queries": res["total_queries"],
            "sample_evaluations": res["total_sample_evaluations"],
        }
        pilot_results.append(pilot_record)
        print(f"Pilot dim={dim_str} -> Val Loss: {val_metrics['nll']:.6f}, Val Acc: {val_metrics['accuracy']:.2f}%")

        if (val_metrics["nll"] < best_pilot_val_loss - 1e-7) or (abs(val_metrics["nll"] - best_pilot_val_loss) <= 1e-7 and val_metrics["accuracy"] > best_pilot_val_acc):
            best_pilot_val_loss = val_metrics["nll"]
            best_pilot_val_acc = val_metrics["accuracy"]
            best_pilot_dim = dim_str

    print(f"\nSelected Pilot Dimension: {best_pilot_dim} (Val Loss: {best_pilot_val_loss:.6f}, Val Acc: {best_pilot_val_acc:.2f}%)")

    # Confirmation Phase: Run multi-seed progressive PSO on selected dimension
    print(f"\n--- Confirmation Phase (Dimension={best_pilot_dim}, Seeds={args.seeds}) ---")
    confirmation_runs = []
    val_candidate_pool = []

    conf_transform = LatentTransform(base_model, latent_dim=best_pilot_dim, device=device)

    for c_seed in args.seeds:
        print(f"Running Confirmation: seed={c_seed}, particles={args.confirmation_particles}, epochs={args.confirmation_epochs}, schedule={args.confirmation_schedule}")
        res = run_latent_pso(
            transform=conf_transform,
            base_model=base_model,
            x_search=x_search,
            y_search=y_search,
            nested_subsets=nested_subsets,
            schedule_str=args.confirmation_schedule,
            epochs=args.confirmation_epochs,
            swarm_size=args.confirmation_particles,
            seed=c_seed,
            device=device,
        )

        # Evaluate EVERY particle's pbest on the 10k VALIDATION set
        P_final = res["final_P"]
        run_best_val_loss = float("inf")
        run_best_val_acc = 0.0

        for p_idx in range(len(P_final)):
            p_z = P_final[p_idx]
            cand_model = make_compact_cnn(seed=41).to(device)
            conf_transform.load_vector_to_model(conf_transform.decode(p_z.unsqueeze(0)).squeeze(0), cand_model)
            probs_val = get_model_probabilities(cand_model, x_val, device)
            val_metrics = evaluate_probabilistic_metrics(probs_val, y_val)

            cand_rec = {
                "seed": c_seed,
                "particle_idx": p_idx,
                "val_loss": val_metrics["nll"],
                "val_acc": val_metrics["accuracy"],
                "val_probs": probs_val,
                "latent_z": p_z,
            }
            val_candidate_pool.append(cand_rec)

            if (val_metrics["nll"] < run_best_val_loss - 1e-7) or (abs(val_metrics["nll"] - run_best_val_loss) <= 1e-7 and val_metrics["accuracy"] > run_best_val_acc):
                run_best_val_loss = val_metrics["nll"]
                run_best_val_acc = val_metrics["accuracy"]

        conf_record = {
            "seed": c_seed,
            "val_loss": run_best_val_loss,
            "val_acc": run_best_val_acc,
            "wall_time_sec": res["wall_time_sec"],
            "queries": res["total_queries"],
            "sample_evaluations": res["total_sample_evaluations"],
            "transition_reevaluations": res["transition_reevaluation_counts"],
            "stage_histories": res["stage_histories"],
        }
        confirmation_runs.append(conf_record)

        print(f"Confirmation seed={c_seed} -> Best Val Loss: {run_best_val_loss:.6f}, Best Val Acc: {run_best_val_acc:.2f}%")

    # Selection on Validation ONLY:
    # 1. Single Final Model (lowest val NLL, then highest val Acc)
    val_candidate_pool.sort(key=lambda c: (c["val_loss"], -c["val_acc"]))
    best_single_candidate = val_candidate_pool[0]

    # 2. Predeclared validation-performing, prediction-diverse Top-5 Ensemble
    top_ensemble_candidates = select_diverse_candidates(val_candidate_pool)

    # Official 10k Test Set Evaluation (EXACTLY ONCE PER ENDPOINT)
    print("\n--- Final Official 10k Test Evaluation ---")

    # Single Final Model Test Evaluation
    single_model = make_compact_cnn(seed=41).to(device)
    conf_transform.load_vector_to_model(conf_transform.decode(best_single_candidate["latent_z"].unsqueeze(0)).squeeze(0), single_model)

    single_test_probs = get_model_probabilities(single_model, x_test, device)
    single_test_metrics = evaluate_probabilistic_metrics(single_test_probs, y_test)
    single_model_fp = compute_model_fingerprint(single_model)

    print(f"Final Single Model (Seed {best_single_candidate['seed']}, Part {best_single_candidate['particle_idx']}) -> Test Acc: {single_test_metrics['accuracy']:.2f}%, Test NLL: {single_test_metrics['nll']:.6f}")

    # Top Ensemble Test Evaluation
    ensemble_test_probs_list = []
    ensemble_preds_list = []

    for cand in top_ensemble_candidates:
        cand_model = make_compact_cnn(seed=41).to(device)
        conf_transform.load_vector_to_model(conf_transform.decode(cand["latent_z"].unsqueeze(0)).squeeze(0), cand_model)
        t_probs = get_model_probabilities(cand_model, x_test, device)
        ensemble_test_probs_list.append(t_probs)
        ensemble_preds_list.append(t_probs.argmax(dim=1).cpu().numpy())

    ensemble_mean_probs = torch.stack(ensemble_test_probs_list).mean(dim=0)
    ensemble_test_metrics = evaluate_probabilistic_metrics(ensemble_mean_probs, y_test)
    ensemble_disagreement = compute_pairwise_disagreement(ensemble_preds_list)

    print(f"Top-{len(top_ensemble_candidates)} Ensemble -> Test Acc: {ensemble_test_metrics['accuracy']:.2f}%, Test NLL: {ensemble_test_metrics['nll']:.6f}, Disagreement: {ensemble_disagreement:.6f}")
    ensemble_val_probs = torch.stack(
        [cand["val_probs"] for cand in top_ensemble_candidates]
    ).mean(dim=0)
    ensemble_val_metrics = evaluate_probabilistic_metrics(
        ensemble_val_probs, y_val
    )
    ensemble_val_disagreement = compute_pairwise_disagreement(
        [
            cand["val_probs"].argmax(dim=1).cpu().numpy()
            for cand in top_ensemble_candidates
        ]
    )


    total_wall_time = sum(c["wall_time_sec"] for c in confirmation_runs) + sum(p["wall_time_sec"] for p in pilot_results)
    total_queries_all = sum(c["queries"] for c in confirmation_runs) + sum(p["queries"] for p in pilot_results)
    total_samples_all = sum(c["sample_evaluations"] for c in confirmation_runs) + sum(p["sample_evaluations"] for p in pilot_results)

    final_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "pso_version": pso_version,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "completed": True,
        "hardware_provenance": hardware_prov,
        "data_provenance": provenance,
        "data_fingerprint": data_fp,
        "base_model_fingerprint": base_fp,
        "configuration": {
            "base_model_seed": 41,
            "split_seed": args.split_seed,
            "pilot_seed": args.pilot_seed,
            "pilot_dimensions": [str(dim) for dim in args.dimensions],
            "pilot_particles": args.pilot_particles,
            "pilot_epochs": args.pilot_epochs,
            "confirmation_seeds": args.seeds,
            "confirmation_particles": args.confirmation_particles,
            "confirmation_epochs": args.confirmation_epochs,
            "confirmation_schedule": args.confirmation_schedule,
            "fitness_objective": "cross_entropy_loss_primary_accuracy_tiebreak",
            "parameterization": {
                "layer_scale": "per_parameter_tensor_std_floor_1e-4",
                "subspace": "deterministic_sparse_signed_hash_count_normalized",
                "initialization": "exact_base_plus_antithetic",
                "initial_radius": 0.5,
                "reflective_bound": 3.0,
            },
            "movement": {
                "name": "latent_adaptive_moment_pso",
                "c0": 1.49618,
                "c1": 1.49618,
                "w": 0.7298,
                "moment_blend": 0.06,
                "moment_step_size": 0.5,
                "moment_beta1": 0.9,
                "moment_beta2": 0.999,
            },
            "objective_transition": "reevaluate_all_pbests_then_rebuild_gbest",
            "validation_selection": {
                "single": "lowest_nll_then_highest_accuracy",
                "ensemble": "within_2_accuracy_points_then_greedy_disagreement",
                "ensemble_size": 5,
            },
            "fitness_batch_size": 1000,
            "ece_bins": 15,
        },
        "pilot_phase": {
            "selected_dimension": str(best_pilot_dim),
            "results": pilot_results,
        },
        "confirmation_phase": {
            "runs": [
                {
                    "seed": r["seed"],
                    "val_loss": r["val_loss"],
                    "val_acc": r["val_acc"],
                    "wall_time_sec": r["wall_time_sec"],
                    "queries": r["queries"],
                    "sample_evaluations": r["sample_evaluations"],
                    "transition_reevaluations": r["transition_reevaluations"],
                    "stage_histories": r["stage_histories"],
                }
                for r in confirmation_runs
            ],
            "validation_summary": {
                "best_pbest_nll": calc_stats(
                    [run["val_loss"] for run in confirmation_runs]
                ),
                "best_pbest_accuracy": calc_stats(
                    [run["val_acc"] for run in confirmation_runs]
                ),
                "wall_time_sec": calc_stats(
                    [run["wall_time_sec"] for run in confirmation_runs]
                ),
            },
        },
        "final_endpoints": {
            "single_model": {
                "selected_seed": best_single_candidate["seed"],
                "selected_particle_idx": best_single_candidate["particle_idx"],
                "model_fingerprint": single_model_fp,
                "val_loss": best_single_candidate["val_loss"],
                "val_acc": best_single_candidate["val_acc"],
                "selection_rule": "lowest_validation_nll_then_highest_accuracy",
                "test_accuracy": single_test_metrics["accuracy"],
                "test_nll": single_test_metrics["nll"],
                "test_brier": single_test_metrics["brier"],
                "test_ece": single_test_metrics["ece"],
                "test_margin": single_test_metrics["margin"],
            },
            "ensemble": {
                "ensemble_size": len(top_ensemble_candidates),
                "members": [
                    {
                        "seed": cand["seed"],
                        "particle_idx": cand["particle_idx"],
                        "val_loss": cand["val_loss"],
                        "val_acc": cand["val_acc"],
                    }
                    for cand in top_ensemble_candidates
                ],
                "validation_accuracy": ensemble_val_metrics["accuracy"],
                "validation_nll": ensemble_val_metrics["nll"],
                "validation_brier": ensemble_val_metrics["brier"],
                "validation_ece": ensemble_val_metrics["ece"],
                "validation_margin": ensemble_val_metrics["margin"],
                "validation_pairwise_disagreement": ensemble_val_disagreement,
                "test_accuracy": ensemble_test_metrics["accuracy"],
                "test_nll": ensemble_test_metrics["nll"],
                "test_brier": ensemble_test_metrics["brier"],
                "test_ece": ensemble_test_metrics["ece"],
                "test_margin": ensemble_test_metrics["margin"],
                "pairwise_disagreement": ensemble_disagreement,
            },
        },
        "accounting": {
            "total_wall_time_sec": round(total_wall_time, 4),
            "total_queries": total_queries_all,
            "total_sample_evaluations": total_samples_all,
            "scope": (
                "pilot_and_confirmation_training_objectives_including_"
                "transition_reevaluations; excludes validation and test"
            ),
        },
        "failure_record": failure_record,
    }

    # Save atomic JSON
    json_path = Path(args.json_path)
    save_json_atomic(final_payload, json_path)
    print(f"Saved atomic JSON to {json_path}")

    # Save CSV
    csv_path = Path(args.csv_path)
    save_csv_summary(final_payload, csv_path)
    print(f"Saved CSV summary to {csv_path}")

    # Save PNG plot
    plot_path = Path(args.plot_path)
    generate_study_plots(final_payload, plot_path)
    print(f"Saved PNG plot to {plot_path}")

    return final_payload


def get_model_probabilities(model: nn.Module, x_data: torch.Tensor, device: torch.device, batch_size: int = 1000) -> torch.Tensor:
    model.eval()
    prob_list = []
    num_samples = len(x_data)
    with torch.inference_mode():
        for b_start in range(0, num_samples, batch_size):
            xb = x_data[b_start:b_start + batch_size].to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            prob_list.append(probs)
    return torch.cat(prob_list, dim=0)


def save_csv_summary(payload: Dict[str, Any], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "metric", "value"])
        writer.writerow(["protocol", "version", payload["protocol_version"]])

        pilot = payload["pilot_phase"]
        writer.writerow(["pilot", "selected_dimension", pilot["selected_dimension"]])

        single = payload["final_endpoints"]["single_model"]
        writer.writerow(["single_model", "test_accuracy", single["test_accuracy"]])
        writer.writerow(["single_model", "test_nll", single["test_nll"]])
        writer.writerow(["single_model", "test_brier", single["test_brier"]])
        writer.writerow(["single_model", "test_ece", single["test_ece"]])

        ens = payload["final_endpoints"]["ensemble"]
        writer.writerow(["ensemble", "test_accuracy", ens["test_accuracy"]])
        writer.writerow(["ensemble", "test_nll", ens["test_nll"]])
        writer.writerow(["ensemble", "pairwise_disagreement", ens["pairwise_disagreement"]])


def generate_study_plots(payload: Dict[str, Any], plot_path: Path):
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Pilot Dimension Selection
    pilot_results = payload["pilot_phase"]["results"]
    dims = [r["dimension"] for r in pilot_results]
    val_losses = [r["val_loss"] for r in pilot_results]
    val_accs = [r["val_acc"] for r in pilot_results]

    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    b1 = ax1.bar(np.arange(len(dims)) - 0.2, val_losses, width=0.4, color="#56B4E9", label="Val NLL")
    b2 = ax1_twin.bar(np.arange(len(dims)) + 0.2, val_accs, width=0.4, color="#009E73", label="Val Acc (%)")
    ax1.set_xticks(range(len(dims)))
    ax1.set_xticklabels(dims)
    ax1.set_xlabel("Latent Dimension")
    ax1.set_ylabel("Validation NLL")
    ax1_twin.set_ylabel("Validation Accuracy (%)")
    ax1.set_title("Pilot Dimension Selection")

    # Panel 2: Confirmation Training Histories
    ax2 = axes[1]
    conf_runs = payload["confirmation_phase"]["runs"]
    for run in conf_runs:
        hist = run["stage_histories"]
        epochs = [h["epoch"] for h in hist]
        losses = [h["gbest_loss"] for h in hist]
        ax2.plot(epochs, losses, label=f"Seed {run['seed']}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Gbest CE Loss")
    ax2.set_title("Confirmation Stage Training Histories")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Final Endpoint Comparison
    ax3 = axes[2]
    single_acc = payload["final_endpoints"]["single_model"]["test_accuracy"]
    ens_acc = payload["final_endpoints"]["ensemble"]["test_accuracy"]
    single_nll = payload["final_endpoints"]["single_model"]["test_nll"]
    ens_nll = payload["final_endpoints"]["ensemble"]["test_nll"]

    x_labels = ["Single Model", "Top-5 Ensemble"]
    accs = [single_acc, ens_acc]
    nlls = [single_nll, ens_nll]

    ax3_twin = ax3.twinx()
    ax3.bar(np.arange(2) - 0.15, accs, width=0.3, color="#CC79A7", label="Test Acc (%)")
    ax3_twin.bar(np.arange(2) + 0.15, nlls, width=0.3, color="#D55E00", label="Test NLL")
    ax3.set_xticks(range(2))
    ax3.set_xticklabels(x_labels)
    ax3.set_ylabel("Test Accuracy (%)")
    ax3_twin.set_ylabel("Test NLL")
    ax3.set_title("Final Official Test Endpoints")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNIST Deep PSO Methods Study (V5)")
    parser.add_argument("--pilot-epochs", type=int, default=160)
    parser.add_argument("--pilot-particles", type=int, default=30)
    parser.add_argument("--pilot-seed", type=int, default=91)
    parser.add_argument("--split-seed", type=int, default=20260902)
    parser.add_argument("--confirmation-epochs", type=int, default=600)
    parser.add_argument("--confirmation-particles", type=int, default=60)
    parser.add_argument("--confirmation-schedule", type=str, default="2000:420,10000:135,50000:45")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--dimensions", nargs="+", type=str, default=["290", "1024", "4096", "full"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--json-path", type=str, default="benchmark_results/pso_v5_deep_methods.json")
    parser.add_argument("--csv-path", type=str, default="benchmark_results/pso_v5_deep_methods.csv")
    parser.add_argument("--plot-path", type=str, default="history_plt/pso_v5_deep_methods.png")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_deep_pso_study(args)
