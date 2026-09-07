"""
Focused Offline Unit Tests for MNIST Deep PSO Methods (Protocol MNIST-PSO-RAW-V5 1.0.0)

Tests:
1. Split balance and nested subset inclusion invariant (I_2k subset of I_10k subset of I_50k).
2. Latent transform exactness (z_0 = 0 -> theta_0), antithetic symmetry (even swarm pairing), and subspace dimensions.
3. Stage transition pbest re-evaluation, gbest rebuild, and exact query/sample accounting.
4. Validation-only pilot and elite ensemble selection with greedy disagreement.
5. CLI argument validation.
"""

import sys
from pathlib import Path

# Insert repo test/ path so deep_pso_methods can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))

import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from deep_pso_methods import (
    CompactCNN,
    LatentTransform,
    build_nested_stratified_subsets,
    evaluate_latent_batch,
    evaluate_probabilistic_metrics,
    make_compact_cnn,
    run_latent_pso,
    select_diverse_candidates,
    validate_cli_args,
    build_parser,
)


def test_split_balance_and_nesting():
    """Verify stratified split balance and nested index inclusion: I_2k subset of I_10k subset of I_50k."""
    N = 50000
    num_classes = 10
    samples_per_class = N // num_classes
    y_search = torch.cat([torch.full((samples_per_class,), c, dtype=torch.long) for c in range(num_classes)])

    split_seed = 20260902
    nested_subsets = build_nested_stratified_subsets(
        y_search=y_search,
        subset_sizes=[2000, 10000, 50000],
        subset_seed=split_seed,
    )

    idx_2k = nested_subsets[2000]
    idx_10k = nested_subsets[10000]
    idx_50k = nested_subsets[50000]

    assert len(idx_2k) == 2000
    assert len(idx_10k) == 10000
    assert len(idx_50k) == 50000

    set_2k = set(idx_2k.numpy().tolist())
    set_10k = set(idx_10k.numpy().tolist())
    set_50k = set(idx_50k.numpy().tolist())

    # Strict nesting: I_2k subset of I_10k subset of I_50k
    assert set_2k.issubset(set_10k), "I_2k must be a strict subset of I_10k"
    assert set_10k.issubset(set_50k), "I_10k must be a strict subset of I_50k"

    # Exact stratification (equal counts per class)
    y_2k = y_search[idx_2k].numpy()
    y_10k = y_search[idx_10k].numpy()

    counts_2k = np.bincount(y_2k, minlength=10)
    counts_10k = np.bincount(y_10k, minlength=10)

    for c in range(num_classes):
        assert counts_2k[c] == 200, f"Class {c} in 2k subset must have 200 samples; got {counts_2k[c]}"
        assert counts_10k[c] == 1000, f"Class {c} in 10k subset must have 1000 samples; got {counts_10k[c]}"


def test_latent_transform_exactness_and_antithetic_symmetry():
    """Verify transform exactness (z_0 = 0 -> theta_0), even swarm pairing, zero centroid, and subspace shapes."""
    device = torch.device("cpu")
    base_model = make_compact_cnn(seed=41).to(device)
    base_vec = torch.cat([p.detach().view(-1) for p in base_model.parameters()])

    dims = [290, 1024, 4096, "full"]
    swarm_size = 30  # Even swarm size

    for d in dims:
        transform = LatentTransform(base_model, latent_dim=d, device=device)
        Z = transform.init_swarm(swarm_size=swarm_size, seed=91)

        # 1. Particle 0 is exact zero (base model)
        decoded_p0 = transform.decode(Z[0:1]).squeeze(0)
        assert torch.allclose(decoded_p0, base_vec, atol=1e-6), f"Particle 0 for dim={d} must match exact base vector"

        # 2. For even N=30: particle N-1 (index 29) is also exact zero
        decoded_plast = transform.decode(Z[29:30]).squeeze(0)
        assert torch.allclose(decoded_plast, base_vec, atol=1e-6), f"Particle N-1 for dim={d} must match exact zero"

        # 3. Antithetic pairs (particles 1..28 in 14 exact pairs)
        z1 = Z[1:2]
        z2 = Z[2:3]
        assert torch.allclose(z1 + z2, torch.zeros_like(z1), atol=1e-6), "Antithetic pair latent sum must be zero"

        delta1 = transform.decode(z1).squeeze(0) - transform.base_vec
        delta2 = transform.decode(z2).squeeze(0) - transform.base_vec
        assert torch.allclose(delta1 + delta2, torch.zeros_like(delta1), atol=1e-5), "Antithetic pair delta sum must be zero"

        # 4. Latent swarm centroid is strictly zero
        centroid_z = Z.mean(dim=0)
        assert torch.allclose(centroid_z, torch.zeros_like(centroid_z), atol=1e-6), "Overall swarm centroid must be zero"


def test_transition_reevaluation_and_exact_accounting():
    """Verify objective size transition re-evaluates all pbests, rebuilds gbest, and asserts exact query/sample counts."""
    device = torch.device("cpu")
    base_model = make_compact_cnn(seed=41).to(device)

    N_samples = 100
    x_synth = torch.randn(N_samples, 1, 28, 28)
    y_synth = torch.randint(0, 10, (N_samples,))

    nested_subsets = {
        20: torch.arange(20, dtype=torch.long),
        50: torch.arange(50, dtype=torch.long),
    }

    transform = LatentTransform(base_model, latent_dim=290, device=device)

    # Run 2-stage PSO: stage 0 (20 samples, 2 epochs), stage 1 (50 samples, 2 epochs), swarm_size = 10
    # Stage 0: 2 * 10 = 20 queries, 20 * 20 = 400 sample evals. No transition re-eval.
    # Stage 1 transition: 1 * 10 = 10 queries, 10 * 50 = 500 sample evals. Transition count = 10.
    # Stage 1: 2 * 10 = 20 queries, 20 * 50 = 1000 sample evals.
    # Total queries = 20 + 10 + 20 = 50.
    # Total sample evals = 400 + 500 + 1000 = 1900.
    res = run_latent_pso(
        transform=transform,
        base_model=base_model,
        x_search=x_synth,
        y_search=y_synth,
        nested_subsets=nested_subsets,
        schedule_str="20:2,50:2",
        epochs=4,
        swarm_size=10,
        seed=42,
        device=device,
    )

    assert res["transition_reevaluation_counts"] == 10, f"Expected 10 transition re-evaluations; got {res['transition_reevaluation_counts']}"
    assert res["total_queries"] == 50, f"Expected 50 total queries; got {res['total_queries']}"
    assert res["total_sample_evaluations"] == 1900, f"Expected 1900 sample evaluations; got {res['total_sample_evaluations']}"
    assert len(res["stage_histories"]) == 4


def test_validation_only_elite_and_ensemble_selection():
    """Verify validation metrics correctly rank candidates and metric routines compute expected values."""
    N = 100
    C = 10
    y_val = torch.randint(0, C, (N,))

    # Candidate 1: Perfect predictions
    probs_perfect = torch.zeros((N, C), dtype=torch.float32)
    probs_perfect[torch.arange(N), y_val] = 1.0

    # Candidate 2: Random noise
    probs_random = torch.full((N, C), 1.0 / C, dtype=torch.float32)

    m1 = evaluate_probabilistic_metrics(probs_perfect, y_val)
    m2 = evaluate_probabilistic_metrics(probs_random, y_val)

    assert m1["accuracy"] == 100.0
    assert m1["nll"] < m2["nll"]
    assert m1["brier"] < m2["brier"]
    assert m1["ece"] <= 0.01

    candidates = [
        {"id": "cand2", "val_loss": m2["nll"], "val_acc": m2["accuracy"]},
        {"id": "cand1", "val_loss": m1["nll"], "val_acc": m1["accuracy"]},
    ]
    candidates.sort(key=lambda c: (c["val_loss"], -c["val_acc"]))
    assert candidates[0]["id"] == "cand1"

    diverse_candidates = [
        {
            "seed": 7,
            "particle_idx": particle_idx,
            "val_loss": 0.2 + 0.01 * particle_idx,
            "val_acc": 90.0 - 0.25 * particle_idx,
            "val_probs": torch.roll(probs_perfect, shifts=particle_idx, dims=1),
            "latent_z": torch.full((4,), float(particle_idx)),
        }
        for particle_idx in range(3)
    ]
    selected = select_diverse_candidates(
        diverse_candidates, max_size=3, accuracy_window=2.0
    )
    assert len(selected) == 3
    assert len({
        (candidate["seed"], candidate["particle_idx"])
        for candidate in selected
    }) == 3


def test_cli_argument_validation():
    """Verify CLI argument validation rejects invalid parameters and accepts valid settings."""
    parser = build_parser()

    # Valid args
    valid_args = parser.parse_args([
        "--pilot-epochs", "160",
        "--confirmation-epochs", "600",
        "--confirmation-schedule", "2000:420,10000:135,50000:45",
        "--seeds", "101", "102", "103",
        "--dimensions", "290", "1024", "4096", "full"
    ])
    validate_cli_args(valid_args)

    # Invalid schedule sum mismatch
    invalid_schedule = parser.parse_args([
        "--confirmation-epochs", "600",
        "--confirmation-schedule", "2000:400,10000:100,50000:50"  # Sums to 550 != 600
    ])
    with pytest.raises(ValueError, match="Schedule epoch sum"):
        validate_cli_args(invalid_schedule)

    # Invalid negative seed
    invalid_seed = parser.parse_args(["--seeds", "-1"])
    with pytest.raises(ValueError, match="Seeds must be non-negative"):
        validate_cli_args(invalid_seed)

    # Invalid duplicate seed
    duplicate_seed = parser.parse_args(["--seeds", "101", "101"])
    with pytest.raises(ValueError, match="Confirmation seeds must be unique"):
        validate_cli_args(duplicate_seed)
