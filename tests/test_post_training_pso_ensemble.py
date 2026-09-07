"""
Unit tests for Post-Training PSO Ensemble Study Runner.

Covers:
1. Protocol version and module export verification.
2. CachedProbabilityEnsemble softmax parameterization, forward log-prob normalization, and NLLLoss integration.
3. Probability cache validation for finite values, non-negativity, and row-sum normalization.
4. Mixture probabilities for uniform and one-hot weight configurations across PyTorch and NumPy arrays.
5. Probabilistic metrics computation (accuracy, NLL, Brier, ECE, margin).
6. Analytical gradient vs central finite-difference gradient verification for simplex NLL.
7. SLSQP solver optimization success, simplex constraint adherence, and NLL improvement.
8. Deterministic PSO optimization and exact query/sample accounting on synthetic probability caches.
9. Development gate boundary checks for safety, accounting, and quality limits.
10. Production-runner enforcement of the official-test seal on development failure.
11. Atomic file and CSV report writers.
"""

import sys
from pathlib import Path

# Ensure test directory and repo root are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name != "PSO" else Path(__file__).resolve().parent
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch
import torch.nn as nn

import post_training_pso_ensemble as study_module

from post_training_pso_ensemble import (
    PROTOCOL_VERSION,
    CachedProbabilityEnsemble,
    CompactCNN,
    atomic_write_file,
    compute_model_fingerprint,
    evaluate_development_gates,
    fit_uniform_temperature,
    mixture_probabilities,
    optimize_slsqp_weights,
    probabilistic_metrics,
    run_pso_weights,
    save_csv_report,
    simplex_nll_and_grad,
    validate_probability_cache,
)


def test_protocol_version():
    """Verify protocol version identifier adheres to required format."""
    assert isinstance(PROTOCOL_VERSION, str)
    assert PROTOCOL_VERSION.startswith("POST-TRAINING-PSO-ENSEMBLE")
    assert "1.1.0" in PROTOCOL_VERSION or "1.0.0" in PROTOCOL_VERSION


def test_cached_probability_ensemble_weights_and_forward():
    """Verify CachedProbabilityEnsemble parameterization, weight normalization, and log-probability output."""
    ensemble = CachedProbabilityEnsemble(num_members=5)
    weights = ensemble.weights()

    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (5,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert (weights >= 0).all()

    # Custom weight initialization
    init_w = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0])
    ensemble_custom = CachedProbabilityEnsemble(num_members=5, init_weights=init_w)
    assert torch.allclose(ensemble_custom.raw_weights, init_w)

    # Invalid init shape
    with pytest.raises(ValueError, match="init_weights must have shape"):
        CachedProbabilityEnsemble(num_members=5, init_weights=torch.tensor([1.0, 2.0]))

    # CachedProbabilityEnsemble has one canonical input shape: (N, M, K).
    N, K = 100, 10
    torch.manual_seed(42)
    raw_probs = torch.rand(5, N, K)
    member_probs_mnk = raw_probs / raw_probs.sum(dim=-1, keepdim=True)
    member_probs_nmk = member_probs_mnk.transpose(0, 1)

    log_probs = ensemble(member_probs_nmk)
    assert log_probs.shape == (N, K)

    # Verify exponentiated log probabilities sum to 1 per sample.
    probs = torch.exp(log_probs)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(N), atol=1e-5)

    # Integration with nn.NLLLoss.
    targets = torch.randint(0, K, (N,))
    loss = nn.NLLLoss()(log_probs, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    # Reject the alternate (M, N, K) orientation instead of guessing.
    with pytest.raises(ValueError, match="canonical"):
        ensemble(member_probs_mnk)

    # Square N == M caches remain unambiguous because the model always weights
    # axis 1 and mixture_probabilities always weights axis 0.
    square_raw = torch.arange(1, 51, dtype=torch.float32).reshape(5, 5, 2)
    square_mnk = square_raw / square_raw.sum(dim=-1, keepdim=True)
    raw_logits = torch.tensor([1.5, -0.5, 0.2, 0.8, -1.0])
    square_ensemble = CachedProbabilityEnsemble(5, init_weights=raw_logits)
    actual_square = torch.exp(square_ensemble(square_mnk.transpose(0, 1)))
    expected_square = mixture_probabilities(
        torch.softmax(raw_logits, dim=0),
        square_mnk,
    )
    assert torch.allclose(actual_square, expected_square, atol=1e-6)

    # Malformed dimension or member count mismatch.
    with pytest.raises(ValueError):
        ensemble(torch.rand(N, K))
    with pytest.raises(ValueError, match="canonical"):
        ensemble(torch.rand(N, 3, K))


def test_validate_probability_cache():
    """Verify probability cache validation logic for valid, negative, unnormalized, and non-finite cases."""
    N, K = 50, 10
    raw = torch.rand(5, N, K)
    valid_tensor = raw / raw.sum(dim=-1, keepdim=True)

    assert validate_probability_cache(valid_tensor) is True
    assert validate_probability_cache(valid_tensor.numpy()) is True

    # Negative values
    invalid_neg = valid_tensor.clone()
    invalid_neg[0, 0, 0] = -0.05
    assert validate_probability_cache(invalid_neg) is False

    # Unnormalized (row sum != 1.0)
    invalid_unnorm = valid_tensor.clone()
    invalid_unnorm[0, 0, :] *= 0.5
    assert validate_probability_cache(invalid_unnorm) is False

    # Non-finite values
    invalid_nan = valid_tensor.clone()
    invalid_nan[0, 0, 0] = float("nan")
    assert validate_probability_cache(invalid_nan) is False


def test_mixture_probabilities_uniform_and_one_hot():
    """Verify mixture_probabilities for uniform and one-hot weight configurations."""
    M, N, K = 5, 40, 10
    rng = np.random.RandomState(42)
    raw = rng.rand(M, N, K)
    member_probs_np = raw / raw.sum(axis=-1, keepdims=True)
    member_probs_torch = torch.from_numpy(member_probs_np).float()

    # 1. Uniform weights [0.2, 0.2, 0.2, 0.2, 0.2]
    uniform_w = np.full(M, 0.2)
    mix_uniform_np = mixture_probabilities(uniform_w, member_probs_np)
    expected_uniform = member_probs_np.mean(axis=0)
    assert np.allclose(mix_uniform_np, expected_uniform, atol=1e-6)

    mix_uniform_torch = mixture_probabilities(uniform_w, member_probs_torch)
    assert torch.allclose(mix_uniform_torch, torch.from_numpy(expected_uniform).float(), atol=1e-5)

    # 2. One-hot weights [1.0, 0.0, 0.0, 0.0, 0.0]
    onehot_0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    mix_onehot_0 = mixture_probabilities(onehot_0, member_probs_np)
    assert np.allclose(mix_onehot_0, member_probs_np[0], atol=1e-6)

    # 3. One-hot weights for model index 2
    onehot_2 = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    mix_onehot_2 = mixture_probabilities(onehot_2, member_probs_np)
    assert np.allclose(mix_onehot_2, member_probs_np[2], atol=1e-6)

    # Alternate (N, M, K) orientation is rejected rather than guessed.
    transposed_np = member_probs_np.transpose(1, 0, 2)
    with pytest.raises(ValueError, match="canonical"):
        mixture_probabilities(uniform_w, transposed_np)

    # Dimension mismatch
    with pytest.raises(ValueError):
        mixture_probabilities(np.array([0.5, 0.5]), member_probs_np)


def test_probabilistic_metrics():
    """Verify calculation of accuracy, NLL, Brier, ECE, and margin metrics."""
    N, K = 100, 10
    targets = np.random.RandomState(42).randint(0, K, size=N)

    # Perfect prediction: prob=1.0 at true target index
    perfect_probs = np.zeros((N, K), dtype=np.float64)
    perfect_probs[np.arange(N), targets] = 1.0

    metrics_perfect = probabilistic_metrics(perfect_probs, targets)
    assert metrics_perfect["accuracy"] == 100.0
    assert metrics_perfect["nll"] < 1e-4
    assert metrics_perfect["brier"] < 1e-4
    assert metrics_perfect["ece"] < 1e-4
    assert metrics_perfect["margin"] == 1.0

    # Uniform prediction (1/K per class)
    uniform_probs = np.full((N, K), 1.0 / K, dtype=np.float64)
    metrics_uniform = probabilistic_metrics(uniform_probs, targets)
    expected_nll = -np.log(1.0 / K)
    assert np.isclose(metrics_uniform["nll"], expected_nll, atol=1e-3)
    assert metrics_uniform["margin"] == 0.0


def test_simplex_nll_and_grad_vs_finite_difference():
    """Verify analytical simplex NLL gradient against central finite differences."""
    M, N, K = 5, 200, 10
    rng = np.random.RandomState(101)
    raw = rng.rand(M, N, K)
    member_probs = raw / raw.sum(axis=-1, keepdims=True)
    targets = rng.randint(0, K, size=N)

    weights = np.array([0.3, 0.2, 0.1, 0.25, 0.15], dtype=np.float64)
    nll_analytical, grad_analytical = simplex_nll_and_grad(weights, member_probs, targets)

    assert np.isfinite(nll_analytical)
    assert grad_analytical.shape == (M,)
    assert np.all(np.isfinite(grad_analytical))

    # Numerical gradient computation via central finite differences
    h = 1e-6
    grad_numerical = np.zeros(M, dtype=np.float64)
    for i in range(M):
        w_plus = weights.copy()
        w_plus[i] += h
        nll_plus, _ = simplex_nll_and_grad(w_plus, member_probs, targets)

        w_minus = weights.copy()
        w_minus[i] -= h
        nll_minus, _ = simplex_nll_and_grad(w_minus, member_probs, targets)

        grad_numerical[i] = (nll_plus - nll_minus) / (2.0 * h)

    assert np.allclose(grad_analytical, grad_numerical, atol=1e-4)


def test_optimize_slsqp_weights():
    """Verify SLSQP solver optimization success, simplex adherence, and NLL non-regression."""
    M, N, K = 5, 300, 10
    rng = np.random.RandomState(202)
    raw = rng.rand(M, N, K)
    member_probs = raw / raw.sum(axis=-1, keepdims=True)
    targets = rng.randint(0, K, size=N)

    # Make member 0 slightly better to give SLSQP a clear target
    member_probs[0, np.arange(N), targets] += 0.5
    member_probs = member_probs / member_probs.sum(axis=-1, keepdims=True)

    result = optimize_slsqp_weights(member_probs, targets)

    assert result["success"] is True
    assert len(result["weights"]) == M
    weights = np.array(result["weights"])
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)

    # Verify optimized NLL is no worse than uniform ensemble NLL
    uniform_p = mixture_probabilities(np.full(M, 1.0 / M), member_probs)
    uniform_nll = probabilistic_metrics(uniform_p, targets)["nll"]
    assert result["metrics"]["nll"] <= uniform_nll + 1e-6
    assert result["evaluations"] > 0
    assert result["wall_time_seconds"] >= 0.0


def test_run_pso_weights_determinism_and_accounting():
    """Verify PSO weight optimization determinism, exact accounting, and output structure."""
    M, N, K = 5, 100, 10
    rng = np.random.RandomState(303)
    raw = rng.rand(M, N, K)
    member_probs = raw / raw.sum(axis=-1, keepdims=True)
    targets = rng.randint(0, K, size=N)

    swarm_seeds = [301, 302]
    res_1 = run_pso_weights(member_probs, targets, swarm_seeds=swarm_seeds, device="cpu")

    # Accounting verification
    assert res_1["queries_per_seed"] == 900
    assert res_1["sample_evaluations_per_seed"] == 900 * N
    assert res_1["total_queries"] == 900 * len(swarm_seeds)
    assert res_1["total_sample_evaluations"] == 900 * N * len(swarm_seeds)

    per_seed = res_1["per_seed_runs"]
    assert len(per_seed) == len(swarm_seeds)
    for run_rec in per_seed:
        assert run_rec["queries"] == 900
        assert run_rec["sample_evaluations"] == 900 * N
        assert np.isclose(sum(run_rec["weights"]), 1.0, atol=1e-5)
        assert run_rec["wall_time_seconds"] >= 0.0

    # Repeatability / Determinism check
    res_2 = run_pso_weights(member_probs, targets, swarm_seeds=swarm_seeds, device="cpu")
    assert res_1["selected_seed"] == res_2["selected_seed"]
    assert np.allclose(res_1["selected_weights"], res_2["selected_weights"], atol=1e-5)
    assert np.isclose(
        res_1["per_seed_runs"][0]["metrics"]["nll"],
        res_2["per_seed_runs"][0]["metrics"]["nll"],
        atol=1e-5,
    )




def test_evaluate_development_gates_pass_and_boundary_failures():
    """Verify development gate boundary evaluations across passing and failing synthetic workloads."""
    def make_valid_workload(seed_nll=1.5, pso_nll=1.0, pso_acc=90.0, slsqp_nll=1.0):
        def make_mets(nll_val, acc_val):
            return {"accuracy": acc_val, "nll": nll_val, "brier": 0.15, "ece": 0.02, "margin": 0.5}

        return {
            "provenance": {"dataset_name": "mnist"},
            "training": {"adam_pool_wall_time_seconds": 100.0},
            "validation_cache": {
                "pool_forward_passes": 5,
                "base_cnn_forward_passes_during_optimization": 0,
            },
            "official_test_data_loaded_before_freeze": False,
            "official_test_evaluations_before_freeze": 0,
            "validation": {
                "methods": {
                    "reference_single_10e": make_mets(seed_nll, 80.0),
                    "best_single_10e": make_mets(1.4, 82.0),
                    "single_50e": make_mets(1.1, 88.0),
                    "uniform_ensemble": make_mets(1.05, 89.9),
                    "uniform_temperature": {
                        "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
                        "metrics": make_mets(1.04, 90.0),
                    },
                    "slsqp_weights": {
                        "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
                        "success": True,
                        "metrics": make_mets(slsqp_nll, 90.0),
                    },
                    "pso_weights": {
                        "selected_seed": 301,
                        "selected_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
                        "metrics": make_mets(pso_nll, pso_acc),
                        "median_one_seed_wall_time_seconds": 2.0,
                        "per_seed_runs": [
                            {
                                "seed": 301,
                                "queries": 900,
                                "sample_evaluations": 9000000,
                                "metrics": make_mets(pso_nll, pso_acc),
                                "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
                            },
                            {
                                "seed": 302,
                                "queries": 900,
                                "sample_evaluations": 9000000,
                                "metrics": make_mets(pso_nll + 0.01, pso_acc),
                                "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
                            },
                            {
                                "seed": 303,
                                "queries": 900,
                                "sample_evaluations": 9000000,
                                "metrics": make_mets(pso_nll + 0.02, pso_acc),
                                "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
                            },
                        ],
                    },
                }
            },
        }

    valid_workloads = {
        "mnist": make_valid_workload(),
        "fashion_mnist": make_valid_workload(),
    }

    eval_pass = evaluate_development_gates(valid_workloads)
    assert eval_pass["pass"] is True
    assert eval_pass["failed_hard_gate_count"] == 0
    assert len(eval_pass["gate_results"]) == 13

    # Assert exact expected gate names
    expected_gate_names = {
        "all_values_finite",
        "validation_pool_forward_passes_exact",
        "optimization_base_model_forward_passes",
        "official_test_data_loaded_before_freeze",
        "slsqp_solver_success",
        "query_and_sample_accounting_exact",
        "maximum_pso_nll_regression_vs_uniform",
        "maximum_pso_accuracy_regression_vs_uniform_pp",
        "pso_nll_below_reference_single",
        "maximum_pso_nll_regression_vs_equal_budget_single",
        "maximum_relative_pso_nll_gap_vs_slsqp",
        "cross_dataset_mean_relative_pso_nll_reduction_vs_uniform_minimum",
        "maximum_median_one_seed_pso_to_pool_training_wall_ratio",
    }
    assert set(eval_pass["gate_results"].keys()) == expected_gate_names

    # 1. Test data loaded before freeze failure
    leak_workloads = {
        "mnist": make_valid_workload(),
        "fashion_mnist": make_valid_workload(),
    }
    leak_workloads["mnist"]["official_test_data_loaded_before_freeze"] = True
    assert evaluate_development_gates(leak_workloads)["pass"] is False

    # 2. PSO accuracy regression > 0.1 pp below uniform
    acc_fail_workloads = {
        "mnist": make_valid_workload(pso_acc=89.0),  # Uniform is 89.9
        "fashion_mnist": make_valid_workload(),
    }
    assert evaluate_development_gates(acc_fail_workloads)["pass"] is False

    # 3. Base model called during optimization
    base_call_fail_workloads = {
        "mnist": make_valid_workload(),
        "fashion_mnist": make_valid_workload(),
    }
    base_call_fail_workloads["mnist"]["validation_cache"][
        "base_cnn_forward_passes_during_optimization"
    ] = 1
    assert evaluate_development_gates(base_call_fail_workloads)["pass"] is False


def test_global_test_seal_monkeypatch(monkeypatch, tmp_path):
    """A failed production development run must never construct train=False data."""
    import torchvision.datasets

    official_constructor_calls = []

    def guarded_dataset(*args, **kwargs):
        train = kwargs.get("train", True)
        official_constructor_calls.append(train)
        if train is False:
            raise RuntimeError("Leakage blocked: train=False requested before pass")
        raise AssertionError("Synthetic split setup must bypass train=True constructors")

    monkeypatch.setattr(torchvision.datasets, "MNIST", guarded_dataset)
    monkeypatch.setattr(torchvision.datasets, "FashionMNIST", guarded_dataset)

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = nn.Parameter(torch.zeros(10))

        def forward(self, x):
            return self.logits.unsqueeze(0).expand(len(x), -1)

    def fake_prepare(dataset_name, split_seed, cache_dir):
        x = torch.zeros(1, 1, 28, 28)
        y = torch.zeros(1, dtype=torch.long)
        return x, y, x.clone(), y.clone(), {
            "dataset_name": dataset_name,
            "split_seed": split_seed,
            "search_samples": 1,
            "validation_samples": 1,
            "normalization": {"mean": 0.0, "std": 1.0},
            "data_fingerprint": "synthetic",
            "split_fingerprint": "synthetic",
        }

    def fake_probabilities(model, x_data, device, batch_size=1000):
        with torch.no_grad():
            return torch.softmax(model(x_data.to(device)), dim=1).cpu(), 0.0

    def fake_temperature(uniform_probs, targets):
        metrics = probabilistic_metrics(uniform_probs, targets)
        return 1.0, {
            "fitted_temperature": 1.0,
            "wall_time_seconds": 0.0,
            "evaluations": 1,
            "metrics": metrics,
        }

    def fake_slsqp(member_probabilities, targets):
        weights = [0.2] * 5
        metrics = probabilistic_metrics(
            mixture_probabilities(weights, member_probabilities),
            targets,
        )
        return {
            "weights": weights,
            "evaluations": 1,
            "wall_time_seconds": 0.0,
            "success": True,
            "message": "synthetic",
            "metrics": metrics,
        }

    def fake_pso(
        member_probabilities,
        targets,
        swarm_seeds,
        particles,
        epochs,
        device,
    ):
        weights = [0.2] * 5
        metrics = probabilistic_metrics(
            mixture_probabilities(weights, member_probabilities),
            targets,
        )
        queries = particles * epochs
        samples = queries * len(targets)
        runs = [
            {
                "seed": seed,
                "queries": queries,
                "sample_evaluations": samples,
                "wall_time_seconds": 0.0,
                "metrics": metrics,
                "weights": weights,
            }
            for seed in swarm_seeds
        ]
        return {
            "per_seed_runs": runs,
            "selected_seed": swarm_seeds[0],
            "selected_weights": weights,
            "metrics": metrics,
            "queries_per_seed": queries,
            "sample_evaluations_per_seed": samples,
            "total_queries": queries * len(swarm_seeds),
            "total_sample_evaluations": samples * len(swarm_seeds),
            "median_one_seed_wall_time_seconds": 0.0,
            "total_wall_time_seconds": 0.0,
        }

    monkeypatch.setattr(study_module, "CompactCNN", TinyCNN)
    monkeypatch.setattr(study_module, "prepare_dataset_splits", fake_prepare)
    monkeypatch.setattr(study_module, "get_model_probabilities", fake_probabilities)
    monkeypatch.setattr(study_module, "fit_uniform_temperature", fake_temperature)
    monkeypatch.setattr(study_module, "optimize_slsqp_weights", fake_slsqp)
    monkeypatch.setattr(study_module, "run_pso_weights", fake_pso)
    monkeypatch.setattr(
        study_module,
        "evaluate_development_gates",
        lambda workloads: {
            "pass": False,
            "failed_hard_gate_count": 1,
            "gate_results": {"synthetic_failure": False},
            "issues": ["forced development failure"],
        },
    )
    monkeypatch.setattr(study_module, "save_csv_report", lambda *args: None)
    monkeypatch.setattr(study_module, "save_publication_plot", lambda *args: None)

    artifact = study_module.run_post_training_study(
        cache_dir=tmp_path / "cache",
        device="cpu",
        output_json=tmp_path / "study.json",
        output_csv=tmp_path / "study.csv",
        output_png=tmp_path / "study.png",
    )

    assert artifact["development_pass"] is False
    assert artifact["official_test_data_loaded"] is False
    assert all(
        workload["confirmation"] is None
        for workload in artifact["workloads"].values()
    )
    assert official_constructor_calls == []


def test_atomic_writers(tmp_path):
    """Verify atomic writing and CSV output formatting."""
    target_file = tmp_path / "report.csv"
    content = "header1,header2\nval1,val2\n"

    atomic_write_file(target_file, content)
    assert target_file.exists()
    assert target_file.read_text() == content

    # Test overwrite
    new_content = "header1,header2\nval3,val4\n"
    atomic_write_file(target_file, new_content)
    assert target_file.read_text() == new_content

    # Synthetic artifact CSV generation
    def make_mets(acc, nll):
        return {"accuracy": acc, "nll": nll, "brier": 0.15, "ece": 0.02, "margin": 0.5}

    artifact = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_frozen": True,
        "development_pass": True,
        "official_test_data_loaded": True,
        "resource_totals": {
            "total_pso_queries": 5400,
            "total_pso_sample_evaluations": 54000000,
            "total_pso_wall_time_seconds": 12.5,
        },
        "workloads": {
            "mnist": {
                "validation": {
                    "methods": {
                        "reference_single_10e": make_mets(85.0, 0.50),
                        "best_single_10e": make_mets(87.0, 0.45),
                        "single_50e": make_mets(89.0, 0.40),
                        "uniform_ensemble": make_mets(89.9, 0.36),
                        "uniform_temperature": {
                            "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
                            "metrics": make_mets(90.0, 0.355),
                        },
                        "slsqp_weights": {
                            "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
                            "wall_time_seconds": 0.5,
                            "metrics": make_mets(90.0, 0.35),
                        },
                        "pso_weights": {
                            "selected_seed": 301,
                            "selected_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
                            "metrics": make_mets(92.5, 0.25),
                            "median_one_seed_wall_time_seconds": 2.0,
                            "per_seed_runs": [
                                {
                                    "seed": 301,
                                    "metrics": make_mets(92.5, 0.25),
                                    "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
                                }
                            ],
                        },
                    }
                }
            }
        },
    }

    csv_path = tmp_path / "summary.csv"
    save_csv_report(artifact, csv_path)
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert len(lines) >= 2
    assert "Workload,Phase,Method" in lines[0]
