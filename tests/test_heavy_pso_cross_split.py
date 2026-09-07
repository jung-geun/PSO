"""
Unit tests for Heavy PSO Cross-Split Runner and Evaluator.

Covers:
1. Split seed propagation through data preparation, confirm runner, and autoresearch.
2. Per-workload projection seed override validation, parsing, and effective seeds.
3. Artifact accounting (exact queries/samples), fingerprint matching, state math, and test seals.
4. Cross-split runner phase seed enforcement (development vs confirmation).
5. All evaluator hard gates, missing confirmation rejection, non-finite rejection, leakage control,
   per-cell non-regression, development gates, confirmation gates, combined gates, and score formula.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

# Ensure test directory and repo root are in Python path
REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name != "PSO" else Path(__file__).resolve().parent
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evaluate_heavy_cross_split as evaluator
import heavy_pso_autoresearch as autoresearch
import heavy_pso_cross_split as runner
import heavy_task_feasibility as heavy_task


@pytest.fixture
def mock_heavy_task_deps(monkeypatch):
    """Mocks data preparation and execution functions for fast, deterministic unit testing."""
    split_records = {}

    def fake_prepare_heavy_task_data(dataset_name: str, split_seed: int = 20260902, cache_dir=None):
        N_search, N_val = 20, 10
        x_search = torch.randn(N_search, 1, 28, 28)
        y_search = torch.randint(0, 10, (N_search,))
        x_val = torch.randn(N_val, 1, 28, 28)
        y_val = torch.randint(0, 10, (N_val,))
        nested_subsets = {2000: torch.arange(10), 10000: torch.arange(20), 50000: torch.arange(20)}
        data_fp = f"data-fp-{dataset_name}-{split_seed}"
        split_fp = f"split-fp-{dataset_name}-{split_seed}"
        split_records[dataset_name] = split_seed
        provenance = {
            "dataset_name": dataset_name,
            "official_test_data_loaded": False,
            "official_test_evaluations": 0,
            "test_samples": 0,
            "search_samples": 50000,
            "val_samples": 10000,
            "split_seed": split_seed,
            "split_fingerprint": split_fp,
            "data_fingerprint": data_fp,
        }
        return x_search, y_search, x_val, y_val, nested_subsets, data_fp, provenance

    monkeypatch.setattr(heavy_task, "prepare_heavy_task_data", fake_prepare_heavy_task_data)
    monkeypatch.setattr(autoresearch, "prepare_heavy_task_data", fake_prepare_heavy_task_data)

    def fake_run_v6_pso(
        transform, base_model, x_search, y_search, x_val, y_val, nested_subsets,
        schedule_str, epochs, swarm_size, seed, device, geom_config=None, val_check_interval=10
    ):
        return {
            "val_selected_loss": 0.50,
            "val_selected_acc": 85.0,
            "gbest_loss": 0.48,
            "gbest_acc": 86.0,
            "wall_time_sec": 0.01,
            "optimization_wall_time_sec": 0.01,
            "validation_wall_time_sec": 0.001,
            "total_queries": swarm_size * epochs,
            "total_sample_evaluations": swarm_size * epochs * 10000,
            "validation_evaluations": 2,
            "val_metrics": {"brier": 0.1, "ece": 0.02},
        }

    def fake_run_g8_optimizer(
        base_model, x_2k, y_2k, x_val, y_val, epochs, swarm_size, seed, device
    ):
        return {
            "val_selected_loss": 0.55,
            "val_selected_acc": 83.0,
            "gbest_loss": 0.52,
            "gbest_acc": 84.0,
            "wall_time_sec": 0.01,
            "optimization_wall_time_sec": 0.01,
            "validation_wall_time_sec": 0.001,
            "total_queries": swarm_size * epochs,
            "total_sample_evaluations": swarm_size * epochs * 10000,
            "validation_evaluations": 2,
            "val_metrics": {"brier": 0.12, "ece": 0.03},
        }

    monkeypatch.setattr(heavy_task, "run_v6_pso", fake_run_v6_pso)
    monkeypatch.setattr(heavy_task, "run_g8_optimizer", fake_run_g8_optimizer)
    monkeypatch.setattr(autoresearch, "run_v6_pso", fake_run_v6_pso)

    return split_records


# =====================================================================
# 1. Split Seed Propagation Tests
# =====================================================================

def test_split_seed_propagation(mock_heavy_task_deps):
    """Verify alternate split seeds reach prepare_heavy_task_data across runners."""
    res_confirm = heavy_task.run_heavy_task_confirm(
        workloads={"mnist_compact": heavy_task.WORKLOADS["mnist_compact"]},
        selected_methods={"mnist_compact": ["G8"]},
        particles=2,
        epochs=2,
        seeds=[101],
        split_seed=20260905,
    )
    assert res_confirm["mnist_compact"]["G8"]["split_seed"] == 20260905
    assert res_confirm["mnist_compact"]["G8"]["data_fingerprint"] == "data-fp-mnist-20260905"

    res_auto = autoresearch.run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=2,
        seeds=[101],
        split_seed=20260906,
        projection_seed_mode="explicit",
        projection_seed=12345,
    )
    meta = res_auto["workloads"]["mnist_compact"]
    assert meta["data_fingerprint"] == "data-fp-mnist-20260906"
    assert meta["split_fingerprint"] == "split-fp-mnist-20260906"


# =====================================================================
# 2. Projection Seed Override Validation & Parsing Tests
# =====================================================================

def test_projection_override_validation_and_parsing():
    """Verify per-workload projection seed dictionary validation and CLI argument parsing."""
    # Valid dict validation
    valid_dict = {
        "mnist_compact": 1800044939,
        "mnist_wide": 592157828,
        "fashion_compact": 1363313651,
        "fashion_wide": 189641451,
    }
    autoresearch.validate_projection_seed_config("explicit", valid_dict)

    # Valid CLI string parsing
    parsed_json = autoresearch.parse_projection_seed_arg(
        '{"mnist_compact": 1800044939, "mnist_wide": 592157828}'
    )
    assert parsed_json["mnist_compact"] == 1800044939

    parsed_kv = autoresearch.parse_projection_seed_arg(
        "mnist_compact:1800044939,mnist_wide:592157828"
    )
    assert parsed_kv["mnist_compact"] == 1800044939
    assert parsed_kv["mnist_wide"] == 592157828

    parsed_int = autoresearch.parse_projection_seed_arg("1800044939")
    assert parsed_int == 1800044939

    # Effective projection seeds in derive_projection_seed
    s1 = autoresearch.derive_projection_seed("mnist_compact", 0.5, 101, mode="explicit", projection_seed=valid_dict)
    s2 = autoresearch.derive_projection_seed("mnist_wide", 0.5, 101, mode="explicit", projection_seed=valid_dict)
    assert s1 == 1800044939
    assert s2 == 592157828

    # Invalid cases
    with pytest.raises(ValueError, match="Invalid projection_seed_mode"):
        autoresearch.validate_projection_seed_config("invalid_mode", None)

    with pytest.raises(ValueError, match="projection_seed must be provided"):
        autoresearch.validate_projection_seed_config("explicit", None)

    with pytest.raises(ValueError, match="Unknown workload_id"):
        autoresearch.validate_projection_seed_config("explicit", {"unknown_wl": 12345})

    with pytest.raises(ValueError, match="non-negative integer"):
        autoresearch.validate_projection_seed_config(
            "explicit",
            {
                "mnist_compact": -5,
                "mnist_wide": 1800044939,
                "fashion_compact": 1363313651,
                "fashion_wide": 189641451,
            },
        )
    with pytest.raises(ValueError, match="projection_seed can only be provided"):
        autoresearch.validate_projection_seed_config("coupled", 12345)


# =====================================================================
# 3. Artifact Accounting & Provenance Tests
# =====================================================================

def test_artifact_accounting_and_provenance(mock_heavy_task_deps):
    """Verify artifact accounting, fingerprint matching, state ratio, and official test seals."""
    payload = runner.run_heavy_pso_cross_split(
        phase="development",
        particles=2,
        epochs=2,
    )
    assert payload["phase"] == "development"
    assert payload["official_test_data_loaded"] is False
    assert payload["official_test_evaluations"] == 0

    res_totals = payload["resource_totals"]
    # 2 splits * 4 workloads * 3 seeds * 2 (baseline + candidate) = 48 runs
    assert res_totals["total_runs"] == 48
    # Each run has 2 particles * 2 epochs = 4 queries, 4 * 10000 = 40000 samples
    assert res_totals["total_queries"] == 48 * 4
    assert res_totals["total_samples_evaluated"] == 48 * 40000

    # Fingerprint matching check in splits payload
    dev_split = payload["splits"]["20260905"]
    b_fp = dev_split["baselines"]["mnist_compact"]["data_fingerprint"]
    c_fp = dev_split["candidates"]["mnist_compact"]["data_fingerprint"]
    assert b_fp == c_fp, "Baseline and candidate data fingerprints must match"


# =====================================================================
# 4. Phase Seed Enforcement Tests
# =====================================================================

def test_cross_split_runner_phase_enforcement(mock_heavy_task_deps):
    """Verify runner enforces exact phase split and swarm seeds."""
    dev_payload = runner.run_heavy_pso_cross_split(phase="development", particles=2, epochs=2)
    assert dev_payload["split_seeds"] == [20260905, 20260906]
    assert dev_payload["swarm_seeds"] == [101, 102, 103]

    conf_payload = runner.run_heavy_pso_cross_split(phase="confirmation", particles=2, epochs=2)
    assert conf_payload["split_seeds"] == [20260907]
    assert conf_payload["swarm_seeds"] == [111, 112, 113]

    with pytest.raises(ValueError, match="Invalid phase"):
        runner.run_heavy_pso_cross_split(phase="invalid_phase")


def test_cross_split_cli_defaults_to_frozen_policy():
    """The CLI must execute the frozen policy when no method flags are supplied."""
    args = runner.build_parser().parse_args([])
    assert args.geometry_policy == "baseline_aligned"
    assert args.projection_scope == "global"
    assert args.projection_seed_mode == "explicit"
    assert args.projection_seed is None


def test_cross_split_mixed_projection_scope(mock_heavy_task_deps):
    """Verify cross-split runner accepts and serializes mixed projection_scope dictionary mapping."""
    mixed_scope = {
        "mnist_compact": "global",
        "mnist_wide": "balanced_global",
        "fashion_compact": "global",
        "fashion_wide": "balanced_global",
    }
    payload = runner.run_heavy_pso_cross_split(
        phase="development",
        projection_scope=mixed_scope,
        particles=2,
        epochs=2,
    )
    assert payload["candidate_config"]["projection_scope"] == mixed_scope
    assert payload["workloads"]["mnist_compact"]["projection_scope"] == "global"
    assert payload["workloads"]["mnist_wide"]["projection_scope"] == "balanced_global"

    dev_split = payload["splits"]["20260905"]
    assert dev_split["candidates"]["mnist_compact"]["projection_scope"] == "global"
    assert dev_split["candidates"]["mnist_wide"]["projection_scope"] == "balanced_global"


def test_cross_split_mixed_projection_scope_two_hash(mock_heavy_task_deps):
    """Verify cross-split runner accepts and serializes mixed projection_scope dictionary mapping with two_hash_global on Wide."""
    mixed_scope = {
        "mnist_compact": "global",
        "mnist_wide": "two_hash_global",
        "fashion_compact": "global",
        "fashion_wide": "two_hash_global",
    }
    payload = runner.run_heavy_pso_cross_split(
        phase="development",
        projection_scope=mixed_scope,
        particles=2,
        epochs=2,
    )
    assert payload["candidate_config"]["projection_scope"] == mixed_scope
    assert payload["workloads"]["mnist_compact"]["projection_scope"] == "global"
    assert payload["workloads"]["mnist_wide"]["projection_scope"] == "two_hash_global"

    dev_split = payload["splits"]["20260905"]
    assert dev_split["candidates"]["mnist_compact"]["projection_scope"] == "global"
    assert dev_split["candidates"]["mnist_wide"]["projection_scope"] == "two_hash_global"


def test_cross_split_mixed_projection_scope_largest_tensor_hash(mock_heavy_task_deps):
    """Verify cross-split runner accepts and serializes mixed projection_scope dictionary mapping with largest_tensor_hash on Wide."""
    mixed_scope = {
        "mnist_compact": "global",
        "mnist_wide": "largest_tensor_hash",
        "fashion_compact": "global",
        "fashion_wide": "largest_tensor_hash",
    }
    payload = runner.run_heavy_pso_cross_split(
        phase="development",
        projection_scope=mixed_scope,
        particles=2,
        epochs=2,
    )
    assert payload["candidate_config"]["projection_scope"] == mixed_scope
    assert payload["workloads"]["mnist_compact"]["projection_scope"] == "global"
    assert payload["workloads"]["mnist_wide"]["projection_scope"] == "largest_tensor_hash"

    dev_split = payload["splits"]["20260905"]
    assert dev_split["candidates"]["mnist_compact"]["projection_scope"] == "global"
    assert dev_split["candidates"]["mnist_wide"]["projection_scope"] == "largest_tensor_hash"
def test_cross_split_mixed_projection_scope_largest_tensor_row_hash(mock_heavy_task_deps):
    """Verify cross-split runner accepts and serializes mixed projection_scope dictionary mapping with largest_tensor_row_hash on Wide."""
    mixed_scope = {
        "mnist_compact": "global",
        "mnist_wide": "largest_tensor_row_hash",
        "fashion_compact": "global",
        "fashion_wide": "largest_tensor_row_hash",
    }
    payload = runner.run_heavy_pso_cross_split(
        phase="development",
        projection_scope=mixed_scope,
        particles=2,
        epochs=2,
    )
    assert payload["candidate_config"]["projection_scope"] == mixed_scope
    assert payload["workloads"]["mnist_compact"]["projection_scope"] == "global"
    assert payload["workloads"]["mnist_wide"]["projection_scope"] == "largest_tensor_row_hash"

    dev_split = payload["splits"]["20260905"]
    assert dev_split["candidates"]["mnist_compact"]["projection_scope"] == "global"
    assert dev_split["candidates"]["mnist_wide"]["projection_scope"] == "largest_tensor_row_hash"
def test_cross_split_mixed_projection_scope_adjacent_pair(mock_heavy_task_deps):
    """Verify cross-split runner accepts and serializes mixed projection_scope dictionary mapping with adjacent_pair on Wide."""
    mixed_scope = {
        "mnist_compact": "global",
        "mnist_wide": "adjacent_pair",
        "fashion_compact": "global",
        "fashion_wide": "adjacent_pair",
    }
    payload = runner.run_heavy_pso_cross_split(
        phase="development",
        projection_scope=mixed_scope,
        particles=2,
        epochs=2,
    )
    assert payload["candidate_config"]["projection_scope"] == mixed_scope
    assert payload["workloads"]["mnist_compact"]["projection_scope"] == "global"
    assert payload["workloads"]["mnist_wide"]["projection_scope"] == "adjacent_pair"

    dev_split = payload["splits"]["20260905"]
    assert dev_split["candidates"]["mnist_compact"]["projection_scope"] == "global"
    assert dev_split["candidates"]["mnist_wide"]["projection_scope"] == "adjacent_pair"
def test_cross_split_mixed_projection_scope_adjacent_difference(mock_heavy_task_deps):
    """Verify cross-split runner accepts and serializes mixed projection_scope dictionary mapping with adjacent_difference on Wide."""
    mixed_scope = {
        "mnist_compact": "global",
        "mnist_wide": "adjacent_difference",
        "fashion_compact": "global",
        "fashion_wide": "adjacent_difference",
    }
    payload = runner.run_heavy_pso_cross_split(
        phase="development",
        projection_scope=mixed_scope,
        particles=2,
        epochs=2,
    )
    assert payload["candidate_config"]["projection_scope"] == mixed_scope
    assert payload["workloads"]["mnist_compact"]["projection_scope"] == "global"
    assert payload["workloads"]["mnist_wide"]["projection_scope"] == "adjacent_difference"

    dev_split = payload["splits"]["20260905"]
    assert dev_split["candidates"]["mnist_compact"]["projection_scope"] == "global"
    assert dev_split["candidates"]["mnist_wide"]["projection_scope"] == "adjacent_difference"

# =====================================================================
# 5. Evaluator Hard Gates & Score Calculation Tests
# =====================================================================

def create_synthetic_artifact(
    phase: str,
    acc_delta: float = 2.0,
    nll_delta: float = -0.10,
) -> Dict[str, Any]:
    split_seeds = [20260905, 20260906] if phase == "development" else [20260907]
    swarm_seeds = [101, 102, 103] if phase == "development" else [111, 112, 113]
    projection_seeds = dict(runner.FROZEN_PROJECTION_SEEDS)
    splits = {}
    for split_seed in split_seeds:
        baselines = {}
        candidates = {}
        for workload in evaluator.WORKLOADS:
            total_dim = evaluator.TOTAL_DIMS[workload]
            latent_dim = evaluator.compute_latent_dim(total_dim, 0.5)
            baseline_states = 5 * 12 + (
                1 if evaluator.BASELINE_METHODS[workload] == "G8" else 0
            )
            baseline_bytes = baseline_states * total_dim * 4
            candidate_bytes = evaluator.compute_core_swarm_state_bytes(12, latent_dim)
            baseline_runs = []
            candidate_runs = []
            for seed in swarm_seeds:
                common = {
                    "seed": seed,
                    "gbest_loss": 0.55,
                    "gbest_acc": 79.0,
                    "wall_time_sec": 1.0,
                    "optimization_wall_time_sec": 0.9,
                    "validation_wall_time_sec": 0.1,
                    "total_queries": 960,
                    "total_sample_evaluations": 9_600_000,
                    "validation_evaluations": 8,
                    "official_test_evaluations": 0,
                    "throughput_samples_per_sec": 10_666_666.0,
                }
                baseline_runs.append(
                    {
                        **common,
                        "val_selected_loss": 0.60,
                        "val_selected_acc": 80.0,
                        "val_metrics": {"nll": 0.60, "brier": 0.20, "ece": 0.05},
                        "core_swarm_state_bytes": baseline_bytes,
                    }
                )
                candidate_runs.append(
                    {
                        **common,
                        "projection_seed": projection_seeds[workload],
                        "projection_scope": "global",
                        "projection_seed_mode": "explicit",
                        "geometry_multiplier": 1.0,
                        "val_selected_loss": 0.60 + nll_delta,
                        "val_selected_acc": 80.0 + acc_delta,
                        "val_metrics": {
                            "nll": 0.60 + nll_delta,
                            "brier": 0.18,
                            "ece": 0.04,
                        },
                        "core_swarm_state_bytes": candidate_bytes,
                        "is_finite": True,
                    }
                )
            fingerprint = f"fp-{workload}-{split_seed}"
            shared_entry = {
                "workload_id": workload,
                "split_seed": split_seed,
                "data_fingerprint": fingerprint,
                "split_fingerprint": f"split-{workload}-{split_seed}",
                "subset_size": 10_000,
                "particles": 12,
                "epochs": 80,
                "seeds": list(swarm_seeds),
            }
            baselines[workload] = {
                **shared_entry,
                "method_id": evaluator.BASELINE_METHODS[workload],
                "stats": {
                    "val_acc": {"mean": 80.0},
                    "val_nll": {"mean": 0.60},
                },
                "per_seed_runs": baseline_runs,
            }
            candidates[workload] = {
                **shared_entry,
                "candidate_id": "pexplicit_aligned_r0.5",
                "ratio": 0.5,
                "geometry_policy": "baseline_aligned",
                "geometry_multiplier": 1.0,
                "projection_scope": "global",
                "projection_seed_mode": "explicit",
                "projection_seed": projection_seeds[workload],
                "total_dim": total_dim,
                "latent_dim": latent_dim,
                "state_ratio": candidate_bytes / baseline_bytes,
                "core_swarm_state_bytes": candidate_bytes,
                "baseline_core_swarm_state_bytes": baseline_bytes,
                "stats": {
                    "val_acc": {"mean": 80.0 + acc_delta},
                    "val_nll": {"mean": 0.60 + nll_delta},
                },
                "per_seed_runs": candidate_runs,
            }
        splits[str(split_seed)] = {
            "split_seed": split_seed,
            "baselines": baselines,
            "candidates": candidates,
        }
    total_runs = len(split_seeds) * 4 * len(swarm_seeds) * 2
    return {
        "version": runner.PROTOCOL_VERSION,
        "protocol_version": runner.PROTOCOL_VERSION,
        "phase": phase,
        "split_seeds": split_seeds,
        "swarm_seeds": swarm_seeds,
        "official_test_data_loaded": False,
        "official_test_evaluations": 0,
        "candidate_config": {
            "ratio": 0.5,
            "geometry_policy": "baseline_aligned",
            "projection_scope": "global",
            "projection_seed_mode": "explicit",
            "projection_seed": projection_seeds,
            "geometry_multiplier": 1.0,
            "particles": 12,
            "epochs": 80,
            "subset_size": 10_000,
        },
        "workloads": {
            workload: {
                "workload_id": workload,
                "dataset_name": (
                    "mnist" if workload.startswith("mnist") else "fashion_mnist"
                ),
                "model_name": (
                    "compact_cnn" if workload.endswith("compact") else "wide_cnn"
                ),
                "baseline_method": evaluator.BASELINE_METHODS[workload],
                "effective_projection_seed": projection_seeds[workload],
            }
            for workload in evaluator.WORKLOADS
        },
        "splits": splits,
        "resource_totals": {
            "total_runs": total_runs,
            "total_queries": total_runs * 960,
            "total_samples_evaluated": total_runs * 9_600_000,
            "official_test_evaluations": 0,
            "wall_time_sec": 1.0,
        },
    }


def test_evaluator_all_gates_and_score():
    """Verify evaluator passes valid dev + conf artifacts and rejects violations."""
    dev_art = create_synthetic_artifact("development", acc_delta=2.5, nll_delta=-0.05)
    conf_art = create_synthetic_artifact("confirmation", acc_delta=2.5, nll_delta=-0.05)

    res_pass = evaluator.evaluate_heavy_cross_split(dev_art, conf_art)
    assert res_pass["pass"] is True
    assert res_pass["failed_hard_gate_count"] == 0
    assert res_pass["score"] > 0

    # Test missing confirmation artifact rejection
    res_no_conf = evaluator.evaluate_heavy_cross_split(dev_art, None)
    assert res_no_conf["pass"] is False
    assert "confirmation_executed" in res_no_conf["failed_gates"]
    assert res_no_conf["failed_hard_gate_count"] > 0

    # Test test leakage rejection
    dev_leak = create_synthetic_artifact("development")
    dev_leak["official_test_data_loaded"] = True
    res_leak = evaluator.evaluate_heavy_cross_split(dev_leak, conf_art)
    assert res_leak["pass"] is False
    assert "official_test_sealed" in res_leak["failed_gates"]

    # Test non-finite metric rejection
    dev_inf = create_synthetic_artifact("development")
    dev_inf["splits"]["20260905"]["candidates"]["mnist_compact"]["per_seed_runs"][0]["val_selected_loss"] = float("nan")
    res_inf = evaluator.evaluate_heavy_cross_split(dev_inf, conf_art)
    assert res_inf["pass"] is False
    assert "all_runs_finite" in res_inf["failed_gates"]

    # Test fingerprint mismatch rejection
    dev_fp_mismatch = create_synthetic_artifact("development")
    dev_fp_mismatch["splits"]["20260905"]["candidates"]["mnist_compact"]["split_fingerprint"] = "bad-fp"
    res_fp_mismatch = evaluator.evaluate_heavy_cross_split(dev_fp_mismatch, conf_art)
    assert res_fp_mismatch["pass"] is False
    assert "split_and_fingerprint_matched" in res_fp_mismatch["failed_gates"]

    # Test per-cell accuracy regression violation (> 1.0 pp)
    dev_reg = create_synthetic_artifact("development", acc_delta=-1.5, nll_delta=0.0)
    res_reg = evaluator.evaluate_heavy_cross_split(dev_reg, conf_art)
    assert res_reg["pass"] is False
    assert "maximum_accuracy_regression_percentage_points_each_split_workload" in res_reg["failed_gates"]


def test_evaluator_accepts_development_only_as_confirmation_eligible():
    development = create_synthetic_artifact(
        "development", acc_delta=2.5, nll_delta=-0.05
    )
    result = evaluator.evaluate_heavy_cross_split(development)
    assert result["pass"] is False
    assert result["development_pass"] is True
    assert result["eligible_for_confirmation"] is True
    assert result["score_failed_gate_count"] == 0


def test_evaluator_rejects_missing_or_inconsistent_evidence():
    development = create_synthetic_artifact(
        "development", acc_delta=2.5, nll_delta=-0.05
    )
    confirmation = create_synthetic_artifact(
        "confirmation", acc_delta=2.5, nll_delta=-0.05
    )

    missing_fingerprint = create_synthetic_artifact(
        "development", acc_delta=2.5, nll_delta=-0.05
    )
    del missing_fingerprint["splits"]["20260905"]["candidates"]["mnist_compact"][
        "data_fingerprint"
    ]
    result = evaluator.evaluate_heavy_cross_split(
        missing_fingerprint, confirmation
    )
    assert "split_and_fingerprint_matched" in result["failed_gates"]

    inconsistent_stats = create_synthetic_artifact(
        "development", acc_delta=2.5, nll_delta=-0.05
    )
    inconsistent_stats["splits"]["20260905"]["candidates"]["mnist_compact"][
        "stats"
    ]["val_acc"]["mean"] += 1.0
    result = evaluator.evaluate_heavy_cross_split(
        inconsistent_stats, confirmation
    )
    assert "schema_and_phase_seeds" in result["failed_gates"]

    confirmation["candidate_config"]["ratio"] = 0.25
    result = evaluator.evaluate_heavy_cross_split(development, confirmation)
    assert "configuration_and_policy_matched" in result["failed_gates"]
def test_evaluator_rejects_missing_selected_metrics():
    conf_art = create_synthetic_artifact("confirmation", acc_delta=2.5, nll_delta=-0.05)

    for metric in ("val_selected_acc", "val_selected_loss"):
        # Test key deletion
        dev_art_del = create_synthetic_artifact("development", acc_delta=2.5, nll_delta=-0.05)
        run_del = dev_art_del["splits"]["20260905"]["candidates"]["mnist_compact"]["per_seed_runs"][0]
        del run_del[metric]
        result_del = evaluator.evaluate_heavy_cross_split(dev_art_del, conf_art)
        assert result_del["pass"] is False
        assert result_del["failed_hard_gate_count"] > 0
        assert (
            "schema_and_phase_seeds" in result_del["failed_gates"]
            or "all_runs_finite" in result_del["failed_gates"]
        )

        # Test non-numeric string
        dev_art_str = create_synthetic_artifact("development", acc_delta=2.5, nll_delta=-0.05)
        run_str = dev_art_str["splits"]["20260905"]["candidates"]["mnist_compact"]["per_seed_runs"][0]
        run_str[metric] = "invalid_string"
        result_str = evaluator.evaluate_heavy_cross_split(dev_art_str, conf_art)
        assert result_str["pass"] is False
        assert result_str["failed_hard_gate_count"] > 0
        assert (
            "schema_and_phase_seeds" in result_str["failed_gates"]
            or "all_runs_finite" in result_str["failed_gates"]
        )

        # Test None
        dev_art_none = create_synthetic_artifact("development", acc_delta=2.5, nll_delta=-0.05)
        run_none = dev_art_none["splits"]["20260905"]["candidates"]["mnist_compact"]["per_seed_runs"][0]
        run_none[metric] = None
        result_none = evaluator.evaluate_heavy_cross_split(dev_art_none, conf_art)
        assert result_none["pass"] is False
        assert result_none["failed_hard_gate_count"] > 0
        assert (
            "schema_and_phase_seeds" in result_none["failed_gates"]
            or "all_runs_finite" in result_none["failed_gates"]
        )
