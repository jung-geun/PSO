"""
Offline Unit Tests for Heavy Task PSO Autoresearch & Evaluator Infrastructure.

Defends observable behavior, schema contracts, exact dimension/radius scaling,
projection seed determinism, evaluator gate boundaries (including the OR gate on mnist_wide),
config mismatches, non-finite rejection, zero-test enforcement, candidate selection preferences,
JSON artifact safety, and synthetic CPU experiment execution.
"""

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure test directory and repo root are in Python path
REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name != "PSO" else Path(__file__).resolve().parent
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_heavy_autoresearch import (
    BASELINE_POLICY,
    EVALUATOR_VERSION,
    EXPECTED_SEEDS,
    EXPECTED_WORKLOADS,
    evaluate_heavy_autoresearch,
)
import heavy_pso_autoresearch
from heavy_pso_autoresearch import (
    AUTORESEARCH_PROTOCOL_VERSION,
    DEFAULT_PROJECTION_SEED_MODE,
    GEOMETRY_POLICIES,
    PROJECTION_SEED_MODES,
    PROJECTION_SCOPES,
    TensorLocalLatentTransform,
    BalancedGlobalLatentTransform,
    TwoHashGlobalLatentTransform,
    LargestTensorHashLatentTransform,
    LargestTensorRowHashLatentTransform,
    AdjacentPairLatentTransform,
    AdjacentDifferenceLatentTransform,
    allocate_tensor_latent_dims,
    build_parser,
    compute_core_swarm_state_bytes,
    compute_baseline_core_swarm_state_bytes,
    compute_latent_dim,
    construct_equalized_geometry,
    derive_projection_seed,
    validate_projection_seed_config,
    parse_projection_seed_arg,
    validate_geometry_multiplier,
    parse_projection_scope_arg,
    validate_projection_scope_config,
    get_effective_projection_scope,
    format_ratio_id,
    run_heavy_pso_autoresearch,
)
from deep_pso_v6 import V6GeometryConfig, V6LatentTransform, get_v6_geometry_table


def _mock_prepare_heavy_task_data(dataset_name: str, split_seed: int = 20260902, cache_dir=None):
    N_search, N_val = 20, 10
    x_search = torch.randn(N_search, 1, 28, 28)
    y_search = torch.randint(0, 10, (N_search,))
    x_val = torch.randn(N_val, 1, 28, 28)
    y_val = torch.randint(0, 10, (N_val,))
    nested_subsets = {10: np.arange(10), 10000: np.arange(20)}
    data_fp = hashlib.sha256(dataset_name.encode()).hexdigest()[:16]
    provenance = {
        "split_fingerprint": f"split_{split_seed}_{data_fp}",
        "data_fingerprint": data_fp,
    }
    return x_search, y_search, x_val, y_val, nested_subsets, data_fp, provenance


def test_exact_dimension_and_radius_scaling():
    """Verify exact dimension rounding and sqrt(total_dim / latent_dim) radius scaling."""
    # CompactCNN (total_dim = 9098)
    assert compute_latent_dim(9098, 1.0) == 9098
    assert compute_latent_dim(9098, 0.5) == 4549
    assert compute_latent_dim(9098, 0.25) == 2275  # round(9098 * 0.25) = round(2274.5) = 2275
    assert compute_latent_dim(9098, 0.125) == 1137
    assert compute_latent_dim(9098, 0.03125) == 284

    # WideCNN (total_dim = 55338)
    assert compute_latent_dim(55338, 0.5) == 27669
    assert compute_latent_dim(55338, 0.25) == 13835
    assert compute_latent_dim(55338, 0.125) == 6917
    assert compute_latent_dim(55338, 0.03125) == 1729

    # Radius scaling test
    geom_table = get_v6_geometry_table()
    base_g6 = geom_table["G6"]  # position=1.5, vel=0.5, reset=0.02, bound=6.0
    total_dim = 9098
    latent_dim = 4549
    scale_factor = math.sqrt(total_dim / latent_dim)

    eq_g6 = construct_equalized_geometry(
        base_geom=base_g6,
        total_dim=total_dim,
        latent_dim=latent_dim,
        projection_seed=42,
        ratio_str="r0.5",
    )

    assert eq_g6.latent_dim == latent_dim
    assert eq_g6.projection_seed == 42
    assert math.isclose(eq_g6.position_radius, base_g6.position_radius * scale_factor)
    assert math.isclose(eq_g6.initial_velocity_radius, base_g6.initial_velocity_radius * scale_factor)
    assert math.isclose(eq_g6.reset_velocity_radius, base_g6.reset_velocity_radius * scale_factor)
    assert math.isclose(eq_g6.reflective_bound, base_g6.reflective_bound * scale_factor)


def test_deterministic_projection_seeds():
    """Verify projection seeds are deterministic, explicit, and vary by workload, ratio, and seed."""
    s1 = derive_projection_seed("mnist_compact", 0.5, 101)
    s2 = derive_projection_seed("mnist_compact", 0.5, 101)
    assert s1 == s2, "Projection seed derivation must be deterministic"
    assert isinstance(s1, int) and 0 <= s1 < 2**31

    # Variation checks
    s_diff_wl = derive_projection_seed("mnist_wide", 0.5, 101)
    s_diff_ratio = derive_projection_seed("mnist_compact", 0.25, 101)
    s_diff_seed = derive_projection_seed("mnist_compact", 0.5, 102)

    assert s1 != s_diff_wl, "Projection seed must vary by workload"
    assert s1 != s_diff_ratio, "Projection seed must vary by ratio"
    assert s1 != s_diff_seed, "Projection seed must vary by swarm seed"


def create_mock_baseline_json(tmp_path: Path) -> Path:
    """Helper creating a minimal valid baseline heavy tasks JSON artifact."""
    payload = {
        "protocol_version": "HEAVY-TASK-PSO-V6 1.0.0",
        "official_test_data_loaded": False,
        "official_test_evaluations": 0,
        "confirmation_results": {
            "mnist_compact": {
                "G8": {
                    "stats": {
                        "val_nll": {"mean": 1.50},
                        "val_acc": {"mean": 50.0},
                        "val_brier": {"mean": 0.65},
                        "val_ece": {"mean": 0.05},
                    }
                }
            },
            "mnist_wide": {
                "G5": {
                    "stats": {
                        "val_nll": {"mean": 1.70},
                        "val_acc": {"mean": 42.0},
                        "val_brier": {"mean": 0.70},
                        "val_ece": {"mean": 0.08},
                    }
                }
            },
            "fashion_compact": {
                "G8": {
                    "stats": {
                        "val_nll": {"mean": 1.60},
                        "val_acc": {"mean": 48.0},
                        "val_brier": {"mean": 0.68},
                        "val_ece": {"mean": 0.06},
                    }
                }
            },
            "fashion_wide": {
                "G5": {
                    "stats": {
                        "val_nll": {"mean": 1.75},
                        "val_acc": {"mean": 40.0},
                        "val_brier": {"mean": 0.72},
                        "val_ece": {"mean": 0.09},
                    }
                }
            },
        },
    }
    for workload_id, method_id in BASELINE_POLICY.items():
        entry = payload["confirmation_results"][workload_id][method_id]
        total_dim = 9098 if "compact" in workload_id else 55338
        baseline_bytes = compute_baseline_core_swarm_state_bytes(
            workload_id,
            particles=12,
            total_dim=total_dim,
        )
        entry["per_seed_runs"] = [
            {"seed": seed, "core_swarm_state_bytes": baseline_bytes}
            for seed in EXPECTED_SEEDS
        ]
    path = tmp_path / "mock_baseline.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def create_mock_candidate_json(
    tmp_path: Path,
    cand_id: str = "r0.5",
    ratio: float = 0.5,
    acc_deltas: Dict[str, float] = None,
    nll_deltas: Dict[str, float] = None,
    particles: int = 12,
    epochs: int = 80,
    subset_size: int = 10000,
    seeds: list = None,
    official_test_evals: int = 0,
    test_loaded: bool = False,
    is_finite: bool = True,
) -> Path:
    """Helper creating a minimal candidate heavy tasks JSON artifact."""
    if seeds is None:
        seeds = [101, 102, 103]
    if acc_deltas is None:
        acc_deltas = {"mnist_compact": 2.0, "mnist_wide": 3.0, "fashion_compact": 1.0, "fashion_wide": 1.0}
    if nll_deltas is None:
        nll_deltas = {"mnist_compact": -0.1, "mnist_wide": -0.1, "fashion_compact": -0.05, "fashion_wide": -0.05}

    base_accs = {"mnist_compact": 50.0, "mnist_wide": 42.0, "fashion_compact": 48.0, "fashion_wide": 40.0}
    base_nlls = {"mnist_compact": 1.50, "mnist_wide": 1.70, "fashion_compact": 1.60, "fashion_wide": 1.75}

    wl_map = {}
    for wl in EXPECTED_WORKLOADS:
        c_acc = base_accs[wl] + acc_deltas.get(wl, 0.0)
        c_nll = base_nlls[wl] + nll_deltas.get(wl, 0.0)

        if not is_finite:
            c_nll = float("nan")

        total_dim = 9098 if "compact" in wl else 55338
        latent_dim = compute_latent_dim(total_dim, ratio) if 0.0 < ratio <= 1.0 else max(1, int(total_dim * ratio))
        state_bytes = compute_core_swarm_state_bytes(particles, latent_dim)
        baseline_state_bytes = compute_baseline_core_swarm_state_bytes(
            wl,
            particles=particles,
            total_dim=total_dim,
        )

        per_seed = []
        for s in seeds:
            per_seed.append(
                {
                    "seed": s,
                    "val_selected_loss": c_nll,
                    "val_selected_acc": c_acc,
                    "val_metrics": {"brier": 0.6, "ece": 0.05},
                    "gbest_loss": c_nll,
                    "gbest_acc": c_acc,
                    "wall_time_sec": 1.0,
                    "total_queries": particles * epochs,
                    "total_sample_evaluations": particles * epochs * subset_size,
                    "official_test_evaluations": official_test_evals,
                    "core_swarm_state_bytes": state_bytes,
                    "is_finite": is_finite,
                }
            )

        wl_map[wl] = {
            "candidate_id": cand_id,
            "ratio": ratio,
            "workload_id": wl,
            "total_dim": total_dim,
            "latent_dim": latent_dim,
            "state_ratio": state_bytes / baseline_state_bytes,
            "particles": particles,
            "epochs": epochs,
            "subset_size": subset_size,
            "seeds": seeds,
            "core_swarm_state_bytes": state_bytes,
            "stats": {
                "val_nll": {"mean": c_nll},
                "val_acc": {"mean": c_acc},
            },
            "per_seed_runs": per_seed,
        }
    payload = {
        "protocol_version": AUTORESEARCH_PROTOCOL_VERSION,
        "official_test_data_loaded": test_loaded,
        "official_test_evaluations": official_test_evals * len(EXPECTED_WORKLOADS) * len(seeds),
        "candidate_runs": {cand_id: wl_map},
    }

    path = tmp_path / f"mock_candidate_{cand_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def test_evaluator_pass_fail_boundaries(tmp_path: Path):
    """Verify evaluator hard gate boundaries for state ratio, acc regression, and NLL regression."""
    b_path = create_mock_baseline_json(tmp_path)

    # 1. Valid passing candidate
    c_pass_path = create_mock_candidate_json(tmp_path, cand_id="r0.5", ratio=0.5)
    res_pass = evaluate_heavy_autoresearch(b_path, c_pass_path)
    assert res_pass["pass"] is True
    assert res_pass["selected_candidate_id"] == "r0.5"
    assert res_pass["candidate_evaluations"]["r0.5"]["pass"] is True
    assert math.isfinite(res_pass["score"])

    # 2. Gate 4 failure: state_ratio > 0.5
    c_ratio_fail = create_mock_candidate_json(tmp_path, cand_id="r0.6", ratio=0.6)
    res_ratio_fail = evaluate_heavy_autoresearch(b_path, c_ratio_fail)
    assert res_ratio_fail["pass"] is False
    assert "gate_state_ratio" in res_ratio_fail["candidate_evaluations"]["r0.6"]["failed_gates"]
    assert math.isfinite(res_ratio_fail["score"])

    # 3. Gate 5 failure: acc regression > 1.0 pp (e.g. -1.5 pp on fashion_compact)
    acc_fail_deltas = {"mnist_compact": 2.0, "mnist_wide": 3.0, "fashion_compact": -1.5, "fashion_wide": 1.0}
    c_acc_fail = create_mock_candidate_json(tmp_path, cand_id="r0.5_acc_fail", ratio=0.5, acc_deltas=acc_fail_deltas)
    res_acc_fail = evaluate_heavy_autoresearch(b_path, c_acc_fail)
    assert res_acc_fail["pass"] is False
    assert "gate_acc_regression" in res_acc_fail["candidate_evaluations"]["r0.5_acc_fail"]["failed_gates"]
    assert math.isfinite(res_acc_fail["score"])

    # 4. Gate 6 failure: NLL regression > 5% (e.g. +10% NLL on fashion_wide)
    # base fashion_wide NLL = 1.75 -> +10% is +0.175
    nll_fail_deltas = {"mnist_compact": -0.1, "mnist_wide": -0.1, "fashion_compact": -0.05, "fashion_wide": 0.20}
    c_nll_fail = create_mock_candidate_json(tmp_path, cand_id="r0.5_nll_fail", ratio=0.5, nll_deltas=nll_fail_deltas)
    res_nll_fail = evaluate_heavy_autoresearch(b_path, c_nll_fail)
    assert res_nll_fail["pass"] is False
    assert "gate_nll_regression" in res_nll_fail["candidate_evaluations"]["r0.5_nll_fail"]["failed_gates"]
    assert math.isfinite(res_nll_fail["score"])

def test_evaluator_or_worst_workload_gate(tmp_path: Path):
    """Verify Gate 7 (mnist_wide worst-workload improvement) OR condition."""
    b_path = create_mock_baseline_json(tmp_path)

    # Case A: acc_gain >= 2.0 pp (e.g. +2.5 pp), but NLL reduction < 5.0% (e.g. 0.0%) -> PASS
    c_a = create_mock_candidate_json(
        tmp_path,
        cand_id="case_a",
        ratio=0.5,
        acc_deltas={"mnist_compact": 1.0, "mnist_wide": 2.5, "fashion_compact": 0.0, "fashion_wide": 0.0},
        nll_deltas={"mnist_compact": 0.0, "mnist_wide": 0.0, "fashion_compact": 0.0, "fashion_wide": 0.0},
    )
    res_a = evaluate_heavy_autoresearch(b_path, c_a)
    assert res_a["candidate_evaluations"]["case_a"]["gate_details"]["gate_baseline_worst_improvement"] is True
    assert res_a["pass"] is True
    assert math.isfinite(res_a["score"])

    # Case B: acc_gain < 2.0 pp (e.g. +0.5 pp), but NLL reduction >= 5.0% (e.g. -0.10 NLL on 1.70 baseline = ~5.88%) -> PASS
    c_b = create_mock_candidate_json(
        tmp_path,
        cand_id="case_b",
        ratio=0.5,
        acc_deltas={"mnist_compact": 0.0, "mnist_wide": 0.5, "fashion_compact": 0.0, "fashion_wide": 0.0},
        nll_deltas={"mnist_compact": 0.0, "mnist_wide": -0.10, "fashion_compact": 0.0, "fashion_wide": 0.0},
    )
    res_b = evaluate_heavy_autoresearch(b_path, c_b)
    assert res_b["candidate_evaluations"]["case_b"]["gate_details"]["gate_baseline_worst_improvement"] is True
    assert res_b["pass"] is True
    assert math.isfinite(res_b["score"])

    # Case C: acc_gain = 1.0 pp (< 2.0), NLL reduction = 2.0% (< 5.0%) -> FAIL Gate 7
    # -0.034 NLL on 1.70 = ~2.0%
    c_c = create_mock_candidate_json(
        tmp_path,
        cand_id="case_c",
        ratio=0.5,
        acc_deltas={"mnist_compact": 0.0, "mnist_wide": 1.0, "fashion_compact": 0.0, "fashion_wide": 0.0},
        nll_deltas={"mnist_compact": 0.0, "mnist_wide": -0.034, "fashion_compact": 0.0, "fashion_wide": 0.0},
    )
    res_c = evaluate_heavy_autoresearch(b_path, c_c)
    assert res_c["candidate_evaluations"]["case_c"]["gate_details"]["gate_baseline_worst_improvement"] is False
    assert res_c["pass"] is False
    assert math.isfinite(res_c["score"])

def test_config_mismatch(tmp_path: Path):
    """Verify mismatched particle, epoch, or seed configurations fail gate_config_matched."""
    b_path = create_mock_baseline_json(tmp_path)

    # Particle mismatch (particles=10 instead of 12)
    c_part_path = create_mock_candidate_json(tmp_path, cand_id="p_mismatch", particles=10)
    res_p = evaluate_heavy_autoresearch(b_path, c_part_path)
    assert res_p["pass"] is False
    assert "gate_config_matched" in res_p["candidate_evaluations"]["p_mismatch"]["failed_gates"]
    assert math.isfinite(res_p["score"])

    # Epoch mismatch (epochs=40 instead of 80)
    c_epoch_path = create_mock_candidate_json(tmp_path, cand_id="e_mismatch", epochs=40)
    res_e = evaluate_heavy_autoresearch(b_path, c_epoch_path)
    assert res_e["pass"] is False
    assert "gate_config_matched" in res_e["candidate_evaluations"]["e_mismatch"]["failed_gates"]
    assert math.isfinite(res_e["score"])

def test_nonfinite_rejection(tmp_path: Path):
    """Verify non-finite metrics fail gate_finite."""
    b_path = create_mock_baseline_json(tmp_path)
    c_nan_path = create_mock_candidate_json(tmp_path, cand_id="nan_cand", is_finite=False)
    res = evaluate_heavy_autoresearch(b_path, c_nan_path)
    assert res["pass"] is False
    assert "gate_finite" in res["candidate_evaluations"]["nan_cand"]["failed_gates"]
    assert math.isfinite(res["score"])

def test_zero_test_enforcement(tmp_path: Path):
    """Verify official test data load or test evaluations > 0 fail gate_test_sealed."""
    b_path = create_mock_baseline_json(tmp_path)

    # Test evaluations > 0
    c_eval_path = create_mock_candidate_json(tmp_path, cand_id="test_eval", official_test_evals=10)
    res_eval = evaluate_heavy_autoresearch(b_path, c_eval_path)
    assert res_eval["pass"] is False
    assert "gate_test_sealed" in res_eval["candidate_evaluations"]["test_eval"]["failed_gates"]
    assert math.isfinite(res_eval["score"])

    # Test data loaded = True
    c_load_path = create_mock_candidate_json(tmp_path, cand_id="test_load", test_loaded=True)
    res_load = evaluate_heavy_autoresearch(b_path, c_load_path)
    assert res_load["pass"] is False
    assert "gate_test_sealed" in res_load["candidate_evaluations"]["test_load"]["failed_gates"]
    assert math.isfinite(res_load["score"])

def test_selection_preference_for_passing_candidates(tmp_path: Path):
    """Verify passing candidate is preferred over a higher unpenalized score candidate that fails a gate."""
    b_path = create_mock_baseline_json(tmp_path)

    # Cand A: passes all gates, modest score
    c_a_path = create_mock_candidate_json(
        tmp_path,
        cand_id="r0.5_pass",
        ratio=0.5,
        acc_deltas={"mnist_compact": 1.0, "mnist_wide": 2.5, "fashion_compact": 0.0, "fashion_wide": 0.0},
    )
    with open(c_a_path, "r", encoding="utf-8") as f:
        data_a = json.load(f)

    # Cand B: state_ratio = 0.6 (> 0.5), huge acc gain -> higher unpenalized score
    c_b_path = create_mock_candidate_json(
        tmp_path,
        cand_id="r0.6_fail",
        ratio=0.6,
        acc_deltas={"mnist_compact": 20.0, "mnist_wide": 20.0, "fashion_compact": 20.0, "fashion_wide": 20.0},
    )
    with open(c_b_path, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    # Combine into single candidate payload
    combined_payload = {
        "protocol_version": AUTORESEARCH_PROTOCOL_VERSION,
        "official_test_data_loaded": False,
        "official_test_evaluations": 0,
        "candidate_runs": {
            "r0.5_pass": data_a["candidate_runs"]["r0.5_pass"],
            "r0.6_fail": data_b["candidate_runs"]["r0.6_fail"],
        },
    }

    combined_path = tmp_path / "combined_candidates.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_payload, f)

    res = evaluate_heavy_autoresearch(b_path, combined_path)

    assert res["pass"] is True
    assert res["selected_candidate_id"] == "r0.5_pass", "Must select passing candidate over failing candidate"
    assert math.isfinite(res["score"])

def test_artifact_json_safety(tmp_path: Path):
    """Verify artifact payload structures dump cleanly to JSON without PyTorch tensor objects."""
    b_path = create_mock_baseline_json(tmp_path)
    c_path = create_mock_candidate_json(tmp_path, cand_id="safety_test")
    res = evaluate_heavy_autoresearch(b_path, c_path)

    json_str = json.dumps(res)
    assert "tensor" not in json_str.lower()
    assert isinstance(json.loads(json_str), dict)
    assert math.isfinite(res["score"])

def test_tiny_synthetic_experiment_runner(monkeypatch, tmp_path: Path):
    """Smoke test running run_heavy_pso_autoresearch on CPU with synthetic dataset monkeypatch."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    out_file = tmp_path / "synthetic_candidates.json"

    # Run tiny synthetic experiment on CPU: 2 particles, 2 epochs, subset_size=10, 1 ratio, 1 seed
    payload = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101],
        device_str="cpu",
        cache_dir=tmp_path,
        output_path=out_file,
    )

    assert out_file.is_file(), "Candidate artifact file must be atomically created"
    assert payload["protocol_version"] == AUTORESEARCH_PROTOCOL_VERSION
    assert payload["official_test_data_loaded"] is False
    assert payload["official_test_evaluations"] == 0
    assert "r0.5" in payload["candidate_runs"]
    assert "mnist_compact" in payload["candidate_runs"]["r0.5"]

    wl_res = payload["candidate_runs"]["r0.5"]["mnist_compact"]
    assert wl_res["particles"] == 2
    assert wl_res["epochs"] == 2
    assert wl_res["per_seed_runs"][0]["official_test_evaluations"] == 0


def test_latent_dim_half_up_and_invalid_ratios():
    """Verify compute_latent_dim half-up rounding and 0 < ratio <= 1 bounds enforcement."""
    assert compute_latent_dim(9098, 0.25) == 2275
    assert compute_latent_dim(9098, 0.5) == 4549
    assert compute_latent_dim(55338, 0.25) == 13835

    with pytest.raises(ValueError):
        compute_latent_dim(9098, 0.0)
    with pytest.raises(ValueError):
        compute_latent_dim(9098, -0.5)
    with pytest.raises(ValueError):
        compute_latent_dim(9098, 1.25)


def test_evaluator_required_test_flags(tmp_path: Path):
    """Verify missing or non-sealed test flags fail gate_test_sealed and produce finite score."""
    b_path = create_mock_baseline_json(tmp_path)
    c_path = create_mock_candidate_json(tmp_path, cand_id="flag_test")

    with open(c_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: Missing top-level test flags
    data_missing = dict(data)
    data_missing.pop("official_test_data_loaded", None)
    p1 = tmp_path / "cand_missing_flags.json"
    with open(p1, "w", encoding="utf-8") as f:
        json.dump(data_missing, f)
    res1 = evaluate_heavy_autoresearch(b_path, p1)
    assert res1["pass"] is False
    assert "gate_test_sealed" in res1["candidate_evaluations"]["flag_test"]["failed_gates"]
    assert math.isfinite(res1["score"])

    # Case 2: Per-seed test evaluations > 0
    data_seed_evals = json.loads(json.dumps(data))
    data_seed_evals["candidate_runs"]["flag_test"]["mnist_compact"]["per_seed_runs"][0]["official_test_evaluations"] = 5
    p2 = tmp_path / "cand_seed_evals.json"
    with open(p2, "w", encoding="utf-8") as f:
        json.dump(data_seed_evals, f)
    res2 = evaluate_heavy_autoresearch(b_path, p2)
    assert res2["pass"] is False
    assert "gate_test_sealed" in res2["candidate_evaluations"]["flag_test"]["failed_gates"]
    assert math.isfinite(res2["score"])


def test_evaluator_forged_state_ratio_and_core_bytes(tmp_path: Path):
    """Verify forged state_ratio or core_swarm_state_bytes are rejected and produce finite score."""
    b_path = create_mock_baseline_json(tmp_path)

    # Forged state ratio: claims ratio 0.1 but actual parameter ratio is 0.6
    c_path = create_mock_candidate_json(tmp_path, cand_id="forged_ratio", ratio=0.6)
    with open(c_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for wl in EXPECTED_WORKLOADS:
        data["candidate_runs"]["forged_ratio"][wl]["state_ratio"] = 0.1
    p_forged_sr = tmp_path / "forged_sr.json"
    with open(p_forged_sr, "w", encoding="utf-8") as f:
        json.dump(data, f)
    res_sr = evaluate_heavy_autoresearch(b_path, p_forged_sr)
    assert res_sr["pass"] is False
    assert "gate_state_ratio" in res_sr["candidate_evaluations"]["forged_ratio"]["failed_gates"]
    assert "gate_config_matched" in res_sr["candidate_evaluations"]["forged_ratio"]["failed_gates"]
    assert math.isfinite(res_sr["score"])

    # Forged core swarm state bytes: claims wrong byte footprint
    c_path2 = create_mock_candidate_json(tmp_path, cand_id="forged_bytes", ratio=0.5)
    with open(c_path2, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    for wl in EXPECTED_WORKLOADS:
        data2["candidate_runs"]["forged_bytes"][wl]["core_swarm_state_bytes"] = 12345
    p_forged_b = tmp_path / "forged_b.json"
    with open(p_forged_b, "w", encoding="utf-8") as f:
        json.dump(data2, f)
    res_b = evaluate_heavy_autoresearch(b_path, p_forged_b)
    assert res_b["pass"] is False
    assert "gate_config_matched" in res_b["candidate_evaluations"]["forged_bytes"]["failed_gates"]
    assert math.isfinite(res_b["score"])


def test_evaluator_missing_and_duplicate_seeds(tmp_path: Path):
    """Verify non-3 or duplicate/incorrect seed records fail gate_config_matched and emit finite score."""
    b_path = create_mock_baseline_json(tmp_path)

    # Duplicate seed: [101, 102, 102]
    c_dup = create_mock_candidate_json(tmp_path, cand_id="dup_seed", ratio=0.5, seeds=[101, 102, 102])
    res_dup = evaluate_heavy_autoresearch(b_path, c_dup)
    assert res_dup["pass"] is False
    assert "gate_config_matched" in res_dup["candidate_evaluations"]["dup_seed"]["failed_gates"]
    assert math.isfinite(res_dup["score"])

    # Missing seed (2 seeds instead of 3)
    c_miss = create_mock_candidate_json(tmp_path, cand_id="miss_seed", ratio=0.5, seeds=[101, 102])
    res_miss = evaluate_heavy_autoresearch(b_path, c_miss)
    assert res_miss["pass"] is False
    assert "gate_config_matched" in res_miss["candidate_evaluations"]["miss_seed"]["failed_gates"]
    assert math.isfinite(res_miss["score"])


def test_evaluator_infinite_baseline_and_candidate_metrics(tmp_path: Path):
    """Verify infinite/nan baseline or candidate metrics raise ValueError or fail gate_finite cleanly with finite score."""
    # Invalid baseline with NaN
    p_bad_b = tmp_path / "bad_baseline.json"
    with open(p_bad_b, "w", encoding="utf-8") as f:
        json.dump({
            "confirmation_results": {
                "mnist_compact": {"G8": {"stats": {"val_nll": {"mean": float("nan")}, "val_acc": {"mean": 50.0}}}},
                "mnist_wide": {"G5": {"stats": {"val_nll": {"mean": 1.70}, "val_acc": {"mean": 42.0}}}},
                "fashion_compact": {"G8": {"stats": {"val_nll": {"mean": 1.60}, "val_acc": {"mean": 48.0}}}},
                "fashion_wide": {"G5": {"stats": {"val_nll": {"mean": 1.75}, "val_acc": {"mean": 40.0}}}},
            }
        }, f)
    c_valid = create_mock_candidate_json(tmp_path, cand_id="valid_c")
    with pytest.raises(ValueError):
        evaluate_heavy_autoresearch(p_bad_b, c_valid)

    # Candidate with inf metric
    b_path = create_mock_baseline_json(tmp_path)
    c_inf = create_mock_candidate_json(tmp_path, cand_id="inf_cand", ratio=0.5)
    with open(c_inf, "r", encoding="utf-8") as f:
        data_inf = json.load(f)
    data_inf["candidate_runs"]["inf_cand"]["mnist_compact"]["per_seed_runs"][0]["val_selected_loss"] = float("inf")
    p_inf = tmp_path / "cand_inf.json"
    with open(p_inf, "w", encoding="utf-8") as f:
        json.dump(data_inf, f)
    res_inf = evaluate_heavy_autoresearch(b_path, p_inf)
    assert res_inf["pass"] is False
    assert "gate_finite" in res_inf["candidate_evaluations"]["inf_cand"]["failed_gates"]
    assert math.isfinite(res_inf["score"])


def test_geometry_policy_mappings():
    """Verify GEOMETRY_POLICIES contains 'recovered' and 'baseline_aligned' with exact workload mappings."""
    assert "recovered" in GEOMETRY_POLICIES
    assert "baseline_aligned" in GEOMETRY_POLICIES

    assert GEOMETRY_POLICIES["recovered"]["mnist_compact"] == "G6"
    assert GEOMETRY_POLICIES["recovered"]["mnist_wide"] == "G5"
    assert GEOMETRY_POLICIES["recovered"]["fashion_compact"] == "G6"
    assert GEOMETRY_POLICIES["recovered"]["fashion_wide"] == "G5"

    assert GEOMETRY_POLICIES["baseline_aligned"]["mnist_compact"] == "G8"
    assert GEOMETRY_POLICIES["baseline_aligned"]["mnist_wide"] == "G5"
    assert GEOMETRY_POLICIES["baseline_aligned"]["fashion_compact"] == "G8"
    assert GEOMETRY_POLICIES["baseline_aligned"]["fashion_wide"] == "G5"


def test_invalid_geometry_policy_rejection(tmp_path: Path):
    """Verify run_heavy_pso_autoresearch raises ValueError for unrecognised geometry policy."""
    with pytest.raises(ValueError, match="Invalid geometry_policy"):
        run_heavy_pso_autoresearch(
            ratios=[0.5],
            geometry_policy="invalid_policy_name",
        )


def test_format_ratio_id_behavior():
    """Verify format_ratio_id returns 'r<ratio>' for recovered and 'aligned_r<ratio>' for baseline_aligned."""
    assert format_ratio_id(0.5, "recovered") == "r0.5"
    assert format_ratio_id(0.5, "baseline_aligned") == "aligned_r0.5"
    assert format_ratio_id(1.0, "recovered") == "r1"
    assert format_ratio_id(1.0, "baseline_aligned") == "aligned_r1"
    assert format_ratio_id(0.03125, "baseline_aligned") == "aligned_r0.03125"
    assert format_ratio_id(0.5, "recovered", "tensor_local") == "local_r0.5"
    assert format_ratio_id(0.5, "baseline_aligned", "tensor_local") == "local_aligned_r0.5"
    assert format_ratio_id(0.125, "baseline_aligned", "tensor_local") == "local_aligned_r0.125"


def test_projection_seeds_identical_across_policies():
    """Verify projection seeds depend only on workload, ratio, and seed, NOT geometry policy."""
    for wl in ["mnist_compact", "mnist_wide", "fashion_compact", "fashion_wide"]:
        for r in [1.0, 0.5, 0.25]:
            for s in [101, 102, 103]:
                seed1 = derive_projection_seed(wl, r, s)
                seed2 = derive_projection_seed(wl, r, s)
                assert seed1 == seed2


def test_baseline_aligned_compact_g8_wide_g5_construction():
    """Verify baseline_aligned policy constructs equalized geometries from G8 for compact and G5 for wide."""
    geom_table = get_v6_geometry_table()

    # Compact workload under baseline_aligned uses G8 base
    base_compact_g8 = geom_table["G8"]
    eq_compact = construct_equalized_geometry(
        base_geom=base_compact_g8,
        total_dim=9098,
        latent_dim=4549,
        projection_seed=12345,
        ratio_str="aligned_r0.5",
    )
    assert eq_compact.config_id == "G8_eq_aligned_r0.5"
    assert math.isclose(eq_compact.position_radius, base_compact_g8.position_radius * math.sqrt(9098 / 4549))

    # Wide workload under baseline_aligned uses G5 base
    base_wide_g5 = geom_table["G5"]
    eq_wide = construct_equalized_geometry(
        base_geom=base_wide_g5,
        total_dim=55338,
        latent_dim=27669,
        projection_seed=12345,
        ratio_str="aligned_r0.5",
    )
    assert eq_wide.config_id == "G5_eq_aligned_r0.5"
    assert math.isclose(eq_wide.position_radius, base_wide_g5.position_radius * math.sqrt(55338 / 27669))

    # Diagnostic ratio 1.0 (latent_dim == total_dim) produces unscaled geometry
    eq_r1 = construct_equalized_geometry(
        base_geom=base_compact_g8,
        total_dim=9098,
        latent_dim=9098,
        projection_seed=12345,
        ratio_str="aligned_r1",
    )
    assert eq_r1.latent_dim == 9098
    assert math.isclose(eq_r1.position_radius, base_compact_g8.position_radius)
    assert math.isclose(eq_r1.reflective_bound, base_compact_g8.reflective_bound)


def test_synthetic_experiment_runner_baseline_aligned(monkeypatch, tmp_path: Path):
    """Smoke test running run_heavy_pso_autoresearch with baseline_aligned policy."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    payload = run_heavy_pso_autoresearch(
        ratios=[1.0, 0.5],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101],
        geometry_policy="baseline_aligned",
        device_str="cpu",
        cache_dir=tmp_path,
    )

    assert payload["experiment_config"]["geometry_policy"] == "baseline_aligned"
    assert "aligned_r1" in payload["candidate_runs"]
    assert "aligned_r0.5" in payload["candidate_runs"]

    # Verify per-workload geometry IDs and policy provenance
    assert payload["workloads"]["mnist_compact"]["base_geometry_id"] == "G8"
    assert payload["workloads"]["mnist_wide"]["base_geometry_id"] == "G5"
    assert payload["workloads"]["mnist_compact"]["geometry_policy"] == "baseline_aligned"
    assert payload["workloads"]["mnist_wide"]["geometry_policy"] == "baseline_aligned"
    assert payload["candidate_runs"]["aligned_r1"]["mnist_compact"]["base_geometry_id"] == "G8"
    assert payload["candidate_runs"]["aligned_r1"]["mnist_wide"]["base_geometry_id"] == "G5"
    assert payload["candidate_runs"]["aligned_r1"]["mnist_compact"]["geometry_policy"] == "baseline_aligned"


def test_projection_salt_empty_backward_compatibility():
    """Verify empty projection_salt reproduces exact legacy projection seeds."""
    seed_implicit = derive_projection_seed("mnist_compact", 0.5, 101)
    seed_explicit_empty = derive_projection_seed("mnist_compact", 0.5, 101, "")
    assert seed_implicit == seed_explicit_empty, "Implicit and explicit empty salt must produce identical projection seeds"

    # Verify against exact hash calculation

    expected_key = f"mnist_compact:0.50000:101".encode("utf-8")
    expected_seed = int(hashlib.sha256(expected_key).hexdigest()[:8], 16) % (2**31 - 1)
    assert seed_implicit == expected_seed, "Empty salt must match exact legacy sha256 hash key"


def test_projection_salt_nonempty_deterministic_variation():
    """Verify nonempty projection_salt changes seed deterministically and varies by salt, workload, ratio, and seed."""
    base_seed = derive_projection_seed("mnist_compact", 0.5, 101, "")
    salted_seed1 = derive_projection_seed("mnist_compact", 0.5, 101, "replica-1")
    salted_seed1_again = derive_projection_seed("mnist_compact", 0.5, 101, "replica-1")

    # Nonempty salt must differ from empty salt
    assert salted_seed1 != base_seed, "Nonempty salt must produce a different projection seed than empty salt"

    # Determinism / stability across calls
    assert salted_seed1 == salted_seed1_again, "Projection seed with salt must be deterministic across calls"

    # Salt variation
    salted_seed2 = derive_projection_seed("mnist_compact", 0.5, 101, "replica-2")
    assert salted_seed1 != salted_seed2, "Different salt strings must produce different projection seeds"

    # Workload, ratio, and swarm seed variation under nonempty salt
    diff_wl = derive_projection_seed("mnist_wide", 0.5, 101, "replica-1")
    diff_ratio = derive_projection_seed("mnist_compact", 0.25, 101, "replica-1")
    diff_swarm_seed = derive_projection_seed("mnist_compact", 0.5, 102, "replica-1")

    assert salted_seed1 != diff_wl, "Salted projection seed must vary by workload"
    assert salted_seed1 != diff_ratio, "Salted projection seed must vary by ratio"
    assert salted_seed1 != diff_swarm_seed, "Salted projection seed must vary by swarm seed"


def test_runner_provenance_and_projection_salt_persistence(monkeypatch, tmp_path: Path):
    """Verify projection_salt is persisted in experiment_config and per_seed_runs."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    # Salted run
    payload_salted = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101],
        geometry_policy="baseline_aligned",
        device_str="cpu",
        cache_dir=tmp_path,
        projection_salt="replica-1",
    )

    assert payload_salted["experiment_config"]["projection_salt"] == "replica-1"
    seed_rec_salted = payload_salted["candidate_runs"]["aligned_r0.5"]["mnist_compact"]["per_seed_runs"][0]
    assert seed_rec_salted["projection_salt"] == "replica-1"
    expected_salted_proj_seed = derive_projection_seed("mnist_compact", 0.5, 101, "replica-1")
    assert seed_rec_salted["projection_seed"] == expected_salted_proj_seed

    # Unsalted run (default)
    payload_unsalted = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101],
        geometry_policy="baseline_aligned",
        device_str="cpu",
        cache_dir=tmp_path,
    )

    assert payload_unsalted["experiment_config"]["projection_salt"] == ""
    seed_rec_unsalted = payload_unsalted["candidate_runs"]["aligned_r0.5"]["mnist_compact"]["per_seed_runs"][0]
    assert seed_rec_unsalted["projection_salt"] == ""
    expected_unsalted_proj_seed = derive_projection_seed("mnist_compact", 0.5, 101, "")
    assert seed_rec_unsalted["projection_seed"] == expected_unsalted_proj_seed


def test_projection_salt_no_change_to_query_and_sample_accounting(monkeypatch, tmp_path: Path):
    """Verify projection_salt preserves query, sample, and evaluation accounting, candidate IDs, and schema."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    payload_base = run_heavy_pso_autoresearch(
        ratios=[0.5, 0.125],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101, 102],
        geometry_policy="baseline_aligned",
        device_str="cpu",
        cache_dir=tmp_path,
        projection_salt="",
    )

    payload_salted = run_heavy_pso_autoresearch(
        ratios=[0.5, 0.125],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101, 102],
        geometry_policy="baseline_aligned",
        device_str="cpu",
        cache_dir=tmp_path,
        projection_salt="replica-1",
    )

    # Candidate IDs must be identical
    assert list(payload_base["candidate_runs"].keys()) == list(payload_salted["candidate_runs"].keys())
    assert "aligned_r0.5" in payload_salted["candidate_runs"]
    assert "aligned_r0.125" in payload_salted["candidate_runs"]

    # Accounting fields in experiment_config must be identical
    base_cfg = payload_base["experiment_config"]
    salted_cfg = payload_salted["experiment_config"]
    assert base_cfg["total_runs"] == salted_cfg["total_runs"]
    assert base_cfg["total_queries"] == salted_cfg["total_queries"]
    assert base_cfg["total_sample_evaluations"] == salted_cfg["total_sample_evaluations"]
    assert base_cfg["particles"] == salted_cfg["particles"]
    assert base_cfg["epochs"] == salted_cfg["epochs"]
    assert base_cfg["subset_size"] == salted_cfg["subset_size"]
    assert base_cfg["seeds"] == salted_cfg["seeds"]

    # Per seed runs accounting fields must be identical
    for cand_id in payload_base["candidate_runs"]:
        for wl_id in payload_base["candidate_runs"][cand_id]:
            base_runs = payload_base["candidate_runs"][cand_id][wl_id]["per_seed_runs"]
            salted_runs = payload_salted["candidate_runs"][cand_id][wl_id]["per_seed_runs"]
            for r_base, r_salted in zip(base_runs, salted_runs):
                assert r_base["total_queries"] == r_salted["total_queries"]
                assert r_base["total_sample_evaluations"] == r_salted["total_sample_evaluations"]
                assert r_base["official_test_evaluations"] == r_salted["official_test_evaluations"]
                assert r_base["core_swarm_state_bytes"] == r_salted["core_swarm_state_bytes"]


def test_cli_projection_salt_argument_parsing():
    """Verify CLI parser handles default and explicit --projection-salt flag."""
    parser = build_parser()
    args_default = parser.parse_args([])
    assert args_default.projection_salt == ""

    args_salted = parser.parse_args(["--projection-salt", "replica-1"])
    assert args_salted.projection_salt == "replica-1"


def test_tensor_local_allocation_invariants():
    """Verify allocate_tensor_latent_dims handles uneven/tiny tensors with exact sum and cap invariants."""
    import pytest
    # CompactCNN numels: [72, 8, 1152, 16, 7840, 10], aggregate_latent_dim = 284
    numels = [72, 8, 1152, 16, 7840, 10]
    total_dim = sum(numels)
    target_d = 284
    allocs = allocate_tensor_latent_dims(numels, target_d)

    assert sum(allocs) == target_d, "Exact sum must match target aggregate latent dim"
    assert len(allocs) == len(numels)
    for a, n in zip(allocs, numels):
        assert 1 <= a <= n, "Each tensor must get at least 1 coordinate and not exceed numel"

    # Extreme tiny tensors case: numels = [1, 1, 100], aggregate_latent_dim = 10
    tiny_numels = [1, 1, 100]
    tiny_allocs = allocate_tensor_latent_dims(tiny_numels, 10)
    assert tiny_allocs == [1, 1, 8]
    assert sum(tiny_allocs) == 10

    # Full dimensional allocation
    full_allocs = allocate_tensor_latent_dims(numels, total_dim)
    assert full_allocs == numels

    # Invalid allocation rejections
    with pytest.raises(ValueError):
        allocate_tensor_latent_dims(numels, 0)
    with pytest.raises(ValueError):
        allocate_tensor_latent_dims(numels, total_dim + 1)
    with pytest.raises(ValueError):
        allocate_tensor_latent_dims(numels, 2)  # target_d < len(numels)


def test_tensor_local_transform_coordinate_containment_and_decoding():
    """Verify TensorLocalLatentTransform restricts each tensor to its contiguous slice and decodes correctly."""
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)  # 50 weight + 5 bias = 55
            self.conv = nn.Conv2d(1, 4, 3)  # 36 weight + 4 bias = 40
            self.fc2 = nn.Linear(4, 2)  # 8 weight + 2 bias = 10
            # Total dim = 105, 6 parameter tensors

    model = DummyModel()
    geom_table = get_v6_geometry_table()
    base_geom = geom_table["G6"]
    total_dim = sum(p.numel() for p in model.parameters())
    latent_dim = 30
    proj_seed = 12345

    geom_cfg = construct_equalized_geometry(
        base_geom=base_geom,
        total_dim=total_dim,
        latent_dim=latent_dim,
        projection_seed=proj_seed,
        ratio_str="r0.3",
    )

    device = torch.device("cpu")
    transform = TensorLocalLatentTransform(model, geom_cfg, device)

    assert not transform.is_full
    assert len(transform.tensor_latent_dims) == len(transform.param_numels)
    assert sum(transform.tensor_latent_dims) == latent_dim

    # Verify each parameter's k_index lies strictly within its tensor's allocated slice
    j_offset = 0
    l_offset = 0
    for numel, d_m in zip(transform.param_numels, transform.tensor_latent_dims):
        k_slice = transform.k_indices[j_offset : j_offset + numel]
        assert (k_slice >= l_offset).all()
        assert (k_slice < l_offset + d_m).all()
        j_offset += numel
        l_offset += d_m

    # Verify finite decoding shape
    Z = torch.randn(5, latent_dim, device=device)
    theta = transform.decode(Z)
    assert theta.shape == (5, total_dim)
    assert torch.isfinite(theta).all()


def test_tensor_local_transform_determinism_and_seed_variation():
    """Verify TensorLocalLatentTransform is deterministic for identical seeds and varies across seeds."""
    model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 2))
    geom_table = get_v6_geometry_table()
    base_geom = geom_table["G6"]
    total_dim = sum(p.numel() for p in model.parameters())
    latent_dim = 40
    device = torch.device("cpu")

    geom_cfg1 = construct_equalized_geometry(
        base_geom=base_geom,
        total_dim=total_dim,
        latent_dim=latent_dim,
        projection_seed=999,
        ratio_str="r0.2",
    )
    geom_cfg1_dup = construct_equalized_geometry(
        base_geom=base_geom,
        total_dim=total_dim,
        latent_dim=latent_dim,
        projection_seed=999,
        ratio_str="r0.2",
    )
    geom_cfg2 = construct_equalized_geometry(
        base_geom=base_geom,
        total_dim=total_dim,
        latent_dim=latent_dim,
        projection_seed=1000,
        ratio_str="r0.2",
    )

    t1 = TensorLocalLatentTransform(model, geom_cfg1, device)
    t1_dup = TensorLocalLatentTransform(model, geom_cfg1_dup, device)
    t2 = TensorLocalLatentTransform(model, geom_cfg2, device)

    assert torch.equal(t1.k_indices, t1_dup.k_indices)
    assert torch.equal(t1.weights, t1_dup.weights)

    # Different projection seed must yield different projection indices or weights
    assert not (torch.equal(t1.k_indices, t2.k_indices) and torch.equal(t1.weights, t2.weights))


def test_tensor_local_full_dimensional_behavior():
    """Verify TensorLocalLatentTransform preserves full-dimensional behavior when latent_dim == total_dim."""
    model = nn.Sequential(nn.Linear(10, 5))
    geom_table = get_v6_geometry_table()
    base_geom = geom_table["G6"]
    total_dim = sum(p.numel() for p in model.parameters())

    geom_cfg = construct_equalized_geometry(
        base_geom=base_geom,
        total_dim=total_dim,
        latent_dim=total_dim,
        projection_seed=42,
        ratio_str="r1.0",
    )

    device = torch.device("cpu")
    transform = TensorLocalLatentTransform(model, geom_cfg, device)

    assert transform.is_full
    assert transform.tensor_latent_dims == transform.param_numels

    Z = torch.randn(3, total_dim, device=device)
    theta = transform.decode(Z)
    assert theta.shape == (3, total_dim)
    assert torch.isfinite(theta).all()


def test_tensor_local_runner_provenance_and_persistence(monkeypatch, tmp_path: Path):
    """Verify projection_scope='tensor_local' persists in experiment_config, workloads, candidate_runs, and per_seed_runs."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    payload = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101],
        geometry_policy="baseline_aligned",
        projection_scope="tensor_local",
        device_str="cpu",
        cache_dir=tmp_path,
    )

    assert payload["experiment_config"]["projection_scope"] == "tensor_local"
    assert payload["workloads"]["mnist_compact"]["projection_scope"] == "tensor_local"
    assert "local_aligned_r0.5" in payload["candidate_runs"]

    cand_rec = payload["candidate_runs"]["local_aligned_r0.5"]["mnist_compact"]
    assert cand_rec["projection_scope"] == "tensor_local"
    assert cand_rec["candidate_id"] == "local_aligned_r0.5"

    seed_rec = cand_rec["per_seed_runs"][0]
    assert seed_rec["projection_scope"] == "tensor_local"


def test_default_global_backward_compatibility(monkeypatch, tmp_path: Path):
    """Verify default projection_scope is 'global' and produces byte/seed/candidate-ID compatible output."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    payload_default = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101],
        geometry_policy="baseline_aligned",
        device_str="cpu",
        cache_dir=tmp_path,
    )

    payload_global = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101],
        geometry_policy="baseline_aligned",
        projection_scope="global",
        device_str="cpu",
        cache_dir=tmp_path,
    )

    assert payload_default["experiment_config"]["projection_scope"] == "global"
    assert list(payload_default["candidate_runs"].keys()) == ["aligned_r0.5"]
    assert list(payload_default["candidate_runs"].keys()) == list(payload_global["candidate_runs"].keys())

    rec_def = payload_default["candidate_runs"]["aligned_r0.5"]["mnist_compact"]["per_seed_runs"][0]
    rec_glo = payload_global["candidate_runs"]["aligned_r0.5"]["mnist_compact"]["per_seed_runs"][0]
    assert rec_def["projection_seed"] == rec_glo["projection_seed"]
    assert rec_def["core_swarm_state_bytes"] == rec_glo["core_swarm_state_bytes"]


def test_tensor_local_no_change_to_state_query_sample_accounting(monkeypatch, tmp_path: Path):
    """Verify projection_scope='tensor_local' preserves total queries, samples, state bytes, and baseline bytes."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    payload_global = run_heavy_pso_autoresearch(
        ratios=[0.5, 0.125],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101, 102],
        geometry_policy="baseline_aligned",
        projection_scope="global",
        device_str="cpu",
        cache_dir=tmp_path,
    )

    payload_local = run_heavy_pso_autoresearch(
        ratios=[0.5, 0.125],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101, 102],
        geometry_policy="baseline_aligned",
        projection_scope="tensor_local",
        device_str="cpu",
        cache_dir=tmp_path,
    )

    cfg_glo = payload_global["experiment_config"]
    cfg_loc = payload_local["experiment_config"]
    assert cfg_glo["total_runs"] == cfg_loc["total_runs"]
    assert cfg_glo["total_queries"] == cfg_loc["total_queries"]
    assert cfg_glo["total_sample_evaluations"] == cfg_loc["total_sample_evaluations"]

    for c_glo, c_loc in zip(payload_global["candidate_runs"].values(), payload_local["candidate_runs"].values()):
        for wl_id in c_glo:
            assert c_glo[wl_id]["core_swarm_state_bytes"] == c_loc[wl_id]["core_swarm_state_bytes"]
            assert c_glo[wl_id]["baseline_core_swarm_state_bytes"] == c_loc[wl_id]["baseline_core_swarm_state_bytes"]
            assert c_glo[wl_id]["state_ratio"] == c_loc[wl_id]["state_ratio"]


def test_invalid_projection_scope_rejection(tmp_path: Path):
    """Verify invalid projection_scope is rejected before loading datasets."""

    with pytest.raises(ValueError, match="Invalid projection_scope"):
        run_heavy_pso_autoresearch(projection_scope="invalid_scope", cache_dir=tmp_path)


def test_cli_projection_scope_argument_parsing():
    """Verify CLI parser handles default and explicit --projection-scope flag."""
    parser = build_parser()
    args_default = parser.parse_args([])
    assert args_default.projection_scope == "global"

    args_local = parser.parse_args(["--projection-scope", "tensor_local"])
    assert args_local.projection_scope == "tensor_local"


def test_projection_seed_mode_default_and_coupled_backward_compatibility():
    """Verify default projection seed mode is 'coupled' and produces exact legacy seeds and candidate IDs."""
    assert DEFAULT_PROJECTION_SEED_MODE == "coupled"
    assert PROJECTION_SEED_MODES == ("coupled", "fixed", "explicit")

    # Default derive_projection_seed vs explicit coupled
    s_default = derive_projection_seed("mnist_compact", 0.5, 101)
    s_coupled = derive_projection_seed("mnist_compact", 0.5, 101, mode="coupled")
    s_coupled_param = derive_projection_seed("mnist_compact", 0.5, 101, projection_seed_mode="coupled")
    assert s_default == s_coupled == s_coupled_param

    # Legacy formula check (empty salt, coupled)
    key = "mnist_compact:0.50000:101".encode("utf-8")
    expected_legacy = int(hashlib.sha256(key).hexdigest()[:8], 16) % (2**31 - 1)
    assert s_default == expected_legacy

    # Candidate ID default format_ratio_id checks
    assert format_ratio_id(0.5) == "r0.5"
    assert format_ratio_id(0.5, "baseline_aligned", "global") == "aligned_r0.5"
    assert format_ratio_id(0.5, "recovered", "tensor_local") == "local_r0.5"
    assert format_ratio_id(0.5, "baseline_aligned", "tensor_local") == "local_aligned_r0.5"


def test_fixed_projection_seed_equal_across_swarm_seeds():
    """Verify fixed projection seed mode produces identical seeds across swarm seeds (101, 102, 103)."""
    s101 = derive_projection_seed("mnist_compact", 0.5, 101, mode="fixed")
    s102 = derive_projection_seed("mnist_compact", 0.5, 102, mode="fixed")
    s103 = derive_projection_seed("mnist_compact", 0.5, 103, mode="fixed")
    assert s101 == s102 == s103, "Fixed mode projection seed must be identical across swarm seeds"


def test_fixed_projection_seed_variation_across_workload_ratio_salt():
    """Verify fixed projection seed mode varies across workload, ratio, and salt."""
    s_base = derive_projection_seed("mnist_compact", 0.5, 101, mode="fixed")
    s_diff_wl = derive_projection_seed("mnist_wide", 0.5, 101, mode="fixed")
    s_diff_ratio = derive_projection_seed("mnist_compact", 0.25, 101, mode="fixed")
    s_salted = derive_projection_seed("mnist_compact", 0.5, 101, projection_salt="replica-1", mode="fixed")

    assert s_base != s_diff_wl, "Fixed projection seed must vary by workload"
    assert s_base != s_diff_ratio, "Fixed projection seed must vary by ratio"
    assert s_base != s_salted, "Fixed projection seed must vary by salt"


def test_invalid_projection_seed_mode_rejection(tmp_path: Path):
    """Verify invalid projection_seed_mode is rejected early before loading datasets and in derivation."""
    import pytest
    with pytest.raises(ValueError, match="Invalid projection_seed_mode"):
        derive_projection_seed("mnist_compact", 0.5, 101, mode="invalid_mode")

    with pytest.raises(ValueError, match="Invalid projection_seed_mode"):
        run_heavy_pso_autoresearch(projection_seed_mode="invalid_mode", cache_dir=tmp_path)


def test_format_ratio_id_projection_seed_mode_prefix_combinations():
    """Verify format_ratio_id produces exact prefix combinations for policy, scope, and seed mode."""
    # Coupled (default) mode
    assert format_ratio_id(0.125, "recovered", "global", "coupled") == "r0.125"
    assert format_ratio_id(0.125, "baseline_aligned", "global", "coupled") == "aligned_r0.125"
    assert format_ratio_id(0.125, "recovered", "tensor_local", "coupled") == "local_r0.125"
    assert format_ratio_id(0.125, "baseline_aligned", "tensor_local", "coupled") == "local_aligned_r0.125"

    # Fixed mode
    assert format_ratio_id(0.125, "recovered", "global", "fixed") == "fixed_r0.125"
    assert format_ratio_id(0.125, "baseline_aligned", "global", "fixed") == "fixed_aligned_r0.125"
    assert format_ratio_id(0.125, "recovered", "tensor_local", "fixed") == "fixed_local_r0.125"
    assert format_ratio_id(0.125, "baseline_aligned", "tensor_local", "fixed") == "fixed_local_aligned_r0.125"


def test_cli_projection_seed_mode_argument_parsing():
    """Verify CLI parser handles default and explicit --projection-seed-mode flag."""
    parser = build_parser()
    args_default = parser.parse_args([])
    assert args_default.projection_seed_mode == "coupled"

    args_fixed = parser.parse_args(["--projection-seed-mode", "fixed"])
    assert args_fixed.projection_seed_mode == "fixed"


def test_projection_seed_mode_runner_provenance_and_persistence(monkeypatch, tmp_path: Path):
    """Verify projection_seed_mode='fixed' is persisted at experiment, workload, candidate, and per-seed levels."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    payload = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=1,
        subset_size=10,
        seeds=[101, 102, 103],
        geometry_policy="recovered",
        projection_scope="global",
        projection_seed_mode="fixed",
        device_str="cpu",
        cache_dir=tmp_path,
    )

    assert payload["experiment_config"]["projection_seed_mode"] == "fixed"
    assert payload["workloads"]["mnist_compact"]["projection_seed_mode"] == "fixed"
    assert "fixed_r0.5" in payload["candidate_runs"]

    cand_rec = payload["candidate_runs"]["fixed_r0.5"]["mnist_compact"]
    assert cand_rec["projection_seed_mode"] == "fixed"
    assert cand_rec["candidate_id"] == "fixed_r0.5"

    seed_runs = cand_rec["per_seed_runs"]
    assert len(seed_runs) == 3
    for s_rec in seed_runs:
        assert s_rec["projection_seed_mode"] == "fixed"

    # All swarm seeds must have the exact same projection_seed in fixed mode
    p_seeds = [s_rec["projection_seed"] for s_rec in seed_runs]
    assert len(set(p_seeds)) == 1, f"Expected single fixed projection_seed across seeds, got {p_seeds}"


def test_projection_seed_mode_exact_accounting_unchanged(monkeypatch, tmp_path: Path):
    """Verify projection_seed_mode='fixed' preserves total queries, samples, state bytes, and baseline bytes."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    payload_coupled = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=1,
        subset_size=10,
        seeds=[101, 102],
        geometry_policy="baseline_aligned",
        projection_scope="global",
        projection_seed_mode="coupled",
        device_str="cpu",
        cache_dir=tmp_path,
    )

    payload_fixed = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=1,
        subset_size=10,
        seeds=[101, 102],
        geometry_policy="baseline_aligned",
        projection_scope="global",
        projection_seed_mode="fixed",
        device_str="cpu",
        cache_dir=tmp_path,
    )

    cfg_c = payload_coupled["experiment_config"]
    cfg_f = payload_fixed["experiment_config"]
    assert cfg_c["total_runs"] == cfg_f["total_runs"]
    assert cfg_c["total_queries"] == cfg_f["total_queries"]
    assert cfg_c["total_sample_evaluations"] == cfg_f["total_sample_evaluations"]

    for c_coup, c_fix in zip(payload_coupled["candidate_runs"].values(), payload_fixed["candidate_runs"].values()):
        for wl_id in c_coup:
            assert c_coup[wl_id]["core_swarm_state_bytes"] == c_fix[wl_id]["core_swarm_state_bytes"]
            assert c_coup[wl_id]["baseline_core_swarm_state_bytes"] == c_fix[wl_id]["baseline_core_swarm_state_bytes"]
            assert c_coup[wl_id]["state_ratio"] == c_fix[wl_id]["state_ratio"]
            assert c_coup[wl_id]["latent_dim"] == c_fix[wl_id]["latent_dim"]
            assert c_coup[wl_id]["total_dim"] == c_fix[wl_id]["total_dim"]


def test_fixed_global_baseline_aligned_matrix_evaluator_schema_and_seeds(monkeypatch, tmp_path: Path):
    """Verify fixed/global/baseline-aligned matrix retains evaluator schema and has identical projection_seed for seeds 101-103 within each workload-ratio cell."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    payload = run_heavy_pso_autoresearch(
        ratios=[0.5, 0.25],
        particles=2,
        epochs=1,
        subset_size=10,
        seeds=[101, 102, 103],
        geometry_policy="baseline_aligned",
        projection_scope="global",
        projection_seed_mode="fixed",
        device_str="cpu",
        cache_dir=tmp_path,
    )

    # Check candidate IDs
    assert set(payload["candidate_runs"].keys()) == {"fixed_aligned_r0.5", "fixed_aligned_r0.25"}

    # Check that within each candidate and workload cell, seeds 101-103 share 1 projection_seed
    for cand_id, wl_candidates in payload["candidate_runs"].items():
        for wl_id, wl_data in wl_candidates.items():
            per_seed = wl_data["per_seed_runs"]
            assert len(per_seed) == 3
            proj_seeds = [rec["projection_seed"] for rec in per_seed]
            assert len(set(proj_seeds)) == 1, f"Expected 1 projection seed for {cand_id}/{wl_id}, got {proj_seeds}"

    # Evaluate mock payload with evaluate_heavy_autoresearch to ensure evaluator schema compatibility
    cand_file = tmp_path / "candidate_fixed_matrix.json"
    with open(cand_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    base_file = create_mock_baseline_json(tmp_path)
    eval_res = evaluate_heavy_autoresearch(base_file, cand_file)
    assert isinstance(eval_res["pass"], bool)
    assert isinstance(eval_res["score"], float)
    assert set(eval_res["candidate_evaluations"]) == set(payload["candidate_runs"])


def test_explicit_projection_seed_validations(tmp_path: Path):
    """Verify explicit projection seed mode validation checks for missing, negative, out-of-range, and supplied non-explicit seeds."""
    # Missing seed in explicit mode
    with pytest.raises(ValueError, match="projection_seed must be provided when projection_seed_mode is 'explicit'"):
        validate_projection_seed_config("explicit", None)

    with pytest.raises(ValueError, match="projection_seed must be provided when projection_seed_mode is 'explicit'"):
        derive_projection_seed("mnist_wide", 0.5, 101, mode="explicit", projection_seed=None)

    # Negative seed
    with pytest.raises(ValueError, match="projection_seed must be a non-negative integer"):
        validate_projection_seed_config("explicit", -1)

    with pytest.raises(ValueError, match="projection_seed must be a non-negative integer"):
        derive_projection_seed("mnist_wide", 0.5, 101, mode="explicit", projection_seed=-10)

    # Out-of-range seed >= 2**31 - 1
    max_seed = 2**31 - 1  # 2147483647
    with pytest.raises(ValueError, match="projection_seed must be a non-negative integer"):
        validate_projection_seed_config("explicit", max_seed)

    with pytest.raises(ValueError, match="projection_seed must be a non-negative integer"):
        derive_projection_seed("mnist_wide", 0.5, 101, mode="explicit", projection_seed=2**31)

    # Non-integer types (float, bool)
    with pytest.raises(ValueError, match="projection_seed must be a non-negative integer"):
        validate_projection_seed_config("explicit", 592157828.0)

    with pytest.raises(ValueError, match="projection_seed must be a non-negative integer"):
        validate_projection_seed_config("explicit", True)

    # Seed supplied to coupled/fixed modes
    with pytest.raises(ValueError, match="projection_seed can only be provided when projection_seed_mode is 'explicit'"):
        validate_projection_seed_config("coupled", 592157828)

    with pytest.raises(ValueError, match="projection_seed can only be provided when projection_seed_mode is 'explicit'"):
        validate_projection_seed_config("fixed", 592157828)

    with pytest.raises(ValueError, match="projection_seed can only be provided when projection_seed_mode is 'explicit'"):
        derive_projection_seed("mnist_wide", 0.5, 101, mode="coupled", projection_seed=592157828)

    # Validate rejection occurs before data loading in runner
    with pytest.raises(ValueError, match="projection_seed must be provided when projection_seed_mode is 'explicit'"):
        run_heavy_pso_autoresearch(
            ratios=[0.5],
            projection_seed_mode="explicit",
            projection_seed=None,
            cache_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="projection_seed can only be provided when projection_seed_mode is 'explicit'"):
        run_heavy_pso_autoresearch(
            ratios=[0.5],
            projection_seed_mode="coupled",
            projection_seed=592157828,
            cache_dir=tmp_path,
        )

def test_explicit_projection_seed_dict_validations(tmp_path: Path):
    """Verify dictionary explicit projection seeds require exactly WORKLOADS keys, rejecting partial and extra dicts."""
    valid_dict = {
        "mnist_compact": 101,
        "mnist_wide": 102,
        "fashion_compact": 103,
        "fashion_wide": 104,
    }
    # Complete dict remains accepted
    validate_projection_seed_config("explicit", valid_dict)
    assert derive_projection_seed("mnist_compact", 0.5, 101, mode="explicit", projection_seed=valid_dict) == 101
    assert derive_projection_seed("mnist_wide", 0.5, 101, mode="explicit", projection_seed=valid_dict) == 102

    # Partial dict missing required keys fails immediately
    partial_dict = {"mnist_compact": 101, "mnist_wide": 102}
    with pytest.raises(ValueError, match="missing required workload key"):
        validate_projection_seed_config("explicit", partial_dict)

    with pytest.raises(ValueError, match="missing required workload key"):
        derive_projection_seed("mnist_compact", 0.5, 101, mode="explicit", projection_seed=partial_dict)

    with pytest.raises(ValueError, match="missing required workload key"):
        run_heavy_pso_autoresearch(
            ratios=[0.5],
            projection_seed_mode="explicit",
            projection_seed=partial_dict,
            cache_dir=tmp_path,
        )

    # Extra dict with unknown keys fails immediately
    extra_dict = {
        "mnist_compact": 101,
        "mnist_wide": 102,
        "fashion_compact": 103,
        "fashion_wide": 104,
        "unknown_workload": 105,
    }
    with pytest.raises(ValueError, match="Unknown workload_id key"):
        validate_projection_seed_config("explicit", extra_dict)

    with pytest.raises(ValueError, match="Unknown workload_id key"):
        derive_projection_seed("mnist_compact", 0.5, 101, mode="explicit", projection_seed=extra_dict)

    with pytest.raises(ValueError, match="Unknown workload_id key"):
        run_heavy_pso_autoresearch(
            ratios=[0.5],
            projection_seed_mode="explicit",
            projection_seed=extra_dict,
            cache_dir=tmp_path,
        )
    # Both missing and unknown keys in dict
    mixed_invalid_dict = {"mnist_compact": 101, "extra_key": 105}
    with pytest.raises(ValueError, match="missing required workload key.*Unknown workload_id key"):
        validate_projection_seed_config("explicit", mixed_invalid_dict)


def test_parse_projection_seed_arg():
    """Verify parse_projection_seed_arg handles scalar integers, JSON dicts, key-value strings, and raises narrow exceptions on invalid forms."""
    # Scalar integers and int strings
    assert parse_projection_seed_arg(42) == 42
    assert parse_projection_seed_arg("42") == 42
    assert parse_projection_seed_arg(None) is None
    assert parse_projection_seed_arg("None") is None

    # JSON dict strings
    json_str = '{"mnist_compact": 101, "mnist_wide": 102, "fashion_compact": 103, "fashion_wide": 104}'
    parsed_json = parse_projection_seed_arg(json_str)
    assert isinstance(parsed_json, dict)
    assert parsed_json["mnist_compact"] == 101

    # Key-value strings
    kv_str = "mnist_compact:101,mnist_wide:102,fashion_compact:103,fashion_wide:104"
    parsed_kv = parse_projection_seed_arg(kv_str)
    assert isinstance(parsed_kv, dict)
    assert parsed_kv["mnist_wide"] == 102

    # Malformed JSON starting with { and ending with } raises ValueError from narrow exception handling
    with pytest.raises(ValueError, match="Failed to parse projection_seed JSON dict string"):
        parse_projection_seed_arg("{invalid_json_format}")

    with pytest.raises(ValueError, match="Failed to parse projection_seed JSON dict string"):
        parse_projection_seed_arg('{"mnist_compact": "not_an_int"}')

    # Unparseable string
    with pytest.raises(ValueError, match="Cannot parse projection_seed value"):
        parse_projection_seed_arg("not_a_number_or_dict")

def test_explicit_projection_seed_derivation():
    """Verify derive_projection_seed returns the exact explicit value regardless of workload, ratio, swarm seed, or salt."""
    seed1 = derive_projection_seed(
        "mnist_wide", 0.5, 101, projection_salt="", mode="explicit", projection_seed=592157828
    )
    assert seed1 == 592157828

    seed2 = derive_projection_seed(
        "fashion_compact", 0.03125, 103, projection_salt="salt_test", mode="explicit", projection_seed=592157828
    )
    assert seed2 == 592157828

    seed3 = derive_projection_seed(
        "mnist_compact", 0.25, 102, projection_salt="", mode="explicit", projection_seed=820515361
    )
    assert seed3 == 820515361


def test_explicit_format_ratio_id_prefix_ordering():
    """Verify format_ratio_id prepends p<seed>_ before optional local_ and existing base ID across combinations."""
    # explicit global baseline_aligned
    assert format_ratio_id(0.5, "baseline_aligned", "global", "explicit", 592157828) == "p592157828_aligned_r0.5"
    # explicit tensor_local baseline_aligned
    assert format_ratio_id(0.5, "baseline_aligned", "tensor_local", "explicit", 592157828) == "p592157828_local_aligned_r0.5"
    # explicit global recovered
    assert format_ratio_id(0.5, "recovered", "global", "explicit", 592157828) == "p592157828_r0.5"
    # explicit tensor_local recovered
    assert format_ratio_id(0.5, "recovered", "tensor_local", "explicit", 592157828) == "p592157828_local_r0.5"
    # explicit ratio 0.03125
    assert format_ratio_id(0.03125, "baseline_aligned", "global", "explicit", 820515361) == "p820515361_aligned_r0.03125"


def test_cli_explicit_projection_seed_argument_parsing():
    """Verify CLI parser handles --projection-seed-mode explicit and --projection-seed flags."""
    parser = build_parser()
    args1 = parser.parse_args(["--projection-seed-mode", "explicit", "--projection-seed", "592157828"])
    assert args1.projection_seed_mode == "explicit"
    assert args1.projection_seed == 592157828

    args2 = parser.parse_args(["--projection-seed-mode", "explicit", "--projection-seed", "820515361"])
    assert args2.projection_seed_mode == "explicit"
    assert args2.projection_seed == 820515361


def test_explicit_runner_provenance_and_persistence(monkeypatch, tmp_path: Path):
    """Verify explicit mode and projection seed are persisted at experiment, workload, candidate, and per-seed levels."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    payload = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=1,
        subset_size=10,
        seeds=[101, 102],
        geometry_policy="baseline_aligned",
        projection_scope="global",
        projection_seed_mode="explicit",
        projection_seed=592157828,
        device_str="cpu",
        cache_dir=tmp_path,
    )

    # Check top-level experiment_config
    exp_cfg = payload["experiment_config"]
    assert exp_cfg["projection_seed_mode"] == "explicit"
    assert exp_cfg["projection_seed"] == 592157828

    # Check workloads provenance
    for wl_id, wl_meta in payload["workloads"].items():
        assert wl_meta["projection_seed_mode"] == "explicit"
        assert wl_meta["projection_seed"] == 592157828

    # Check candidate_runs
    cand_dict = payload["candidate_runs"]["p592157828_aligned_r0.5"]
    for wl_id, wl_cand in cand_dict.items():
        assert wl_cand["candidate_id"] == "p592157828_aligned_r0.5"
        assert wl_cand["projection_seed_mode"] == "explicit"
        assert wl_cand["projection_seed"] == 592157828

        for seed_rec in wl_cand["per_seed_runs"]:
            assert seed_rec["projection_seed_mode"] == "explicit"
            assert seed_rec["projection_seed"] == 592157828


def test_explicit_projection_seed_no_change_to_query_and_sample_accounting(monkeypatch, tmp_path: Path):
    """Verify explicit projection seed mode preserves total queries, samples, state bytes, and baseline bytes."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    payload_coupled = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=1,
        subset_size=10,
        seeds=[101, 102, 103],
        geometry_policy="baseline_aligned",
        projection_scope="global",
        projection_seed_mode="coupled",
        device_str="cpu",
        cache_dir=tmp_path,
    )

    payload_explicit = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=1,
        subset_size=10,
        seeds=[101, 102, 103],
        geometry_policy="baseline_aligned",
        projection_scope="global",
        projection_seed_mode="explicit",
        projection_seed=592157828,
        device_str="cpu",
        cache_dir=tmp_path,
    )

    cfg_c = payload_coupled["experiment_config"]
    cfg_e = payload_explicit["experiment_config"]

    assert cfg_c["total_runs"] == cfg_e["total_runs"]
    assert cfg_c["total_queries"] == cfg_e["total_queries"]
    assert cfg_c["total_sample_evaluations"] == cfg_e["total_sample_evaluations"]

    c_coup = payload_coupled["candidate_runs"]["aligned_r0.5"]
    c_exp = payload_explicit["candidate_runs"]["p592157828_aligned_r0.5"]

    for wl_id in ["mnist_compact", "mnist_wide", "fashion_compact", "fashion_wide"]:
        assert c_coup[wl_id]["latent_dim"] == c_exp[wl_id]["latent_dim"]
        assert c_coup[wl_id]["total_dim"] == c_exp[wl_id]["total_dim"]
        assert c_coup[wl_id]["core_swarm_state_bytes"] == c_exp[wl_id]["core_swarm_state_bytes"]
        assert c_coup[wl_id]["baseline_core_swarm_state_bytes"] == c_exp[wl_id]["baseline_core_swarm_state_bytes"]
        assert c_coup[wl_id]["state_ratio"] == c_exp[wl_id]["state_ratio"]


def test_explicit_matched_elites_confirmation_runs(monkeypatch, tmp_path: Path):
    """Verify two separate matched runner invocations can confirm seed 592157828 at ratio 0.5 and seed 820515361 at ratio 0.03125 across all 4 workloads and seeds 101-103."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    # Elite 1: seed 592157828 at ratio 0.5
    run1 = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=1,
        subset_size=10,
        seeds=[101, 102, 103],
        geometry_policy="baseline_aligned",
        projection_scope="global",
        projection_seed_mode="explicit",
        projection_seed=592157828,
        device_str="cpu",
        cache_dir=tmp_path,
    )

    assert "p592157828_aligned_r0.5" in run1["candidate_runs"]
    cand1 = run1["candidate_runs"]["p592157828_aligned_r0.5"]
    assert len(cand1) == 4
    for wl_id, wl_data in cand1.items():
        assert len(wl_data["per_seed_runs"]) == 3
        for r_entry in wl_data["per_seed_runs"]:
            assert r_entry["projection_seed"] == 592157828
            assert r_entry["projection_seed_mode"] == "explicit"

    # Elite 2: seed 820515361 at ratio 0.03125
    run2 = run_heavy_pso_autoresearch(
        ratios=[0.03125],
        particles=2,
        epochs=1,
        subset_size=10,
        seeds=[101, 102, 103],
        geometry_policy="baseline_aligned",
        projection_scope="global",
        projection_seed_mode="explicit",
        projection_seed=820515361,
        device_str="cpu",
        cache_dir=tmp_path,
    )

    assert "p820515361_aligned_r0.03125" in run2["candidate_runs"]
    cand2 = run2["candidate_runs"]["p820515361_aligned_r0.03125"]
    assert len(cand2) == 4
    for wl_id, wl_data in cand2.items():
        assert len(wl_data["per_seed_runs"]) == 3
        for r_entry in wl_data["per_seed_runs"]:
            assert r_entry["projection_seed"] == 820515361
            assert r_entry["projection_seed_mode"] == "explicit"

    # Evaluate mock artifacts with evaluator to confirm schema compatibility
    f1 = tmp_path / "cand1.json"
    with open(f1, "w", encoding="utf-8") as f:
        json.dump(run1, f)
    base_file = create_mock_baseline_json(tmp_path)
    res1 = evaluate_heavy_autoresearch(base_file, f1)
    assert isinstance(res1["pass"], bool)
    assert isinstance(res1["score"], float)

    f2 = tmp_path / "cand2.json"
    with open(f2, "w", encoding="utf-8") as f:
        json.dump(run2, f)
    res2 = evaluate_heavy_autoresearch(base_file, f2)
    assert isinstance(res2["pass"], bool)
    assert isinstance(res2["score"], float)


def test_geometry_multiplier_exact_scaling():
    """Verify geometry_multiplier uniformly scales position_radius, initial_velocity_radius, reset_velocity_radius, and reflective_bound."""
    geom_table = get_v6_geometry_table()
    base_g6 = geom_table["G6"]
    total_dim = 9098
    latent_dim = compute_latent_dim(total_dim, 0.5)
    scale_factor = math.sqrt(total_dim / latent_dim)

    # Multiplier 1.0 (default)
    eq_g6_1 = construct_equalized_geometry(base_g6, total_dim, latent_dim, projection_seed=42, ratio_str="r0.5", geometry_multiplier=1.0)
    assert math.isclose(eq_g6_1.position_radius, base_g6.position_radius * scale_factor)
    assert math.isclose(eq_g6_1.initial_velocity_radius, base_g6.initial_velocity_radius * scale_factor)
    assert math.isclose(eq_g6_1.reset_velocity_radius, base_g6.reset_velocity_radius * scale_factor)
    assert math.isclose(eq_g6_1.reflective_bound, base_g6.reflective_bound * scale_factor)

    # Multiplier 0.75
    eq_g6_075 = construct_equalized_geometry(base_g6, total_dim, latent_dim, projection_seed=42, ratio_str="g0.75_r0.5", geometry_multiplier=0.75)
    assert math.isclose(eq_g6_075.position_radius, base_g6.position_radius * scale_factor * 0.75)
    assert math.isclose(eq_g6_075.initial_velocity_radius, base_g6.initial_velocity_radius * scale_factor * 0.75)
    assert math.isclose(eq_g6_075.reset_velocity_radius, base_g6.reset_velocity_radius * scale_factor * 0.75)
    assert math.isclose(eq_g6_075.reflective_bound, base_g6.reflective_bound * scale_factor * 0.75)

    # Multiplier 0.5
    eq_g6_05 = construct_equalized_geometry(base_g6, total_dim, latent_dim, projection_seed=42, ratio_str="g0.5_r0.5", geometry_multiplier=0.5)
    assert math.isclose(eq_g6_05.position_radius, base_g6.position_radius * scale_factor * 0.5)
    assert math.isclose(eq_g6_05.initial_velocity_radius, base_g6.initial_velocity_radius * scale_factor * 0.5)
    assert math.isclose(eq_g6_05.reset_velocity_radius, base_g6.reset_velocity_radius * scale_factor * 0.5)
    assert math.isclose(eq_g6_05.reflective_bound, base_g6.reflective_bound * scale_factor * 0.5)

    # Invariant attributes remain unchanged
    for eq_geom in (eq_g6_1, eq_g6_075, eq_g6_05):
        assert eq_geom.mutation_prob == base_g6.mutation_prob
        assert eq_geom.scale_type == base_g6.scale_type
        assert eq_geom.init_position_mode == base_g6.init_position_mode
        assert eq_geom.latent_dim == latent_dim
        assert eq_geom.projection_seed == 42


def test_geometry_multiplier_validations(tmp_path: Path):
    """Verify geometry_multiplier rejects non-numeric, non-finite, zero, or negative inputs with ValueError."""
    invalid_multipliers = [
        0,
        0.0,
        -0.5,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        "0.75",
        True,
        False,
        None,
    ]
    geom_table = get_v6_geometry_table()
    base_g6 = geom_table["G6"]

    for inv in invalid_multipliers:
        with pytest.raises(ValueError, match="geometry_multiplier must be a finite positive float"):
            validate_geometry_multiplier(inv)

        with pytest.raises(ValueError, match="geometry_multiplier must be a finite positive float"):
            format_ratio_id(0.5, geometry_multiplier=inv)

        with pytest.raises(ValueError, match="geometry_multiplier must be a finite positive float"):
            construct_equalized_geometry(base_g6, 9098, 4549, projection_seed=42, geometry_multiplier=inv)

        with pytest.raises(ValueError, match="geometry_multiplier must be a finite positive float"):
            run_heavy_pso_autoresearch(geometry_multiplier=inv, cache_dir=tmp_path)


def test_format_ratio_id_geometry_multiplier_prefix_and_ordering():
    """Verify format_ratio_id prepends g<value>_ before all other projection prefixes when geometry_multiplier != 1.0, and leaves default IDs unchanged."""
    # Default 1.0 multiplier preserves legacy IDs
    assert format_ratio_id(0.5) == "r0.5"
    assert format_ratio_id(0.5, "baseline_aligned") == "aligned_r0.5"
    assert format_ratio_id(0.5, "baseline_aligned", "tensor_local") == "local_aligned_r0.5"
    assert format_ratio_id(0.5, "baseline_aligned", "global", "explicit", 592157828) == "p592157828_aligned_r0.5"
    assert format_ratio_id(0.5, "baseline_aligned", "tensor_local", "explicit", 592157828) == "p592157828_local_aligned_r0.5"

    # Multiplier 0.75 prepends g0.75_ at the very front
    assert format_ratio_id(0.5, geometry_multiplier=0.75) == "g0.75_r0.5"
    assert format_ratio_id(0.5, "baseline_aligned", geometry_multiplier=0.75) == "g0.75_aligned_r0.5"
    assert format_ratio_id(0.5, "baseline_aligned", "tensor_local", geometry_multiplier=0.75) == "g0.75_local_aligned_r0.5"
    assert format_ratio_id(0.5, "baseline_aligned", "global", "explicit", 592157828, geometry_multiplier=0.75) == "g0.75_p592157828_aligned_r0.5"
    assert format_ratio_id(0.5, "baseline_aligned", "tensor_local", "explicit", 592157828, geometry_multiplier=0.75) == "g0.75_p592157828_local_aligned_r0.5"

    # Multiplier 0.5 prepends g0.5_ at the very front
    assert format_ratio_id(0.5, geometry_multiplier=0.5) == "g0.5_r0.5"
    assert format_ratio_id(0.5, "baseline_aligned", "global", "explicit", 592157828, geometry_multiplier=0.5) == "g0.5_p592157828_aligned_r0.5"
    assert format_ratio_id(0.5, "baseline_aligned", "tensor_local", "explicit", 592157828, geometry_multiplier=0.5) == "g0.5_p592157828_local_aligned_r0.5"


def test_cli_geometry_multiplier_argument_parsing():
    """Verify CLI parser handles default and explicit --geometry-multiplier flags."""
    parser = build_parser()

    args_def = parser.parse_args([])
    assert args_def.geometry_multiplier == 1.0

    args_075 = parser.parse_args(["--geometry-multiplier", "0.75"])
    assert args_075.geometry_multiplier == 0.75

    args_05 = parser.parse_args(["--geometry-multiplier", "0.5"])
    assert args_05.geometry_multiplier == 0.5


def test_geometry_multiplier_runner_provenance_and_persistence(monkeypatch, tmp_path: Path):
    """Verify geometry_multiplier is persisted at experiment_config, workloads, candidate_runs, and per_seed_runs levels."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    res = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=1,
        subset_size=10,
        seeds=[101, 102],
        geometry_policy="baseline_aligned",
        projection_scope="global",
        projection_seed_mode="explicit",
        projection_seed=592157828,
        geometry_multiplier=0.75,
        device_str="cpu",
        cache_dir=tmp_path,
    )

    # 1. Experiment level
    assert res["experiment_config"]["geometry_multiplier"] == 0.75

    # 2. Workload level
    for wl_id in EXPECTED_WORKLOADS:
        assert res["workloads"][wl_id]["geometry_multiplier"] == 0.75

    # 3. Candidate level
    cand_id = "g0.75_p592157828_aligned_r0.5"
    assert cand_id in res["candidate_runs"]
    cand_entry = res["candidate_runs"][cand_id]
    for wl_id in EXPECTED_WORKLOADS:
        assert cand_entry[wl_id]["geometry_multiplier"] == 0.75

        # 4. Per-seed level
        for seed_rec in cand_entry[wl_id]["per_seed_runs"]:
            assert seed_rec["geometry_multiplier"] == 0.75


def test_geometry_multiplier_unchanged_accounting(monkeypatch, tmp_path: Path):
    """Verify geometry_multiplier preserves total queries, samples, state bytes, baseline bytes, and dimension accounting."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    kwargs = dict(
        ratios=[0.5],
        particles=2,
        epochs=1,
        subset_size=10,
        seeds=[101, 102],
        geometry_policy="baseline_aligned",
        projection_scope="global",
        projection_seed_mode="explicit",
        projection_seed=592157828,
        device_str="cpu",
        cache_dir=tmp_path,
    )

    res_default = run_heavy_pso_autoresearch(geometry_multiplier=1.0, **kwargs)
    res_mult = run_heavy_pso_autoresearch(geometry_multiplier=0.75, **kwargs)

    assert res_default["experiment_config"]["total_queries"] == res_mult["experiment_config"]["total_queries"]
    assert res_default["experiment_config"]["total_sample_evaluations"] == res_mult["experiment_config"]["total_sample_evaluations"]

    cand_def = res_default["candidate_runs"]["p592157828_aligned_r0.5"]
    cand_mult = res_mult["candidate_runs"]["g0.75_p592157828_aligned_r0.5"]

    for wl_id in EXPECTED_WORKLOADS:
        assert cand_def[wl_id]["total_dim"] == cand_mult[wl_id]["total_dim"]
        assert cand_def[wl_id]["latent_dim"] == cand_mult[wl_id]["latent_dim"]
        assert cand_def[wl_id]["state_ratio"] == cand_mult[wl_id]["state_ratio"]
        assert cand_def[wl_id]["core_swarm_state_bytes"] == cand_mult[wl_id]["core_swarm_state_bytes"]
        assert cand_def[wl_id]["baseline_core_swarm_state_bytes"] == cand_mult[wl_id]["baseline_core_swarm_state_bytes"]


def test_explicit_projection_seed_multiplier_elites_confirmation_runs(monkeypatch, tmp_path: Path):
    """Verify explicit seed 592157828 ratio 0.5 can run at multipliers 0.75 and 0.5 under the matched evaluator schema."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    base_file = create_mock_baseline_json(tmp_path)

    for mult in (0.75, 0.5):
        cand_run = run_heavy_pso_autoresearch(
            ratios=[0.5],
            particles=2,
            epochs=1,
            subset_size=10,
            seeds=[101, 102, 103],
            geometry_policy="baseline_aligned",
            projection_scope="global",
            projection_seed_mode="explicit",
            projection_seed=592157828,
            geometry_multiplier=mult,
            device_str="cpu",
            cache_dir=tmp_path,
        )

        expected_cand_id = f"g{mult:g}_p592157828_aligned_r0.5"
        assert expected_cand_id in cand_run["candidate_runs"]

        cand_file = tmp_path / f"cand_mult_{mult}.json"
        with open(cand_file, "w", encoding="utf-8") as f:
            json.dump(cand_run, f)

        eval_res = evaluate_heavy_autoresearch(base_file, cand_file)
        assert isinstance(eval_res["pass"], bool)
        assert isinstance(eval_res["score"], float)
        assert eval_res["evaluator_version"] == EVALUATOR_VERSION
        assert expected_cand_id in eval_res["candidate_evaluations"]
        assert isinstance(eval_res["candidate_evaluations"][expected_cand_id]["gate_details"]["gate_config_matched"], bool)
def test_balanced_global_latent_transform_occupancy_balance():
    """Verify BalancedGlobalLatentTransform produces occupancy differing by at most one and valid weights."""
    base_model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))
    total_dim = sum(p.numel() for p in base_model.parameters())
    latent_dim = 2780

    geom_cfg = V6GeometryConfig(
        config_id="test_balanced",
        projection_seed=12345,
        latent_dim=latent_dim,
    )
    transform = BalancedGlobalLatentTransform(base_model, geom_cfg, torch.device("cpu"))
    assert transform.is_full is False
    k_indices = transform.k_indices.cpu().numpy()
    weights = transform.weights.cpu().numpy()

    bin_counts = np.bincount(k_indices, minlength=latent_dim)
    assert bin_counts.max() - bin_counts.min() <= 1, "Bucket occupancies must differ by at most 1"
    assert len(weights) == total_dim
    assert np.all(np.isfinite(weights))


def test_balanced_global_seed_reproducibility_and_variation():
    """Verify BalancedGlobalLatentTransform reproduces identically for same seed and varies for different seed."""
    base_model = nn.Sequential(nn.Linear(50, 20), nn.ReLU(), nn.Linear(20, 5))

    geom_cfg1 = V6GeometryConfig(config_id="g1", projection_seed=42, latent_dim=100)
    geom_cfg2 = V6GeometryConfig(config_id="g2", projection_seed=42, latent_dim=100)
    geom_cfg3 = V6GeometryConfig(config_id="g3", projection_seed=43, latent_dim=100)

    t1 = BalancedGlobalLatentTransform(base_model, geom_cfg1, torch.device("cpu"))
    t2 = BalancedGlobalLatentTransform(base_model, geom_cfg2, torch.device("cpu"))
    t3 = BalancedGlobalLatentTransform(base_model, geom_cfg3, torch.device("cpu"))

    assert torch.equal(t1.k_indices, t2.k_indices)
    assert torch.equal(t1.weights, t2.weights)

    assert not torch.equal(t1.k_indices, t3.k_indices) or not torch.equal(t1.weights, t3.weights)


def test_projection_scope_parsing_and_validation():
    """Verify projection scope parsing and validation for strings and workload dictionaries."""
    # Parsing
    assert parse_projection_scope_arg("global") == "global"
    assert parse_projection_scope_arg("balanced_global") == "balanced_global"
    dict_str = '{"mnist_compact": "global", "mnist_wide": "balanced_global", "fashion_compact": "global", "fashion_wide": "balanced_global"}'
    parsed = parse_projection_scope_arg(dict_str)
    assert parsed["mnist_wide"] == "balanced_global"

    kv_str = "mnist_compact:global,mnist_wide:balanced_global,fashion_compact:global,fashion_wide:balanced_global"
    parsed_kv = parse_projection_scope_arg(kv_str)
    assert parsed_kv["mnist_wide"] == "balanced_global"

    # Validation
    validate_projection_scope_config("global")
    validate_projection_scope_config("tensor_local")
    validate_projection_scope_config("balanced_global")
    validate_projection_scope_config("adjacent_difference")
    mixed_dict = {
        "mnist_compact": "global",
        "mnist_wide": "balanced_global",
        "fashion_compact": "global",
        "fashion_wide": "balanced_global",
    }
    validate_projection_scope_config(mixed_dict)

    with pytest.raises(ValueError, match="Invalid projection_scope"):
        validate_projection_scope_config("unknown_scope")

    with pytest.raises(ValueError, match="missing keys"):
        validate_projection_scope_config({"mnist_compact": "global"})

    with pytest.raises(ValueError, match="unknown keys"):
        validate_projection_scope_config({
            "mnist_compact": "global",
            "mnist_wide": "balanced_global",
            "fashion_compact": "global",
            "fashion_wide": "balanced_global",
            "extra_key": "global",
        })

    with pytest.raises(ValueError, match="Invalid projection_scope"):
        validate_projection_scope_config({
            "mnist_compact": "global",
            "mnist_wide": "invalid_scope",
            "fashion_compact": "global",
            "fashion_wide": "balanced_global",
        })


def test_mixed_projection_scope_transform_selection_and_candidate_id(monkeypatch, tmp_path: Path):
    """Verify mixed scope dict candidate ID formatting and transform selection/serialization."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    mixed_scope = {
        "mnist_compact": "global",
        "mnist_wide": "balanced_global",
        "fashion_compact": "global",
        "fashion_wide": "balanced_global",
    }
    projection_seeds = {
        "mnist_compact": 1800044939,
        "mnist_wide": 592157828,
        "fashion_compact": 1363313651,
        "fashion_wide": 189641451,
    }

    cand_id = format_ratio_id(
        0.5,
        "baseline_aligned",
        mixed_scope,
        "explicit",
        projection_seeds,
    )
    assert cand_id == "pexplicit_mixed_aligned_r0.5"

    payload = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=1,
        subset_size=10,
        seeds=[101, 102, 103],
        geometry_policy="baseline_aligned",
        projection_scope=mixed_scope,
        projection_seed_mode="explicit",
        projection_seed=projection_seeds,
        device_str="cpu",
        cache_dir=tmp_path,
    )

    assert payload["experiment_config"]["projection_scope"] == mixed_scope
    assert payload["workloads"]["mnist_compact"]["projection_scope"] == "global"
    assert payload["workloads"]["mnist_wide"]["projection_scope"] == "balanced_global"

    cand_runs = payload["candidate_runs"]["pexplicit_mixed_aligned_r0.5"]
    assert cand_runs["mnist_compact"]["projection_scope"] == "global"
    assert cand_runs["mnist_wide"]["projection_scope"] == "balanced_global"
    assert cand_runs["mnist_compact"]["per_seed_runs"][0]["projection_scope"] == "global"
    assert cand_runs["mnist_wide"]["per_seed_runs"][0]["projection_scope"] == "balanced_global"


def test_balanced_global_state_math_unchanged():
    """Verify state bytes accounting is identical for global, tensor_local, and balanced_global scopes."""
    bytes_global = compute_core_swarm_state_bytes(12, 4549)
    bytes_balanced = compute_core_swarm_state_bytes(12, 4549)
    assert bytes_global == bytes_balanced == 5 * 12 * 4549 * 4


def test_two_hash_global_transform_same_seed_reproducibility_and_variation():
    """Verify TwoHashGlobalLatentTransform is deterministic for identical seeds and varies across seeds."""
    model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 2))
    geom_table = get_v6_geometry_table()
    base_geom = geom_table["G6"]
    total_dim = sum(p.numel() for p in model.parameters())
    geom_cfg1 = construct_equalized_geometry(
        base_geom=base_geom,
        total_dim=total_dim,
        latent_dim=15,
        projection_seed=42,
        ratio_str="r0.5",
    )
    geom_cfg2 = construct_equalized_geometry(
        base_geom=base_geom,
        total_dim=total_dim,
        latent_dim=15,
        projection_seed=42,
        ratio_str="r0.5",
    )
    geom_cfg3 = construct_equalized_geometry(
        base_geom=base_geom,
        total_dim=total_dim,
        latent_dim=15,
        projection_seed=43,
        ratio_str="r0.5",
    )
    device = torch.device("cpu")
    t1 = TwoHashGlobalLatentTransform(model, geom_cfg1, device)
    t2 = TwoHashGlobalLatentTransform(model, geom_cfg2, device)
    t3 = TwoHashGlobalLatentTransform(model, geom_cfg3, device)

    assert torch.equal(t1.k1_indices, t2.k1_indices)
    assert torch.allclose(t1.weights1, t2.weights1)
    assert torch.equal(t1.k2_indices, t2.k2_indices)
    assert torch.allclose(t1.weights2, t2.weights2)

    Z = torch.randn(5, 15)
    assert torch.allclose(t1.decode(Z), t2.decode(Z))

    assert not (torch.equal(t1.k1_indices, t3.k1_indices) and torch.equal(t1.k2_indices, t3.k2_indices))
    assert not torch.allclose(t1.decode(Z), t3.decode(Z))


def test_two_hash_global_transform_distinct_coordinates():
    """Verify TwoHashGlobalLatentTransform assigns distinct coordinates (k1 != k2) when latent_dim > 1."""
    model = nn.Sequential(nn.Linear(30, 20), nn.Linear(20, 5))
    geom_table = get_v6_geometry_table()
    base_geom = geom_table["G5"]
    total_dim = sum(p.numel() for p in model.parameters())
    device = torch.device("cpu")

    for d in [2, 10, 50, 100]:
        geom_cfg = construct_equalized_geometry(
            base_geom=base_geom,
            total_dim=total_dim,
            latent_dim=d,
            projection_seed=123,
            ratio_str=f"d{d}",
        )
        transform = TwoHashGlobalLatentTransform(model, geom_cfg, device)
        assert (transform.k1_indices != transform.k2_indices).all()


def test_two_hash_global_transform_finite_weights_and_decode_formula():
    """Verify TwoHashGlobalLatentTransform weights are finite and decode formula matches math specification."""
    model = nn.Sequential(nn.Linear(10, 4))
    geom_table = get_v6_geometry_table()
    base_geom = geom_table["G6"]
    total_dim = sum(p.numel() for p in model.parameters())
    geom_cfg = construct_equalized_geometry(
        base_geom=base_geom,
        total_dim=total_dim,
        latent_dim=8,
        projection_seed=999,
        ratio_str="r0.5",
    )
    device = torch.device("cpu")
    transform = TwoHashGlobalLatentTransform(model, geom_cfg, device)

    assert torch.isfinite(transform.weights1).all()
    assert torch.isfinite(transform.weights2).all()
    assert not torch.equal(
        torch.sign(transform.weights1), torch.sign(transform.weights2)
    )

    Z = torch.randn(4, 8)
    decoded = transform.decode(Z)
    assert decoded.shape == (4, total_dim)

    term1 = Z[:, transform.k1_indices] * transform.weights1
    term2 = Z[:, transform.k2_indices] * transform.weights2
    expected_delta = (term1 + term2) / math.sqrt(2.0)
    expected_decoded = transform.base_vec + transform.scale_vec * expected_delta

    assert torch.allclose(decoded, expected_decoded, atol=1e-6)


def test_two_hash_global_full_dimensional_parity():
    """Verify TwoHashGlobalLatentTransform preserves full-dimensional parity when latent_dim == total_dim."""
    model = nn.Sequential(nn.Linear(10, 5))
    geom_table = get_v6_geometry_table()
    base_geom = geom_table["G6"]
    total_dim = sum(p.numel() for p in model.parameters())
    geom_cfg = construct_equalized_geometry(
        base_geom=base_geom,
        total_dim=total_dim,
        latent_dim=total_dim,
        projection_seed=777,
        ratio_str="r1.0",
    )
    device = torch.device("cpu")
    transform_two_hash = TwoHashGlobalLatentTransform(model, geom_cfg, device)
    transform_v6 = V6LatentTransform(model, geom_cfg, device)

    Z = torch.randn(3, total_dim)
    out_two_hash = transform_two_hash.decode(Z)
    out_v6 = transform_v6.decode(Z)

    assert torch.allclose(out_two_hash, out_v6)


def test_two_hash_global_unchanged_core_state_bytes():
    """Verify state bytes accounting is identical for two_hash_global, global, and other scopes."""
    bytes_global = compute_core_swarm_state_bytes(12, 4549)
    bytes_two_hash = compute_core_swarm_state_bytes(12, 4549)
    assert bytes_global == bytes_two_hash == 5 * 12 * 4549 * 4


def test_two_hash_global_runner_provenance_and_persistence(monkeypatch, tmp_path: Path):
    """Verify projection_scope='two_hash_global' persists in experiment_config, workloads, candidate_runs, and per_seed_runs."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    payload = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101],
        geometry_policy="baseline_aligned",
        projection_scope="two_hash_global",
        device_str="cpu",
        cache_dir=tmp_path,
    )

    assert payload["experiment_config"]["projection_scope"] == "two_hash_global"
    assert payload["workloads"]["mnist_compact"]["projection_scope"] == "two_hash_global"
    assert "two_hash_aligned_r0.5" in payload["candidate_runs"]

    cand_rec = payload["candidate_runs"]["two_hash_aligned_r0.5"]["mnist_compact"]
    assert cand_rec["projection_scope"] == "two_hash_global"
    assert cand_rec["candidate_id"] == "two_hash_aligned_r0.5"

    seed_rec = cand_rec["per_seed_runs"][0]
    assert seed_rec["projection_scope"] == "two_hash_global"


def test_largest_tensor_hash_transform_mapping_and_containment():
    """Verify LargestTensorHashLatentTransform maps non-largest tensors 1-to-1/direct and largest tensor into residual range."""
    from heavy_task_feasibility import create_model
    model = create_model("wide_cnn")
    param_numels = [p.numel() for p in model.parameters()]
    total_dim = sum(param_numels)
    largest_idx = int(np.argmax(param_numels))
    protected_dim = sum(numel for i, numel in enumerate(param_numels) if i != largest_idx)

    latent_dim = math.ceil(total_dim * 0.5)
    geom_cfg = V6GeometryConfig(
        config_id="test_lth_containment",
        projection_seed=12345,
        latent_dim=latent_dim,
    )
    device = torch.device("cpu")
    transform = LargestTensorHashLatentTransform(model, geom_cfg, device)

    residual_dim = latent_dim - protected_dim
    assert protected_dim == 5162
    assert residual_dim == 22507
    assert residual_dim > 0

    k_indices = transform.k_indices.cpu().numpy()
    weights = transform.weights.cpu().numpy()

    j_offset = 0
    direct_coord = 0
    for i, numel in enumerate(param_numels):
        j_slice = slice(j_offset, j_offset + numel)
        if i != largest_idx:
            expected_coords = np.arange(direct_coord, direct_coord + numel)
            assert np.array_equal(k_indices[j_slice], expected_coords)
            assert np.array_equal(weights[j_slice], np.ones(numel, dtype=np.float32))
            direct_coord += numel
        else:
            assert np.all(k_indices[j_slice] >= protected_dim)
            assert np.all(k_indices[j_slice] < latent_dim)
        j_offset += numel


def test_largest_tensor_hash_transform_determinism_and_seed_variation():
    """Verify LargestTensorHashLatentTransform reproduces for same seed and varies only hashed mapping/signs for changed seed."""
    from heavy_task_feasibility import create_model
    model = create_model("wide_cnn")
    param_numels = [p.numel() for p in model.parameters()]
    largest_idx = int(np.argmax(param_numels))
    largest_start = sum(param_numels[:largest_idx])
    largest_end = largest_start + param_numels[largest_idx]
    latent_dim = math.ceil(sum(param_numels) * 0.5)

    geom1 = V6GeometryConfig(config_id="g1", projection_seed=100, latent_dim=latent_dim)
    geom2 = V6GeometryConfig(config_id="g2", projection_seed=100, latent_dim=latent_dim)
    geom3 = V6GeometryConfig(config_id="g3", projection_seed=999, latent_dim=latent_dim)

    device = torch.device("cpu")
    t1 = LargestTensorHashLatentTransform(model, geom1, device)
    t2 = LargestTensorHashLatentTransform(model, geom2, device)
    t3 = LargestTensorHashLatentTransform(model, geom3, device)

    # Identical seed produces identical transform
    assert torch.equal(t1.k_indices, t2.k_indices)
    assert torch.equal(t1.weights, t2.weights)

    # Changed seed keeps direct/non-largest parameters identical
    direct_mask = torch.ones(sum(param_numels), dtype=torch.bool)
    direct_mask[largest_start:largest_end] = False
    assert torch.equal(t1.k_indices[direct_mask], t3.k_indices[direct_mask])
    assert torch.equal(t1.weights[direct_mask], t3.weights[direct_mask])

    # Changed seed varies hashed mapping/signs for the largest tensor
    largest_slice = slice(largest_start, largest_end)
    assert (
        not torch.equal(t1.k_indices[largest_slice], t3.k_indices[largest_slice])
        or not torch.equal(t1.weights[largest_slice], t3.weights[largest_slice])
    )


def test_largest_tensor_hash_transform_decode_formula():
    """Verify LargestTensorHashLatentTransform decode matches the mapped formula."""
    model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 2))
    total_dim = sum(p.numel() for p in model.parameters())
    latent_dim = math.ceil(total_dim * 0.5)
    geom_cfg = V6GeometryConfig(config_id="g_decode", projection_seed=42, latent_dim=latent_dim)
    device = torch.device("cpu")
    transform = LargestTensorHashLatentTransform(model, geom_cfg, device)

    Z = torch.randn(4, latent_dim)
    delta = Z[:, transform.k_indices] * transform.weights
    expected = transform.base_vec + transform.scale_vec * delta
    actual = transform.decode(Z)
    assert torch.allclose(actual, expected)


def test_largest_tensor_hash_transform_invalid_latent_budgets():
    """Verify LargestTensorHashLatentTransform rejects configurations where latent_dim cannot provide >= 1 coordinate for largest tensor."""
    model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 2))
    param_numels = [p.numel() for p in model.parameters()]
    largest_idx = int(np.argmax(param_numels))
    protected_dim = sum(numel for i, numel in enumerate(param_numels) if i != largest_idx)

    # latent_dim <= protected_dim should fail
    geom_invalid = V6GeometryConfig(config_id="g_inv", projection_seed=42, latent_dim=protected_dim)
    device = torch.device("cpu")
    with pytest.raises(ValueError, match="latent_dim .* must be greater than protected"):
        LargestTensorHashLatentTransform(model, geom_invalid, device)


def test_largest_tensor_hash_full_dimensional_parity():
    """Verify LargestTensorHashLatentTransform preserves full-dimensional parity when latent_dim == total_dim."""
    model = nn.Sequential(nn.Linear(10, 5))
    total_dim = sum(p.numel() for p in model.parameters())
    geom_cfg = V6GeometryConfig(config_id="g_full", projection_seed=42, latent_dim=total_dim)
    device = torch.device("cpu")

    transform_lth = LargestTensorHashLatentTransform(model, geom_cfg, device)
    transform_v6 = V6LatentTransform(model, geom_cfg, device)

    assert transform_lth.is_full is True
    Z = torch.randn(3, total_dim)
    assert torch.allclose(transform_lth.decode(Z), transform_v6.decode(Z))


def test_largest_tensor_hash_unchanged_core_state_bytes():
    """Verify state bytes accounting is identical for largest_tensor_hash, global, and other scopes."""
    bytes_global = compute_core_swarm_state_bytes(12, 4549)
    bytes_lth = compute_core_swarm_state_bytes(12, 4549)
    assert bytes_global == bytes_lth == 5 * 12 * 4549 * 4


def test_largest_tensor_hash_runner_provenance_and_persistence(monkeypatch, tmp_path: Path):
    """Verify mixed projection_scope {global, largest_tensor_hash, global, largest_tensor_hash} is accepted and persists at all levels."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    mixed_scope = {
        "mnist_compact": "global",
        "mnist_wide": "largest_tensor_hash",
        "fashion_compact": "global",
        "fashion_wide": "largest_tensor_hash",
    }

    payload = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101],
        geometry_policy="baseline_aligned",
        projection_scope=mixed_scope,
        device_str="cpu",
        cache_dir=tmp_path,
    )

    assert payload["experiment_config"]["projection_scope"] == mixed_scope
    assert payload["workloads"]["mnist_compact"]["projection_scope"] == "global"
    assert payload["workloads"]["mnist_wide"]["projection_scope"] == "largest_tensor_hash"

    cand_runs = payload["candidate_runs"]["mixed_aligned_r0.5"]
    assert cand_runs["mnist_compact"]["projection_scope"] == "global"
    assert cand_runs["mnist_wide"]["projection_scope"] == "largest_tensor_hash"
    assert cand_runs["mnist_compact"]["per_seed_runs"][0]["projection_scope"] == "global"
    assert cand_runs["mnist_wide"]["per_seed_runs"][0]["projection_scope"] == "largest_tensor_hash"
def test_largest_tensor_row_hash_transform_mapping_allocation_and_containment():
    """Verify LargestTensorRowHashLatentTransform maps non-largest tensors direct, partitions residual range across rows without overlap, and allocates exact sum."""
    from heavy_task_feasibility import create_model
    model = create_model("wide_cnn")
    param_numels = [p.numel() for p in model.parameters()]
    param_shapes = [p.shape for p in model.parameters()]
    total_dim = sum(param_numels)
    largest_idx = int(np.argmax(param_numels))
    protected_dim = sum(numel for i, numel in enumerate(param_numels) if i != largest_idx)

    latent_dim = math.ceil(total_dim * 0.5)
    geom_cfg = V6GeometryConfig(
        config_id="test_ltrh_containment",
        projection_seed=12345,
        latent_dim=latent_dim,
    )
    device = torch.device("cpu")
    transform = LargestTensorRowHashLatentTransform(model, geom_cfg, device)

    residual_dim = latent_dim - protected_dim
    assert protected_dim == 5162
    assert residual_dim == 22507

    shape = param_shapes[largest_idx]
    num_rows = shape[0] if len(shape) >= 2 else 1
    assert num_rows == 32
    assert len(transform.row_latent_dims) == 32
    assert sum(transform.row_latent_dims) == residual_dim

    k_indices = transform.k_indices.cpu().numpy()
    weights = transform.weights.cpu().numpy()

    # 1. Protected direct mapping
    j_offset = 0
    direct_coord = 0
    for i, numel in enumerate(param_numels):
        if i != largest_idx:
            j_slice = slice(j_offset, j_offset + numel)
            expected_coords = np.arange(direct_coord, direct_coord + numel)
            assert np.array_equal(k_indices[j_slice], expected_coords)
            assert np.array_equal(weights[j_slice], np.ones(numel, dtype=np.float32))
            direct_coord += numel
            j_offset += numel
        else:
            j_largest_start = j_offset
            j_offset += numel

    # 2. Row slices non-overlap and containment
    elements_per_row = param_numels[largest_idx] // num_rows
    row_start_coord = protected_dim
    for r in range(num_rows):
        r_dim = transform.row_latent_dims[r]
        assert r_dim >= 1
        r_end_coord = row_start_coord + r_dim
        r_j_slice = slice(
            j_largest_start + r * elements_per_row,
            j_largest_start + (r + 1) * elements_per_row,
        )
        r_k = k_indices[r_j_slice]
        assert np.all(r_k >= row_start_coord)
        assert np.all(r_k < r_end_coord)
        row_start_coord = r_end_coord

    assert row_start_coord == latent_dim


def test_largest_tensor_row_hash_transform_determinism_and_seed_variation():
    """Verify LargestTensorRowHashLatentTransform reproduces for same seed and varies only hashed mapping/signs for changed seed."""
    from heavy_task_feasibility import create_model
    model = create_model("wide_cnn")
    param_numels = [p.numel() for p in model.parameters()]
    largest_idx = int(np.argmax(param_numels))
    largest_start = sum(param_numels[:largest_idx])
    largest_end = largest_start + param_numels[largest_idx]
    latent_dim = math.ceil(sum(param_numels) * 0.5)

    geom1 = V6GeometryConfig(config_id="g1", projection_seed=100, latent_dim=latent_dim)
    geom2 = V6GeometryConfig(config_id="g2", projection_seed=100, latent_dim=latent_dim)
    geom3 = V6GeometryConfig(config_id="g3", projection_seed=999, latent_dim=latent_dim)

    device = torch.device("cpu")
    t1 = LargestTensorRowHashLatentTransform(model, geom1, device)
    t2 = LargestTensorRowHashLatentTransform(model, geom2, device)
    t3 = LargestTensorRowHashLatentTransform(model, geom3, device)

    # Identical seed produces identical transform
    assert torch.equal(t1.k_indices, t2.k_indices)
    assert torch.equal(t1.weights, t2.weights)

    # Changed seed keeps direct/non-largest parameters identical
    direct_mask = torch.ones(sum(param_numels), dtype=torch.bool)
    direct_mask[largest_start:largest_end] = False
    assert torch.equal(t1.k_indices[direct_mask], t3.k_indices[direct_mask])
    assert torch.equal(t1.weights[direct_mask], t3.weights[direct_mask])

    # Changed seed varies hashed mapping/signs for the largest tensor
    largest_slice = slice(largest_start, largest_end)
    assert (
        not torch.equal(t1.k_indices[largest_slice], t3.k_indices[largest_slice])
        or not torch.equal(t1.weights[largest_slice], t3.weights[largest_slice])
    )


def test_largest_tensor_row_hash_transform_decode_formula():
    """Verify LargestTensorRowHashLatentTransform decode matches the mapped formula."""
    model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 2))
    total_dim = sum(p.numel() for p in model.parameters())
    latent_dim = math.ceil(total_dim * 0.5)
    geom_cfg = V6GeometryConfig(config_id="g_decode_row", projection_seed=42, latent_dim=latent_dim)
    device = torch.device("cpu")
    transform = LargestTensorRowHashLatentTransform(model, geom_cfg, device)

    Z = torch.randn(4, latent_dim)
    delta = Z[:, transform.k_indices] * transform.weights
    expected = transform.base_vec + transform.scale_vec * delta
    actual = transform.decode(Z)
    assert torch.allclose(actual, expected)


def test_largest_tensor_row_hash_transform_invalid_latent_budgets():
    """Verify LargestTensorRowHashLatentTransform rejects configurations where residual_dim < num_rows."""
    model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 2))
    param_numels = [p.numel() for p in model.parameters()]
    largest_idx = int(np.argmax(param_numels))
    protected_dim = sum(numel for i, numel in enumerate(param_numels) if i != largest_idx)

    num_rows = list(model.parameters())[largest_idx].shape[0]
    # One fewer than the minimum residual coordinate count must fail.
    geom_invalid = V6GeometryConfig(
        config_id="g_inv_row",
        projection_seed=42,
        latent_dim=protected_dim + num_rows - 1,
    )
    device = torch.device("cpu")
    with pytest.raises(ValueError, match="latent_dim .* must be at least protected dimension"):
        LargestTensorRowHashLatentTransform(model, geom_invalid, device)


def test_largest_tensor_row_hash_full_dimensional_parity():
    """Verify LargestTensorRowHashLatentTransform preserves full-dimensional parity when latent_dim == total_dim."""
    model = nn.Sequential(nn.Linear(10, 5))
    total_dim = sum(p.numel() for p in model.parameters())
    geom_cfg = V6GeometryConfig(config_id="g_full_row", projection_seed=42, latent_dim=total_dim)
    device = torch.device("cpu")

    transform_ltrh = LargestTensorRowHashLatentTransform(model, geom_cfg, device)
    transform_v6 = V6LatentTransform(model, geom_cfg, device)

    assert transform_ltrh.is_full is True
    Z = torch.randn(3, total_dim)
    assert torch.allclose(transform_ltrh.decode(Z), transform_v6.decode(Z))


def test_largest_tensor_row_hash_unchanged_core_state_bytes():
    """Verify state bytes accounting is identical for largest_tensor_row_hash, global, and other scopes."""
    bytes_global = compute_core_swarm_state_bytes(12, 4549)
    bytes_ltrh = compute_core_swarm_state_bytes(12, 4549)
    assert bytes_global == bytes_ltrh == 5 * 12 * 4549 * 4


def test_largest_tensor_row_hash_runner_provenance_and_persistence(monkeypatch, tmp_path: Path):
    """Verify mixed projection_scope {global, largest_tensor_row_hash, global, largest_tensor_row_hash} is accepted and persists at all levels."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    mixed_scope = {
        "mnist_compact": "global",
        "mnist_wide": "largest_tensor_row_hash",
        "fashion_compact": "global",
        "fashion_wide": "largest_tensor_row_hash",
    }

    payload = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101],
        geometry_policy="baseline_aligned",
        projection_scope=mixed_scope,
        device_str="cpu",
        cache_dir=tmp_path,
    )

    assert payload["experiment_config"]["projection_scope"] == mixed_scope
    assert payload["workloads"]["mnist_compact"]["projection_scope"] == "global"
    assert payload["workloads"]["mnist_wide"]["projection_scope"] == "largest_tensor_row_hash"

    cand_runs = payload["candidate_runs"]["mixed_aligned_r0.5"]
    assert cand_runs["mnist_compact"]["projection_scope"] == "global"
    assert cand_runs["mnist_wide"]["projection_scope"] == "largest_tensor_row_hash"
    assert cand_runs["mnist_compact"]["per_seed_runs"][0]["projection_scope"] == "global"
    assert cand_runs["mnist_wide"]["per_seed_runs"][0]["projection_scope"] == "largest_tensor_row_hash"
def test_adjacent_pair_transform_mapping_allocation_and_containment():
    """Verify AdjacentPairLatentTransform pairs parameters tensor-locally, allocates sum(ceil(numel/2)), and prevents cross-tensor coordinate sharing."""
    model = nn.Sequential(nn.Linear(5, 4), nn.Linear(4, 3))
    # param_numels: [20, 4, 12, 3] -> ceil(numel/2): [10, 2, 6, 2], total required_dim = 20
    total_dim = sum(p.numel() for p in model.parameters())
    geom_cfg = V6GeometryConfig(config_id="g_adj", projection_seed=42, latent_dim=20)
    device = torch.device("cpu")

    transform = AdjacentPairLatentTransform(model, geom_cfg, device)
    assert transform.latent_dim == 20
    assert transform.tensor_latent_dims == [10, 2, 6, 2]

    # Verify tensor bounds and coordinate containment
    k_indices = transform.k_indices.cpu().numpy()
    j_offsets = [0, 20, 24, 36, 39]
    l_offsets = [0, 10, 12, 18, 20]

    for i in range(4):
        tensor_k = k_indices[j_offsets[i]:j_offsets[i+1]]
        assert np.all(tensor_k >= l_offsets[i])
        assert np.all(tensor_k < l_offsets[i+1])


def test_adjacent_pair_transform_pairing_and_normalized_weights():
    """Verify AdjacentPairLatentTransform maps consecutive pairs to same latent coordinate with weight 1/sqrt(2), unpaired final parameter to 1.0, and column norm == 1.0."""
    class OddEvenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.p1 = nn.Parameter(torch.randn(5))  # odd -> 3 latent coords
            self.p2 = nn.Parameter(torch.randn(4))  # even -> 2 latent coords

    model = OddEvenModel()
    # param_numels: [5, 4] -> ceil(numel/2): [3, 2], required_dim = 5
    geom_cfg = V6GeometryConfig(config_id="g_adj", projection_seed=42, latent_dim=5)
    device = torch.device("cpu")

    transform = AdjacentPairLatentTransform(model, geom_cfg, device)
    k_indices = transform.k_indices.cpu().numpy()
    weights = transform.weights.cpu().numpy()
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    # Tensor 0 (size 5):
    # p=0,1 -> k=0, w=inv_sqrt2
    # p=2,3 -> k=1, w=inv_sqrt2
    # p=4   -> k=2, w=1.0
    assert k_indices[0] == k_indices[1] == 0
    assert k_indices[2] == k_indices[3] == 1
    assert k_indices[4] == 2

    assert np.isclose(weights[0], inv_sqrt2)
    assert np.isclose(weights[1], inv_sqrt2)
    assert np.isclose(weights[2], inv_sqrt2)
    assert np.isclose(weights[3], inv_sqrt2)
    assert np.isclose(weights[4], 1.0)

    # Tensor 1 (size 4):
    # p=5,6 -> k=3, w=inv_sqrt2
    # p=7,8 -> k=4, w=inv_sqrt2
    assert k_indices[5] == k_indices[6] == 3
    assert k_indices[7] == k_indices[8] == 4
    assert np.isclose(weights[5], inv_sqrt2)
    assert np.isclose(weights[6], inv_sqrt2)
    assert np.isclose(weights[7], inv_sqrt2)
    assert np.isclose(weights[8], inv_sqrt2)

    # Verify column norm = 1.0 for every latent coordinate
    for k in range(5):
        j_col = np.where(k_indices == k)[0]
        col_norm = math.sqrt(sum(weights[j]**2 for j in j_col))
        assert np.isclose(col_norm, 1.0)


def test_adjacent_pair_transform_seed_independence():
    """Verify AdjacentPairLatentTransform mapping and weights are completely deterministic and seed-independent."""
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
    req_dim = sum(math.ceil(p.numel() / 2) for p in model.parameters())
    device = torch.device("cpu")

    geom_cfg1 = V6GeometryConfig(config_id="g1", projection_seed=101, latent_dim=req_dim)
    geom_cfg2 = V6GeometryConfig(config_id="g2", projection_seed=999999, latent_dim=req_dim)
    geom_cfg3 = V6GeometryConfig(config_id="g3", projection_seed=None, latent_dim=req_dim)

    t1 = AdjacentPairLatentTransform(model, geom_cfg1, device)
    t2 = AdjacentPairLatentTransform(model, geom_cfg2, device)
    t3 = AdjacentPairLatentTransform(model, geom_cfg3, device)

    assert torch.equal(t1.k_indices, t2.k_indices)
    assert torch.equal(t1.k_indices, t3.k_indices)
    assert torch.allclose(t1.weights, t2.weights)
    assert torch.allclose(t1.weights, t3.weights)


def test_adjacent_pair_transform_decode_formula():
    """Verify AdjacentPairLatentTransform decode matches base_vec + scale_vec * (Z[:, k_indices] * weights)."""
    model = nn.Sequential(nn.Linear(6, 4), nn.Linear(4, 2))
    req_dim = sum(math.ceil(p.numel() / 2) for p in model.parameters())
    geom_cfg = V6GeometryConfig(config_id="g_adj", projection_seed=42, latent_dim=req_dim)
    device = torch.device("cpu")

    transform = AdjacentPairLatentTransform(model, geom_cfg, device)
    Z = torch.randn(5, req_dim)
    decoded = transform.decode(Z)

    expected_delta = Z[:, transform.k_indices] * transform.weights
    expected_theta = transform.base_vec + transform.scale_vec * expected_delta

    assert torch.allclose(decoded, expected_theta)


def test_adjacent_pair_transform_required_dimension_rejection():
    """Verify AdjacentPairLatentTransform rejects non-full configurations where latent_dim != sum(ceil(numel/2))."""
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
    req_dim = sum(math.ceil(p.numel() / 2) for p in model.parameters())
    device = torch.device("cpu")

    invalid_latent_dim = req_dim - 1
    geom_invalid = V6GeometryConfig(config_id="g_inv", projection_seed=42, latent_dim=invalid_latent_dim)

    with pytest.raises(ValueError, match="AdjacentPairLatentTransform requires latent_dim == sum\\(ceil\\(numel_i/2\\)\\)"):
        AdjacentPairLatentTransform(model, geom_invalid, device)


def test_adjacent_pair_full_dimensional_parity():
    """Verify AdjacentPairLatentTransform preserves full-dimensional parity when latent_dim == total_dim."""
    model = nn.Sequential(nn.Linear(10, 5))
    total_dim = sum(p.numel() for p in model.parameters())
    geom_cfg = V6GeometryConfig(config_id="g_full_adj", projection_seed=42, latent_dim=total_dim)
    device = torch.device("cpu")

    transform_adj = AdjacentPairLatentTransform(model, geom_cfg, device)
    transform_v6 = V6LatentTransform(model, geom_cfg, device)

    assert transform_adj.is_full is True
    assert transform_adj.tensor_latent_dims == [p.numel() for p in model.parameters()]
    Z = torch.randn(3, total_dim)
    assert torch.allclose(transform_adj.decode(Z), transform_v6.decode(Z))


def test_adjacent_pair_unchanged_core_state_bytes():
    """Verify state bytes accounting is identical for adjacent_pair, global, and other scopes."""
    bytes_global = compute_core_swarm_state_bytes(12, 4549)
    bytes_adj = compute_core_swarm_state_bytes(12, 4549)
    assert bytes_global == bytes_adj == 5 * 12 * 4549 * 4


def test_adjacent_pair_mixed_wide_only_runner_provenance_and_persistence(monkeypatch, tmp_path: Path):
    """Verify mixed projection_scope {global, adjacent_pair, global, adjacent_pair} is accepted and persists at all levels."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    mixed_scope = {
        "mnist_compact": "global",
        "mnist_wide": "adjacent_pair",
        "fashion_compact": "global",
        "fashion_wide": "adjacent_pair",
    }

    payload = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101],
        geometry_policy="baseline_aligned",
        projection_scope=mixed_scope,
        device_str="cpu",
        cache_dir=tmp_path,
    )

    assert payload["experiment_config"]["projection_scope"] == mixed_scope
    assert payload["workloads"]["mnist_compact"]["projection_scope"] == "global"
    assert payload["workloads"]["mnist_wide"]["projection_scope"] == "adjacent_pair"

    cand_runs = payload["candidate_runs"]["mixed_aligned_r0.5"]
    assert cand_runs["mnist_compact"]["projection_scope"] == "global"
    assert cand_runs["mnist_wide"]["projection_scope"] == "adjacent_pair"
    assert cand_runs["mnist_compact"]["per_seed_runs"][0]["projection_scope"] == "global"
    assert cand_runs["mnist_wide"]["per_seed_runs"][0]["projection_scope"] == "adjacent_pair"


def test_adjacent_pair_compact_and_wide_acceptance_criteria():
    """Verify that for both CompactCNN and WideCNN architectures at ratio 0.5, AdjacentPairLatentTransform covers every tensor, has column norm 1 for every latent column, and matches half-dim."""
    from heavy_task_feasibility import create_model

    for wl_name in ["compact_cnn", "wide_cnn"]:
        model = create_model(wl_name)
        total_dim = sum(p.numel() for p in model.parameters())
        runner_half_dim = compute_latent_dim(total_dim, 0.5)

        req_dim = sum(math.ceil(p.numel() / 2) for p in model.parameters())
        assert req_dim == runner_half_dim, f"For model {wl_name}, required pair dim {req_dim} must equal runner half-dim {runner_half_dim}"

        geom_cfg = V6GeometryConfig(config_id=f"g_{wl_name}", projection_seed=42, latent_dim=runner_half_dim)
        device = torch.device("cpu")

        transform = AdjacentPairLatentTransform(model, geom_cfg, device)
        assert transform.latent_dim == runner_half_dim

        # Check total parameter coverage
        assert len(transform.k_indices) == total_dim
        assert len(transform.weights) == total_dim

        k_indices = transform.k_indices.cpu().numpy()
        weights = transform.weights.cpu().numpy()

        j_offset = 0
        for p in model.parameters():
            numel = p.numel()
            tensor_k = k_indices[j_offset:j_offset + numel]
            # Check tensor-local contiguous latent coords
            assert np.min(tensor_k) >= 0
            assert np.max(tensor_k) < runner_half_dim
            j_offset += numel

        # Check column norm = 1.0 for all latent columns
        for k in range(runner_half_dim):
            j_col = np.where(k_indices == k)[0]
            assert 1 <= len(j_col) <= 2, f"Latent column {k} must map to 1 or 2 parameters, got {len(j_col)}"
            if len(j_col) == 2:
                # Non-singleton: adjacent parameters from one tensor
                assert j_col[1] == j_col[0] + 1
            col_norm = math.sqrt(sum(weights[j]**2 for j in j_col))
            assert np.isclose(col_norm, 1.0), f"Latent column {k} column norm must be 1.0, got {col_norm}"
def test_adjacent_difference_transform_mapping_allocation_and_containment():
    """Verify AdjacentDifferenceLatentTransform pairs parameters tensor-locally, allocates sum(ceil(numel/2)), and prevents cross-tensor coordinate sharing."""
    model = nn.Sequential(nn.Linear(5, 4), nn.Linear(4, 3))
    # param_numels: [20, 4, 12, 3] -> ceil(numel/2): [10, 2, 6, 2], total required_dim = 20
    total_dim = sum(p.numel() for p in model.parameters())
    geom_cfg = V6GeometryConfig(config_id="g_adj_diff", projection_seed=42, latent_dim=20)
    device = torch.device("cpu")

    transform = AdjacentDifferenceLatentTransform(model, geom_cfg, device)
    assert transform.latent_dim == 20
    assert transform.tensor_latent_dims == [10, 2, 6, 2]

    # Verify tensor bounds and coordinate containment
    k_indices = transform.k_indices.cpu().numpy()
    j_offsets = [0, 20, 24, 36, 39]
    l_offsets = [0, 10, 12, 18, 20]

    for i in range(4):
        tensor_k = k_indices[j_offsets[i]:j_offsets[i+1]]
        assert np.all(tensor_k >= l_offsets[i])
        assert np.all(tensor_k < l_offsets[i+1])


def test_adjacent_difference_transform_pairing_and_normalized_weights():
    """Verify AdjacentDifferenceLatentTransform maps consecutive pairs to same latent coordinate with opposite weights (+1/sqrt(2), -1/sqrt(2)), zero pair-column sums, unpaired final parameter to 1.0, and column norm == 1.0."""
    class OddEvenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.p1 = nn.Parameter(torch.randn(5))  # odd -> 3 latent coords
            self.p2 = nn.Parameter(torch.randn(4))  # even -> 2 latent coords

    model = OddEvenModel()
    # param_numels: [5, 4] -> ceil(numel/2): [3, 2], required_dim = 5
    geom_cfg = V6GeometryConfig(config_id="g_adj_diff", projection_seed=42, latent_dim=5)
    device = torch.device("cpu")

    transform = AdjacentDifferenceLatentTransform(model, geom_cfg, device)
    k_indices = transform.k_indices.cpu().numpy()
    weights = transform.weights.cpu().numpy()
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    # Tensor 0 (size 5):
    # p=0,1 -> k=0, w=(+inv_sqrt2, -inv_sqrt2)
    # p=2,3 -> k=1, w=(+inv_sqrt2, -inv_sqrt2)
    # p=4   -> k=2, w=+1.0
    assert k_indices[0] == k_indices[1] == 0
    assert k_indices[2] == k_indices[3] == 1
    assert k_indices[4] == 2

    assert np.isclose(weights[0], +inv_sqrt2)
    assert np.isclose(weights[1], -inv_sqrt2)
    assert np.isclose(weights[2], +inv_sqrt2)
    assert np.isclose(weights[3], -inv_sqrt2)
    assert np.isclose(weights[4], 1.0)

    # Tensor 1 (size 4):
    # p=5,6 -> k=3, w=(+inv_sqrt2, -inv_sqrt2)
    # p=7,8 -> k=4, w=(+inv_sqrt2, -inv_sqrt2)
    assert k_indices[5] == k_indices[6] == 3
    assert k_indices[7] == k_indices[8] == 4
    assert np.isclose(weights[5], +inv_sqrt2)
    assert np.isclose(weights[6], -inv_sqrt2)
    assert np.isclose(weights[7], +inv_sqrt2)
    assert np.isclose(weights[8], -inv_sqrt2)

    # Verify zero pair-column sums and column norm = 1.0 for every latent coordinate
    for k in range(5):
        j_col = np.where(k_indices == k)[0]
        if len(j_col) == 2:
            col_sum = sum(weights[j] for j in j_col)
            assert np.isclose(col_sum, 0.0), f"Pair column {k} sum must be 0.0, got {col_sum}"
        elif len(j_col) == 1:
            assert np.isclose(weights[j_col[0]], 1.0)
        col_norm = math.sqrt(sum(weights[j]**2 for j in j_col))
        assert np.isclose(col_norm, 1.0), f"Column {k} norm must be 1.0, got {col_norm}"


def test_adjacent_difference_transform_seed_independence():
    """Verify AdjacentDifferenceLatentTransform mapping and weights are completely deterministic and seed-independent."""
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
    req_dim = sum(math.ceil(p.numel() / 2) for p in model.parameters())
    device = torch.device("cpu")

    geom_cfg1 = V6GeometryConfig(config_id="g1", projection_seed=101, latent_dim=req_dim)
    geom_cfg2 = V6GeometryConfig(config_id="g2", projection_seed=999999, latent_dim=req_dim)
    geom_cfg3 = V6GeometryConfig(config_id="g3", projection_seed=None, latent_dim=req_dim)

    t1 = AdjacentDifferenceLatentTransform(model, geom_cfg1, device)
    t2 = AdjacentDifferenceLatentTransform(model, geom_cfg2, device)
    t3 = AdjacentDifferenceLatentTransform(model, geom_cfg3, device)

    assert torch.equal(t1.k_indices, t2.k_indices)
    assert torch.equal(t1.k_indices, t3.k_indices)
    assert torch.allclose(t1.weights, t2.weights)
    assert torch.allclose(t1.weights, t3.weights)


def test_adjacent_difference_transform_decode_formula():
    """Verify AdjacentDifferenceLatentTransform decode matches base_vec + scale_vec * (Z[:, k_indices] * weights)."""
    model = nn.Sequential(nn.Linear(6, 4), nn.Linear(4, 2))
    req_dim = sum(math.ceil(p.numel() / 2) for p in model.parameters())
    geom_cfg = V6GeometryConfig(config_id="g_adj_diff", projection_seed=42, latent_dim=req_dim)
    device = torch.device("cpu")

    transform = AdjacentDifferenceLatentTransform(model, geom_cfg, device)
    Z = torch.randn(5, req_dim)
    decoded = transform.decode(Z)

    expected_delta = Z[:, transform.k_indices] * transform.weights
    expected_theta = transform.base_vec + transform.scale_vec * expected_delta

    assert torch.allclose(decoded, expected_theta)


def test_adjacent_difference_transform_required_dimension_rejection():
    """Verify AdjacentDifferenceLatentTransform rejects non-full configurations where latent_dim != sum(ceil(numel/2))."""
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
    req_dim = sum(math.ceil(p.numel() / 2) for p in model.parameters())
    device = torch.device("cpu")

    invalid_latent_dim = req_dim - 1
    geom_invalid = V6GeometryConfig(config_id="g_inv", projection_seed=42, latent_dim=invalid_latent_dim)

    with pytest.raises(ValueError, match="AdjacentDifferenceLatentTransform requires latent_dim == sum\\(ceil\\(numel_i/2\\)\\)"):
        AdjacentDifferenceLatentTransform(model, geom_invalid, device)


def test_adjacent_difference_full_dimensional_parity():
    """Verify AdjacentDifferenceLatentTransform preserves full-dimensional parity when latent_dim == total_dim."""
    model = nn.Sequential(nn.Linear(10, 5))
    total_dim = sum(p.numel() for p in model.parameters())
    geom_cfg = V6GeometryConfig(config_id="g_full_adj_diff", projection_seed=42, latent_dim=total_dim)
    device = torch.device("cpu")

    transform_adj = AdjacentDifferenceLatentTransform(model, geom_cfg, device)
    transform_v6 = V6LatentTransform(model, geom_cfg, device)

    assert transform_adj.is_full is True
    assert transform_adj.tensor_latent_dims == [p.numel() for p in model.parameters()]
    Z = torch.randn(3, total_dim)
    assert torch.allclose(transform_adj.decode(Z), transform_v6.decode(Z))


def test_adjacent_difference_unchanged_core_state_bytes():
    """Verify state bytes accounting is identical for adjacent_difference, adjacent_pair, global, and other scopes."""
    bytes_global = compute_core_swarm_state_bytes(12, 4549)
    bytes_adj_diff = compute_core_swarm_state_bytes(12, 4549)
    assert bytes_global == bytes_adj_diff == 5 * 12 * 4549 * 4


def test_adjacent_difference_mixed_wide_only_runner_provenance_and_persistence(monkeypatch, tmp_path: Path):
    """Verify mixed projection_scope {global, adjacent_difference, global, adjacent_difference} is accepted and persists at all levels."""
    monkeypatch.setattr(heavy_pso_autoresearch, "prepare_heavy_task_data", _mock_prepare_heavy_task_data)

    mixed_scope = {
        "mnist_compact": "global",
        "mnist_wide": "adjacent_difference",
        "fashion_compact": "global",
        "fashion_wide": "adjacent_difference",
    }

    payload = run_heavy_pso_autoresearch(
        ratios=[0.5],
        particles=2,
        epochs=2,
        subset_size=10,
        seeds=[101],
        geometry_policy="baseline_aligned",
        projection_scope=mixed_scope,
        device_str="cpu",
        cache_dir=tmp_path,
    )

    assert payload["experiment_config"]["projection_scope"] == mixed_scope
    assert payload["workloads"]["mnist_compact"]["projection_scope"] == "global"
    assert payload["workloads"]["mnist_wide"]["projection_scope"] == "adjacent_difference"

    cand_runs = payload["candidate_runs"]["mixed_aligned_r0.5"]
    assert cand_runs["mnist_compact"]["projection_scope"] == "global"
    assert cand_runs["mnist_wide"]["projection_scope"] == "adjacent_difference"
    assert cand_runs["mnist_compact"]["per_seed_runs"][0]["projection_scope"] == "global"
    assert cand_runs["mnist_wide"]["per_seed_runs"][0]["projection_scope"] == "adjacent_difference"


def test_adjacent_difference_compact_and_wide_acceptance_criteria():
    """Verify that for both CompactCNN and WideCNN architectures at ratio 0.5, AdjacentDifferenceLatentTransform covers every tensor, has column norm 1 for every latent column, zero pair-column sums, and matches half-dim."""
    from heavy_task_feasibility import create_model

    for wl_name in ["compact_cnn", "wide_cnn"]:
        model = create_model(wl_name)
        total_dim = sum(p.numel() for p in model.parameters())
        runner_half_dim = compute_latent_dim(total_dim, 0.5)

        req_dim = sum(math.ceil(p.numel() / 2) for p in model.parameters())
        assert req_dim == runner_half_dim, f"For model {wl_name}, required pair dim {req_dim} must equal runner half-dim {runner_half_dim}"

        geom_cfg = V6GeometryConfig(config_id=f"g_{wl_name}", projection_seed=42, latent_dim=runner_half_dim)
        device = torch.device("cpu")

        transform = AdjacentDifferenceLatentTransform(model, geom_cfg, device)
        assert transform.latent_dim == runner_half_dim

        # Check total parameter coverage
        assert len(transform.k_indices) == total_dim
        assert len(transform.weights) == total_dim

        k_indices = transform.k_indices.cpu().numpy()
        weights = transform.weights.cpu().numpy()

        j_offset = 0
        for p in model.parameters():
            numel = p.numel()
            tensor_k = k_indices[j_offset:j_offset + numel]
            # Check tensor-local contiguous latent coords
            assert np.min(tensor_k) >= 0
            assert np.max(tensor_k) < runner_half_dim
            j_offset += numel

        # Check column norm = 1.0 and pair column sums = 0.0 for all latent columns
        for k in range(runner_half_dim):
            j_col = np.where(k_indices == k)[0]
            assert 1 <= len(j_col) <= 2, f"Latent column {k} must map to 1 or 2 parameters, got {len(j_col)}"
            if len(j_col) == 2:
                # Non-singleton: adjacent parameters from one tensor with opposite weights
                assert j_col[1] == j_col[0] + 1
                pair_sum = weights[j_col[0]] + weights[j_col[1]]
                assert np.isclose(pair_sum, 0.0), f"Latent pair column {k} sum must be 0.0, got {pair_sum}"
            else:
                assert np.isclose(weights[j_col[0]], 1.0)
            col_norm = math.sqrt(sum(weights[j]**2 for j in j_col))
            assert np.isclose(col_norm, 1.0), f"Latent column {k} column norm must be 1.0, got {col_norm}"
