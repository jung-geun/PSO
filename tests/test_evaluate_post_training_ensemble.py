"""
Unit tests for Strict Evaluator of Post-Training PSO Ensemble Study.

Covers:
1. Evaluator version and constant exports.
2. Complete valid study artifact evaluation (pass=True, 0 failed hard gates, valid score).
3. Schema tampering (non-dict, missing top-level keys, missing workloads).
4. Config tampering (wrong split seed, sample counts, pool seeds, PSO parameters).
5. Non-finite value scan (NaN or Inf values in nested metrics or weights).
6. Simplex weight validation failure (non-unit sum, negative elements).
7. Query and sample accounting mismatch.
8. Base-model forward count gate failure.
9. Data leakage contradictions, frozen-policy drift, and post-test tuning.
10. Duplicate or missing frozen swarm seeds.
11. SLSQP gap, uniform ensemble accuracy/NLL regression, and baseline NLL gates.
12. Per-workload wall-time ratio gate enforcement.
13. Missing-confirmation failure and evaluator CLI output.
"""

import json
import sys
from pathlib import Path

# Ensure test directory and repo root are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name != "PSO" else Path(__file__).resolve().parent
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from evaluate_post_training_ensemble import (
    EVALUATOR_VERSION,
    EXPECTED_DATASETS,
    EXPECTED_SPLIT_SEED,
    evaluate_artifact,
    main,
    save_json_atomic,
)


def make_valid_metrics(nll: float = 0.35, accuracy: float = 90.0):
    return {
        "accuracy": accuracy,
        "nll": nll,
        "brier": 0.15,
        "ece": 0.02,
        "margin": 0.5,
    }


def make_valid_method(
    nll: float = 0.35,
    accuracy: float = 90.0,
    weights: list = None,
    method_type: str = "base",
):
    if weights is None:
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]

    metrics = make_valid_metrics(nll, accuracy)

    if method_type == "pso":
        return {
            "selected_seed": 301,
            "selected_weights": weights,
            "weights": weights,
            "metrics": metrics,
            "queries_per_seed": 900,
            "sample_evaluations_per_seed": 9000000,
            "median_one_seed_wall_time_seconds": 2.0,
            "total_wall_time_seconds": 6.0,
            "per_seed_runs": [
                {
                    "seed": 301,
                    "queries": 900,
                    "sample_evaluations": 9000000,
                    "wall_time_seconds": 2.0,
                    "metrics": metrics,
                    "weights": weights,
                },
                {
                    "seed": 302,
                    "queries": 900,
                    "sample_evaluations": 9000000,
                    "wall_time_seconds": 2.0,
                    "metrics": make_valid_metrics(nll + 0.01, accuracy),
                    "weights": weights,
                },
                {
                    "seed": 303,
                    "queries": 900,
                    "sample_evaluations": 9000000,
                    "wall_time_seconds": 2.0,
                    "metrics": make_valid_metrics(nll + 0.02, accuracy),
                    "weights": weights,
                },
            ],
        }
    elif method_type == "slsqp":
        return {
            "weights": weights,
            "success": True,
            "wall_time_seconds": 0.5,
            "metrics": metrics,
        }
    elif method_type == "temp":
        return {
            "weights": weights,
            "fitted_temperature": 1.0,
            "metrics": metrics,
        }

    return metrics


def make_valid_workload_entry():
    return {
        "provenance": {"dataset_name": "mnist", "split_seed": EXPECTED_SPLIT_SEED},
        "training": {
            "adam_pool_model_epochs": 50,
            "adam_pool_wall_time_seconds": 100.0,
            "equal_budget_50e_single_wall_time_seconds": 25.0,
        },
        "validation_cache": {
            "pool_forward_passes": 5,
            "long_single_forward_passes": 1,
            "base_cnn_forward_passes_during_optimization": 0,
            "size_bytes": 2000000,
        },
        "validation": {
            "methods": {
                "reference_single_10e": make_valid_method(nll=0.50, accuracy=85.0, method_type="base"),
                "best_single_10e": make_valid_method(nll=0.45, accuracy=87.0, method_type="base"),
                "single_50e": make_valid_method(nll=0.40, accuracy=89.0, method_type="base"),
                "uniform_ensemble": make_valid_method(nll=0.36, accuracy=89.9, method_type="base"),
                "uniform_temperature": make_valid_method(nll=0.355, accuracy=90.0, method_type="temp"),
                "slsqp_weights": make_valid_method(nll=0.35, accuracy=90.0, method_type="slsqp"),
                "pso_weights": make_valid_method(nll=0.35, accuracy=90.0, method_type="pso"),
            }
        },
        "official_test_data_loaded_before_freeze": False,
        "official_test_evaluations_before_freeze": 0,
        "confirmation": {
            "test_cache_counts": {
                "dataset_loads": 1,
                "pool_forward_passes": 5,
                "long_single_forward_passes": 1,
            },
            "frozen_methods": {
                "selected_pso_seed": 301,
                "selected_pso_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
                "slsqp_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
                "fitted_temperature": 1.0,
            },
            "methods": {
                "reference_single_10e": make_valid_metrics(nll=0.52, accuracy=84.5),
                "best_single_10e": make_valid_metrics(nll=0.47, accuracy=86.5),
                "single_50e": make_valid_metrics(nll=0.42, accuracy=88.5),
                "uniform_ensemble": make_valid_metrics(nll=0.37, accuracy=89.5),
                "uniform_temperature": make_valid_metrics(nll=0.365, accuracy=89.6),
                "slsqp_weights": make_valid_metrics(nll=0.36, accuracy=89.7),
                "pso_weights": make_valid_metrics(nll=0.36, accuracy=89.7),
            },
        },
    }


def make_valid_study_artifact():
    return {
        "protocol_version": "POST-TRAINING-PSO-ENSEMBLE 1.1.0",
        "config": {
            "datasets": ["mnist", "fashion_mnist"],
            "split_seed": 20260904,
            "search_samples": 50000,
            "validation_samples": 10000,
            "pool_seeds": [201, 202, 203, 204, 205],
            "reference_single_seed": 201,
            "equal_budget_single_epochs": 50,
            "pso": {
                "method": "constriction",
                "evaluation": "full",
                "renewal": "loss",
                "particles": 30,
                "epochs": 30,
                "swarm_seeds": [301, 302, 303],
                "particle_bounds": [-4.0, 4.0],
                "boundary_strategy": "reflect",
                "velocity_limit_ratio": 0.1,
                "initial_position_noise": 0.0,
                "queries_per_seed": 900,
                "sample_evaluations_per_seed": 9000000,
            },
        },
        "development_pass": True,
        "policy_frozen": True,
        "official_test_data_loaded": True,
        "official_test_data_loaded_before_freeze": False,
        "official_test_evaluations_before_freeze": 0,
        "post_test_tuning_or_reruns": 0,
        "resource_totals": {
            "total_adam_pool_model_epochs": 100,
            "total_pso_queries": 5400,
            "total_pso_sample_evaluations": 54000000,
            "total_pso_wall_time_seconds": 12.0,
            "pso_to_pool_wall_ratio": 0.02,
        },
        "workloads": {
            "mnist": make_valid_workload_entry(),
            "fashion_mnist": make_valid_workload_entry(),
        },
    }


def test_evaluator_version_and_imports():
    """Verify evaluator version identifier."""
    assert isinstance(EVALUATOR_VERSION, str)
    assert EVALUATOR_VERSION.startswith("POST-TRAINING-PSO-ENSEMBLE-EVALUATOR")


def test_evaluate_artifact_valid_passing_study():
    """Verify evaluator approves valid study artifact with zero hard gate failures."""
    artifact = make_valid_study_artifact()
    result = evaluate_artifact(artifact)

    assert result["pass"] is True
    assert result["development_pass"] is True
    assert result["confirmation_pass"] is True
    assert result["failed_hard_gate_count"] == 0
    assert isinstance(result["score"], float)
    assert result["score"] > -100.0


def test_evaluate_artifact_schema_tampering():
    """Verify evaluator rejects non-dict, missing config, and missing workload structures."""
    # 1. Non-dict artifact
    res_non_dict = evaluate_artifact("invalid_string_artifact")
    assert res_non_dict["pass"] is False
    assert res_non_dict["failed_hard_gate_count"] >= 1
    assert "schema" in res_non_dict["issues"]
    assert len(res_non_dict["issues"]["schema"]) > 0

    # 2. Missing config
    art_no_cfg = make_valid_study_artifact()
    del art_no_cfg["config"]
    res_no_cfg = evaluate_artifact(art_no_cfg)
    assert res_no_cfg["pass"] is False
    assert len(res_no_cfg["issues"]["schema"]) > 0

    # 3. Missing dataset in workloads
    art_missing_ds = make_valid_study_artifact()
    del art_missing_ds["workloads"]["fashion_mnist"]
    res_missing_ds = evaluate_artifact(art_missing_ds)
    assert res_missing_ds["pass"] is False
    assert len(res_missing_ds["issues"]["schema"]) > 0


def test_evaluate_artifact_config_tampering():
    """Verify evaluator flags mismatched split seed, sample counts, or PSO parameters."""
    art = make_valid_study_artifact()
    art["config"]["split_seed"] = 99999999  # Mismatched seed
    art["config"]["pso"]["particles"] = 15    # Expected 30
    art["config"]["pso"]["epochs"] = 15       # Expected 30

    res = evaluate_artifact(art)
    assert res["pass"] is False
    assert len(res["issues"]["config"]) >= 2


def test_evaluate_artifact_non_finite_tampering():
    """Verify evaluator detects non-finite values (NaN / Inf) in nested metrics or weights."""
    art = make_valid_study_artifact()
    # Inject NaN into validation NLL
    art["workloads"]["mnist"]["validation"]["methods"]["pso_weights"]["per_seed_runs"][0]["metrics"]["nll"] = float("nan")

    res = evaluate_artifact(art)
    assert res["pass"] is False
    assert res["failed_hard_gate_count"] >= 1
    assert len(res["issues"]["finite"]) >= 1


def test_evaluate_artifact_simplex_weights_tampering():
    """Verify evaluator rejects weight vectors that do not sum to 1.0 within tolerance."""
    art = make_valid_study_artifact()
    # Set weights that sum to 1.5
    art["workloads"]["mnist"]["validation"]["methods"]["slsqp_weights"]["weights"] = [0.3, 0.3, 0.3, 0.3, 0.3]

    res = evaluate_artifact(art)
    assert res["pass"] is False
    assert res["failed_hard_gate_count"] >= 1
    assert len(res["issues"]["weights"]) >= 1


def test_evaluate_artifact_accounting_tampering():
    """Verify evaluator flags invalid PSO queries or sample evaluations accounting."""
    art = make_valid_study_artifact()
    art["config"]["pso"]["queries_per_seed"] = 899  # Expected 900

    res = evaluate_artifact(art)
    assert res["pass"] is False
    assert len(res["issues"]["accounting"]) >= 1

def test_evaluate_artifact_requires_each_frozen_swarm_seed_once():
    """Duplicate seed records cannot stand in for independent replication."""
    art = make_valid_study_artifact()
    runs = art["workloads"]["mnist"]["validation"]["methods"]["pso_weights"][
        "per_seed_runs"
    ]
    runs[1]["seed"] = 301

    result = evaluate_artifact(art)

    assert result["pass"] is False
    assert any(
        "each frozen seed exactly once" in issue
        for issue in result["issues"]["config"]
    )


def test_evaluate_artifact_base_model_forward_count_tampering():
    """Verify evaluator flags non-zero base model forward passes during optimization."""
    art = make_valid_study_artifact()
    art["workloads"]["mnist"]["validation_cache"]["base_cnn_forward_passes_during_optimization"] = 2

    res = evaluate_artifact(art)
    assert res["pass"] is False
    assert res["failed_hard_gate_count"] >= 1
    assert len(res["issues"]["accounting"]) >= 1


def test_evaluate_artifact_leakage_and_post_test_tuning_tampering():
    """Global/local leakage contradictions and post-test tuning must fail."""
    art_loaded = make_valid_study_artifact()
    art_loaded["official_test_data_loaded_before_freeze"] = True
    res_loaded = evaluate_artifact(art_loaded)
    assert res_loaded["pass"] is False
    assert len(res_loaded["issues"]["leakage"]) >= 1

    art_evals = make_valid_study_artifact()
    art_evals["official_test_evaluations_before_freeze"] = 1
    res_evals = evaluate_artifact(art_evals)
    assert res_evals["pass"] is False
    assert len(res_evals["issues"]["leakage"]) >= 1

    art_tune = make_valid_study_artifact()
    art_tune["post_test_tuning_or_reruns"] = 1
    res_tune = evaluate_artifact(art_tune)
    assert res_tune["pass"] is False
    assert len(res_tune["issues"]["tuning"]) >= 1

def test_evaluate_artifact_rejects_confirmation_policy_drift():
    """Confirmation must identify the exact validation-frozen method parameters."""
    mutations = [
        ("policy_frozen", False),
        (
            "selected_pso_seed",
            302,
        ),
        (
            "selected_pso_weights",
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "slsqp_weights",
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        ("fitted_temperature", 2.0),
    ]

    for field, value in mutations:
        art = make_valid_study_artifact()
        if field == "policy_frozen":
            art[field] = value
        else:
            art["workloads"]["mnist"]["confirmation"]["frozen_methods"][
                field
            ] = value

        result = evaluate_artifact(art)

        assert result["pass"] is False, field
        assert result["confirmation_gates"]["frozen_policy_consistency"] is False


def test_evaluate_artifact_slsqp_gap_and_uniform_regression_tampering():
    """Verify evaluator flags PSO NLL gap vs SLSQP > 0.5% or accuracy regression > 0.1 pp vs uniform."""
    # 1. SLSQP gap > 0.005
    art_slsqp = make_valid_study_artifact()
    # SLSQP NLL = 0.30, PSO NLL = 0.35 -> relative gap (0.35 - 0.30)/0.30 = 0.1667 > 0.005
    art_slsqp["workloads"]["mnist"]["validation"]["methods"]["slsqp_weights"]["metrics"]["nll"] = 0.30
    art_slsqp["workloads"]["mnist"]["validation"]["methods"]["pso_weights"]["metrics"]["nll"] = 0.35
    art_slsqp["workloads"]["mnist"]["validation"]["methods"]["pso_weights"]["per_seed_runs"][0]["metrics"]["nll"] = 0.35

    res_slsqp = evaluate_artifact(art_slsqp)
    assert res_slsqp["pass"] is False
    assert len(res_slsqp["issues"]["gates"]) >= 1

    # 2. PSO accuracy regression > 0.1 pp below uniform
    art_acc = make_valid_study_artifact()
    art_acc["workloads"]["mnist"]["validation"]["methods"]["uniform_ensemble"]["accuracy"] = 90.0
    # Set PSO accuracy to 89.5 (0.5 pp regression)
    art_acc["workloads"]["mnist"]["validation"]["methods"]["pso_weights"]["metrics"]["accuracy"] = 89.5
    art_acc["workloads"]["mnist"]["validation"]["methods"]["pso_weights"]["per_seed_runs"][0]["metrics"]["accuracy"] = 89.5

    res_acc = evaluate_artifact(art_acc)
    assert res_acc["pass"] is False
    assert len(res_acc["issues"]["gates"]) >= 1


def test_evaluate_artifact_reference_single_and_equal_budget_tampering():
    """Verify evaluator flags PSO validation NLL >= reference single or > equal-budget single NLL + 1e-7."""
    # PSO NLL > reference single NLL
    art_ref = make_valid_study_artifact()
    art_ref["workloads"]["mnist"]["validation"]["methods"]["reference_single_10e"]["nll"] = 0.30
    art_ref["workloads"]["mnist"]["validation"]["methods"]["pso_weights"]["metrics"]["nll"] = 0.35
    art_ref["workloads"]["mnist"]["validation"]["methods"]["pso_weights"]["per_seed_runs"][0]["metrics"]["nll"] = 0.35

    res_ref = evaluate_artifact(art_ref)
    assert res_ref["pass"] is False
    assert len(res_ref["issues"]["gates"]) >= 1


def test_evaluate_artifact_wall_time_ratio_tampering():
    """Each workload's recomputed median PSO/Adam ratio must stay at most 10%."""
    art_time = make_valid_study_artifact()
    art_time["workloads"]["mnist"]["training"]["adam_pool_wall_time_seconds"] = 10.0
    runs = art_time["workloads"]["mnist"]["validation"]["methods"][
        "pso_weights"
    ]["per_seed_runs"]
    for run in runs:
        run["wall_time_seconds"] = 2.0

    result = evaluate_artifact(art_time)

    assert result["pass"] is False
    assert result["development_gates"][
        "maximum_median_one_seed_pso_to_pool_training_wall_ratio"
    ] is False
    assert len(result["issues"]["gates"]) >= 1


def test_evaluate_artifact_missing_confirmation_on_dev_pass():
    """Verify missing confirmation on development pass fails overall study evaluation."""
    art_no_conf = make_valid_study_artifact()
    art_no_conf["official_test_data_loaded"] = False
    art_no_conf["workloads"]["mnist"]["confirmation"] = None
    art_no_conf["workloads"]["fashion_mnist"]["confirmation"] = None

    res = evaluate_artifact(art_no_conf)
    assert res["pass"] is False
    assert res["confirmation_pass"] is False
    assert len(res["issues"]["gates"]) >= 1 or len(res["issues"]["leakage"]) >= 1


def test_evaluator_cli(tmp_path, monkeypatch):
    """Verify CLI main entrypoint writes evaluation payload atomically."""
    art = make_valid_study_artifact()
    art_path = tmp_path / "study_artifact.json"
    save_json_atomic(art, art_path)

    out_path = tmp_path / "evaluation_output.json"

    # Simulate command-line arguments: --artifact <art_path> --output <out_path>
    test_args = [
        "evaluate_post_training_ensemble.py",
        "--artifact",
        str(art_path),
        "--output",
        str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert out_path.exists()

    eval_data = json.loads(out_path.read_text())
    assert eval_data["pass"] is True
    assert eval_data["failed_hard_gate_count"] == 0
    assert isinstance(eval_data["score"], float)
