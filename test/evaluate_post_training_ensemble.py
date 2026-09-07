"""Strict Evaluator for Post-Training PSO Ensemble Study.

Evaluates experiment artifacts against the frozen mission and evaluator contract
defined in .omc/autoresearch/post-training-pso-ensemble/evaluator.json and mission.md.

Recomputes 14 development hard gates and 9 confirmation hard gates, verifies
leakage control, frozen-policy consistency, exact query/sample/cache accounting,
simplex probability weight constraints, SLSQP solver status, finiteness, and metric consistency.

Produces a structured evaluation payload containing score, pass/fail status, gate
results, and issue categories. Never trusts self-reported artifact pass flags.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

EVALUATOR_VERSION = "POST-TRAINING-PSO-ENSEMBLE-EVALUATOR 1.2.0"
EXPECTED_PROTOCOL_VERSION = "POST-TRAINING-PSO-ENSEMBLE 1.1.0"
EXPECTED_DATASETS = ["mnist", "fashion_mnist"]
EXPECTED_SPLIT_SEED = 20260904
EXPECTED_SEARCH_SAMPLES = 50000
EXPECTED_VAL_SAMPLES = 10000
EXPECTED_POOL_SEEDS = [201, 202, 203, 204, 205]
EXPECTED_REF_SINGLE_SEED = 201
EXPECTED_50E_SINGLE_EPOCHS = 50
EXPECTED_SWARM_SEEDS = [301, 302, 303]
EXPECTED_PARTICLES = 30
EXPECTED_EPOCHS = 30
EXPECTED_QUERIES_PER_SEED = 900
EXPECTED_SAMPLES_PER_SEED = 9000000

REQUIRED_BASELINES = [
    "reference_single_10e",
    "best_single_10e",
    "single_50e",
    "uniform_ensemble",
    "uniform_temperature",
    "slsqp_weights",
    "pso_weights",
]
REQUIRED_BASELINES_SET = set(REQUIRED_BASELINES)

# Weighted methods that store explicit simplex weight vectors
WEIGHTED_METHODS = [
    "slsqp_weights",
    "pso_weights",
]


def _is_finite_number(value: Any) -> bool:
    """Returns True if value is a numeric int/float (not bool) and finite."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _append_issue(issues: Dict[str, List[str]], category: str, message: str) -> None:
    """Appends an issue string to the given category list."""
    if category not in issues:
        issues[category] = []
    issues[category].append(message)


def save_json_atomic(data: Dict[str, Any], json_path: Union[str, Path]) -> None:
    """Atomically writes JSON payload to destination path using a temporary file."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp_{os.getpid()}_{time.time_ns()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def _to_pp(acc: float) -> float:
    """Converts accuracy to percentage points (0-100 scale)."""
    return acc * 100.0 if acc <= 1.0 else acc


def _unwrap_metrics(entry: Any) -> Tuple[Optional[float], Optional[float]]:
    """Unwraps accuracy and NLL/loss from base method dict or nested weighted method metrics dict."""
    if not isinstance(entry, dict):
        return None, None
    metrics_dict = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else entry

    acc = None
    for key in ("accuracy", "acc", "val_acc", "test_acc", "val_selected_acc"):
        if key in metrics_dict and _is_finite_number(metrics_dict[key]):
            acc = float(metrics_dict[key])
            break

    nll = None
    for key in ("nll", "loss", "val_nll", "test_nll", "val_loss", "val_selected_loss"):
        if key in metrics_dict and _is_finite_number(metrics_dict[key]):
            nll = float(metrics_dict[key])
            break

    return acc, nll


def _validate_simplex_weights(weights: Any, tolerance: float = 1e-6) -> bool:
    """Validates that weights form a 5-element probability simplex summing to 1 within tolerance."""
    if not isinstance(weights, (list, tuple)) or len(weights) != 5:
        return False
    for w in weights:
        if not _is_finite_number(w) or float(w) < -tolerance:
            return False
    total = math.fsum([float(w) for w in weights])
    return abs(total - 1.0) <= tolerance

def _sequences_close(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    """Return whether two finite numeric sequences agree elementwise."""
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    if len(left) != len(right):
        return False
    return all(
        _is_finite_number(a)
        and _is_finite_number(b)
        and math.isclose(float(a), float(b), abs_tol=tolerance, rel_tol=tolerance)
        for a, b in zip(left, right)
    )


def _scan_for_non_finite(data: Any, path: str = "") -> List[str]:
    """Recursively scans a data structure for any NaN/Inf values."""
    non_finites: List[str] = []
    if isinstance(data, float):
        if not math.isfinite(data):
            non_finites.append(f"{path}: {data}")
    elif isinstance(data, dict):
        for k, v in data.items():
            non_finites.extend(_scan_for_non_finite(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(data, (list, tuple)):
        for idx, item in enumerate(data):
            non_finites.extend(_scan_for_non_finite(item, f"{path}[{idx}]"))
    return non_finites


def evaluate_artifact(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Strictly evaluates a post-training PSO ensemble experiment artifact.

    Args:
        artifact: Parsed JSON experiment artifact dictionary.

    Returns:
        Structured evaluation payload with score, pass/fail status, gate counts,
        and categorized issues. Never relies on self-reported artifact pass flags.
    """
    issues: Dict[str, List[str]] = {
        "schema": [],
        "config": [],
        "finite": [],
        "weights": [],
        "accounting": [],
        "leakage": [],
        "tuning": [],
        "slsqp": [],
        "consistency": [],
        "gates": [],
    }

    if not isinstance(artifact, dict):
        _append_issue(issues, "schema", "Artifact must be a JSON object")
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "pass": False,
            "score": -1000.0,
            "development_pass": False,
            "confirmation_pass": False,
            "failed_hard_gate_count": 1,
            "issues": issues,
            "development_gates": {},
            "confirmation_gates": None,
            "metrics": {},
        }

    # 1. Non-finite value scan
    non_finite_locations = _scan_for_non_finite(artifact)
    if non_finite_locations:
        for loc in non_finite_locations[:10]:
            _append_issue(issues, "finite", f"Non-finite value found at {loc}")

    # Protocol version check
    protocol_version = artifact.get("protocol_version")
    if protocol_version != EXPECTED_PROTOCOL_VERSION:
        _append_issue(
            issues,
            "config",
            f"Artifact protocol_version must be '{EXPECTED_PROTOCOL_VERSION}', got '{protocol_version}'",
        )

    # 2. Config & Protocol verification
    config = artifact.get("config")
    if not isinstance(config, dict):
        _append_issue(issues, "schema", "Missing or non-object top-level 'config'")
        config = {}

    datasets = config.get("datasets")
    if not isinstance(datasets, list) or sorted(datasets) != sorted(EXPECTED_DATASETS):
        _append_issue(issues, "config", f"Config 'datasets' must be {EXPECTED_DATASETS}")

    if config.get("split_seed") != EXPECTED_SPLIT_SEED:
        _append_issue(issues, "config", f"Config 'split_seed' must be {EXPECTED_SPLIT_SEED}")

    if config.get("search_samples") != EXPECTED_SEARCH_SAMPLES:
        _append_issue(
            issues, "config", f"Config 'search_samples' must be {EXPECTED_SEARCH_SAMPLES}"
        )

    if config.get("validation_samples") != EXPECTED_VAL_SAMPLES:
        _append_issue(
            issues, "config", f"Config 'validation_samples' must be {EXPECTED_VAL_SAMPLES}"
        )

    if config.get("pool_seeds") != EXPECTED_POOL_SEEDS:
        _append_issue(issues, "config", f"Config 'pool_seeds' must be {EXPECTED_POOL_SEEDS}")

    if config.get("reference_single_seed") != EXPECTED_REF_SINGLE_SEED:
        _append_issue(
            issues, "config", f"Config 'reference_single_seed' must be {EXPECTED_REF_SINGLE_SEED}"
        )

    if config.get("equal_budget_single_epochs") != EXPECTED_50E_SINGLE_EPOCHS:
        _append_issue(
            issues,
            "config",
            f"Config 'equal_budget_single_epochs' must be {EXPECTED_50E_SINGLE_EPOCHS}",
        )

    pso_cfg = config.get("pso", {}) if isinstance(config.get("pso"), dict) else {}
    if pso_cfg.get("particles") != EXPECTED_PARTICLES:
        _append_issue(
            issues, "config", f"Config 'pso.particles' must be {EXPECTED_PARTICLES}"
        )
    if pso_cfg.get("epochs") != EXPECTED_EPOCHS:
        _append_issue(issues, "config", f"Config 'pso.epochs' must be {EXPECTED_EPOCHS}")
    if pso_cfg.get("swarm_seeds") != EXPECTED_SWARM_SEEDS:
        _append_issue(
            issues, "config", f"Config 'pso.swarm_seeds' must be {EXPECTED_SWARM_SEEDS}"
        )
    if pso_cfg.get("queries_per_seed") != EXPECTED_QUERIES_PER_SEED:
        _append_issue(
            issues,
            "accounting",
            f"Config 'pso.queries_per_seed' must be {EXPECTED_QUERIES_PER_SEED}",
        )
    if pso_cfg.get("sample_evaluations_per_seed") != EXPECTED_SAMPLES_PER_SEED:
        _append_issue(
            issues,
            "accounting",
            f"Config 'pso.sample_evaluations_per_seed' must be {EXPECTED_SAMPLES_PER_SEED}",
        )

    # Validate frozen PSO hyperparameters in config
    if pso_cfg.get("method") != "constriction":
        _append_issue(issues, "config", f"Config 'pso.method' must be 'constriction', got '{pso_cfg.get('method')}'")
    if pso_cfg.get("evaluation") != "full":
        _append_issue(issues, "config", f"Config 'pso.evaluation' must be 'full', got '{pso_cfg.get('evaluation')}'")
    if pso_cfg.get("renewal") != "loss":
        _append_issue(issues, "config", f"Config 'pso.renewal' must be 'loss', got '{pso_cfg.get('renewal')}'")
    if pso_cfg.get("particle_bounds") != [-4.0, 4.0]:
        _append_issue(issues, "config", f"Config 'pso.particle_bounds' must be [-4.0, 4.0], got '{pso_cfg.get('particle_bounds')}'")
    if pso_cfg.get("boundary_strategy") != "reflect":
        _append_issue(issues, "config", f"Config 'pso.boundary_strategy' must be 'reflect', got '{pso_cfg.get('boundary_strategy')}'")
    if pso_cfg.get("velocity_limit_ratio") != 0.1:
        _append_issue(issues, "config", f"Config 'pso.velocity_limit_ratio' must be 0.1, got '{pso_cfg.get('velocity_limit_ratio')}'")
    if pso_cfg.get("initial_position_noise") != 0.0:
        _append_issue(issues, "config", f"Config 'pso.initial_position_noise' must be 0.0, got '{pso_cfg.get('initial_position_noise')}'")

    # 3. Leakage and Post-Test Tuning checks
    post_test_tuning = artifact.get("post_test_tuning_or_reruns", 0)
    if post_test_tuning != 0:
        _append_issue(
            issues,
            "tuning",
            f"post_test_tuning_or_reruns must be 0, got {post_test_tuning}",
        )

    if "official_test_data_loaded_before_freeze" in artifact:
        global_pre_loaded = artifact["official_test_data_loaded_before_freeze"]
        if global_pre_loaded is not False:
            _append_issue(
                issues,
                "leakage",
                "Top-level official_test_data_loaded_before_freeze must be False "
                f"when present, got {global_pre_loaded}",
            )
    if "official_test_evaluations_before_freeze" in artifact:
        global_pre_evals = artifact["official_test_evaluations_before_freeze"]
        if global_pre_evals != 0:
            _append_issue(
                issues,
                "leakage",
                "Top-level official_test_evaluations_before_freeze must be 0 "
                f"when present, got {global_pre_evals}",
            )

    # 4. Workloads & Validation Analysis
    workloads = artifact.get("workloads")
    if not isinstance(workloads, dict):
        _append_issue(issues, "schema", "Missing or non-object top-level 'workloads'")
        workloads = {}

    pre_freeze_loaded_ok = (
        artifact.get("official_test_data_loaded_before_freeze", False) is False
    )
    pre_freeze_evals_ok = (
        artifact.get("official_test_evaluations_before_freeze", 0) == 0
    )

    val_metrics_by_dataset: Dict[str, Dict[str, Dict[str, float]]] = {}
    test_metrics_by_dataset: Dict[str, Dict[str, Dict[str, float]]] = {}
    val_cache_counts: Dict[str, Dict[str, int]] = {}

    pso_wall_times: Dict[str, List[float]] = {}
    adam_pool_wall_times: Dict[str, float] = {}

    for ds in EXPECTED_DATASETS:
        if ds not in workloads:
            _append_issue(issues, "schema", f"Workloads missing dataset '{ds}'")
            pre_freeze_loaded_ok = False
            pre_freeze_evals_ok = False
            continue
        wl = workloads[ds]
        if not isinstance(wl, dict):
            _append_issue(issues, "schema", f"Workload '{ds}' must be a JSON object")
            pre_freeze_loaded_ok = False
            pre_freeze_evals_ok = False
            continue

        # Per-workload declarations are mandatory and cannot mask contradictory
        # top-level leakage counters.
        wl_pre_loaded = wl.get("official_test_data_loaded_before_freeze")
        if wl_pre_loaded is not False:
            pre_freeze_loaded_ok = False
            _append_issue(
                issues,
                "leakage",
                f"Dataset '{ds}' official_test_data_loaded_before_freeze must be False, got {wl_pre_loaded}",
            )

        wl_pre_evals = wl.get("official_test_evaluations_before_freeze")
        if wl_pre_evals != 0:
            pre_freeze_evals_ok = False
            _append_issue(
                issues,
                "leakage",
                f"Dataset '{ds}' official_test_evaluations_before_freeze must be 0, got {wl_pre_evals}",
            )

        # Validation cache key checks:
        # pool_forward_passes (5), long_single_forward_passes (1), base_cnn_forward_passes_during_optimization (0)
        val_cache = wl.get("validation_cache") if isinstance(wl.get("validation_cache"), dict) else wl
        val_pool_passes = val_cache.get("pool_forward_passes", val_cache.get("validation_pool_forward_passes"))
        long_single_passes = val_cache.get("long_single_forward_passes", val_cache.get("val_long_single_passes", 1))
        opt_base_passes = val_cache.get("base_cnn_forward_passes_during_optimization", val_cache.get("optimization_base_model_forward_passes"))

        val_cache_counts[ds] = {
            "pool_forward_passes": int(val_pool_passes) if _is_finite_number(val_pool_passes) else -1,
            "long_single_forward_passes": int(long_single_passes) if _is_finite_number(long_single_passes) else -1,
            "base_cnn_forward_passes_during_optimization": int(opt_base_passes) if _is_finite_number(opt_base_passes) else -1,
        }

        if val_cache_counts[ds]["pool_forward_passes"] != 5:
            _append_issue(
                issues,
                "accounting",
                f"Dataset '{ds}' validation pool_forward_passes must be 5, got {val_pool_passes}",
            )
        if val_cache_counts[ds]["base_cnn_forward_passes_during_optimization"] != 0:
            _append_issue(
                issues,
                "accounting",
                f"Dataset '{ds}' base_cnn_forward_passes_during_optimization must be 0, got {opt_base_passes}",
            )

        # Extract Adam pool training wall time (key: adam_pool_wall_time_seconds)
        training_info = wl.get("training") if isinstance(wl.get("training"), dict) else wl
        adam_wall = training_info.get("adam_pool_wall_time_seconds", training_info.get("adam_pool_wall_time"))
        if _is_finite_number(adam_wall):
            adam_pool_wall_times[ds] = float(adam_wall)

        # Validate validation methods dict & exact method set
        val_sec = wl.get("validation") if isinstance(wl.get("validation"), dict) else wl
        methods_dict = val_sec.get("methods") if isinstance(val_sec.get("methods"), dict) else val_sec

        if not isinstance(methods_dict, dict):
            _append_issue(issues, "schema", f"Dataset '{ds}' validation methods must be a dictionary")
            methods_dict = {}

        present_methods = set(methods_dict.keys())
        if present_methods != REQUIRED_BASELINES_SET:
            _append_issue(
                issues,
                "schema",
                f"Dataset '{ds}' validation methods set {present_methods} does not match required {REQUIRED_BASELINES_SET}",
            )

        ds_val_metrics: Dict[str, Dict[str, float]] = {}
        for method in REQUIRED_BASELINES:
            if method not in methods_dict:
                _append_issue(issues, "schema", f"Dataset '{ds}' validation missing method '{method}'")
                continue

            entry = methods_dict[method]
            acc, nll = _unwrap_metrics(entry)

            if acc is None or nll is None:
                _append_issue(
                    issues,
                    "finite",
                    f"Dataset '{ds}' validation method '{method}' has missing or non-finite acc/nll",
                )
            else:
                ds_val_metrics[method] = {"acc": acc, "nll": nll}

            # Simplex weight checks for weighted methods (slsqp_weights, pso_weights)
            if method in WEIGHTED_METHODS:
                weights = entry.get("weights", entry.get("selected_weights")) if isinstance(entry, dict) else None
                if not _validate_simplex_weights(weights):
                    _append_issue(
                        issues,
                        "weights",
                        f"Dataset '{ds}' validation method '{method}' has invalid simplex weights: {weights}",
                    )

            # SLSQP solver success check
            if method == "slsqp_weights":
                solver_success = entry.get("success", entry.get("status") in (0, "success", True)) if isinstance(entry, dict) else False
                if solver_success is False:
                    _append_issue(
                        issues,
                        "slsqp",
                        f"Dataset '{ds}' SLSQP solver failed (success=False)",
                    )

            # PSO detailed run & seed verification
            if method == "pso_weights":
                seed_runs = entry.get("per_seed_runs") if isinstance(entry, dict) else None
                if not isinstance(seed_runs, list) or len(seed_runs) != len(EXPECTED_SWARM_SEEDS):
                    _append_issue(
                        issues,
                        "schema",
                        f"Dataset '{ds}' PSO pso_weights per_seed_runs must contain {len(EXPECTED_SWARM_SEEDS)} seed runs",
                    )
                else:
                    recorded_seeds: List[int] = []
                    run_times: List[float] = []
                    best_run_nll = float("inf")
                    best_run_entry: Optional[Dict[str, Any]] = None

                    for idx, run in enumerate(seed_runs):
                        if not isinstance(run, dict):
                            _append_issue(
                                issues, "schema", f"Dataset '{ds}' PSO seed run {idx} is non-dict"
                            )
                            continue

                        seed = run.get("seed")
                        if seed not in EXPECTED_SWARM_SEEDS:
                            _append_issue(
                                issues, "config", f"Dataset '{ds}' PSO seed run seed {seed} unexpected"
                            )
                        if isinstance(seed, int) and not isinstance(seed, bool):
                            recorded_seeds.append(seed)

                        queries = run.get("queries", run.get("total_queries"))
                        # PSO per-seed sample key: sample_evaluations, samples, or total_sample_evaluations
                        samples = run.get("sample_evaluations", run.get("samples", run.get("total_sample_evaluations")))
                        if queries != EXPECTED_QUERIES_PER_SEED:
                            _append_issue(
                                issues,
                                "accounting",
                                f"Dataset '{ds}' PSO seed {seed} queries must be {EXPECTED_QUERIES_PER_SEED}, got {queries}",
                            )
                        if samples != EXPECTED_SAMPLES_PER_SEED:
                            _append_issue(
                                issues,
                                "accounting",
                                f"Dataset '{ds}' PSO seed {seed} samples must be {EXPECTED_SAMPLES_PER_SEED}, got {samples}",
                            )

                        r_weights = run.get("weights")
                        if not _validate_simplex_weights(r_weights):
                            _append_issue(
                                issues,
                                "weights",
                                f"Dataset '{ds}' PSO seed {seed} weights invalid: {r_weights}",
                            )

                        w_time = run.get("wall_time_seconds", run.get("wall_time", run.get("time")))
                        if _is_finite_number(w_time):
                            run_times.append(float(w_time))
                        else:
                            _append_issue(
                                issues,
                                "finite",
                                f"Dataset '{ds}' PSO seed {seed} missing or non-finite wall_time_seconds",
                            )

                        r_acc, r_nll = _unwrap_metrics(run)
                        if r_nll is not None and r_nll < best_run_nll:
                            best_run_nll = r_nll
                            best_run_entry = run

                    if sorted(recorded_seeds) != EXPECTED_SWARM_SEEDS:
                        _append_issue(
                            issues,
                            "config",
                            f"Dataset '{ds}' PSO seed runs must contain each frozen seed exactly once; "
                            f"got {recorded_seeds}",
                        )

                    if run_times:
                        pso_wall_times[ds] = run_times

                    # Validate selected PSO top metrics/weights/seed equal best per-seed NLL record
                    if isinstance(entry, dict):
                        top_selected_seed = entry.get("selected_seed")
                        top_weights = entry.get("selected_weights", entry.get("weights"))
                        top_acc, top_nll = _unwrap_metrics(entry)

                        if best_run_entry is not None:
                            best_seed = best_run_entry.get("seed")
                            best_weights = best_run_entry.get("weights")
                            best_acc, _ = _unwrap_metrics(best_run_entry)

                            if top_selected_seed != best_seed:
                                _append_issue(
                                    issues,
                                    "consistency",
                                    f"Dataset '{ds}' pso_weights selected_seed ({top_selected_seed}) != best seed ({best_seed})",
                                )
                            if top_weights != best_weights and not (
                                isinstance(top_weights, list)
                                and isinstance(best_weights, list)
                                and len(top_weights) == len(best_weights)
                                and all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(top_weights, best_weights))
                            ):
                                _append_issue(
                                    issues,
                                    "consistency",
                                    f"Dataset '{ds}' pso_weights weights disagree with best seed run weights",
                                )
                            if top_nll is not None and not math.isclose(top_nll, best_run_nll, abs_tol=1e-6, rel_tol=1e-5):
                                _append_issue(
                                    issues,
                                    "consistency",
                                    f"Dataset '{ds}' pso_weights top NLL ({top_nll}) != best seed run NLL ({best_run_nll})",
                                )
                            if top_acc is not None and best_acc is not None and not math.isclose(top_acc, best_acc, abs_tol=1e-6, rel_tol=1e-5):
                                _append_issue(
                                    issues,
                                    "consistency",
                                    f"Dataset '{ds}' pso_weights top acc ({top_acc}) != best seed run acc ({best_acc})",
                                )

        val_metrics_by_dataset[ds] = ds_val_metrics

    # 5. Development Hard Gates Recomputation
    # Named booleans assess their own fields directly
    dev_gates: Dict[str, bool] = {
        "all_values_finite": len(issues["finite"]) == 0,
        "simplex_tolerance": len(issues["weights"]) == 0,
        "validation_pool_forward_passes_each_dataset": all(
            val_cache_counts.get(ds, {}).get("pool_forward_passes") == 5
            for ds in EXPECTED_DATASETS
        ),
        "optimization_base_model_forward_passes": all(
            val_cache_counts.get(ds, {}).get("base_cnn_forward_passes_during_optimization") == 0
            for ds in EXPECTED_DATASETS
        ),
        "official_test_data_loaded_before_freeze": pre_freeze_loaded_ok,
        "official_test_evaluations_before_freeze": pre_freeze_evals_ok,
        "query_and_sample_accounting_exact": len(issues["accounting"]) == 0,
        "maximum_pso_nll_regression_vs_uniform": True,
        "maximum_pso_accuracy_regression_vs_uniform_pp": True,
        "pso_nll_below_reference_single": True,
        "maximum_pso_nll_regression_vs_equal_budget_single": True,
        "maximum_relative_pso_nll_gap_vs_slsqp": True,
        "cross_dataset_mean_relative_pso_nll_reduction_vs_uniform_minimum": True,
        "maximum_median_one_seed_pso_to_pool_training_wall_ratio": True,
    }

    # Evaluate metric-dependent development gates across datasets
    rel_nll_reductions_vs_uniform: List[float] = []
    pso_wall_ratios: Dict[str, float] = {}

    for ds in EXPECTED_DATASETS:
        m = val_metrics_by_dataset.get(ds, {})
        pso_nll = m.get("pso_weights", {}).get("nll")
        pso_acc = m.get("pso_weights", {}).get("acc")
        unif_nll = m.get("uniform_ensemble", {}).get("nll")
        unif_acc = m.get("uniform_ensemble", {}).get("acc")
        ref_nll = m.get("reference_single_10e", {}).get("nll")
        s50_nll = m.get("single_50e", {}).get("nll")
        slsqp_nll = m.get("slsqp_weights", {}).get("nll")

        # Nominal gate booleans cannot stay True if required inputs are missing!
        if any(v is None for v in (pso_nll, unif_nll, ref_nll, s50_nll, slsqp_nll, pso_acc, unif_acc)):
            dev_gates["maximum_pso_nll_regression_vs_uniform"] = False
            dev_gates["maximum_pso_accuracy_regression_vs_uniform_pp"] = False
            dev_gates["pso_nll_below_reference_single"] = False
            dev_gates["maximum_pso_nll_regression_vs_equal_budget_single"] = False
            dev_gates["maximum_relative_pso_nll_gap_vs_slsqp"] = False
            dev_gates["cross_dataset_mean_relative_pso_nll_reduction_vs_uniform_minimum"] = False

        # Gate 8: maximum PSO NLL regression vs uniform <= 1e-7
        if pso_nll is not None and unif_nll is not None:
            if (pso_nll - unif_nll) > 1e-7:
                dev_gates["maximum_pso_nll_regression_vs_uniform"] = False
                _append_issue(
                    issues,
                    "gates",
                    f"Dataset '{ds}' val PSO NLL ({pso_nll:.6f}) > uniform NLL ({unif_nll:.6f}) by > 1e-7",
                )
            rel_nll_reductions_vs_uniform.append((unif_nll - pso_nll) / unif_nll)

        # Gate 9: maximum PSO accuracy regression vs uniform <= 0.10 pp
        if pso_acc is not None and unif_acc is not None:
            acc_diff_pp = _to_pp(unif_acc) - _to_pp(pso_acc)
            if acc_diff_pp > 0.10:
                dev_gates["maximum_pso_accuracy_regression_vs_uniform_pp"] = False
                _append_issue(
                    issues,
                    "gates",
                    f"Dataset '{ds}' val PSO acc regression vs uniform ({acc_diff_pp:.4f} pp) > 0.10 pp",
                )

        # Gate 10: PSO NLL strictly below reference single
        if pso_nll is not None and ref_nll is not None:
            if pso_nll >= ref_nll:
                dev_gates["pso_nll_below_reference_single"] = False
                _append_issue(
                    issues,
                    "gates",
                    f"Dataset '{ds}' val PSO NLL ({pso_nll:.6f}) >= reference single NLL ({ref_nll:.6f})",
                )

        # Gate 11: maximum PSO NLL regression vs equal-budget 50e single <= 1e-7
        if pso_nll is not None and s50_nll is not None:
            if (pso_nll - s50_nll) > 1e-7:
                dev_gates["maximum_pso_nll_regression_vs_equal_budget_single"] = False
                _append_issue(
                    issues,
                    "gates",
                    f"Dataset '{ds}' val PSO NLL ({pso_nll:.6f}) > 50e single NLL ({s50_nll:.6f}) by > 1e-7",
                )

        # Gate 12: maximum relative PSO NLL gap vs SLSQP <= 0.005
        if pso_nll is not None and slsqp_nll is not None and slsqp_nll > 0:
            rel_gap = (pso_nll - slsqp_nll) / slsqp_nll
            if rel_gap > 0.005:
                dev_gates["maximum_relative_pso_nll_gap_vs_slsqp"] = False
                _append_issue(
                    issues,
                    "gates",
                    f"Dataset '{ds}' val PSO NLL gap vs SLSQP ({rel_gap:.4%}) > 0.5%",
                )

        # Gate 14 wall time ratio accounting
        if ds in pso_wall_times and ds in adam_pool_wall_times and adam_pool_wall_times[ds] > 0:
            sorted_times = sorted(pso_wall_times[ds])
            median_pso = sorted_times[len(sorted_times) // 2]
            pso_wall_ratios[ds] = median_pso / adam_pool_wall_times[ds]

    # Gate 13: cross-dataset mean relative PSO NLL reduction vs uniform >= 0.0
    if rel_nll_reductions_vs_uniform:
        mean_reduction = math.fsum(rel_nll_reductions_vs_uniform) / len(rel_nll_reductions_vs_uniform)
        if mean_reduction < 0.0:
            dev_gates["cross_dataset_mean_relative_pso_nll_reduction_vs_uniform_minimum"] = False
            _append_issue(
                issues,
                "gates",
                f"Mean relative val PSO NLL reduction vs uniform ({mean_reduction:.4%}) < 0.0",
            )
    else:
        dev_gates["cross_dataset_mean_relative_pso_nll_reduction_vs_uniform_minimum"] = False

    # Gate 14: every workload must satisfy the frozen 10% wall-time ceiling.
    res_totals = (
        artifact.get("resource_totals")
        if isinstance(artifact.get("resource_totals"), dict)
        else {}
    )
    if set(pso_wall_ratios) != set(EXPECTED_DATASETS):
        dev_gates["maximum_median_one_seed_pso_to_pool_training_wall_ratio"] = False
        _append_issue(
            issues,
            "accounting",
            "Cannot recompute a finite positive PSO/Adam wall ratio for every dataset",
        )
    else:
        for ds, ratio in pso_wall_ratios.items():
            if not math.isfinite(ratio) or ratio > 0.10:
                dev_gates["maximum_median_one_seed_pso_to_pool_training_wall_ratio"] = False
                _append_issue(
                    issues,
                    "gates",
                    f"Dataset '{ds}' median PSO wall time to Adam pool wall ratio "
                    f"({ratio:.2%}) > 10%",
                )

        sorted_ratios = sorted(pso_wall_ratios.values())
        mid = len(sorted_ratios) // 2
        recomputed_ratio = (
            sorted_ratios[mid]
            if len(sorted_ratios) % 2
            else 0.5 * (sorted_ratios[mid - 1] + sorted_ratios[mid])
        )
        reported_ratio = res_totals.get("pso_to_pool_wall_ratio")
        if not _is_finite_number(reported_ratio) or not math.isclose(
            float(reported_ratio),
            recomputed_ratio,
            abs_tol=1e-12,
            rel_tol=1e-9,
        ):
            dev_gates["maximum_median_one_seed_pso_to_pool_training_wall_ratio"] = False
            _append_issue(
                issues,
                "accounting",
                "resource_totals.pso_to_pool_wall_ratio does not match the "
                f"per-dataset recomputation ({recomputed_ratio:.12f}); got {reported_ratio}",
            )

    # Check structural/config/accounting/leakage/SLSQP/weights/finite/tuning errors
    dev_has_structural_errors = (
        len(issues["schema"]) > 0
        or len(issues["config"]) > 0
        or len(issues["finite"]) > 0
        or len(issues["weights"]) > 0
        or len(issues["accounting"]) > 0
        or len(issues["leakage"]) > 0
        or len(issues["tuning"]) > 0
        or len(issues["slsqp"]) > 0
        or len(issues["consistency"]) > 0
    )

    development_pass = all(dev_gates.values()) and not dev_has_structural_errors

    # 6. Confirmation Phase Verification
    official_test_data_loaded = artifact.get("official_test_data_loaded")
    confirmation_pass = False
    conf_gates: Optional[Dict[str, bool]] = None

    if not development_pass:
        if official_test_data_loaded is not False and official_test_data_loaded is True:
            _append_issue(
                issues,
                "leakage",
                "official_test_data_loaded must be False when development fails",
            )

        conf_improper = False
        for ds in EXPECTED_DATASETS:
            wl = workloads.get(ds) if isinstance(workloads.get(ds), dict) else {}
            conf_wl = wl.get("confirmation")
            if conf_wl is not None and conf_wl != {}:
                conf_improper = True
                _append_issue(
                    issues,
                    "gates",
                    f"Dataset '{ds}' confirmation present despite development failure",
                )

        if conf_improper or (official_test_data_loaded is not False and official_test_data_loaded is True):
            conf_gates = {"confirmation_absent_when_dev_failed": False}
        else:
            conf_gates = None
    else:  # development_pass is True
        if official_test_data_loaded is not True:
            _append_issue(
                issues,
                "leakage",
                "official_test_data_loaded must be True when development passed",
            )

        conf_missing = False
        for ds in EXPECTED_DATASETS:
            wl = workloads.get(ds) if isinstance(workloads.get(ds), dict) else {}
            conf_wl = wl.get("confirmation")
            if conf_wl is None or not isinstance(conf_wl, dict) or conf_wl == {}:
                conf_missing = True
                _append_issue(
                    issues,
                    "gates",
                    f"Dataset '{ds}' confirmation missing when development passed",
                )

        if conf_missing or official_test_data_loaded is not True:
            conf_gates = {"confirmation_present_and_loaded": False}
        else:
            conf_gates = {
                "all_values_finite": True,
                "official_test_dataset_loads_each_dataset": True,
                "official_test_pool_forward_passes_each_dataset": True,
                "official_test_long_single_forward_passes_each_dataset": True,
                "frozen_policy_consistency": artifact.get("policy_frozen") is True,
                "maximum_pso_accuracy_regression_vs_uniform_pp": True,
                "pso_nll_below_reference_single": True,
                "maximum_pso_nll_regression_vs_equal_budget_single": True,
                "post_test_tuning_or_reruns": artifact.get("post_test_tuning_or_reruns", 0) == 0,
            }
            if not conf_gates["frozen_policy_consistency"]:
                _append_issue(
                    issues,
                    "leakage",
                    "policy_frozen must be True before official confirmation",
                )

            for ds in EXPECTED_DATASETS:
                wl = workloads.get(ds, {})
                conf_wl = wl.get("confirmation", {})
                val_methods = wl.get("validation", {}).get("methods", {})
                frozen_methods = conf_wl.get("frozen_methods")
                expected_pso = val_methods.get("pso_weights", {})
                expected_slsqp = val_methods.get("slsqp_weights", {})
                expected_temp = val_methods.get("uniform_temperature", {})
                frozen_ok = (
                    isinstance(frozen_methods, dict)
                    and frozen_methods.get("selected_pso_seed")
                    == expected_pso.get("selected_seed")
                    and _sequences_close(
                        frozen_methods.get("selected_pso_weights"),
                        expected_pso.get("selected_weights"),
                    )
                    and _sequences_close(
                        frozen_methods.get("slsqp_weights"),
                        expected_slsqp.get("weights"),
                    )
                    and _is_finite_number(frozen_methods.get("fitted_temperature"))
                    and _is_finite_number(expected_temp.get("fitted_temperature"))
                    and math.isclose(
                        float(frozen_methods["fitted_temperature"]),
                        float(expected_temp["fitted_temperature"]),
                        abs_tol=1e-6,
                        rel_tol=1e-6,
                    )
                )
                if not frozen_ok:
                    conf_gates["frozen_policy_consistency"] = False
                    _append_issue(
                        issues,
                        "consistency",
                        f"Dataset '{ds}' confirmation frozen_methods do not match "
                        "the validation-frozen PSO seed/weights, SLSQP weights, and temperature",
                    )

                # Per-workload confirmation cache is test_cache_counts with dataset_loads, pool_forward_passes, long_single_forward_passes
                c_cache = conf_wl.get("test_cache_counts") if isinstance(conf_wl.get("test_cache_counts"), dict) else conf_wl.get("cache", conf_wl)
                t_loads = c_cache.get("dataset_loads", c_cache.get("official_test_dataset_loads"))
                t_pool_passes = c_cache.get("pool_forward_passes", c_cache.get("official_test_pool_forward_passes"))
                t_single_passes = c_cache.get("long_single_forward_passes", c_cache.get("official_test_long_single_forward_passes"))

                if t_loads != 1:
                    conf_gates["official_test_dataset_loads_each_dataset"] = False
                    _append_issue(issues, "accounting", f"Dataset '{ds}' test dataset_loads must be 1, got {t_loads}")
                if t_pool_passes != 5:
                    conf_gates["official_test_pool_forward_passes_each_dataset"] = False
                    _append_issue(issues, "accounting", f"Dataset '{ds}' test pool_forward_passes must be 5, got {t_pool_passes}")
                if t_single_passes != 1:
                    conf_gates["official_test_long_single_forward_passes_each_dataset"] = False
                    _append_issue(issues, "accounting", f"Dataset '{ds}' test long_single_forward_passes must be 1, got {t_single_passes}")

                # Test methods dict verification
                methods_dict = conf_wl.get("methods") if isinstance(conf_wl.get("methods"), dict) else conf_wl
                if not isinstance(methods_dict, dict):
                    conf_gates["all_values_finite"] = False
                    _append_issue(issues, "schema", f"Dataset '{ds}' confirmation methods must be a dictionary")
                    methods_dict = {}

                present_methods = set(methods_dict.keys())
                if present_methods != REQUIRED_BASELINES_SET:
                    conf_gates["all_values_finite"] = False
                    _append_issue(
                        issues,
                        "schema",
                        f"Dataset '{ds}' confirmation methods set {present_methods} does not match required {REQUIRED_BASELINES_SET}",
                    )

                ds_test_metrics: Dict[str, Dict[str, float]] = {}
                for method in REQUIRED_BASELINES:
                    if method not in methods_dict:
                        conf_gates["all_values_finite"] = False
                        _append_issue(issues, "schema", f"Dataset '{ds}' confirmation missing method '{method}'")
                        continue

                    entry = methods_dict[method]
                    acc, nll = _unwrap_metrics(entry)

                    if acc is None or nll is None:
                        conf_gates["all_values_finite"] = False
                        _append_issue(
                            issues,
                            "finite",
                            f"Dataset '{ds}' confirmation method '{method}' has missing or non-finite acc/nll",
                        )
                    else:
                        ds_test_metrics[method] = {"acc": acc, "nll": nll}

                test_metrics_by_dataset[ds] = ds_test_metrics

                pso_test_nll = ds_test_metrics.get("pso_weights", {}).get("nll")
                pso_test_acc = ds_test_metrics.get("pso_weights", {}).get("acc")
                unif_test_acc = ds_test_metrics.get("uniform_ensemble", {}).get("acc")
                ref_test_nll = ds_test_metrics.get("reference_single_10e", {}).get("nll")
                s50_test_nll = ds_test_metrics.get("single_50e", {}).get("nll")

                if any(v is None for v in (pso_test_nll, pso_test_acc, unif_test_acc, ref_test_nll, s50_test_nll)):
                    conf_gates["all_values_finite"] = False
                    conf_gates["maximum_pso_accuracy_regression_vs_uniform_pp"] = False
                    conf_gates["pso_nll_below_reference_single"] = False
                    conf_gates["maximum_pso_nll_regression_vs_equal_budget_single"] = False

                if pso_test_acc is not None and unif_test_acc is not None:
                    diff_pp = _to_pp(unif_test_acc) - _to_pp(pso_test_acc)
                    if diff_pp > 0.20:
                        conf_gates["maximum_pso_accuracy_regression_vs_uniform_pp"] = False
                        _append_issue(
                            issues,
                            "gates",
                            f"Dataset '{ds}' test PSO acc regression vs uniform ({diff_pp:.4f} pp) > 0.20 pp",
                        )

                if pso_test_nll is not None and ref_test_nll is not None:
                    if pso_test_nll >= ref_test_nll:
                        conf_gates["pso_nll_below_reference_single"] = False
                        _append_issue(
                            issues,
                            "gates",
                            f"Dataset '{ds}' test PSO NLL ({pso_test_nll:.6f}) >= reference single NLL ({ref_test_nll:.6f})",
                        )

                if pso_test_nll is not None and s50_test_nll is not None:
                    if (pso_test_nll - s50_test_nll) > 1e-7:
                        conf_gates["maximum_pso_nll_regression_vs_equal_budget_single"] = False
                        _append_issue(
                            issues,
                            "gates",
                            f"Dataset '{ds}' test PSO NLL ({pso_test_nll:.6f}) > 50e single NLL ({s50_test_nll:.6f}) by > 1e-7",
                        )

            confirmation_pass = all(conf_gates.values())

    # 7. Failed Gate Counting & Numeric Score Calculation
    # Count structural/config/accounting/leakage/SLSQP/weights/finite/tuning errors as hard failures
    structural_issue_count = sum(len(lst) for lst in issues.values())
    failed_dev_gates = sum(1 for v in dev_gates.values() if not v)
    failed_conf_gates = (
        sum(1 for v in conf_gates.values() if not v) if conf_gates is not None else (1 if development_pass else 0)
    )
    failed_hard_gate_count = max(failed_dev_gates + failed_conf_gates, structural_issue_count)

    # Calculate validation score metrics
    val_rel_nll_reductions: List[float] = []
    val_acc_gains_pp: List[float] = []
    test_rel_nll_reductions: List[float] = []
    test_acc_gains_pp: List[float] = []

    for ds in EXPECTED_DATASETS:
        m_val = val_metrics_by_dataset.get(ds, {})
        pso_v_nll = m_val.get("pso_weights", {}).get("nll")
        pso_v_acc = m_val.get("pso_weights", {}).get("acc")
        s50_v_nll = m_val.get("single_50e", {}).get("nll")
        s50_v_acc = m_val.get("single_50e", {}).get("acc")

        if pso_v_nll is not None and s50_v_nll is not None and s50_v_nll > 0:
            val_rel_nll_reductions.append((s50_v_nll - pso_v_nll) / s50_v_nll)
        if pso_v_acc is not None and s50_v_acc is not None:
            val_acc_gains_pp.append(_to_pp(pso_v_acc) - _to_pp(s50_v_acc))

        m_test = test_metrics_by_dataset.get(ds, {})
        pso_t_nll = m_test.get("pso_weights", {}).get("nll")
        pso_t_acc = m_test.get("pso_weights", {}).get("acc")
        s50_t_nll = m_test.get("single_50e", {}).get("nll")
        s50_t_acc = m_test.get("single_50e", {}).get("acc")

        if pso_t_nll is not None and s50_t_nll is not None and s50_t_nll > 0:
            test_rel_nll_reductions.append((s50_t_nll - pso_t_nll) / s50_t_nll)
        if pso_t_acc is not None and s50_t_acc is not None:
            test_acc_gains_pp.append(_to_pp(pso_t_acc) - _to_pp(s50_t_acc))

    mean_val_rel_nll = (
        math.fsum(val_rel_nll_reductions) / len(val_rel_nll_reductions) if val_rel_nll_reductions else 0.0
    )
    mean_val_acc_gain = (
        math.fsum(val_acc_gains_pp) / len(val_acc_gains_pp) if val_acc_gains_pp else 0.0
    )

    mean_test_rel_nll = (
        math.fsum(test_rel_nll_reductions) / len(test_rel_nll_reductions) if test_rel_nll_reductions else None
    )
    mean_test_acc_gain = (
        math.fsum(test_acc_gains_pp) / len(test_acc_gains_pp) if test_acc_gains_pp else None
    )

    if confirmation_pass and mean_test_rel_nll is not None and mean_test_acc_gain is not None:
        raw_score = 100.0 * mean_test_rel_nll + mean_test_acc_gain
    else:
        raw_score = 100.0 * mean_val_rel_nll + mean_val_acc_gain

    score = float(raw_score - 1000.0 * failed_hard_gate_count)
    if not math.isfinite(score):
        score = -1000.0 * float(failed_hard_gate_count if failed_hard_gate_count > 0 else 1)

    overall_pass = development_pass and confirmation_pass and (failed_hard_gate_count == 0)

    return {
        "evaluator_version": EVALUATOR_VERSION,
        "pass": overall_pass,
        "score": score,
        "development_pass": development_pass,
        "confirmation_pass": confirmation_pass,
        "failed_hard_gate_count": failed_hard_gate_count,
        "issues": issues,
        "development_gates": dev_gates,
        "confirmation_gates": conf_gates,
        "metrics": {
            "mean_val_relative_nll_reduction_vs_equal_budget_single": mean_val_rel_nll,
            "mean_val_accuracy_gain_vs_equal_budget_single_pp": mean_val_acc_gain,
            "mean_test_relative_nll_reduction_vs_equal_budget_single": mean_test_rel_nll,
            "mean_test_accuracy_gain_vs_equal_budget_single_pp": mean_test_acc_gain,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict Evaluator for Post-Training PSO Ensemble Study"
    )
    parser.add_argument("--artifact", type=Path, required=True, help="Path to study artifact JSON")
    parser.add_argument("--output", type=Path, required=True, help="Path to output evaluation JSON")
    args = parser.parse_args()

    if not args.artifact.is_file():
        print(f"Error: Artifact file not found at '{args.artifact}'", file=sys.stderr)
        sys.exit(1)

    try:
        with args.artifact.open("r", encoding="utf-8") as f:
            artifact = json.load(f)
    except Exception as exc:
        print(f"Error reading artifact JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    eval_result = evaluate_artifact(artifact)

    save_json_atomic(eval_result, args.output)
    print(
        f"Evaluation complete. Pass: {eval_result['pass']}, Score: {eval_result['score']:.6f}, Failed Gates: {eval_result['failed_hard_gate_count']}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
