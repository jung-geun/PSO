"""Strict evaluator for the Heavy PSO cross-split robustness mission.

Every candidate is compared with a baseline rerun on the same train/validation
partition and swarm seeds. Official test data must remain sealed. Development
may qualify a policy for one-shot confirmation, but mission ``pass`` is true
only when both phases satisfy the frozen evaluator contract.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_suite import save_json_atomic
from heavy_pso_autoresearch import compute_core_swarm_state_bytes, compute_latent_dim

EVALUATOR_VERSION = "HEAVY-PSO-CROSS-SPLIT-EVALUATOR 1.0.0"
EXPECTED_PARTICLES = 12
EXPECTED_EPOCHS = 80
EXPECTED_SUBSET_SIZE = 10000
EXPECTED_QUERIES = EXPECTED_PARTICLES * EXPECTED_EPOCHS
EXPECTED_SAMPLES = EXPECTED_QUERIES * EXPECTED_SUBSET_SIZE
EXPECTED_DEV_SPLIT_SEEDS = [20260905, 20260906]
EXPECTED_DEV_SWARM_SEEDS = [101, 102, 103]
EXPECTED_CONF_SPLIT_SEEDS = [20260907]
EXPECTED_CONF_SWARM_SEEDS = [111, 112, 113]
WORKLOADS = ["mnist_compact", "mnist_wide", "fashion_compact", "fashion_wide"]
BASELINE_METHODS = {
    "mnist_compact": "G8",
    "mnist_wide": "G5",
    "fashion_compact": "G8",
    "fashion_wide": "G5",
}
TOTAL_DIMS = {
    "mnist_compact": 9098,
    "mnist_wide": 55338,
    "fashion_compact": 9098,
    "fashion_wide": 55338,
}
PHASE_SPECS = {
    "development": (EXPECTED_DEV_SPLIT_SEEDS, EXPECTED_DEV_SWARM_SEEDS),
    "confirmation": (EXPECTED_CONF_SPLIT_SEEDS, EXPECTED_CONF_SWARM_SEEDS),
}


def load_artifact(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Artifact file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Artifact at {path} must be a JSON object")
    return data


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mean(values: Sequence[float]) -> float:
    return float(math.fsum(values) / len(values))


def _append_issue(issues: Dict[str, List[str]], category: str, message: str) -> None:
    issues[category].append(message)


def _validate_stats(
    entry: Dict[str, Any],
    per_seed_runs: List[Dict[str, Any]],
    label: str,
    issues: Dict[str, List[str]],
) -> Optional[Tuple[float, float]]:
    stats = entry.get("stats")
    if not isinstance(stats, dict):
        _append_issue(issues, "schema", f"{label}: missing stats object")
        return None

    try:
        acc_mean = stats["val_acc"]["mean"]
        nll_mean = stats["val_nll"]["mean"]
    except (KeyError, TypeError):
        _append_issue(issues, "schema", f"{label}: missing val_acc/val_nll means")
        return None

    if not (_is_finite_number(acc_mean) and _is_finite_number(nll_mean)):
        _append_issue(issues, "finite", f"{label}: non-finite aggregate metrics")
        return None

    if len(per_seed_runs) > 0:
        run_accs: List[float] = []
        run_nlls: List[float] = []
        for run in per_seed_runs:
            if not isinstance(run, dict):
                continue
            seed_label = f"{label}/seed={run.get('seed')}"

            if "val_selected_acc" not in run or run.get("val_selected_acc") is None:
                _append_issue(issues, "schema", f"{seed_label}: missing val_selected_acc")
            else:
                val_acc = run["val_selected_acc"]
                if not isinstance(val_acc, (int, float)) or isinstance(val_acc, bool):
                    _append_issue(issues, "schema", f"{seed_label}: non-numeric val_selected_acc")
                elif not math.isfinite(float(val_acc)):
                    _append_issue(issues, "finite", f"{seed_label}: non-finite val_selected_acc")
                else:
                    run_accs.append(float(val_acc))

            if "val_selected_loss" not in run or run.get("val_selected_loss") is None:
                _append_issue(issues, "schema", f"{seed_label}: missing val_selected_loss")
            else:
                val_nll = run["val_selected_loss"]
                if not isinstance(val_nll, (int, float)) or isinstance(val_nll, bool):
                    _append_issue(issues, "schema", f"{seed_label}: non-numeric val_selected_loss")
                elif not math.isfinite(float(val_nll)):
                    _append_issue(issues, "finite", f"{seed_label}: non-finite val_selected_loss")
                else:
                    run_nlls.append(float(val_nll))

        if len(run_accs) == len(per_seed_runs):
            if not math.isclose(float(acc_mean), _mean(run_accs), rel_tol=1e-6, abs_tol=1e-5):
                _append_issue(issues, "schema", f"{label}: val_acc mean disagrees with per-seed runs")
        if len(run_nlls) == len(per_seed_runs):
            if not math.isclose(float(nll_mean), _mean(run_nlls), rel_tol=1e-6, abs_tol=1e-5):
                _append_issue(issues, "schema", f"{label}: val_nll mean disagrees with per-seed runs")
    return float(acc_mean), float(nll_mean)


def _validate_runs(
    entry: Dict[str, Any],
    expected_seeds: List[int],
    expected_state_bytes: int,
    label: str,
    candidate: bool,
    issues: Dict[str, List[str]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    runs = entry.get("per_seed_runs")
    if not isinstance(runs, list) or len(runs) != len(expected_seeds):
        _append_issue(issues, "schema", f"{label}: expected {len(expected_seeds)} per-seed runs")
        return [], 0, 0

    if [run.get("seed") if isinstance(run, dict) else None for run in runs] != expected_seeds:
        _append_issue(issues, "config", f"{label}: per-seed run order/content does not match {expected_seeds}")

    total_queries = 0
    total_samples = 0
    numeric_fields = (
        "val_selected_loss",
        "val_selected_acc",
        "gbest_loss",
        "gbest_acc",
        "wall_time_sec",
        "optimization_wall_time_sec",
        "validation_wall_time_sec",
        "throughput_samples_per_sec",
    )

    valid_runs: List[Dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            _append_issue(issues, "schema", f"{label}: non-object seed record")
            continue
        seed_label = f"{label}/seed={run.get('seed')}"
        for field in numeric_fields:
            if not _is_finite_number(run.get(field)):
                _append_issue(issues, "finite", f"{seed_label}: missing/non-finite {field}")

        val_metrics = run.get("val_metrics")
        if (
            not isinstance(val_metrics, dict)
            or not val_metrics
            or any(not _is_finite_number(value) for value in val_metrics.values())
        ):
            _append_issue(issues, "finite", f"{seed_label}: missing/non-finite val_metrics")

        if run.get("official_test_evaluations") != 0:
            _append_issue(issues, "test", f"{seed_label}: official_test_evaluations must be 0")
        if candidate and run.get("is_finite") is not True:
            _append_issue(issues, "finite", f"{seed_label}: candidate is_finite must be true")
        if run.get("total_queries") != EXPECTED_QUERIES:
            _append_issue(issues, "accounting", f"{seed_label}: total_queries must be {EXPECTED_QUERIES}")
        if run.get("total_sample_evaluations") != EXPECTED_SAMPLES:
            _append_issue(issues, "accounting", f"{seed_label}: total_sample_evaluations must be {EXPECTED_SAMPLES}")
        if run.get("core_swarm_state_bytes") != expected_state_bytes:
            _append_issue(issues, "state", f"{seed_label}: incorrect core_swarm_state_bytes")

        if isinstance(run.get("total_queries"), int):
            total_queries += run["total_queries"]
        if isinstance(run.get("total_sample_evaluations"), int):
            total_samples += run["total_sample_evaluations"]
        valid_runs.append(run)

    return valid_runs, total_queries, total_samples


def _validate_artifact(artifact: Dict[str, Any], phase: str) -> Dict[str, Any]:
    expected_splits, expected_seeds = PHASE_SPECS[phase]
    issues: Dict[str, List[str]] = {
        "schema": [],
        "test": [],
        "finite": [],
        "provenance": [],
        "config": [],
        "accounting": [],
        "state": [],
    }
    cells: List[Dict[str, Any]] = []
    state_ratios: Dict[str, float] = {}
    expected_split_keys = {str(seed) for seed in expected_splits}

    if not isinstance(artifact, dict):
        _append_issue(issues, "schema", f"{phase}: artifact must be an object")
        return {
            "issues": issues,
            "cells": cells,
            "state_ratios": state_ratios,
            "max_state_ratio": 1.0,
            "policy_signature": None,
        }

    if artifact.get("phase") != phase:
        _append_issue(issues, "schema", f"{phase}: phase field mismatch")
    if artifact.get("split_seeds") != expected_splits:
        _append_issue(issues, "config", f"{phase}: split_seeds must be {expected_splits}")
    if artifact.get("swarm_seeds") != expected_seeds:
        _append_issue(issues, "config", f"{phase}: swarm_seeds must be {expected_seeds}")
    if artifact.get("official_test_data_loaded") is not False:
        _append_issue(issues, "test", f"{phase}: official_test_data_loaded must be false")
    if artifact.get("official_test_evaluations") != 0:
        _append_issue(issues, "test", f"{phase}: official_test_evaluations must be 0")

    candidate_config = artifact.get("candidate_config")
    if not isinstance(candidate_config, dict):
        _append_issue(issues, "schema", f"{phase}: missing candidate_config")
        candidate_config = {}
    for field, expected in (
        ("particles", EXPECTED_PARTICLES),
        ("epochs", EXPECTED_EPOCHS),
        ("subset_size", EXPECTED_SUBSET_SIZE),
    ):
        if candidate_config.get(field) != expected:
            _append_issue(issues, "config", f"{phase}: candidate_config.{field} must be {expected}")

    workload_config = artifact.get("workloads")
    if not isinstance(workload_config, dict) or set(workload_config) != set(WORKLOADS):
        _append_issue(issues, "schema", f"{phase}: workloads metadata must contain exactly {WORKLOADS}")
        workload_config = {}

    splits = artifact.get("splits")
    if not isinstance(splits, dict) or set(splits) != expected_split_keys:
        _append_issue(issues, "schema", f"{phase}: splits must contain exactly {sorted(expected_split_keys)}")
        splits = splits if isinstance(splits, dict) else {}

    observed_runs = 0
    observed_queries = 0
    observed_samples = 0

    for split_seed in expected_splits:
        split_key = str(split_seed)
        split_entry = splits.get(split_key)
        if not isinstance(split_entry, dict):
            _append_issue(issues, "schema", f"{phase}/{split_key}: missing split object")
            continue
        if split_entry.get("split_seed") != split_seed:
            _append_issue(issues, "provenance", f"{phase}/{split_key}: split_seed mismatch")

        baselines = split_entry.get("baselines")
        candidates = split_entry.get("candidates")
        if not isinstance(baselines, dict) or set(baselines) != set(WORKLOADS):
            _append_issue(issues, "schema", f"{phase}/{split_key}: baseline workloads incomplete")
            baselines = baselines if isinstance(baselines, dict) else {}
        if not isinstance(candidates, dict) or set(candidates) != set(WORKLOADS):
            _append_issue(issues, "schema", f"{phase}/{split_key}: candidate workloads incomplete")
            candidates = candidates if isinstance(candidates, dict) else {}

        for workload in WORKLOADS:
            baseline = baselines.get(workload)
            candidate_entry = candidates.get(workload)
            label = f"{phase}/{split_key}/{workload}"
            if not isinstance(baseline, dict) or not isinstance(candidate_entry, dict):
                _append_issue(issues, "schema", f"{label}: missing baseline or candidate entry")
                continue

            if baseline.get("method_id") != BASELINE_METHODS[workload]:
                _append_issue(issues, "config", f"{label}: wrong baseline method")
            for mode, entry in (("baseline", baseline), ("candidate", candidate_entry)):
                if entry.get("workload_id") != workload:
                    _append_issue(issues, "schema", f"{label}/{mode}: workload_id mismatch")
                if entry.get("split_seed") != split_seed:
                    _append_issue(issues, "provenance", f"{label}/{mode}: split_seed mismatch")
                if entry.get("particles") != EXPECTED_PARTICLES:
                    _append_issue(issues, "config", f"{label}/{mode}: particles mismatch")
                if entry.get("epochs") != EXPECTED_EPOCHS:
                    _append_issue(issues, "config", f"{label}/{mode}: epochs mismatch")
                if entry.get("subset_size") != EXPECTED_SUBSET_SIZE:
                    _append_issue(issues, "config", f"{label}/{mode}: subset_size mismatch")
                if entry.get("seeds") != expected_seeds:
                    _append_issue(issues, "config", f"{label}/{mode}: seeds mismatch")

            fingerprints = (
                baseline.get("split_fingerprint"),
                candidate_entry.get("split_fingerprint"),
                baseline.get("data_fingerprint"),
                candidate_entry.get("data_fingerprint"),
            )
            if any(not isinstance(value, str) or not value for value in fingerprints):
                _append_issue(issues, "provenance", f"{label}: fingerprints must be non-empty strings")
            elif fingerprints[0] != fingerprints[1] or fingerprints[2] != fingerprints[3]:
                _append_issue(issues, "provenance", f"{label}: baseline/candidate fingerprints differ")

            total_dim = TOTAL_DIMS[workload]
            baseline_states = 5 * EXPECTED_PARTICLES + (1 if BASELINE_METHODS[workload] == "G8" else 0)
            expected_baseline_bytes = baseline_states * total_dim * 4
            raw_ratio = candidate_entry.get("ratio")
            if not _is_finite_number(raw_ratio) or not (0.0 < float(raw_ratio) <= 1.0):
                _append_issue(issues, "state", f"{label}: invalid candidate ratio")
                expected_candidate_bytes = -1
            else:
                expected_latent_dim = compute_latent_dim(total_dim, float(raw_ratio))
                expected_candidate_bytes = compute_core_swarm_state_bytes(EXPECTED_PARTICLES, expected_latent_dim)
                if candidate_entry.get("total_dim") != total_dim:
                    _append_issue(issues, "state", f"{label}: total_dim mismatch")
                if candidate_entry.get("latent_dim") != expected_latent_dim:
                    _append_issue(issues, "state", f"{label}: latent_dim mismatch")
                if candidate_entry.get("core_swarm_state_bytes") != expected_candidate_bytes:
                    _append_issue(issues, "state", f"{label}: candidate state bytes mismatch")
                if candidate_entry.get("baseline_core_swarm_state_bytes") != expected_baseline_bytes:
                    _append_issue(issues, "state", f"{label}: candidate baseline state bytes mismatch")
                ratio = expected_candidate_bytes / expected_baseline_bytes
                state_ratios[workload] = max(state_ratios.get(workload, 0.0), ratio)
                if not _is_finite_number(candidate_entry.get("state_ratio")) or not math.isclose(
                    float(candidate_entry.get("state_ratio", -1.0)), ratio, rel_tol=1e-6, abs_tol=1e-6
                ):
                    _append_issue(issues, "state", f"{label}: reported state_ratio mismatch")

            baseline_runs, b_queries, b_samples = _validate_runs(
                baseline,
                expected_seeds,
                expected_baseline_bytes,
                f"{label}/baseline",
                False,
                issues,
            )
            candidate_runs, c_queries, c_samples = _validate_runs(
                candidate_entry,
                expected_seeds,
                expected_candidate_bytes,
                f"{label}/candidate",
                True,
                issues,
            )
            observed_runs += len(baseline_runs) + len(candidate_runs)
            observed_queries += b_queries + c_queries
            observed_samples += b_samples + c_samples

            baseline_stats = _validate_stats(baseline, baseline_runs, f"{label}/baseline", issues)
            candidate_stats = _validate_stats(candidate_entry, candidate_runs, f"{label}/candidate", issues)
            if baseline_stats is not None and candidate_stats is not None:
                baseline_acc, baseline_nll = baseline_stats
                candidate_acc, candidate_nll = candidate_stats
                nll_reduction = (
                    (baseline_nll - candidate_nll) / baseline_nll
                    if baseline_nll > 0.0
                    else float("nan")
                )
                if not math.isfinite(nll_reduction):
                    _append_issue(issues, "finite", f"{label}: NLL reduction is non-finite")
                else:
                    cells.append(
                        {
                            "phase": phase,
                            "split_seed": split_seed,
                            "workload_id": workload,
                            "baseline_acc": baseline_acc,
                            "candidate_acc": candidate_acc,
                            "baseline_nll": baseline_nll,
                            "candidate_nll": candidate_nll,
                            "acc_gain_pp": candidate_acc - baseline_acc,
                            "nll_reduction_fraction": nll_reduction,
                        }
                    )

    expected_runs = len(expected_splits) * len(WORKLOADS) * len(expected_seeds) * 2
    if observed_runs != expected_runs:
        _append_issue(issues, "accounting", f"{phase}: observed {observed_runs} runs, expected {expected_runs}")
    resources = artifact.get("resource_totals")
    if not isinstance(resources, dict):
        _append_issue(issues, "accounting", f"{phase}: missing resource_totals")
        resources = {}
    if resources.get("total_runs") != observed_runs:
        _append_issue(issues, "accounting", f"{phase}: total_runs does not match records")
    if resources.get("total_queries") != observed_queries:
        _append_issue(issues, "accounting", f"{phase}: total_queries does not match records")
    if resources.get("total_samples_evaluated") != observed_samples:
        _append_issue(issues, "accounting", f"{phase}: total_samples_evaluated does not match records")
    if resources.get("official_test_evaluations") != 0:
        _append_issue(issues, "test", f"{phase}: resource official_test_evaluations must be 0")

    max_state_ratio = max(state_ratios.values(), default=1.0)
    policy_signature = {
        "candidate_config": candidate_config,
        "workloads": workload_config,
    }
    return {
        "issues": issues,
        "cells": cells,
        "state_ratios": state_ratios,
        "max_state_ratio": max_state_ratio,
        "policy_signature": policy_signature,
    }


def evaluate_heavy_cross_split(
    development_artifact: Dict[str, Any],
    confirmation_artifact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dev_result = _validate_artifact(development_artifact, "development")
    conf_result = (
        _validate_artifact(confirmation_artifact, "confirmation")
        if confirmation_artifact is not None
        else None
    )

    gates: Dict[str, Dict[str, Any]] = {}
    failed_gates: List[str] = []
    development_gate_names: List[str] = []

    def record_gate(
        name: str,
        passed: bool,
        observed: Any,
        expected: Any,
        details: str,
        development_gate: bool = False,
    ) -> None:
        gates[name] = {
            "pass": bool(passed),
            "observed": observed,
            "expected": expected,
            "details": details,
        }
        if not passed:
            failed_gates.append(name)
        if development_gate:
            development_gate_names.append(name)

    all_results = [dev_result] + ([conf_result] if conf_result is not None else [])
    record_gate(
        "schema_and_phase_seeds",
        all(not result["issues"]["schema"] for result in all_results),
        [message for result in all_results for message in result["issues"]["schema"]][:10],
        "complete artifacts with exact declared phase/split/swarm seeds",
        "Missing evidence is rejected rather than defaulted",
        True,
    )
    record_gate(
        "official_test_sealed",
        all(not result["issues"]["test"] for result in all_results),
        [message for result in all_results for message in result["issues"]["test"]][:10],
        "loaded=false and evaluations=0 at artifact, resource, and run levels",
        "Official test data must never be loaded or evaluated",
        True,
    )
    record_gate(
        "all_runs_finite",
        all(not result["issues"]["finite"] for result in all_results),
        [message for result in all_results for message in result["issues"]["finite"]][:10],
        "all aggregate and per-run validation metrics and times finite",
        "Every recorded validation metric must be finite",
        True,
    )
    record_gate(
        "split_and_fingerprint_matched",
        all(not result["issues"]["provenance"] for result in all_results),
        [message for result in all_results for message in result["issues"]["provenance"]][:10],
        "non-empty matching split/data fingerprints and split seeds per baseline/candidate cell",
        "Every delta must use a baseline rerun on the identical partition",
        True,
    )

    policy_matches = conf_result is None or (
        dev_result["policy_signature"] == conf_result["policy_signature"]
    )
    config_issues = [message for result in all_results for message in result["issues"]["config"]]
    if not policy_matches:
        config_issues.append("confirmation candidate policy differs from the frozen development policy")
    record_gate(
        "configuration_and_policy_matched",
        not config_issues,
        config_issues[:10],
        "12p x 80e x fixed10k, exact seeds, matching baseline, identical frozen candidate policy",
        "Confirmation cannot change the development-selected policy",
        True,
    )
    record_gate(
        "query_and_sample_accounting_exact",
        all(not result["issues"]["accounting"] for result in all_results),
        [message for result in all_results for message in result["issues"]["accounting"]][:10],
        f"{EXPECTED_QUERIES} queries and {EXPECTED_SAMPLES} sample evaluations per run",
        "Per-run and aggregate accounting must agree exactly",
        True,
    )

    max_state_ratio = max(result["max_state_ratio"] for result in all_results)
    state_issues = [message for result in all_results for message in result["issues"]["state"]]
    state_ok = not state_issues and 0.0 < max_state_ratio <= 0.5 + 1e-12
    record_gate(
        "maximum_state_ratio_each_workload",
        state_ok,
        {"max_state_ratio": max_state_ratio, "issues": state_issues[:10]},
        "analytically verified state ratio <= 0.5 for every cell",
        "No absent or reported-only state evidence is accepted",
        True,
    )

    dev_cells = dev_result["cells"]
    conf_cells = conf_result["cells"] if conf_result is not None else []
    all_cells = dev_cells + conf_cells
    expected_dev_cells = len(EXPECTED_DEV_SPLIT_SEEDS) * len(WORKLOADS)
    complete_dev_cells = len(dev_cells) == expected_dev_cells

    def nonregression(cells: List[Dict[str, Any]]) -> Tuple[bool, float, float]:
        if not cells:
            return False, float("inf"), float("inf")
        max_acc_regression = max(-cell["acc_gain_pp"] for cell in cells)
        max_nll_regression = max(-cell["nll_reduction_fraction"] for cell in cells)
        return (
            max_acc_regression <= 1.0 + 1e-12 and max_nll_regression <= 0.05 + 1e-12,
            max_acc_regression,
            max_nll_regression,
        )

    all_nonreg, max_acc_reg, max_nll_reg = nonregression(all_cells)
    record_gate(
        "maximum_accuracy_regression_percentage_points_each_split_workload",
        bool(all_cells) and max_acc_reg <= 1.0 + 1e-12,
        max_acc_reg,
        "<= 1.0 pp",
        "No evaluated split-workload cell may regress accuracy by more than 1 pp",
        True,
    )
    record_gate(
        "maximum_nll_regression_fraction_each_split_workload",
        bool(all_cells) and max_nll_reg <= 0.05 + 1e-12,
        max_nll_reg,
        "<= 0.05",
        "No evaluated split-workload cell may regress NLL by more than 5%",
        True,
    )
    dev_nonreg, dev_acc_reg, dev_nll_reg = nonregression(dev_cells)

    dev_acc_mean = _mean([cell["acc_gain_pp"] for cell in dev_cells]) if dev_cells else float("-inf")
    dev_nll_mean = _mean([cell["nll_reduction_fraction"] for cell in dev_cells]) if dev_cells else float("-inf")
    record_gate(
        "development_grand_mean_accuracy_gain_minimum_pp",
        complete_dev_cells and dev_acc_mean >= 0.0,
        dev_acc_mean,
        ">= 0.0 pp",
        "Development grand mean accuracy must not regress",
        True,
    )
    record_gate(
        "development_grand_mean_nll_reduction_minimum_fraction",
        complete_dev_cells and dev_nll_mean >= 0.0,
        dev_nll_mean,
        ">= 0.0",
        "Development grand mean NLL must not regress",
        True,
    )

    dev_mw = [cell for cell in dev_cells if cell["workload_id"] == "mnist_wide"]
    dev_mw_acc = _mean([cell["acc_gain_pp"] for cell in dev_mw]) if dev_mw else float("-inf")
    dev_mw_nll = _mean([cell["nll_reduction_fraction"] for cell in dev_mw]) if dev_mw else float("-inf")
    record_gate(
        "development_mnist_wide_improvement",
        len(dev_mw) == len(EXPECTED_DEV_SPLIT_SEEDS) and (dev_mw_acc >= 2.0 or dev_mw_nll >= 0.05),
        {"accuracy_gain_pp": dev_mw_acc, "nll_reduction_fraction": dev_mw_nll},
        "mean accuracy gain >=2pp OR mean NLL reduction >=5%",
        "The prior worst baseline workload must materially improve across development partitions",
        True,
    )

    development_pass = (
        all(not messages for messages in dev_result["issues"].values())
        and 0.0 < dev_result["max_state_ratio"] <= 0.5 + 1e-12
        and complete_dev_cells
        and dev_nonreg
        and dev_acc_mean >= 0.0
        and dev_nll_mean >= 0.0
        and len(dev_mw) == len(EXPECTED_DEV_SPLIT_SEEDS)
        and (dev_mw_acc >= 2.0 or dev_mw_nll >= 0.05)
    )
    if confirmation_artifact is None:
        record_gate(
            "confirmation_executed",
            False,
            "not executed",
            "one exact confirmation artifact after development_pass",
            "A development pass only qualifies the frozen policy for one-shot confirmation",
        )
    else:
        record_gate(
            "confirmation_executed",
            True,
            {"split_seeds": confirmation_artifact.get("split_seeds"), "swarm_seeds": confirmation_artifact.get("swarm_seeds")},
            {"split_seeds": EXPECTED_CONF_SPLIT_SEEDS, "swarm_seeds": EXPECTED_CONF_SWARM_SEEDS},
            "Confirmation evidence is evaluated only with the exact sealed phase contract",
        )

        expected_conf_cells = len(EXPECTED_CONF_SPLIT_SEEDS) * len(WORKLOADS)
        complete_conf_cells = len(conf_cells) == expected_conf_cells
        conf_nonreg, conf_acc_reg, conf_nll_reg = nonregression(conf_cells)
        record_gate(
            "confirmation_per_cell_non_regression",
            complete_conf_cells and conf_nonreg,
            {"cells": len(conf_cells), "max_acc_regression_pp": conf_acc_reg, "max_nll_regression_fraction": conf_nll_reg},
            "4 cells; accuracy regression <=1pp and NLL regression <=5% in each",
            "The sealed partition must remain safe workload by workload",
        )

        conf_acc_mean = _mean([cell["acc_gain_pp"] for cell in conf_cells]) if conf_cells else float("-inf")
        conf_nll_mean = _mean([cell["nll_reduction_fraction"] for cell in conf_cells]) if conf_cells else float("-inf")
        record_gate(
            "confirmation_grand_mean_accuracy_gain_minimum_pp",
            complete_conf_cells and conf_acc_mean >= 1.5,
            conf_acc_mean,
            ">= 1.5 pp",
            "One-shot confirmation must retain the predeclared accuracy effect",
        )
        record_gate(
            "confirmation_grand_mean_nll_reduction_minimum_fraction",
            complete_conf_cells and conf_nll_mean >= 0.02,
            conf_nll_mean,
            ">= 0.02",
            "One-shot confirmation must retain the predeclared NLL effect",
        )

        conf_mw = [cell for cell in conf_cells if cell["workload_id"] == "mnist_wide"]
        conf_mw_acc = _mean([cell["acc_gain_pp"] for cell in conf_mw]) if conf_mw else float("-inf")
        conf_mw_nll = _mean([cell["nll_reduction_fraction"] for cell in conf_mw]) if conf_mw else float("-inf")
        record_gate(
            "confirmation_mnist_wide_improvement",
            len(conf_mw) == 1 and (conf_mw_acc >= 1.0 or conf_mw_nll >= 0.03),
            {"accuracy_gain_pp": conf_mw_acc, "nll_reduction_fraction": conf_mw_nll},
            "accuracy gain >=1pp OR NLL reduction >=3%",
            "The prior worst workload must improve on the sealed partition",
        )

        combined_mw = dev_mw + conf_mw
        combined_mw_acc = _mean([cell["acc_gain_pp"] for cell in combined_mw]) if combined_mw else float("-inf")
        combined_mw_nll = _mean([cell["nll_reduction_fraction"] for cell in combined_mw]) if combined_mw else float("-inf")
        record_gate(
            "combined_mnist_wide_improvement",
            len(combined_mw) == 3 and (combined_mw_acc >= 2.0 or combined_mw_nll >= 0.05),
            {"accuracy_gain_pp": combined_mw_acc, "nll_reduction_fraction": combined_mw_nll},
            "three-split mean accuracy gain >=2pp OR NLL reduction >=5%",
            "The material worst-workload improvement must hold across all new partitions",
        )

    score_cells = all_cells if confirmation_artifact is not None else dev_cells
    mean_acc_gain = _mean([cell["acc_gain_pp"] for cell in score_cells]) if score_cells else 0.0
    mean_nll_reduction = _mean([cell["nll_reduction_fraction"] for cell in score_cells]) if score_cells else 0.0
    score_gate_names = list(gates) if confirmation_artifact is not None else development_gate_names
    score_failed_gates = sum(not gates[name]["pass"] for name in score_gate_names)
    state_points = 10.0 * math.log2(1.0 / max_state_ratio) if 0.0 < max_state_ratio <= 1.0 else 0.0
    score = (
        100.0 * mean_nll_reduction
        + mean_acc_gain
        + state_points
        - 100.0 * score_failed_gates
    )
    mission_pass = confirmation_artifact is not None and not failed_gates

    return {
        "pass": bool(mission_pass),
        "development_pass": bool(development_pass),
        "eligible_for_confirmation": bool(development_pass and confirmation_artifact is None),
        "score": float(score),
        "evaluator_version": EVALUATOR_VERSION,
        "failed_hard_gate_count": len(failed_gates),
        "failed_gates": failed_gates,
        "score_failed_gate_count": score_failed_gates,
        "gates": gates,
        "score_components": {
            "mean_relative_nll_reduction_pct": 100.0 * mean_nll_reduction,
            "mean_accuracy_gain_pp": mean_acc_gain,
            "state_efficiency_points": state_points,
            "gate_penalty_points": 100.0 * score_failed_gates,
            "max_state_ratio": max_state_ratio,
        },
        "summary_metrics": {
            "development_cells": len(dev_cells),
            "confirmation_cells": len(conf_cells),
            "development_grand_mean_accuracy_gain_pp": dev_acc_mean,
            "development_grand_mean_nll_reduction_fraction": dev_nll_mean,
            "development_mnist_wide_accuracy_gain_pp": dev_mw_acc,
            "development_mnist_wide_nll_reduction_fraction": dev_mw_nll,
        },
        "cell_metrics": all_cells,
        "state_ratios": {
            "development": dev_result["state_ratios"],
            "confirmation": conf_result["state_ratios"] if conf_result is not None else None,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict Heavy PSO cross-split evaluator")
    parser.add_argument("--development", required=True, help="Development artifact JSON")
    parser.add_argument("--confirmation", default=None, help="Optional one-shot confirmation artifact JSON")
    parser.add_argument("--output", default=None, help="Optional evaluation JSON output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    development = load_artifact(Path(args.development))
    confirmation = load_artifact(Path(args.confirmation)) if args.confirmation else None
    result = evaluate_heavy_cross_split(development, confirmation)
    if args.output:
        save_json_atomic(result, Path(args.output))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
