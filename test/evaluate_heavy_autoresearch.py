"""
Strict Pareto Evaluator for Heavy Task PSO Autoresearch.

Evaluates candidate equalized signed-hash subspace experiment artifacts against
baseline heavy task benchmark results (benchmark_results/pso_v6_heavy_tasks.json).

Baseline Policy:
  - mnist_compact: G8
  - mnist_wide: G5
  - fashion_compact: G8
  - fashion_wide: G5

Evaluates 7 Hard Gates:
  1. Finite Metrics (all validation metrics finite across runs)
  2. Test-Sealed (zero official test data loaded & 0 test evaluations)
  3. Config Matched (12 particles, 80 epochs, 10k subset, seeds 101-103)
  4. State Ratio Boundary (max state ratio <= 0.5 across all workloads)
  5. Workload Accuracy Regression Boundary (each workload acc regression <= 1.0 pp)
  6. Workload NLL Regression Boundary (each workload NLL regression <= 5.0%)
  7. Worst-Workload (mnist_wide) Improvement (acc gain >= 2.0 pp OR NLL reduction >= 5.0%)

Numeric Score:
  Score = mean_rel_nll_reduction_pct + mean_acc_gain_pp + 10 * log2(1 / max_state_ratio) - 100 * failed_gate_count
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure test directory and repo root are in Python path
REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name != "PSO" else Path(__file__).resolve().parent
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_suite import save_json_atomic
from heavy_pso_autoresearch import compute_latent_dim, compute_core_swarm_state_bytes
EVALUATOR_VERSION = "EVALUATE-HEAVY-AUTORESEARCH 1.0.0"

BASELINE_POLICY: Dict[str, str] = {
    "mnist_compact": "G8",
    "mnist_wide": "G5",
    "fashion_compact": "G8",
    "fashion_wide": "G5",
}

EXPECTED_SEEDS = [101, 102, 103]
EXPECTED_PARTICLES = 12
EXPECTED_EPOCHS = 80
EXPECTED_SUBSET_SIZE = 10000
EXPECTED_WORKLOADS = ["mnist_compact", "mnist_wide", "fashion_compact", "fashion_wide"]


def evaluate_heavy_autoresearch(
    baseline_path: Union[str, Path],
    candidate_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Evaluates a candidate experiment artifact against baseline heavy task results.
    Validates schemas, configurations, seeds, test-sealed constraints, finiteness,
    calculates every hard gate and numeric score, selects the winning candidate,
    and returns a structured evaluator payload.
    """
    baseline_path = Path(baseline_path)
    candidate_path = Path(candidate_path)

    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline artifact not found at '{baseline_path}'")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate artifact not found at '{candidate_path}'")

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    with open(candidate_path, "r", encoding="utf-8") as f:
        candidate_data = json.load(f)

    WORKLOAD_TOTAL_DIMS: Dict[str, int] = {
        "mnist_compact": 9098,
        "mnist_wide": 55338,
        "fashion_compact": 9098,
        "fashion_wide": 55338,
    }

    if (
        baseline_data.get("official_test_data_loaded") is not False
        or baseline_data.get("official_test_evaluations") != 0
    ):
        raise ValueError("Baseline artifact must explicitly seal official test data.")

    # Validate baseline JSON schema & confirmation results
    if "confirmation_results" not in baseline_data:
        raise KeyError("Baseline JSON missing top-level 'confirmation_results' key")
    
    baseline_confirm = baseline_data["confirmation_results"]
    baseline_metrics: Dict[str, Dict[str, float]] = {}
    baseline_core_bytes: Dict[str, int] = {}

    for wl in EXPECTED_WORKLOADS:
        if wl not in baseline_confirm:
            raise KeyError(f"Baseline confirmation results missing workload '{wl}'")
        selected_method = BASELINE_POLICY[wl]
        if selected_method not in baseline_confirm[wl]:
            raise KeyError(
                f"Baseline confirmation results for '{wl}' missing policy method '{selected_method}'"
            )
        entry = baseline_confirm[wl][selected_method]
        stats = entry.get("stats", {}) if isinstance(entry, dict) else {}
        val_nll_mean = float(stats.get("val_nll", {}).get("mean", float("nan")))
        val_acc_mean = float(stats.get("val_acc", {}).get("mean", float("nan")))
        if not (math.isfinite(val_nll_mean) and math.isfinite(val_acc_mean)):
            raise ValueError(f"Baseline metrics for '{wl}' method '{selected_method}' contain NaN or non-finite value")
        
        baseline_metrics[wl] = {
            "val_nll": val_nll_mean,
            "val_acc": val_acc_mean,
            "val_brier": float(stats.get("val_brier", {}).get("mean", 0.0) or 0.0),
            "val_ece": float(stats.get("val_ece", {}).get("mean", 0.0) or 0.0),
            "method_id": selected_method,
        }

        bytes_list = []
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("per_seed_runs"), list)
        ):
            bytes_list = [
                int(run["core_swarm_state_bytes"])
                for run in entry["per_seed_runs"]
                if (
                    isinstance(run, dict)
                    and "core_swarm_state_bytes" in run
                    and math.isfinite(float(run["core_swarm_state_bytes"]))
                )
            ]
        if bytes_list and len(set(bytes_list)) != 1:
            raise ValueError(f"Baseline core-state bytes vary across seeds for '{wl}'.")
        if bytes_list:
            b_bytes = bytes_list[0]
        else:
            states = 5 * EXPECTED_PARTICLES + (1 if selected_method == "G8" else 0)
            b_bytes = states * WORKLOAD_TOTAL_DIMS[wl] * 4
        baseline_core_bytes[wl] = b_bytes

    # Validate candidate JSON schema
    if "candidate_runs" not in candidate_data:
        raise KeyError("Candidate JSON missing top-level 'candidate_runs' key")

    candidate_runs = candidate_data["candidate_runs"]
    if not isinstance(candidate_runs, dict) or len(candidate_runs) == 0:
        raise ValueError("Candidate JSON 'candidate_runs' must be a non-empty dictionary")

    # Global candidate test-sealed checks
    top_test_loaded_present = "official_test_data_loaded" in candidate_data
    top_test_loaded_val = candidate_data.get("official_test_data_loaded")
    top_test_evals_present = "official_test_evaluations" in candidate_data
    top_test_evals_val = candidate_data.get("official_test_evaluations")

    top_test_sealed = (
        top_test_loaded_present
        and top_test_loaded_val is False
        and top_test_evals_present
        and top_test_evals_val == 0
    )

    candidate_evaluations: Dict[str, Dict[str, Any]] = {}

    for cand_id, wl_map in candidate_runs.items():
        gate_finite = True
        gate_test_sealed = bool(top_test_sealed)
        gate_config_matched = True

        if not isinstance(wl_map, dict):
            gate_config_matched = False
            gate_finite = False
            wl_map = {}

        if set(wl_map) != set(EXPECTED_WORKLOADS):
            gate_config_matched = False

        for wl in EXPECTED_WORKLOADS:
            if wl not in wl_map:
                gate_config_matched = False
                gate_finite = False

        derived_ratios: Dict[str, float] = {}
        per_workload_deltas: Dict[str, Dict[str, float]] = {}
        acc_regressions_valid = True
        nll_regressions_valid = True

        for wl in EXPECTED_WORKLOADS:
            if wl not in wl_map or not isinstance(wl_map[wl], dict):
                acc_regressions_valid = False
                nll_regressions_valid = False
                continue

            wl_entry = wl_map[wl]
            p_val = wl_entry.get("particles")
            e_val = wl_entry.get("epochs")
            sub_val = wl_entry.get("subset_size")
            seeds_val = wl_entry.get("seeds")

            if (
                p_val != EXPECTED_PARTICLES
                or e_val != EXPECTED_EPOCHS
                or sub_val != EXPECTED_SUBSET_SIZE
                or seeds_val != EXPECTED_SEEDS
            ):
                gate_config_matched = False

            per_seed = wl_entry.get("per_seed_runs")
            if not isinstance(per_seed, list) or len(per_seed) != len(EXPECTED_SEEDS):
                gate_config_matched = False
                gate_finite = False
                actual_seeds = []
            else:
                actual_seeds = [
                    s_run.get("seed") for s_run in per_seed if isinstance(s_run, dict) and "seed" in s_run
                ]
                if actual_seeds != EXPECTED_SEEDS:
                    gate_config_matched = False

                expected_queries = EXPECTED_PARTICLES * EXPECTED_EPOCHS
                expected_samples = expected_queries * EXPECTED_SUBSET_SIZE
                for s_run in per_seed:
                    if (
                        not isinstance(s_run, dict)
                        or s_run.get("total_queries") != expected_queries
                        or s_run.get("total_sample_evaluations") != expected_samples
                    ):
                        gate_config_matched = False

            stats = wl_entry.get("stats") if isinstance(wl_entry.get("stats"), dict) else {}
            val_nll_dict = stats.get("val_nll") if isinstance(stats.get("val_nll"), dict) else {}
            val_acc_dict = stats.get("val_acc") if isinstance(stats.get("val_acc"), dict) else {}
            
            c_nll_raw = val_nll_dict.get("mean")
            c_acc_raw = val_acc_dict.get("mean")

            if (
                c_nll_raw is None
                or c_acc_raw is None
                or not isinstance(c_nll_raw, (int, float))
                or not isinstance(c_acc_raw, (int, float))
                or not (math.isfinite(float(c_nll_raw)) and math.isfinite(float(c_acc_raw)))
            ):
                gate_finite = False
                c_nll = float("nan")
                c_acc = float("nan")
            else:
                c_nll = float(c_nll_raw)
                c_acc = float(c_acc_raw)

            if isinstance(per_seed, list):
                for s_run in per_seed:
                    if not isinstance(s_run, dict):
                        gate_finite = False
                        gate_config_matched = False
                        continue
                    if "official_test_evaluations" not in s_run or s_run["official_test_evaluations"] != 0:
                        gate_test_sealed = False
                    if s_run.get("is_finite") is not True:
                        gate_finite = False

                    s_nll = s_run.get("val_selected_loss")
                    s_acc = s_run.get("val_selected_acc")
                    g_nll = s_run.get("gbest_loss")
                    g_acc = s_run.get("gbest_acc")
                    w_time = s_run.get("wall_time_sec")
                    val_m = s_run.get("val_metrics")

                    val_m_ok = (
                        isinstance(val_m, dict)
                        and len(val_m) > 0
                        and all(
                            isinstance(v, (int, float)) and math.isfinite(float(v))
                            for v in val_m.values()
                        )
                    )

                    scalars_for_s_run = [s_nll, s_acc, g_nll, g_acc, w_time]
                    s_scalars_ok = all(
                        v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))
                        for v in scalars_for_s_run
                    )

                    if not (val_m_ok and s_scalars_ok):
                        gate_finite = False

            expected_total_dim = WORKLOAD_TOTAL_DIMS[wl]
            total_dim = wl_entry.get("total_dim")
            raw_ratio = wl_entry.get("ratio")
            if (
                total_dim != expected_total_dim
                or not isinstance(raw_ratio, (int, float))
                or not (0.0 < float(raw_ratio) <= 1.0)
            ):
                gate_config_matched = False
                total_dim = expected_total_dim
                raw_ratio = 1.0

            latent_dim = compute_latent_dim(total_dim, float(raw_ratio))
            if wl_entry.get("latent_dim") != latent_dim:
                gate_config_matched = False

            analytical_cand_bytes = compute_core_swarm_state_bytes(EXPECTED_PARTICLES, latent_dim)
            b_bytes = baseline_core_bytes[wl]

            derived_ratio = float(analytical_cand_bytes) / float(b_bytes)
            derived_ratios[wl] = derived_ratio

            if wl_entry.get("core_swarm_state_bytes") != analytical_cand_bytes:
                gate_config_matched = False

            if isinstance(per_seed, list):
                for s_run in per_seed:
                    if isinstance(s_run, dict) and s_run.get("core_swarm_state_bytes") != analytical_cand_bytes:
                        gate_config_matched = False

            reported_state_ratio = wl_entry.get("state_ratio")
            if (
                not isinstance(reported_state_ratio, (int, float))
                or not math.isfinite(float(reported_state_ratio))
                or not math.isclose(
                    float(reported_state_ratio),
                    derived_ratio,
                    rel_tol=1e-5,
                    abs_tol=1e-5,
                )
            ):
                gate_config_matched = False

            b_metrics = baseline_metrics[wl]
            b_nll = b_metrics["val_nll"]
            b_acc = b_metrics["val_acc"]

            if math.isfinite(c_nll) and math.isfinite(c_acc) and math.isfinite(b_nll) and math.isfinite(b_acc):
                nll_delta = c_nll - b_nll
                rel_nll_reduction_pct = ((b_nll - c_nll) / b_nll) * 100.0 if b_nll > 0 else 0.0
                acc_gain_pp = c_acc - b_acc

                per_workload_deltas[wl] = {
                    "state_ratio": derived_ratio,
                    "baseline_nll": b_nll,
                    "candidate_nll": c_nll,
                    "nll_delta": nll_delta,
                    "rel_nll_reduction_pct": rel_nll_reduction_pct,
                    "baseline_acc": b_acc,
                    "candidate_acc": c_acc,
                    "acc_gain_pp": acc_gain_pp,
                }

                if acc_gain_pp < -1.0:
                    acc_regressions_valid = False
                if rel_nll_reduction_pct < -5.0:
                    nll_regressions_valid = False
            else:
                acc_regressions_valid = False
                nll_regressions_valid = False
                gate_finite = False

        if derived_ratios and len(derived_ratios) == len(EXPECTED_WORKLOADS):
            max_state_ratio = max(derived_ratios.values())
        else:
            max_state_ratio = 1.0

        gate_state_ratio = bool(0.0 < max_state_ratio <= 0.5 and math.isfinite(max_state_ratio))
        gate_acc_regression = bool(acc_regressions_valid)
        gate_nll_regression = bool(nll_regressions_valid)

        if "mnist_wide" in per_workload_deltas:
            mw_delta = per_workload_deltas["mnist_wide"]
            mw_acc_gain = mw_delta["acc_gain_pp"]
            mw_nll_red = mw_delta["rel_nll_reduction_pct"]
            gate_baseline_worst_improvement = bool((mw_acc_gain >= 2.0) or (mw_nll_red >= 5.0))
        else:
            gate_baseline_worst_improvement = False

        gates = {
            "gate_finite": bool(gate_finite),
            "gate_test_sealed": bool(gate_test_sealed),
            "gate_config_matched": bool(gate_config_matched),
            "gate_state_ratio": bool(gate_state_ratio),
            "gate_acc_regression": bool(gate_acc_regression),
            "gate_nll_regression": bool(gate_nll_regression),
            "gate_baseline_worst_improvement": bool(gate_baseline_worst_improvement),
        }

        failed_gates = [g_name for g_name, g_pass in gates.items() if not g_pass]
        failed_gate_count = len(failed_gates)
        cand_pass = bool(failed_gate_count == 0)

        if len(per_workload_deltas) == len(EXPECTED_WORKLOADS):
            mean_rel_nll_reduction_pct = float(
                sum(d["rel_nll_reduction_pct"] for d in per_workload_deltas.values())
                / len(per_workload_deltas)
            )
            mean_acc_gain_pp = float(
                sum(d["acc_gain_pp"] for d in per_workload_deltas.values())
                / len(per_workload_deltas)
            )
        else:
            mean_rel_nll_reduction_pct = 0.0
            mean_acc_gain_pp = 0.0

        if 0.0 < max_state_ratio <= 1.0 and math.isfinite(max_state_ratio):
            state_efficiency_bonus = 10.0 * math.log2(1.0 / max_state_ratio)
        else:
            state_efficiency_bonus = 0.0

        score = (
            mean_rel_nll_reduction_pct
            + mean_acc_gain_pp
            + state_efficiency_bonus
            - (100.0 * failed_gate_count)
        )

        if not math.isfinite(score):
            score = -100.0 * max(1, failed_gate_count)

        candidate_evaluations[cand_id] = {
            "candidate_id": cand_id,
            "pass": cand_pass,
            "score": float(score),
            "failed_gates": failed_gates,
            "failed_gate_count": failed_gate_count,
            "gate_details": gates,
            "per_workload": per_workload_deltas,
            "state_ratios": derived_ratios,
            "summary_metrics": {
                "mean_rel_nll_reduction_pct": mean_rel_nll_reduction_pct,
                "mean_acc_gain_pp": mean_acc_gain_pp,
                "max_state_ratio": max_state_ratio,
                "state_efficiency_bonus": state_efficiency_bonus,
            },
        }

    passing_cand_ids = [
        c_id for c_id, c_eval in candidate_evaluations.items() if c_eval["pass"]
    ]

    if passing_cand_ids:
        selected_candidate_id = max(
            passing_cand_ids, key=lambda c_id: candidate_evaluations[c_id]["score"]
        )
        overall_pass = True
    else:
        selected_candidate_id = max(
            candidate_evaluations.keys(),
            key=lambda c_id: candidate_evaluations[c_id]["score"],
        )
        overall_pass = False

    selected_eval = candidate_evaluations[selected_candidate_id]

    evaluator_output = {
        "pass": overall_pass,
        "score": float(selected_eval["score"]),
        "selected_candidate_id": selected_candidate_id,
        "evaluator_version": EVALUATOR_VERSION,
        "source_paths": {
            "baseline_path": str(baseline_path),
            "candidate_path": str(candidate_path),
        },
        "baseline_policy": BASELINE_POLICY,
        "candidate_evaluations": candidate_evaluations,
        "per_workload_deltas": selected_eval["per_workload"],
        "state_ratios": selected_eval["state_ratios"],
    }

    if output_path is not None:
        save_json_atomic(evaluator_output, Path(output_path))

    return evaluator_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict Pareto Evaluator for Heavy Task PSO Autoresearch"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="benchmark_results/pso_v6_heavy_tasks.json",
        help="Path to baseline heavy tasks JSON artifact",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="Path to candidate heavy autoresearch JSON artifact",
    )
    parser.add_argument(
        "candidate_pos",
        nargs="?",
        type=str,
        default=None,
        help="Positional candidate JSON path fallback",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write evaluator result JSON",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cand_path = args.candidate or args.candidate_pos
    if not cand_path:
        parser.error("Candidate JSON path must be supplied via --candidate or positional argument.")

    out_path = Path(args.output) if args.output else None

    result = evaluate_heavy_autoresearch(
        baseline_path=args.baseline,
        candidate_path=cand_path,
        output_path=out_path,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
