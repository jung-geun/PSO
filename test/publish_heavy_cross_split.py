"""
Heavy PSO Cross-Split Experiment Results Publisher.

Protocol Version: HEAVY-PSO-CROSS-SPLIT-PUBLISH 1.0.0

Reads raw candidate and evaluation artifacts from a completed cross-split mission run,
validates integrity, accounting, test seals, and candidate/evaluation alignment across
all development variants, and exports deterministic public benchmark artifacts:
1. benchmark_results/pso_v7_heavy_cross_split.json
2. benchmark_results/pso_v7_heavy_cross_split.csv
3. history_plt/pso_v7_heavy_cross_split.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure test directory and repo root are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name != "PSO" else Path(__file__).resolve().parent
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_suite import save_json_atomic
from pso import __version__ as pso_version

PUBLISH_PROTOCOL_VERSION = "HEAVY-PSO-CROSS-SPLIT-PUBLISH 1.0.0"
DEFAULT_SOURCE_DIR = Path(".omc/autoresearch/heavy-pso-cross-split-robustness/runs/20260903T162504Z")
DEFAULT_JSON_OUTPUT = Path("benchmark_results/pso_v7_heavy_cross_split.json")
DEFAULT_CSV_OUTPUT = Path("benchmark_results/pso_v7_heavy_cross_split.csv")
DEFAULT_PLOT_OUTPUT = Path("history_plt/pso_v7_heavy_cross_split.png")

WORKLOADS = ["mnist_compact", "mnist_wide", "fashion_compact", "fashion_wide"]
BASELINE_METHODS = {
    "mnist_compact": "G8",
    "mnist_wide": "G5",
    "fashion_compact": "G8",
    "fashion_wide": "G5",
}
EXPECTED_DEV_SPLITS = [20260905, 20260906]
EXPECTED_DEV_SWARM_SEEDS = [101, 102, 103]
EXPECTED_VARIANTS_COUNT = 9
EXPECTED_VARIANT_IDS = (
    "iteration-0001-development",
    "iteration-0002-development",
    "iteration-0003-replica1-development",
    "iteration-0003-replica2-development",
    "iteration-0004-development",
    "iteration-0005-development",
    "iteration-0006-development",
    "iteration-0007-development",
    "iteration-0008-development",
)
EXPECTED_CELLS_PER_VARIANT = 8
EXPECTED_RUNS_PER_VARIANT = 48
EXPECTED_QUERIES_PER_VARIANT = 46080
EXPECTED_SAMPLES_PER_VARIANT = 460800000

EXPECTED_TOTAL_RUNS = EXPECTED_VARIANTS_COUNT * EXPECTED_RUNS_PER_VARIANT  # 432
EXPECTED_TOTAL_QUERIES = EXPECTED_VARIANTS_COUNT * EXPECTED_QUERIES_PER_VARIANT  # 414720
EXPECTED_TOTAL_SAMPLES = EXPECTED_VARIANTS_COUNT * EXPECTED_SAMPLES_PER_VARIANT  # 4147200000


def discover_and_load_variants(
    source_dir: Path,
) -> List[Tuple[Path, Path, Dict[str, Any], Dict[str, Any]]]:
    """
    Discovers candidate and evaluation JSON file pairs in source_dir.
    Supports both subdirectories (candidates/ & evaluations/) and direct directory structure.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    cand_dir = source_dir / "candidates"
    eval_dir = source_dir / "evaluations"

    if cand_dir.is_dir() and eval_dir.is_dir():
        candidate_files = sorted(cand_dir.glob("*.json"))
    else:
        candidate_files = sorted(source_dir.glob("*candidate*.json"))
        if not candidate_files:
            candidate_files = sorted(source_dir.glob("*.json"))

    if not candidate_files:
        raise ValueError(f"No candidate JSON files found in {source_dir}")

    observed_ids = tuple(path.stem for path in candidate_files)
    if observed_ids != EXPECTED_VARIANT_IDS:
        raise ValueError(
            f"Expected exact development variants {list(EXPECTED_VARIANT_IDS)}, "
            f"got {list(observed_ids)}"
        )

    pairs = []
    for cf in candidate_files:
        if cand_dir.is_dir() and eval_dir.is_dir():
            ef = eval_dir / cf.name
        else:
            ef_name = cf.name.replace("candidate", "evaluation")
            ef = source_dir / ef_name
            if not ef.exists():
                ef = cf

        if not ef.exists():
            raise FileNotFoundError(f"Missing corresponding evaluation file for candidate {cf.name}: {ef}")

        with cf.open("r", encoding="utf-8") as f:
            cdata = json.load(f)
        with ef.open("r", encoding="utf-8") as f:
            edata = json.load(f)

        pairs.append((cf, ef, cdata, edata))

    return pairs


def validate_variant_pair(
    cf_path: Path,
    ef_path: Path,
    cdata: Dict[str, Any],
    edata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validates each source candidate/evaluation artifact pair for expected IDs,
    phase, test seals, no development pass, cell count 8, and resource accounting.
    """
    variant_id = cf_path.stem

    # Phase check
    c_phase = cdata.get("phase")
    e_phase = edata.get("phase")
    if c_phase != "development":
        raise ValueError(f"[{variant_id}] Candidate phase must be 'development', got: {c_phase}")
    if e_phase not in (None, "development"):
        raise ValueError(f"[{variant_id}] Evaluation phase must be 'development', got: {e_phase}")

    # Official test sealed check
    if cdata.get("official_test_data_loaded") is not False:
        raise ValueError(f"[{variant_id}] candidate.official_test_data_loaded must be False")
    if cdata.get("official_test_evaluations") != 0:
        raise ValueError(f"[{variant_id}] candidate.official_test_evaluations must be 0")

    gates = edata.get("gates", {})
    if gates and "official_test_sealed" in gates:
        if not gates["official_test_sealed"].get("pass", False):
            raise ValueError(f"[{variant_id}] evaluation gate 'official_test_sealed' must pass")

    # No development pass check
    if edata.get("development_pass") is not False:
        raise ValueError(f"[{variant_id}] development_pass must be False for all variants")
    if edata.get("pass") is not False:
        raise ValueError(f"[{variant_id}] pass must be False for all variants")
    if edata.get("eligible_for_confirmation") is not False:
        raise ValueError(f"[{variant_id}] eligible_for_confirmation must be False for all variants")

    # Matching cell count 8
    summary_metrics = edata.get("summary_metrics", {})
    dev_cells = summary_metrics.get("development_cells")
    cell_metrics = edata.get("cell_metrics", [])
    if dev_cells != EXPECTED_CELLS_PER_VARIANT or len(cell_metrics) != EXPECTED_CELLS_PER_VARIANT:
        raise ValueError(
            f"[{variant_id}] Expected {EXPECTED_CELLS_PER_VARIANT} development cells, got "
            f"summary_metrics.development_cells={dev_cells}, len(cell_metrics)={len(cell_metrics)}"
        )

    # Resource accounting check across per-seed runs
    splits = cdata.get("splits", {})
    c_runs = 0
    c_queries = 0
    c_samples = 0
    c_test_evals = 0

    for split_key, split_data in splits.items():
        for role in ("baselines", "candidates"):
            for wl_key, wl_data in split_data.get(role, {}).items():
                for run in wl_data.get("per_seed_runs", []):
                    c_runs += 1
                    c_queries += run.get("total_queries", 0)
                    c_samples += run.get("total_sample_evaluations", 0)
                    c_test_evals += run.get("official_test_evaluations", 0)

    if c_runs != EXPECTED_RUNS_PER_VARIANT:
        raise ValueError(f"[{variant_id}] Expected {EXPECTED_RUNS_PER_VARIANT} runs, got {c_runs}")
    if c_queries != EXPECTED_QUERIES_PER_VARIANT:
        raise ValueError(f"[{variant_id}] Expected {EXPECTED_QUERIES_PER_VARIANT} total queries, got {c_queries}")
    if c_samples != EXPECTED_SAMPLES_PER_VARIANT:
        raise ValueError(f"[{variant_id}] Expected {EXPECTED_SAMPLES_PER_VARIANT} total sample evaluations, got {c_samples}")
    if c_test_evals != 0:
        raise ValueError(f"[{variant_id}] Official test evaluations must be 0, got {c_test_evals}")
    resource_totals = cdata.get("resource_totals")
    if not isinstance(resource_totals, dict):
        raise ValueError(f"[{variant_id}] candidate.resource_totals must be present")
    expected_resource_totals = {
        "total_runs": c_runs,
        "total_queries": c_queries,
        "total_samples_evaluated": c_samples,
        "official_test_evaluations": c_test_evals,
    }
    for field, expected in expected_resource_totals.items():
        if resource_totals.get(field) != expected:
            raise ValueError(
                f"[{variant_id}] candidate.resource_totals.{field} must be {expected}, "
                f"got {resource_totals.get(field)!r}"
            )
    wall_time = resource_totals.get("wall_time_sec")
    if (
        not isinstance(wall_time, (int, float))
        or isinstance(wall_time, bool)
        or not math.isfinite(float(wall_time))
        or wall_time < 0
    ):
        raise ValueError(
            f"[{variant_id}] candidate.resource_totals.wall_time_sec must be finite and non-negative"
        )


    return {
        "variant_id": variant_id,
        "candidate_file": cf_path.name,
        "evaluation_file": ef_path.name,
        "candidate_path": _repository_relative_path(cf_path),
        "evaluation_path": _repository_relative_path(ef_path),
        "phase": "development",
        "candidate_config": cdata.get("candidate_config", {}),
        "score": float(edata.get("score", 0.0)),
        "pass": False,
        "development_pass": False,
        "eligible_for_confirmation": False,
        "failed_hard_gate_count": edata.get("failed_hard_gate_count", 0),
        "failed_gates": edata.get("failed_gates", []),
        "summary_metrics": summary_metrics,
        "state_ratios": edata.get("state_ratios", {}),
        "cell_metrics": cell_metrics,
        "cdata": cdata,
        "edata": edata,
        "resources": {
            "runs": c_runs,
            "queries": c_queries,
            "sample_evaluations": c_samples,
            "wall_time_sec": float(wall_time),
            "official_test_evaluations": 0,
        },
    }


def validate_all_variants(
    variant_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validates cumulative resources across all variants, identifies the best-observed variant,
    and constructs cumulative summary dictionary.
    """
    observed_ids = tuple(v["variant_id"] for v in variant_summaries)
    if observed_ids != EXPECTED_VARIANT_IDS:
        raise ValueError(
            f"Expected exact development variants {list(EXPECTED_VARIANT_IDS)}, got {list(observed_ids)}"
        )

    total_runs = sum(v["resources"]["runs"] for v in variant_summaries)
    total_queries = sum(v["resources"]["queries"] for v in variant_summaries)
    total_samples = sum(v["resources"]["sample_evaluations"] for v in variant_summaries)
    official_test_evals = sum(v["resources"]["official_test_evaluations"] for v in variant_summaries)

    wall_time_sec = round(
        math.fsum(v["resources"]["wall_time_sec"] for v in variant_summaries),
        4,
    )
    if total_runs != EXPECTED_TOTAL_RUNS:
        raise ValueError(f"Cumulative total runs must be {EXPECTED_TOTAL_RUNS}, got {total_runs}")
    if total_queries != EXPECTED_TOTAL_QUERIES:
        raise ValueError(f"Cumulative total queries must be {EXPECTED_TOTAL_QUERIES}, got {total_queries}")
    if total_samples != EXPECTED_TOTAL_SAMPLES:
        raise ValueError(f"Cumulative total sample evaluations must be {EXPECTED_TOTAL_SAMPLES}, got {total_samples}")
    if official_test_evals != 0:
        raise ValueError(f"Cumulative official test evaluations must be 0, got {official_test_evals}")

    # Mark best-observed-but-rejected variant (highest evaluation score)
    best_variant = max(variant_summaries, key=lambda v: v["score"])
    for v in variant_summaries:
        v["is_best_observed"] = (v["variant_id"] == best_variant["variant_id"])

    return {
        "n_variants": len(variant_summaries),
        "total_runs": total_runs,
        "total_queries": total_queries,
        "total_sample_evaluations": total_samples,
        "official_test_evaluations": 0,
        "best_observed_variant_id": best_variant["variant_id"],
        "wall_time_sec": wall_time_sec,
        "best_observed_score": best_variant["score"],
    }


def build_publish_json(
    source_dir: Path,
    variant_summaries: List[Dict[str, Any]],
    cum_resources: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Constructs the compact JSON dictionary matching all publication criteria.
    """
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    clean_variants = []
    for v in sorted(variant_summaries, key=lambda item: item["variant_id"]):
        clean_variants.append({
            "variant_id": v["variant_id"],
            "candidate_file": v["candidate_file"],
            "evaluation_file": v["evaluation_file"],
            "phase": v["phase"],
            "candidate_config": v["candidate_config"],
            "score": v["score"],
            "pass": v["pass"],
            "development_pass": v["development_pass"],
            "eligible_for_confirmation": v["eligible_for_confirmation"],
            "failed_hard_gate_count": v["failed_hard_gate_count"],
            "failed_gates": v["failed_gates"],
            "is_best_observed": v["is_best_observed"],
            "summary_metrics": v["summary_metrics"],
            "state_ratios": v["state_ratios"],
            "cell_metrics": v["cell_metrics"],
            "resources": v["resources"],
        })

    payload = {
        "protocol_version": PUBLISH_PROTOCOL_VERSION,
        "pso_version": pso_version,
        "timestamp": now_iso,
        "mission_contract": {
            "phase": "development",
            "official_test_data_loaded": False,
            "official_test_evaluations": 0,
            "confirmation_executed": False,
            "retained_policy": None,
        },
        "official_test_data_loaded": False,
        "official_test_evaluations": 0,
        "confirmation_executed": False,
        "retained_policy": None,
        "verdict": {
            "status": "NO_RETAINED_POLICY_NO_CONFIRMATION",
            "retained_policy": None,
            "confirmation_executed": False,
            "official_test_data_loaded": False,
            "official_test_evaluations": 0,
            "best_observed_variant_id": cum_resources["best_observed_variant_id"],
            "best_observed_score": cum_resources["best_observed_score"],
            "description": (
                "All 9 development candidates failed the frozen evaluator gates (specifically maximum accuracy "
                "regression and/or development wide CNN improvement). No candidate qualified for confirmation. "
                "Confirmation split 20260907 and official test data remained completely unexecuted and sealed."
            ),
        },
        "source_provenance": {
            "source_dir": _repository_relative_path(source_dir),
            "n_variants": cum_resources["n_variants"],
            "candidate_files": [v["candidate_file"] for v in clean_variants],
            "evaluation_files": [v["evaluation_file"] for v in clean_variants],
        },
        "cumulative_resources": cum_resources,
        "total_runs": cum_resources["total_runs"],
        "total_wall_time_sec": cum_resources["wall_time_sec"],
        "total_queries": cum_resources["total_queries"],
        "total_sample_evaluations": cum_resources["total_sample_evaluations"],
        "variants": clean_variants,
    }
    return payload


def build_publish_csv(
    variant_summaries: List[Dict[str, Any]],
    csv_path: Path,
) -> None:
    """
    Writes CSV summary with header + exactly 72 data rows (9 variants x 2 splits x 4 workloads).
    Uses deterministic ordering and atomic writing.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant_id",
        "phase",
        "split_seed",
        "workload_id",
        "baseline_method",
        "baseline_acc",
        "candidate_acc",
        "acc_gain_pp",
        "baseline_nll",
        "candidate_nll",
        "nll_reduction_fraction",
        "state_ratio",
        "score",
        "pass",
        "is_best_observed",
        "candidate_path",
        "evaluation_path",
    ]

    rows = []
    # Sort variants deterministically
    sorted_variants = sorted(variant_summaries, key=lambda v: v["variant_id"])

    for v in sorted_variants:
        dev_ratios = v.get("state_ratios", {}).get("development", {})
        cell_metrics = v.get("cell_metrics", [])

        # Sort cells deterministically by split_seed then workload_id order
        def cell_sort_key(cm):
            wl_idx = WORKLOADS.index(cm["workload_id"]) if cm["workload_id"] in WORKLOADS else 99
            return (cm["split_seed"], wl_idx)

        sorted_cells = sorted(cell_metrics, key=cell_sort_key)

        for cm in sorted_cells:
            workload_id = cm["workload_id"]
            baseline_method = BASELINE_METHODS.get(workload_id, "G8" if "compact" in workload_id else "G5")
            st_ratio = dev_ratios.get(workload_id, 0.5) if isinstance(dev_ratios, dict) else 0.5

            row = {
                "variant_id": v["variant_id"],
                "phase": cm["phase"],
                "split_seed": cm["split_seed"],
                "workload_id": workload_id,
                "baseline_method": baseline_method,
                "baseline_acc": cm["baseline_acc"],
                "candidate_acc": cm["candidate_acc"],
                "acc_gain_pp": cm["acc_gain_pp"],
                "baseline_nll": cm["baseline_nll"],
                "candidate_nll": cm["candidate_nll"],
                "nll_reduction_fraction": cm["nll_reduction_fraction"],
                "state_ratio": st_ratio,
                "score": v["score"],
                "pass": False,
                "is_best_observed": v["is_best_observed"],
                "candidate_path": v["candidate_path"],
                "evaluation_path": v["evaluation_path"],
            }
            rows.append(row)

    tmp_csv = csv_path.with_suffix(".csv.tmp")
    with tmp_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tmp_csv.replace(csv_path)

def _repository_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def render_publish_plot(
    variant_summaries: List[Dict[str, Any]],
    plot_path: Path,
) -> None:
    """
    Renders readable 2-panel figure comparing mean gains and worst-cell regressions with frozen thresholds.
    Clearly marks all variants failed and confirmation withheld.
    """
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_variants = sorted(variant_summaries, key=lambda v: v["variant_id"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    variant_labels = [
        v["variant_id"].replace("-development", "").replace("iteration-", "iter-")
        for v in sorted_variants
    ]
    x = np.arange(len(variant_labels))
    width = 0.35

    # Panel 1: Development Grand Mean Performance Gains
    acc_gains = [v["summary_metrics"].get("development_grand_mean_accuracy_gain_pp", 0.0) for v in sorted_variants]
    nll_reductions = [v["summary_metrics"].get("development_grand_mean_nll_reduction_fraction", 0.0) * 100.0 for v in sorted_variants]

    ax1.bar(x - width/2, acc_gains, width, label="Grand Mean Acc Gain (pp)", color="#1f77b4")
    ax1.bar(x + width/2, nll_reductions, width, label="Grand Mean NLL Red. (%)", color="#2ca02c")

    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(variant_labels, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("Percentage Points (pp) / Percentage (%)")
    ax1.set_title("Panel A: Development Grand Mean Performance Gains")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Worst-Cell Regressions & Wide CNN Improvement vs Gate Thresholds
    worst_acc_regs = []
    worst_nll_regs = []
    mnist_wide_accs = []

    for v in sorted_variants:
        cell_acc_gains = [cm["acc_gain_pp"] for cm in v["cell_metrics"]]
        cell_nll_reds = [cm["nll_reduction_fraction"] * 100.0 for cm in v["cell_metrics"]]
        worst_acc_regs.append(min(cell_acc_gains))
        worst_nll_regs.append(min(cell_nll_reds))
        mnist_wide_accs.append(v["summary_metrics"].get("development_mnist_wide_accuracy_gain_pp", 0.0))

    ax2.plot(x, worst_acc_regs, "o-", color="#d62728", linewidth=2, label="Worst-Cell Acc Delta (pp)")
    ax2.plot(x, worst_nll_regs, "s--", color="#ff7f0e", linewidth=2, label="Worst-Cell NLL Red. (%)")
    ax2.plot(x, mnist_wide_accs, "^-.", color="#9467bd", linewidth=2, label="MNIST Wide Acc Gain (pp)")

    # Gate threshold lines
    ax2.axhline(-1.0, color="#d62728", linestyle=":", linewidth=1.5, label="Gate: Max Acc Reg. (-1.0 pp)")
    ax2.axhline(-5.0, color="#ff7f0e", linestyle=":", linewidth=1.5, label="Gate: Max NLL Reg. (-5.0%)")
    ax2.axhline(2.0, color="#2ca02c", linestyle="--", linewidth=1.5, label="Gate: MNIST Wide Gain (>= +2 pp)")
    panel_two_values = worst_acc_regs + worst_nll_regs + mnist_wide_accs + [-5.0, 2.0]
    panel_two_span = max(panel_two_values) - min(panel_two_values)
    panel_two_margin = max(1.0, panel_two_span * 0.08)
    ax2.set_ylim(
        min(panel_two_values) - panel_two_margin,
        max(panel_two_values) + panel_two_margin,
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(variant_labels, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("Metrics vs Gate Thresholds")
    ax2.set_title("Panel B: Worst-Cell Regressions & Wide CNN Improvement vs Gates")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Highlight best observed candidate
    best_idx = next(i for i, v in enumerate(sorted_variants) if v["is_best_observed"])
    ax1.annotate(
        f"Best observed, still rejected\nScore: {sorted_variants[best_idx]['score']:.2f}",
        xy=(best_idx, acc_gains[best_idx]),
        xycoords="data",
        xytext=(0.62, 0.94),
        textcoords="axes fraction",
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.95),
        fontsize=8,
        ha="center",
        va="top",
        weight="bold",
    )

    # Mission Outcome Text Banner
    fig.suptitle(
        "HEAVY PSO CROSS-SPLIT ROBUSTNESS MISSION REPORT\n"
        "STATUS: ALL 9 VARIANTS FAILED FROZEN EVALUATOR GATES  |  CONFIRMATION WITHHELD & SEALED OFFICIAL TEST UNEXECUTED  |  RETAINED POLICY: NONE",
        fontsize=11,
        weight="bold",
        color="#8b0000",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    tmp_plot = plot_path.with_name(plot_path.stem + "_tmp.png")
    fig.savefig(tmp_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)
    tmp_plot.replace(plot_path)


def publish_heavy_cross_split(
    source_dir: Union[str, Path] = DEFAULT_SOURCE_DIR,
    output_json: Union[str, Path] = DEFAULT_JSON_OUTPUT,
    output_csv: Union[str, Path] = DEFAULT_CSV_OUTPUT,
    output_plot: Union[str, Path] = DEFAULT_PLOT_OUTPUT,
) -> Dict[str, Any]:
    """
    Main programmatic interface for Heavy PSO Cross-Split results publication.
    Parses artifacts, validates all contracts and resource totals, and writes
    the compact JSON, CSV summary, and PNG plot atomically.
    """
    source_dir = Path(source_dir).resolve()
    output_json = Path(output_json).resolve()
    output_csv = Path(output_csv).resolve()
    output_plot = Path(output_plot).resolve()

    pairs = discover_and_load_variants(source_dir)

    variant_summaries = []
    for cf, ef, cdata, edata in pairs:
        summary = validate_variant_pair(cf, ef, cdata, edata)
        variant_summaries.append(summary)

    cum_resources = validate_all_variants(variant_summaries)

    json_payload = build_publish_json(source_dir, variant_summaries, cum_resources)
    save_json_atomic(json_payload, output_json)

    build_publish_csv(variant_summaries, output_csv)

    render_publish_plot(variant_summaries, output_plot)

    return json_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish Heavy PSO Cross-Split Robustness Mission Results"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Raw experiment runs directory (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help=f"Output compact JSON path (default: {DEFAULT_JSON_OUTPUT})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help=f"Output CSV summary path (default: {DEFAULT_CSV_OUTPUT})",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=DEFAULT_PLOT_OUTPUT,
        help=f"Output PNG plot path (default: {DEFAULT_PLOT_OUTPUT})",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    publish_heavy_cross_split(
        source_dir=args.source_dir,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_plot=args.output_plot,
    )
    print(f"[{PUBLISH_PROTOCOL_VERSION}] Successfully published cross-split results!")
    print(f"  JSON: {args.output_json}")
    print(f"  CSV:  {args.output_csv}")
    print(f"  Plot: {args.output_plot}")


if __name__ == "__main__":
    main()
