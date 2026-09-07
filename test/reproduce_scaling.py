"""
Adaptive Moment 120-Particle x 80-Epoch MNIST Scaling Replication Check

Validates the published 120-particle x 80-epoch fixed-epoch Adaptive Moment MNIST scaling result.
Performs exact replay on seeds 71-75 and fresh independent cohort evaluation on seeds 81-85.

Predeclared Acceptance Criteria:
1. Exact Replay (seeds 71-75): Max per-seed absolute test accuracy delta <= 0.005 (0.5%p).
2. Independent Cohort (seeds 81-85): Mean test accuracy absolute difference <= 0.03 (3%p)
   AND 95% t-confidence intervals overlap between baseline and independent cohorts.
"""

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from sklearn.decomposition import PCA

# Path setup for imports from test/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_suite import (
    calc_stats,
    compute_data_fingerprint,
    get_hardware_provenance,
    resolve_execution_device,
    save_json_atomic,
)
from pso import __version__ as pso_version
from tuning_suite import (
    TUNING_PROTOCOL_VERSION,
    CandidateConfig,
    get_mnist_raw_data,
    get_search_candidates,
    run_single_experiment,
)

REPLAY_TOLERANCE = 0.005
INDEPENDENT_MEAN_MARGIN = 0.03
REPLICATION_PROTOCOL_VERSION = "1.0.0"
REPLAY_SEEDS = [71, 72, 73, 74, 75]
INDEPENDENT_SEEDS = [81, 82, 83, 84, 85]


def validate_and_load_baseline(
    baseline_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], CandidateConfig, str]:
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline JSON file not found at: {baseline_path}")

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    if baseline_data.get("tuning_protocol_version") != TUNING_PROTOCOL_VERSION:
        raise ValueError(
            "Baseline tuning protocol mismatch: "
            f"expected {TUNING_PROTOCOL_VERSION}, "
            f"got {baseline_data.get('tuning_protocol_version')}"
        )
    if baseline_data.get("quick") is not False:
        raise ValueError("Replication requires the full, non-quick tuning baseline.")

    winners = baseline_data.get("winners", {})
    if "adaptive_moment" not in winners:
        raise ValueError(f"Baseline JSON {baseline_path} missing 'adaptive_moment' winner entry.")

    am_winner_info = winners["adaptive_moment"]
    winner_label = am_winner_info.get("candidate_label")

    all_candidates = get_search_candidates()
    am_candidates = all_candidates.get("adaptive_moment", [])
    winner_cfg = None
    for cfg in am_candidates:
        if cfg.candidate_label == winner_label:
            winner_cfg = cfg
            break

    if winner_cfg is None:
        raise ValueError(
            f"Could not find CandidateConfig matching label '{winner_label}' in search candidates."
        )
    expected_optimizer_config = winner_cfg.to_optimizer_kwargs(quick=False)
    if am_winner_info.get("config") != expected_optimizer_config:
        raise ValueError(
            "Adaptive Moment winner configuration in the baseline no longer matches "
            f"CandidateConfig '{winner_label}'."
        )


    scaling_runs = baseline_data.get("scaling_runs", [])
    baseline_records = []
    for r in scaling_runs:
        if (
            r.get("completed")
            and r.get("method") == "adaptive_moment"
            and r.get("candidate_label") == winner_label
            and r.get("n_particles") == 120
            and r.get("epochs") == 80
            and r.get("regimen") == "fixed_epoch"
            and r.get("seed") in REPLAY_SEEDS
        ):
            baseline_records.append(r)

    baseline_records.sort(key=lambda x: x["seed"])

    if len(baseline_records) != 5:
        raise ValueError(
            f"Expected exactly 5 baseline records for seeds {REPLAY_SEEDS}, "
            f"found {len(baseline_records)} in {baseline_path}."
        )

    expected_seeds = sorted(REPLAY_SEEDS)
    actual_seeds = [r["seed"] for r in baseline_records]
    if actual_seeds != expected_seeds:
        raise ValueError(f"Baseline seeds mismatch: expected {expected_seeds}, got {actual_seeds}")

    expected_fp = baseline_data.get("split_fingerprints", {}).get("full")
    if not isinstance(expected_fp, str) or not expected_fp:
        raise ValueError("Baseline JSON is missing split_fingerprints.full.")
    for r in baseline_records:
        if r.get("data_fingerprint") != expected_fp:
            raise ValueError(
                f"Baseline run seed {r['seed']} data_fingerprint {r.get('data_fingerprint')} "
                f"does not match split_fingerprints.full {expected_fp}"
            )
        run_config = r.get("config", {})
        for key, value in expected_optimizer_config.items():
            if run_config.get(key) != value:
                raise ValueError(
                    f"Baseline run seed {r['seed']} config[{key!r}]={run_config.get(key)!r} "
                    f"does not match selected winner value {value!r}."
                )
        expected_run_config = {
            "n_particles": 120,
            "epochs": 80,
            "batch_size": 1000,
            "renewal": "loss",
        }
        for key, value in expected_run_config.items():
            if run_config.get(key) != value:
                raise ValueError(
                    f"Baseline run seed {r['seed']} config[{key!r}]={run_config.get(key)!r}; "
                    f"expected {value!r}."
                )

    return baseline_data, baseline_records, winner_cfg, expected_fp


def prepare_full_pca_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    x_train_raw, x_test_raw, y_train_3000, y_test_1000 = get_mnist_raw_data()
    pca_full = PCA(n_components=32, whiten=True, random_state=42)
    x_full_tr = torch.tensor(pca_full.fit_transform(x_train_raw), dtype=torch.float32)
    x_full_test = torch.tensor(pca_full.transform(x_test_raw), dtype=torch.float32)
    data_fp = compute_data_fingerprint(x_full_tr, x_full_test, y_train_3000, y_test_1000)
    return x_full_tr, y_train_3000, x_full_test, y_test_1000, data_fp


def run_replication_cohort(
    cfg: CandidateConfig,
    seeds: List[int],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    device: torch.device,
    data_fp: str,
    run_type: str,
) -> List[Dict[str, Any]]:
    runs = []
    for seed in seeds:
        res = run_single_experiment(
            cfg=cfg,
            seed=seed,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            y_eval=y_eval,
            n_particles=120,
            epochs=80,
            batch_size=1000,
            device=device,
            quick=False,
            eval_metric_name="test",
            data_fp=data_fp,
            run_type=run_type,
            extra_meta={"regimen": "fixed_epoch"},
        )
        runs.append(res)
    return runs


def evaluate_replication(
    baseline_records: List[Dict[str, Any]],
    replay_runs: List[Dict[str, Any]],
    independent_runs: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float], Dict[str, float]]:
    base_acc_by_seed = {r["seed"]: float(r["test_acc"]) for r in baseline_records}
    replay_acc_by_seed = {r["seed"]: float(r["test_acc"]) for r in replay_runs}
    expected_seeds = set(REPLAY_SEEDS)
    if set(base_acc_by_seed) != expected_seeds or set(replay_acc_by_seed) != expected_seeds:
        raise ValueError("Baseline and replay cohorts must each contain exactly seeds 71-75.")
    if len(independent_runs) != len(INDEPENDENT_SEEDS) or {
        r["seed"] for r in independent_runs
    } != set(INDEPENDENT_SEEDS):
        raise ValueError("Independent cohort must contain exactly seeds 81-85.")

    baseline_model_fp = {r["seed"]: r.get("model_fingerprint") for r in baseline_records}
    replay_model_fp = {r["seed"]: r.get("model_fingerprint") for r in replay_runs}
    replay_model_fingerprint_match = baseline_model_fp == replay_model_fp
    replay_deltas = {}
    max_replay_delta = 0.0
    for seed in sorted(base_acc_by_seed.keys()):
        b_acc = base_acc_by_seed[seed]
        r_acc = replay_acc_by_seed[seed]
        delta = abs(r_acc - b_acc)
        replay_deltas[str(seed)] = round(delta, 6)
        if delta > max_replay_delta:
            max_replay_delta = delta

    replay_pass = bool(max_replay_delta <= REPLAY_TOLERANCE)

    baseline_accs = [base_acc_by_seed[s] for s in sorted(base_acc_by_seed.keys())]
    replay_accs = [replay_acc_by_seed[s] for s in sorted(replay_acc_by_seed.keys())]
    indep_accs = [float(r["test_acc"]) for r in independent_runs]

    baseline_stats = calc_stats(baseline_accs)
    replay_stats = calc_stats(replay_accs)
    independent_stats = calc_stats(indep_accs)

    indep_mean_diff = abs(independent_stats["mean"] - baseline_stats["mean"])
    independent_mean_pass = bool(indep_mean_diff <= INDEPENDENT_MEAN_MARGIN)

    baseline_ci_low = round(baseline_stats["mean"] - baseline_stats["ci95_t"], 6)
    baseline_ci_high = round(baseline_stats["mean"] + baseline_stats["ci95_t"], 6)

    indep_ci_low = round(independent_stats["mean"] - independent_stats["ci95_t"], 6)
    indep_ci_high = round(independent_stats["mean"] + independent_stats["ci95_t"], 6)

    ci_overlap_pass = bool(max(baseline_ci_low, indep_ci_low) <= min(baseline_ci_high, indep_ci_high))
    independent_pass = bool(independent_mean_pass and ci_overlap_pass)

    comparison = {
        "replay_per_seed_deltas": replay_deltas,
        "replay_max_abs_delta": round(max_replay_delta, 6),
        "replay_model_fingerprint_match": replay_model_fingerprint_match,
        "replay_pass": bool(replay_pass and replay_model_fingerprint_match),
        "independent_mean_abs_diff": round(indep_mean_diff, 6),
        "independent_mean_pass": independent_mean_pass,
        "baseline_ci95_t_interval": [baseline_ci_low, baseline_ci_high],
        "independent_ci95_t_interval": [indep_ci_low, indep_ci_high],
        "ci_overlap_pass": ci_overlap_pass,
        "independent_pass": independent_pass,
        "overall_pass": bool(
            replay_pass and replay_model_fingerprint_match and independent_pass
        ),
    }

    return comparison, baseline_stats, replay_stats, independent_stats


def write_replication_csv(
    baseline_records: List[Dict[str, Any]],
    replay_runs: List[Dict[str, Any]],
    independent_runs: List[Dict[str, Any]],
    output_csv: Path,
):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cohort",
        "method",
        "candidate_label",
        "regimen",
        "seed",
        "n_particles",
        "epochs",
        "particle_epochs",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "test_mse",
        "fit_time_sec",
        "data_fingerprint",
        "model_fingerprint",
        "device",
        "completed",
        "error",
    ]
    all_rows = []
    for r in baseline_records:
        row = dict(r)
        row["cohort"] = "baseline"
        all_rows.append(row)
    for r in replay_runs:
        row = dict(r)
        row["cohort"] = "replay"
        all_rows.append(row)
    for r in independent_runs:
        row = dict(r)
        row["cohort"] = "independent"
        all_rows.append(row)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(
        description="Replicate and verify published Adaptive Moment 120p x 80e MNIST scaling result"
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=Path("benchmark_results/pso_v4_tuning.json"),
        help="Path to baseline tuning JSON",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmark_results/pso_v4_120p80_replication.json"),
        help="Path for replication output JSON",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_results/pso_v4_120p80_replication.csv"),
        help="Path for replication output CSV",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device (cpu, cuda, mps)",
    )

    args = parser.parse_args()
    device = resolve_execution_device(args.device)

    print("=== Adaptive Moment 120p x 80e Replication Check ===")
    print(f"Device: {device}")
    print(f"Baseline JSON: {args.baseline_json}")
    print(f"Output JSON: {args.output_json}")
    print(f"Output CSV: {args.output_csv}")

    baseline_data, baseline_records, winner_cfg, expected_fp = validate_and_load_baseline(
        args.baseline_json
    )
    print(f"Validated baseline winner '{winner_cfg.candidate_label}' across 5 records.")
    if baseline_data.get("device") != str(device):
        raise ValueError(
            f"Exact replay requires baseline device {baseline_data.get('device')!r}; "
            f"got {str(device)!r}."
        )
    if baseline_data.get("pso_version") != pso_version:
        raise ValueError(
            f"Exact replay requires pso version {baseline_data.get('pso_version')!r}; "
            f"got {pso_version!r}."
        )
    if baseline_data.get("torch_version") != torch.__version__:
        raise ValueError(
            f"Exact replay requires torch version {baseline_data.get('torch_version')!r}; "
            f"got {torch.__version__!r}."
        )

    x_full_tr, y_train_3000, x_full_test, y_test_1000, data_fp = prepare_full_pca_data()
    if expected_fp and data_fp != expected_fp:
        raise ValueError(
            f"Reconstructed data fingerprint {data_fp} does not match baseline {expected_fp}"
        )
    print(f"Reconstructed PCA32 data (fingerprint: {data_fp})")

    print("\n--- Running Cohort 1: Exact Replay (Seeds 71-75) ---")
    replay_runs = run_replication_cohort(
        cfg=winner_cfg,
        seeds=REPLAY_SEEDS,
        x_train=x_full_tr,
        y_train=y_train_3000,
        x_eval=x_full_test,
        y_eval=y_test_1000,
        device=device,
        data_fp=data_fp,
        run_type="replication_replay",
    )

    print("\n--- Running Cohort 2: Independent Fresh Seeds (Seeds 81-85) ---")
    independent_runs = run_replication_cohort(
        cfg=winner_cfg,
        seeds=INDEPENDENT_SEEDS,
        x_train=x_full_tr,
        y_train=y_train_3000,
        x_eval=x_full_test,
        y_eval=y_test_1000,
        device=device,
        data_fp=data_fp,
        run_type="replication_independent",
    )

    comparison, baseline_stats, replay_stats, independent_stats = evaluate_replication(
        baseline_records, replay_runs, independent_runs
    )

    payload = {
        "replication_protocol_version": REPLICATION_PROTOCOL_VERSION,
        "source_tuning_protocol_version": baseline_data["tuning_protocol_version"],
        "pso_version": pso_version,
        "torch_version": torch.__version__,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "hardware": get_hardware_provenance(device),
        "baseline_json": str(args.baseline_json),
        "source_tuning_timestamp": baseline_data.get("timestamp"),
        "candidate_label": winner_cfg.candidate_label,
        "config": winner_cfg.to_optimizer_kwargs(),
        "data_fingerprint": data_fp,
        "criteria": {
            "replay_seeds": REPLAY_SEEDS,
            "replay_max_abs_delta_tolerance": REPLAY_TOLERANCE,
            "require_replay_model_fingerprint_match": True,
            "independent_seeds": INDEPENDENT_SEEDS,
            "independent_mean_abs_diff_margin": INDEPENDENT_MEAN_MARGIN,
            "require_ci_overlap": True,
        },
        "summaries": {
            "baseline": baseline_stats,
            "replay": replay_stats,
            "independent": independent_stats,
        },
        "comparison": comparison,
        "baseline_runs": baseline_records,
        "replay_runs": replay_runs,
        "independent_runs": independent_runs,
        "completed": True,
        "error": None,
    }

    save_json_atomic(payload, args.output_json)
    write_replication_csv(baseline_records, replay_runs, independent_runs, args.output_csv)

    print("\n=== Replication Results Summary ===")
    print(f"Baseline Mean Test Acc:    {baseline_stats['mean']:.4f} ± {baseline_stats['std']:.4f}")
    print(f"Replay Mean Test Acc:      {replay_stats['mean']:.4f} ± {replay_stats['std']:.4f}")
    print(f"Independent Mean Test Acc: {independent_stats['mean']:.4f} ± {independent_stats['std']:.4f}")
    print(f"Max Replay Delta: {comparison['replay_max_abs_delta']:.6f} (Limit: {REPLAY_TOLERANCE}) -> Pass: {comparison['replay_pass']}")
    print(f"Indep Mean Diff:  {comparison['independent_mean_abs_diff']:.6f} (Limit: {INDEPENDENT_MEAN_MARGIN}) -> Pass: {comparison['independent_mean_pass']}")
    print(f"CI Overlap Pass:  {comparison['ci_overlap_pass']} (Baseline CI: {comparison['baseline_ci95_t_interval']}, Indep CI: {comparison['independent_ci95_t_interval']})")
    print(f"OVERALL PASS:     {comparison['overall_pass']}")

    if not comparison["overall_pass"]:
        print("\nREPLICATION CHECK FAILED!")
        sys.exit(1)
    else:
        print("\nREPLICATION CHECK PASSED!")


if __name__ == "__main__":
    main()
