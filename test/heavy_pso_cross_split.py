"""
Heavy PSO Cross-Split Experiment Runner.

Protocol Version: HEAVY-PSO-CROSS-SPLIT 1.0.0

Runs matching baseline and candidate PSO experiments across development or confirmation
data splits under sealed official test conditions (official_test_evaluations = 0).

Phase Specifications:
- Development: split_seeds = (20260905, 20260906), swarm_seeds = (101, 102, 103)
- Confirmation: split_seeds = (20260907,), swarm_seeds = (111, 112, 113)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure test directory and repo root are in Python path
REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name != "PSO" else Path(__file__).resolve().parent
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_suite import (
    get_hardware_provenance,
    resolve_execution_device,
    save_json_atomic,
)
from heavy_pso_autoresearch import (
    DEFAULT_GEOMETRY_MULTIPLIER,
    parse_projection_seed_arg,
    parse_projection_scope_arg,
    validate_projection_scope_config,
    get_effective_projection_scope,
    run_heavy_pso_autoresearch,
)
from heavy_task_feasibility import (
    WORKLOADS,
    run_heavy_task_confirm,
)
from pso import __version__ as pso_version

PROTOCOL_VERSION = "HEAVY-PSO-CROSS-SPLIT 1.0.0"

PHASE_CONFIGS = {
    "development": {
        "split_seeds": [20260905, 20260906],
        "swarm_seeds": [101, 102, 103],
    },
    "confirmation": {
        "split_seeds": [20260907],
        "swarm_seeds": [111, 112, 113],
    },
}

WORKLOAD_BASELINE_METHODS = {
    "mnist_compact": "G8",
    "mnist_wide": "G5",
    "fashion_compact": "G8",
    "fashion_wide": "G5",
}

FROZEN_PROJECTION_SEEDS = {
    "mnist_compact": 1800044939,
    "mnist_wide": 592157828,
    "fashion_compact": 1363313651,
    "fashion_wide": 189641451,
}


def run_heavy_pso_cross_split(
    phase: str = "development",
    ratio: float = 0.5,
    geometry_policy: str = "baseline_aligned",
    projection_scope: Union[str, Dict[str, str]] = "global",
    projection_seed_mode: str = "explicit",
    projection_seed: Optional[Union[int, Dict[str, int]]] = None,
    geometry_multiplier: float = DEFAULT_GEOMETRY_MULTIPLIER,
    particles: int = 12,
    epochs: int = 80,
    subset_size: int = 10000,
    device_str: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Runs cross-split evaluation for development or confirmation phase.
    Enforces exact split and swarm seed contracts for each phase.
    Reruns matching baseline and candidate models per split seed.
    """
    projection_scope = parse_projection_scope_arg(projection_scope)
    validate_projection_scope_config(projection_scope)
    if phase not in PHASE_CONFIGS:
        raise ValueError(
            f"Invalid phase '{phase}'. Must be one of {list(PHASE_CONFIGS.keys())}"
        )

    phase_spec = PHASE_CONFIGS[phase]
    split_seeds = phase_spec["split_seeds"]
    swarm_seeds = phase_spec["swarm_seeds"]

    if projection_seed_mode == "explicit" and projection_seed is None:
        projection_seed = dict(FROZEN_PROJECTION_SEEDS)

    start_time = time.time()
    device = resolve_execution_device(device_str)
    hardware_info = get_hardware_provenance(device)

    if cache_dir is None:
        cache_dir = REPO_ROOT / "result" / "cache"

    splits_payload: Dict[str, Any] = {}
    total_runs = 0
    total_queries = 0
    total_samples_evaluated = 0

    for split_seed in split_seeds:
        # 1. Baseline runs for this split seed
        selected_baseline_methods = {
            wl_id: [WORKLOAD_BASELINE_METHODS[wl_id]] for wl_id in WORKLOADS
        }
        baseline_res = run_heavy_task_confirm(
            workloads=WORKLOADS,
            selected_methods=selected_baseline_methods,
            particles=particles,
            epochs=epochs,
            seeds=swarm_seeds,
            split_seed=split_seed,
            device=device,
            cache_dir=cache_dir,
        )

        # 2. Candidate runs for this split seed
        candidate_res = run_heavy_pso_autoresearch(
            ratios=[ratio],
            particles=particles,
            epochs=epochs,
            subset_size=subset_size,
            seeds=swarm_seeds,
            geometry_policy=geometry_policy,
            device_str=device_str,
            cache_dir=cache_dir,
            split_seed=split_seed,
            projection_scope=projection_scope,
            projection_seed_mode=projection_seed_mode,
            projection_seed=projection_seed,
            geometry_multiplier=geometry_multiplier,
        )

        # Extract candidate ratio payload
        candidate_ratio_runs = list(candidate_res["candidate_runs"].values())[0]

        baselines_split: Dict[str, Any] = {}
        candidates_split: Dict[str, Any] = {}
        dataset_fingerprints: Dict[str, str] = {}
        split_fingerprints: Dict[str, str] = {}

        for wl_id in WORKLOADS:
            b_method = WORKLOAD_BASELINE_METHODS[wl_id]
            b_entry = baseline_res[wl_id][b_method]
            baselines_split[wl_id] = b_entry

            c_entry = candidate_ratio_runs[wl_id]
            candidates_split[wl_id] = c_entry

            dataset_name = WORKLOADS[wl_id].dataset_name
            dataset_fingerprints[dataset_name] = c_entry["data_fingerprint"]
            split_fingerprints[dataset_name] = c_entry["split_fingerprint"]

            # Resource accumulation
            for r in b_entry["per_seed_runs"]:
                total_runs += 1
                total_queries += r["total_queries"]
                total_samples_evaluated += r["total_sample_evaluations"]
            for r in c_entry["per_seed_runs"]:
                total_runs += 1
                total_queries += r["total_queries"]
                total_samples_evaluated += r["total_sample_evaluations"]

        splits_payload[str(split_seed)] = {
            "split_seed": split_seed,
            "data_fingerprints": dataset_fingerprints,
            "split_fingerprints": split_fingerprints,
            "baselines": baselines_split,
            "candidates": candidates_split,
        }

    payload = {
        "version": PROTOCOL_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "phase": phase,
        "split_seeds": list(split_seeds),
        "swarm_seeds": list(swarm_seeds),
        "official_test_data_loaded": False,
        "official_test_evaluations": 0,
        "candidate_config": {
            "ratio": ratio,
            "geometry_policy": geometry_policy,
            "projection_scope": projection_scope,
            "projection_seed_mode": projection_seed_mode,
            "projection_seed": projection_seed,
            "geometry_multiplier": float(geometry_multiplier),
            "particles": particles,
            "epochs": epochs,
            "subset_size": subset_size,
        },
        "workloads": {
            wl_id: {
                "workload_id": wl_id,
                "dataset_name": wl_cfg.dataset_name,
                "model_name": wl_cfg.model_name,
                "baseline_method": WORKLOAD_BASELINE_METHODS[wl_id],
                "projection_scope": get_effective_projection_scope(projection_scope, wl_id),
                "effective_projection_seed": (
                    projection_seed[wl_id]
                    if isinstance(projection_seed, dict)
                    else projection_seed
                ),
            }
            for wl_id, wl_cfg in WORKLOADS.items()
        },
        "splits": splits_payload,
        "resource_totals": {
            "total_runs": total_runs,
            "total_queries": total_queries,
            "total_samples_evaluated": total_samples_evaluated,
            "official_test_evaluations": 0,
            "wall_time_sec": round(time.time() - start_time, 4),
        },
        "provenance": {
            "hardware": hardware_info,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pso_version": pso_version,
        },
    }

    if output_path is not None:
        save_json_atomic(payload, output_path)

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Heavy PSO Cross-Split Experiment Runner (Development / Confirmation)"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="development",
        choices=list(PHASE_CONFIGS.keys()),
        help="Experiment phase ('development' or 'confirmation')",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Dataset cache directory")
    parser.add_argument("--output", type=str, default=None, help="Output artifact JSON path")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Subspace ratio for candidate PSO (default: 0.5)",
    )
    parser.add_argument(
        "--geometry-policy",
        type=str,
        default="baseline_aligned",
        help="Geometry policy (default: 'baseline_aligned')",
    )
    parser.add_argument(
        "--projection-scope",
        type=parse_projection_scope_arg,
        default="global",
        help="Projection scope ('global', 'tensor_local', 'balanced_global', 'two_hash_global', 'largest_tensor_hash', 'largest_tensor_row_hash', 'adjacent_pair', 'adjacent_difference', or workload dict)",
    )
    parser.add_argument(
        "--projection-seed-mode",
        type=str,
        default="explicit",
        help="Projection seed mode (default: 'explicit')",
    )
    parser.add_argument(
        "--projection-seed",
        type=parse_projection_seed_arg,
        default=None,
        help="Exact projection seed (int or dict) for explicit mode",
    )
    parser.add_argument(
        "--geometry-multiplier",
        type=float,
        default=DEFAULT_GEOMETRY_MULTIPLIER,
        help="Geometry multiplier (default: 1.0)",
    )
    parser.add_argument("--particles", type=int, default=12, help="Swarm size (default: 12)")
    parser.add_argument("--epochs", type=int, default=80, help="PSO epochs (default: 80)")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=10000,
        help="Subset size (default: 10000)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cache_path = Path(args.cache_dir) if args.cache_dir else None
    out_path = Path(args.output) if args.output else None

    run_heavy_pso_cross_split(
        phase=args.phase,
        ratio=args.ratio,
        geometry_policy=args.geometry_policy,
        projection_scope=args.projection_scope,
        projection_seed_mode=args.projection_seed_mode,
        projection_seed=args.projection_seed,
        geometry_multiplier=args.geometry_multiplier,
        particles=args.particles,
        epochs=args.epochs,
        subset_size=args.subset_size,
        device_str=args.device,
        cache_dir=cache_path,
        output_path=out_path,
    )


if __name__ == "__main__":
    main()
