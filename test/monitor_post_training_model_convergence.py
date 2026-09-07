"""Emit live, read-only TensorBoard progress for a convergence-study run."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

BASE_SEEDS = (501, 502, 503)
SWARM_SEEDS = (601, 602, 603)
CIFAR_WORKLOADS = ("cifar10_resnet18", "cifar10_resnet50")
YOLO_WORKLOAD = "voc_yolo11n"


def _cifar_seed(root: Path, seed: int) -> dict[str, Any]:
    baseline = root / f"baseline-{seed}.pt"
    searches = sum(
        (root / f"feature-{method}-{seed}-{swarm}.json").is_file()
        for method in ("pso", "random")
        for swarm in SWARM_SEEDS
    )
    controls = sum(
        (root / f"{method}-{seed}.pt").is_file()
        for method in ("feature-adam", "head-adam")
    )
    complete = baseline.is_file() and searches == 6 and controls == 2
    if complete:
        stage = "complete"
    elif not baseline.is_file():
        stage = "baseline_training"
    elif searches < 6:
        stage = f"pso_random_search_{searches}_of_6"
    else:
        stage = f"adam_controls_{controls}_of_2"
    return {
        "seed": seed,
        "stage": stage,
        "complete": complete,
        "searches": searches,
        "controls": controls,
    }


def _read_yolo_metrics(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = []
            for source in csv.DictReader(stream):
                rows.append(
                    {
                        key.strip(): float(value)
                        for key, value in source.items()
                        if key is not None
                        and value is not None
                        and value.strip()
                    }
                )
            return rows
    except (OSError, ValueError):
        return []


def _yolo_seed(
    root: Path,
    run_root: Path,
    seed: int,
) -> dict[str, Any]:
    baseline = root / "baselines" / str(seed) / "ema_fp32.pt"
    training_root = (
        run_root / "ultralytics" / f"base-{seed}-100e"
    )
    metrics = _read_yolo_metrics(training_root / "results.csv")
    baseline_complete = (
        baseline.is_file()
        and len(metrics) >= 100
        and (training_root / "weights" / "last.pt").is_file()
    )
    arm_root = root / "arms" / str(seed)
    searches = sum(
        (arm_root / f"feature_{method}-{swarm}.pt").is_file()
        for method in ("pso", "random")
        for swarm in SWARM_SEEDS
    )
    controls = sum(
        (arm_root / f"{method}.pt").is_file()
        for method in ("feature_adam", "head_adam")
    )
    complete = (arm_root / "record.json").is_file()
    if complete:
        stage = "complete"
    elif not baseline_complete:
        stage = f"baseline_training_epoch_{len(metrics)}_of_100"
    elif searches < 6:
        stage = f"pso_random_search_{searches}_of_6"
    elif controls < 2:
        stage = f"adam_controls_{controls}_of_2"
    else:
        stage = "selection"
    return {
        "seed": seed,
        "stage": stage,
        "complete": complete,
        "searches": searches,
        "controls": controls,
        "training_metrics": metrics,
    }


def snapshot(run_root: Path) -> dict[str, Any]:
    workloads = run_root / "workloads"
    status: dict[str, Any] = {}
    completed = 0
    for workload in CIFAR_WORKLOADS:
        seeds = [
            _cifar_seed(workloads / workload, seed)
            for seed in BASE_SEEDS
        ]
        completed += sum(item["complete"] for item in seeds)
        status[workload] = seeds
    yolo = [
        _yolo_seed(
            workloads / YOLO_WORKLOAD,
            run_root,
            seed,
        )
        for seed in BASE_SEEDS
    ]
    completed += sum(item["complete"] for item in yolo)
    status[YOLO_WORKLOAD] = yolo
    state_path = run_root / "state.json"
    state = "missing"
    if state_path.is_file():
        try:
            state = str(
                json.loads(
                    state_path.read_text(encoding="utf-8")
                ).get("state", "unknown")
            )
        except (OSError, ValueError):
            state = "unreadable"
    active = next(
        (
            f"{workload}/seed-{item['seed']}/{item['stage']}"
            for workload, items in status.items()
            for item in items
            if not item["complete"]
        ),
        "",
    )
    if not active:
        for workload in CIFAR_WORKLOADS:
            selected = sum(
                (
                    workloads
                    / workload
                    / f"selected-feature_{method}-{seed}.pt"
                ).is_file()
                for method in ("pso", "random")
                for seed in BASE_SEEDS
            )
            if selected < 6:
                active = (
                    f"{workload}/development_selection_"
                    f"{selected}_of_6"
                )
                break
    if not active:
        active = "development_complete"
    return {
        "state": state,
        "active": active,
        "completed_base_seeds": int(completed),
        "total_base_seeds": 9,
        "workloads": status,
        "observed_at": time.time(),
    }


def emit(
    writer: SummaryWriter,
    value: dict[str, Any],
    step: int,
) -> None:
    writer.add_scalar(
        "progress/completed_base_seeds",
        value["completed_base_seeds"],
        step,
    )
    writer.add_scalar(
        "progress/completion_fraction",
        value["completed_base_seeds"] / value["total_base_seeds"],
        step,
    )
    for workload, seeds in value["workloads"].items():
        writer.add_scalar(
            f"progress/{workload}/completed_base_seeds",
            sum(item["complete"] for item in seeds),
            step,
        )
    for item in value["workloads"][YOLO_WORKLOAD]:
        seed = item["seed"]
        for row in item["training_metrics"]:
            epoch = int(row["epoch"])
            for metric, metric_value in row.items():
                if metric in {"epoch", "time"}:
                    continue
                writer.add_scalar(
                    f"training/{YOLO_WORKLOAD}/seed_{seed}/{metric}",
                    metric_value,
                    epoch,
                )
    writer.add_text(
        "progress/current",
        f"`{value['active']}`",
        step,
    )
    writer.add_text(
        "progress/snapshot",
        f"```json\n{json.dumps(value, indent=2)}\n```",
        step,
    )
    writer.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    log_dir = args.run_root / "tensorboard"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"monitoring {args.run_root} -> {log_dir}", flush=True)
    step = 0
    try:
        while True:
            value = snapshot(args.run_root)
            emit(writer, value, step)
            print(
                json.dumps(
                    {
                        "active": value["active"],
                        "completed_base_seeds": value[
                            "completed_base_seeds"
                        ],
                        "state": value["state"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            step += 1
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
