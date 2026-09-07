"""
Unit tests for Heavy PSO Cross-Split Results Publisher.

Covers:
1. Valid publication pipeline execution on synthetic 9-variant cross-split source data.
2. Verification of compact JSON schema, cumulative resources (432 runs, 414720 queries, 4147200000 samples),
   official test seals (0 evaluations), confirmation_executed=false, retained_policy=null.
3. Verification of exact CSV output shape (72 data rows) and deterministic byte-for-byte reproducibility.
4. Validation error enforcement for cell count mismatch, official test unsealing, unexpected development pass,
   and variant count mismatch.
5. CLI entrypoint invocation.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

# Ensure test directory and repo root are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name != "PSO" else Path(__file__).resolve().parent
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import publish_heavy_cross_split as publisher
from pso import __version__ as pso_version


def make_synthetic_source(
    source_path: Path,
    num_variants: int = 9,
    cell_count: int = 8,
    test_evals: int = 0,
    test_loaded: bool = False,
    dev_pass: bool = False,
    queries_per_run: int = 960,
    samples_per_run: int = 9600000,
) -> Path:
    """
    Creates a synthetic cross-split experiment run directory with candidates/ and evaluations/
    matching expected structure for testing publisher integrity checks.
    """
    cand_dir = source_path / "candidates"
    eval_dir = source_path / "evaluations"
    cand_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    splits = [20260905, 20260906]
    workloads = ["mnist_compact", "mnist_wide", "fashion_compact", "fashion_wide"]
    swarm_seeds = [101, 102, 103]

    for variant_id in publisher.EXPECTED_VARIANT_IDS[:num_variants]:
        cf_path = cand_dir / f"{variant_id}.json"
        ef_path = eval_dir / f"{variant_id}.json"

        # Construct Candidate Artifact
        splits_dict = {}
        for split_seed in splits:
            split_key = str(split_seed)
            baselines_dict = {}
            candidates_dict = {}

            for wl in workloads:
                baseline_method = "G8" if "compact" in wl else "G5"

                def make_runs():
                    runs_list = []
                    for seed in swarm_seeds:
                        runs_list.append({
                            "seed": seed,
                            "val_selected_loss": 0.50,
                            "val_selected_acc": 80.0,
                            "gbest_loss": 0.48,
                            "gbest_acc": 81.0,
                            "wall_time_sec": 0.01,
                            "optimization_wall_time_sec": 0.01,
                            "validation_wall_time_sec": 0.001,
                            "throughput_samples_per_sec": 1000000.0,
                            "total_queries": queries_per_run,
                            "total_sample_evaluations": samples_per_run,
                            "official_test_evaluations": test_evals,
                            "val_metrics": {"brier": 0.1, "ece": 0.02},
                            "is_finite": True,
                            "core_swarm_state_bytes": 1000,
                        })
                    return runs_list

                baselines_dict[wl] = {
                    "workload_id": wl,
                    "method_id": baseline_method,
                    "subset_size": 10000,
                    "particles": 12,
                    "epochs": 80,
                    "seeds": swarm_seeds,
                    "split_seed": split_seed,
                    "data_fingerprint": f"data-fp-{wl}-{split_seed}",
                    "split_fingerprint": f"split-fp-{wl}-{split_seed}",
                    "per_seed_runs": make_runs(),
                }
                candidates_dict[wl] = {
                    "workload_id": wl,
                    "ratio": 0.5,
                    "subset_size": 10000,
                    "particles": 12,
                    "epochs": 80,
                    "seeds": swarm_seeds,
                    "split_seed": split_seed,
                    "data_fingerprint": f"data-fp-{wl}-{split_seed}",
                    "split_fingerprint": f"split-fp-{wl}-{split_seed}",
                    "per_seed_runs": make_runs(),
                }

            splits_dict[split_key] = {
                "split_seed": split_seed,
                "baselines": baselines_dict,
                "candidates": candidates_dict,
            }

        candidate_payload = {
            "version": "HEAVY-PSO-CROSS-SPLIT 1.0.0",
            "protocol_version": "HEAVY-PSO-CROSS-SPLIT 1.0.0",
            "phase": "development",
            "split_seeds": splits,
            "swarm_seeds": swarm_seeds,
            "official_test_data_loaded": test_loaded,
            "official_test_evaluations": test_evals * 48,
            "candidate_config": {
                "ratio": 0.5,
                "geometry_policy": "baseline_aligned",
                "particles": 12,
                "epochs": 80,
                "subset_size": 10000,
            },
            "workloads": {wl: {"workload_id": wl, "baseline_method": "G8" if "compact" in wl else "G5"} for wl in workloads},
            "splits": splits_dict,
            "resource_totals": {
                "total_runs": 48,
                "total_queries": queries_per_run * 48,
                "total_samples_evaluated": samples_per_run * 48,
                "official_test_evaluations": test_evals * 48,
                "wall_time_sec": 1.0,
            },
        }

        # Construct Evaluation Artifact
        cell_metrics = []
        for s_idx, split_seed in enumerate(splits):
            for wl in workloads:
                cell_metrics.append({
                    "phase": "development",
                    "split_seed": split_seed,
                    "workload_id": wl,
                    "baseline_acc": 80.0,
                    "candidate_acc": 80.5,
                    "baseline_nll": 0.50,
                    "candidate_nll": 0.49,
                    "acc_gain_pp": 0.5,
                    "nll_reduction_fraction": 0.02,
                })

        cell_metrics = cell_metrics[:cell_count]

        evaluation_payload = {
            "pass": False,
            "development_pass": dev_pass,
            "eligible_for_confirmation": False,
            "score": -300.0 + publisher.EXPECTED_VARIANT_IDS.index(variant_id) * 10.0,
            "evaluator_version": "HEAVY-PSO-CROSS-SPLIT-EVALUATOR 1.0.0",
            "failed_hard_gate_count": 3,
            "failed_gates": ["maximum_accuracy_regression_percentage_points_each_split_workload"],
            "gates": {
                "official_test_sealed": {
                    "pass": not test_loaded and test_evals == 0,
                }
            },
            "summary_metrics": {
                "development_cells": len(cell_metrics),
                "development_grand_mean_accuracy_gain_pp": 0.5,
                "development_grand_mean_nll_reduction_fraction": 0.02,
                "development_mnist_wide_accuracy_gain_pp": -0.5,
                "development_mnist_wide_nll_reduction_fraction": -0.01,
            },
            "state_ratios": {
                "development": {
                    "mnist_compact": 0.4918032786885246,
                    "mnist_wide": 0.5,
                    "fashion_compact": 0.4918032786885246,
                    "fashion_wide": 0.5,
                }
            },
            "cell_metrics": cell_metrics,
        }

        with cf_path.open("w", encoding="utf-8") as f:
            json.dump(candidate_payload, f, indent=2)
        with ef_path.open("w", encoding="utf-8") as f:
            json.dump(evaluation_payload, f, indent=2)

    return source_path


def test_publish_synthetic_success(tmp_path):
    """Verifies successful end-to-end publication on valid synthetic 9-variant cross-split source data."""
    source_dir = make_synthetic_source(tmp_path / "source")
    out_json = tmp_path / "pso_v7_heavy_cross_split.json"
    out_csv = tmp_path / "pso_v7_heavy_cross_split.csv"
    out_plot = tmp_path / "pso_v7_heavy_cross_split.png"

    payload = publisher.publish_heavy_cross_split(
        source_dir=source_dir,
        output_json=out_json,
        output_csv=out_csv,
        output_plot=out_plot,
    )

    # 1. JSON Verification
    assert out_json.is_file()
    assert payload["protocol_version"] == publisher.PUBLISH_PROTOCOL_VERSION
    assert payload["pso_version"] == pso_version
    assert payload["official_test_data_loaded"] is False
    assert payload["official_test_evaluations"] == 0
    assert payload["confirmation_executed"] is False
    assert payload["retained_policy"] is None

    assert payload["total_runs"] == 432
    assert payload["total_queries"] == 414720
    assert payload["total_sample_evaluations"] == 4147200000
    assert payload["total_wall_time_sec"] == 9.0
    assert len(payload["variants"]) == 9
    assert payload["verdict"]["status"] == "NO_RETAINED_POLICY_NO_CONFIRMATION"

    # Best-observed variant should be the final expected variant.
    best_v = [v for v in payload["variants"] if v["is_best_observed"]]
    assert len(best_v) == 1
    assert best_v[0]["variant_id"] == publisher.EXPECTED_VARIANT_IDS[-1]

    # 2. CSV Verification
    assert out_csv.is_file()
    with out_csv.open("r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    # 1 header line + 72 data rows = 73 lines
    assert len(reader) == 73
    header = reader[0]
    assert "variant_id" in header
    assert "baseline_acc" in header
    assert "candidate_acc" in header
    assert "acc_gain_pp" in header

    # 3. Plot Verification
    assert out_plot.is_file()
    assert out_plot.stat().st_size > 0


def test_publish_mismatch_cell_count(tmp_path):
    """Verifies ValueError when an evaluation artifact has a cell count other than 8."""
    source_dir = make_synthetic_source(tmp_path / "source", cell_count=7)
    out_json = tmp_path / "out.json"
    out_csv = tmp_path / "out.csv"
    out_plot = tmp_path / "out.png"

    with pytest.raises(ValueError, match="Expected 8 development cells"):
        publisher.publish_heavy_cross_split(source_dir, out_json, out_csv, out_plot)


def test_publish_official_test_unsealed(tmp_path):
    """Verifies ValueError when official test evaluations > 0 or official_test_data_loaded is True."""
    source_dir = make_synthetic_source(tmp_path / "source", test_evals=10)
    out_json = tmp_path / "out.json"
    out_csv = tmp_path / "out.csv"
    out_plot = tmp_path / "out.png"

    with pytest.raises(ValueError, match="official_test_evaluations must be 0"):
        publisher.publish_heavy_cross_split(source_dir, out_json, out_csv, out_plot)


def test_publish_unexpected_pass(tmp_path):
    """Verifies ValueError when development_pass is True."""
    source_dir = make_synthetic_source(tmp_path / "source", dev_pass=True)
    out_json = tmp_path / "out.json"
    out_csv = tmp_path / "out.csv"
    out_plot = tmp_path / "out.png"

    with pytest.raises(ValueError, match="development_pass must be False"):
        publisher.publish_heavy_cross_split(source_dir, out_json, out_csv, out_plot)


def test_publish_variant_count_mismatch(tmp_path):
    """Verifies that omitting an expected variant is rejected."""
    source_dir = make_synthetic_source(tmp_path / "source", num_variants=8)
    out_json = tmp_path / "out.json"
    out_csv = tmp_path / "out.csv"
    out_plot = tmp_path / "out.png"

    with pytest.raises(ValueError, match="Expected exact development variants"):
        publisher.publish_heavy_cross_split(source_dir, out_json, out_csv, out_plot)


def test_deterministic_csv_shape(tmp_path):
    """Verifies that running publication twice yields identical CSV byte content."""
    source_dir = make_synthetic_source(tmp_path / "source")
    out_json = tmp_path / "out.json"
    out_csv1 = tmp_path / "out1.csv"
    out_csv2 = tmp_path / "out2.csv"
    out_plot = tmp_path / "out.png"

    publisher.publish_heavy_cross_split(source_dir, out_json, out_csv1, out_plot)
    publisher.publish_heavy_cross_split(source_dir, out_json, out_csv2, out_plot)

    assert out_csv1.read_bytes() == out_csv2.read_bytes()

def test_publish_rejects_wrong_variant_identity(tmp_path):
    source_dir = make_synthetic_source(tmp_path / "source")
    candidate = source_dir / "candidates" / f"{publisher.EXPECTED_VARIANT_IDS[-1]}.json"
    evaluation = source_dir / "evaluations" / f"{publisher.EXPECTED_VARIANT_IDS[-1]}.json"
    candidate.rename(candidate.with_name("iteration-9999-development.json"))
    evaluation.rename(evaluation.with_name("iteration-9999-development.json"))

    with pytest.raises(ValueError, match="Expected exact development variants"):
        publisher.publish_heavy_cross_split(
            source_dir,
            tmp_path / "out.json",
            tmp_path / "out.csv",
            tmp_path / "out.png",
        )


def test_publish_rejects_resource_total_mismatch(tmp_path):
    source_dir = make_synthetic_source(tmp_path / "source")
    candidate = source_dir / "candidates" / f"{publisher.EXPECTED_VARIANT_IDS[0]}.json"
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["resource_totals"]["total_queries"] -= 1
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"resource_totals\.total_queries"):
        publisher.publish_heavy_cross_split(
            source_dir,
            tmp_path / "out.json",
            tmp_path / "out.csv",
            tmp_path / "out.png",
        )


def test_publish_rejects_invalid_wall_time(tmp_path):
    source_dir = make_synthetic_source(tmp_path / "source")
    candidate = source_dir / "candidates" / f"{publisher.EXPECTED_VARIANT_IDS[0]}.json"
    candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
    candidate_payload["resource_totals"]["wall_time_sec"] = float("nan")
    candidate.write_text(json.dumps(candidate_payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"resource_totals\.wall_time_sec"):
        publisher.publish_heavy_cross_split(
            source_dir,
            tmp_path / "out.json",
            tmp_path / "out.csv",
            tmp_path / "out.png",
        )


def test_publish_uses_repository_relative_source_path(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    source_dir = make_synthetic_source(tmp_path / "source")
    csv_path = tmp_path / "out.csv"
    payload = publisher.publish_heavy_cross_split(
        source_dir,
        tmp_path / "out.json",
        csv_path,
        tmp_path / "out.png",
    )
    assert payload["source_provenance"]["source_dir"] == "source"
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["candidate_path"].startswith("source/candidates/")
    assert rows[0]["evaluation_path"].startswith("source/evaluations/")


def test_cli_invocation(tmp_path, monkeypatch):
    """Verifies CLI main entrypoint executes cleanly."""
    source_dir = make_synthetic_source(tmp_path / "source")
    out_json = tmp_path / "cli.json"
    out_csv = tmp_path / "cli.csv"
    out_plot = tmp_path / "cli.png"

    cli_args = [
        "publish_heavy_cross_split.py",
        "--source-dir", str(source_dir),
        "--output-json", str(out_json),
        "--output-csv", str(out_csv),
        "--output-plot", str(out_plot),
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    publisher.main()

    assert out_json.is_file()
    assert out_csv.is_file()
    assert out_plot.is_file()
