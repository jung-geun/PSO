"""Behavioral tests for the independent model-convergence evaluator.

These fixtures intentionally stay in prediction/artifact space: no dataset, model,
optional detection dependency, or network access is needed.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evaluate_post_training_model_convergence as evaluator


CLASSIFICATION_RECORDS = [
    {"image_id": "a", "probabilities": [0.80, 0.20], "target": 0},
    {"image_id": "b", "probabilities": [0.40, 0.60], "target": 1},
    {"image_id": "c", "probabilities": [0.70, 0.30], "target": 1},
    {"image_id": "d", "probabilities": [0.55, 0.45], "target": 0},
]


def _secondary_classification_metrics(records):
    brier_terms = []
    confidences = []
    correctness = []
    for record in records:
        probabilities = record["probabilities"]
        target = record["target"]
        brier_terms.append(
            sum((probability - (index == target)) ** 2 for index, probability in enumerate(probabilities))
        )
        prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
        confidences.append(max(probabilities))
        correctness.append(prediction == target)

    # Match the protocol's 15 equal-width confidence bins, including the
    # right-most endpoint in the final bin.
    ece = 0.0
    for bin_index in range(15):
        lower, upper = bin_index / 15.0, (bin_index + 1) / 15.0
        members = [
            index
            for index, confidence in enumerate(confidences)
            if (confidence >= lower and (confidence < upper or bin_index == 14 and confidence <= upper))
        ]
        if members:
            accuracy = sum(correctness[index] for index in members) / len(members)
            confidence = sum(confidences[index] for index in members) / len(members)
            ece += abs(accuracy - confidence) * len(members) / len(records)
    return sum(brier_terms) / len(records), ece


def test_classification_metrics_recompute_exact_nll_accuracy_brier_and_ece():
    """Per-example probabilities determine all classification statistics without rounding."""
    metrics = evaluator.classification_metrics(CLASSIFICATION_RECORDS)
    expected_nll = -math.fsum(math.log(record["probabilities"][record["target"]]) for record in CLASSIFICATION_RECORDS) / 4
    expected_brier, expected_ece = _secondary_classification_metrics(CLASSIFICATION_RECORDS)

    assert metrics["n"] == 4
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["nll"] == pytest.approx(expected_nll, rel=0, abs=1e-15)
    assert metrics["brier"] == pytest.approx(expected_brier, rel=0, abs=1e-15)
    assert metrics["ece15"] == pytest.approx(expected_ece, rel=0, abs=1e-15)
    # The evaluator must retain the unrounded probability/target evidence used
    # for the secondary metrics rather than substituting aggregate values.
    assert metrics["probabilities"] == [record["probabilities"] for record in CLASSIFICATION_RECORDS]
    assert metrics["targets"] == [record["target"] for record in CLASSIFICATION_RECORDS]


def test_classification_metrics_reject_invalid_probability_contracts():
    with pytest.raises(ValueError, match="sum to one"):
        evaluator.classification_metrics([{"probabilities": [0.8, 0.3], "target": 0}])
    with pytest.raises(ValueError, match="invalid classification"):
        evaluator.classification_metrics([{"probabilities": [1.0, 0.0], "target": True}])
    with pytest.raises(ZeroDivisionError):
        evaluator.classification_metrics([])


def test_detection_metrics_deduplicates_predictions_and_counts_empty_images():
    """One image may have duplicate detections while other images are empty."""
    box = [0.0, 0.0, 10.0, 10.0]
    records = [
        {
            "image_id": "duplicate",
            "ground_truth": [{"class_id": 0, "box": box}],
            # Deliberately preserve a low-score row first: matching is one-to-one,
            # then confidence ranking makes the duplicate a false positive.
            "predictions": [
                {"class_id": 0, "score": 0.10, "box": box},
                {"class_id": 0, "score": 0.90, "box": box},
            ],
        },
        {"image_id": "empty-predictions", "ground_truth": [{"class_id": 1, "box": box}], "predictions": []},
        {"image_id": "empty-image", "ground_truth": [], "predictions": []},
    ]

    metrics = evaluator.detection_metrics(records, class_count=2)

    assert metrics["n"] == 3
    assert metrics["ground_truth"] == 2
    assert metrics["predictions"] == 2
    expected_ap = 0.49750000000000033
    assert metrics["per_class_ap"]["0"] == pytest.approx(
        [expected_ap] * 10,
        rel=0,
        abs=1e-12,
    )
    assert metrics["per_class_ap"]["1"] == pytest.approx(
        [0.0] * 10,
        rel=0,
        abs=1e-12,
    )
    assert metrics["map50"] == pytest.approx(expected_ap / 2, abs=1e-12)
    assert metrics["map50_95"] == pytest.approx(expected_ap / 2, abs=1e-12)


def test_detection_metrics_empty_dataset_is_a_finite_zero_result():
    metrics = evaluator.detection_metrics([], class_count=3)
    assert metrics["n"] == 0
    assert metrics["ground_truth"] == 0
    assert metrics["predictions"] == 0
    assert metrics["map50"] == 0.0
    assert metrics["map50_95"] == 0.0
    assert set(metrics["per_class_ap"]) == {"0", "1", "2"}
    assert all(value == [0.0] * 10 for value in metrics["per_class_ap"].values())


def test_bootstrap_helper_is_deterministic_and_uses_improvement_orientation():
    base = [
        {"image_id": "0", "probabilities": [0.60, 0.40], "target": 0},
        {"image_id": "1", "probabilities": [0.40, 0.60], "target": 1},
        {"image_id": "2", "probabilities": [0.60, 0.40], "target": 0},
        {"image_id": "3", "probabilities": [0.40, 0.60], "target": 1},
    ]
    improved = [
        {**record, "probabilities": [0.90, 0.10] if record["target"] == 0 else [0.10, 0.90]}
        for record in base
    ]
    pairs = [(base, improved), (base, improved)]

    first = evaluator._bootstrap_from_records(pairs, "classification")
    second = evaluator._bootstrap_from_records(pairs, "classification")

    assert first == second
    assert first["available"] is True
    assert first["seed"] == evaluator.BOOTSTRAP_SEED
    assert first["resamples"] == evaluator.BOOTSTRAP_RESAMPLES
    assert first["alpha"] == evaluator.BOOTSTRAP_ALPHA
    assert first["statistic"] > 0.0
    assert first["lower"] > 0.0
    assert first["excludes_zero"] is True


def test_bootstrap_helper_rejects_misaligned_image_identity():
    base = [{"image_id": "a", "probabilities": [1.0, 0.0], "target": 0}]
    reordered = [{"image_id": "b", "probabilities": [1.0, 0.0], "target": 0}]
    result = evaluator._bootstrap_from_records([(base, reordered)], "classification")
    assert result["available"] is False
    assert "image IDs/order differ" in result["reason"]


def test_global_query_and_candidate_sample_constants_are_exact():
    expected_queries = (
        len(evaluator.WORKLOADS)
        * len(evaluator.BASE_SEEDS)
        * len(evaluator.SWARM_SEEDS)
        * evaluator.PRIMARY_QUERIES
        + len(evaluator.WORKLOADS)
        * len(evaluator.SWARM_SEEDS)
        * evaluator.ENSEMBLE_QUERIES
    )
    expected_samples = sum(
        (
            len(evaluator.BASE_SEEDS) * len(evaluator.SWARM_SEEDS) * evaluator.PRIMARY_QUERIES
            + len(evaluator.SWARM_SEEDS) * evaluator.ENSEMBLE_QUERIES
        )
        * evaluator.OBJECTIVE_SAMPLES[workload]
        for workload in evaluator.WORKLOADS
    )

    assert evaluator.PRIMARY_QUERIES == 720
    assert evaluator.ENSEMBLE_QUERIES == 240
    assert evaluator.TOTAL_PSO_QUERIES == expected_queries == 21_600
    assert evaluator.TOTAL_CANDIDATE_SAMPLES == expected_samples == 18_432_000


def test_pt_prediction_artifact_is_resolved_without_model_import(tmp_path):
    torch = pytest.importorskip("torch")
    records = [{"image_id": "one", "probabilities": [0.25, 0.75], "target": 1}]
    artifact = tmp_path / "predictions.pt"
    torch.save({"predictions": records}, artifact)

    resolved = evaluator._prediction_records({"prediction_artifact": "predictions.pt"}, tmp_path)

    assert resolved == records


def _write_compact_results(root: Path, *, leakage_bad: bool, malformed_matrix: bool) -> None:
    leakage = {
        "official_test_data_loaded_before_freeze": True if leakage_bad else False,
        "official_test_evaluations_before_freeze": 1 if leakage_bad else 0,
        "official_test_construction": 0 if leakage_bad else 1,
        "official_test_forward_passes": 0 if leakage_bad else 1,
    }
    for workload in evaluator.WORKLOADS:
        workload_dir = root / "workloads" / workload
        workload_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "workload_id": workload,
            "family": "detection" if workload == evaluator.DETECTION_WORKLOAD else "classification",
            "manifests": {},
            "provenance": {},
            "baselines": {},
            "arms": {} if malformed_matrix else {"feature_pso": []},
            "ensemble": {},
            "development_selection": {},
            "confirmation": {},
            "integrity": {},
            "leakage_counters": leakage,
            "resource_ledger": {},
            "artifact_hashes": {},
        }
        (workload_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")


def test_evaluate_run_rejects_incomplete_matrix_from_temp_fixture(tmp_path):
    _write_compact_results(tmp_path, leakage_bad=False, malformed_matrix=True)

    result = evaluator.evaluate_run(tmp_path)

    assert result["pass"] is False
    assert result["issue_counts"]["matrix"] >= len(evaluator.WORKLOADS)
    assert any("missing arms" in issue for issue in result["issues"]["matrix"])


def test_evaluate_run_rejects_pre_freeze_test_leakage_from_temp_fixture(tmp_path):
    _write_compact_results(tmp_path, leakage_bad=True, malformed_matrix=False)

    result = evaluator.evaluate_run(tmp_path)

    assert result["pass"] is False
    assert result["issue_counts"]["leakage"] >= len(evaluator.WORKLOADS)
    assert any("must be explicitly marked not loaded" in issue for issue in result["issues"]["leakage"])
    assert any("exposure before freeze" in issue for issue in result["issues"]["leakage"])
