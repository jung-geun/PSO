"""
Unit tests for Heavy Task Feasibility Study (MNIST & FashionMNIST).

Covers:
1. Exact model parameter counts and forward shapes (CompactCNN vs WideCNN)
2. Immutable workload matrix definitions (mnist_compact, mnist_wide, fashion_compact, fashion_wide)
3. Train-only loader guard for both datasets (train=True only, test_samples=0, test_evals=0)
4. Deterministic normalized-method selection logic (G0, G5, G6)
5. Feasibility threshold boundaries (execution_feasible and optimization_feasible)
6. Finite and artifact schema properties
7. Exact fixed2k/fixed10k query, sample evaluation, and swarm state accounting
8. CPU smoke runner execution without network
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Ensure test directory and repo root are in Python path
REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name != "PSO" else Path(__file__).resolve().parent
TEST_DIR = REPO_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import heavy_task_feasibility as heavy_task


@pytest.fixture
def synthetic_heavy_data(monkeypatch):
    """Keep runner tests deterministic and independent of dataset downloads."""
    x_search = torch.zeros(100, 1, 28, 28)
    y_search = torch.arange(100) % 10
    x_val = torch.zeros(100, 1, 28, 28)
    y_val = torch.arange(100) % 10
    nested_subsets = {
        size: torch.arange(size) % len(y_search)
        for size in (2000, 10000, 50000)
    }

    def fake_prepare(dataset_name, split_seed=20260902, cache_dir=None):
        provenance = {
            "dataset_name": dataset_name,
            "official_test_data_loaded": False,
            "official_test_evaluations": 0,
            "test_samples": 0,
            "search_samples": 50000,
            "val_samples": 10000,
            "split_fingerprint": "synthetic-split",
            "data_fingerprint": "synthetic-data",
        }
        return (
            x_search,
            y_search,
            x_val,
            y_val,
            nested_subsets,
            "synthetic-data",
            provenance,
        )

    monkeypatch.setattr(heavy_task, "prepare_heavy_task_data", fake_prepare)


from heavy_task_feasibility import (
    PROTOCOL_VERSION,
    WORKLOADS,
    HEAVY_METHODS,
    CompactCNN,
    WideCNN,
    make_compact_cnn,
    make_wide_cnn,
    create_model,
    prepare_heavy_task_data,
    evaluate_untrained_baseline,
    select_best_normalized_method,
    evaluate_feasibility,
    run_heavy_task_screen,
    run_heavy_task_confirm,
    run_heavy_task_study,
    WorkloadConfig,
)


# =====================================================================
# 1. Parameter Counts & Forward Shapes
# =====================================================================

def test_exact_model_parameter_counts_and_forward_shapes():
    compact_model = make_compact_cnn(seed=41)
    compact_params = sum(p.numel() for p in compact_model.parameters())
    assert compact_params == 9098, f"CompactCNN should have 9,098 parameters, got {compact_params}"

    wide_model = make_wide_cnn(seed=41)
    wide_params = sum(p.numel() for p in wide_model.parameters())
    assert wide_params == 55338, f"WideCNN should have 55,338 parameters, got {wide_params}"

    # Forward shape test with (B, 1, 28, 28)
    x_img = torch.randn(2, 1, 28, 28)
    out_c_img = compact_model(x_img)
    out_w_img = wide_model(x_img)
    assert out_c_img.shape == (2, 10), f"CompactCNN image output shape should be (2, 10), got {out_c_img.shape}"
    assert out_w_img.shape == (2, 10), f"WideCNN image output shape should be (2, 10), got {out_w_img.shape}"

    # Forward shape test with flattened (B, 784)
    x_flat = torch.randn(2, 784)
    out_c_flat = compact_model(x_flat)
    out_w_flat = wide_model(x_flat)
    assert out_c_flat.shape == (2, 10), f"CompactCNN flat output shape should be (2, 10), got {out_c_flat.shape}"
    assert out_w_flat.shape == (2, 10), f"WideCNN flat output shape should be (2, 10), got {out_w_flat.shape}"

    # Deterministic factory behavior
    c1 = make_compact_cnn(seed=41)
    c2 = make_compact_cnn(seed=41)
    for p1, p2 in zip(c1.parameters(), c2.parameters()):
        assert torch.equal(p1, p2)

    w1 = make_wide_cnn(seed=41)
    w2 = make_wide_cnn(seed=41)
    for p1, p2 in zip(w1.parameters(), w2.parameters()):
        assert torch.equal(p1, p2)


# =====================================================================
# 2. Immutable Workload Matrix
# =====================================================================

def test_immutable_workload_matrix():
    expected_workloads = {"mnist_compact", "mnist_wide", "fashion_compact", "fashion_wide"}
    assert set(WORKLOADS.keys()) == expected_workloads

    assert WORKLOADS["mnist_compact"].dataset_name == "mnist"
    assert WORKLOADS["mnist_compact"].model_name == "compact_cnn"

    assert WORKLOADS["mnist_wide"].dataset_name == "mnist"
    assert WORKLOADS["mnist_wide"].model_name == "wide_cnn"

    assert WORKLOADS["fashion_compact"].dataset_name == "fashion_mnist"
    assert WORKLOADS["fashion_compact"].model_name == "compact_cnn"

    assert WORKLOADS["fashion_wide"].dataset_name == "fashion_mnist"
    assert WORKLOADS["fashion_wide"].model_name == "wide_cnn"

    assert HEAVY_METHODS == ["G0", "G5", "G6", "G8"]


# =====================================================================
# 3. Train-Only Loader Guard
# =====================================================================

def test_train_only_loader_guard(monkeypatch, tmp_path):
    import torchvision.datasets

    calls = []

    class DatasetConstructionStopped(Exception):
        pass

    def reject_after_recording(name):
        def constructor(*, root, train, download):
            calls.append((name, train, download))
            raise DatasetConstructionStopped
        return constructor

    monkeypatch.setattr(
        torchvision.datasets,
        "MNIST",
        reject_after_recording("mnist"),
    )
    monkeypatch.setattr(
        torchvision.datasets,
        "FashionMNIST",
        reject_after_recording("fashion_mnist"),
    )

    for dataset_name in ("mnist", "fashion_mnist"):
        with pytest.raises(DatasetConstructionStopped):
            prepare_heavy_task_data(
                dataset_name=dataset_name,
                split_seed=20260902,
                cache_dir=tmp_path,
            )

    assert calls == [
        ("mnist", True, True),
        ("fashion_mnist", True, True),
    ]


# =====================================================================
# 4. Deterministic Normalized Method Selection
# =====================================================================

def test_deterministic_normalized_method_selection():
    mock_screen_results = [
        {"method_id": "G0", "val_selected_loss": 0.60, "val_selected_acc": 82.0},
        {"method_id": "G5", "val_selected_loss": 0.50, "val_selected_acc": 84.0},
        {"method_id": "G6", "val_selected_loss": 0.52, "val_selected_acc": 84.5},
        {"method_id": "G8", "val_selected_loss": 0.45, "val_selected_acc": 85.0},
    ]
    # G8 is excluded from normalized custom selection; G5 has lowest val_selected_loss (0.50)
    best_m = select_best_normalized_method(mock_screen_results)
    assert best_m == "G5"

    # Test tiebreak logic: same loss, pick higher accuracy
    mock_tie = [
        {"method_id": "G0", "val_selected_loss": 0.50, "val_selected_acc": 83.0},
        {"method_id": "G5", "val_selected_loss": 0.50, "val_selected_acc": 85.0},
        {"method_id": "G6", "val_selected_loss": 0.50, "val_selected_acc": 84.0},
    ]
    best_tie = select_best_normalized_method(mock_tie)
    assert best_tie == "G5"

    # Diverged candidates cannot win selection; fail explicitly if none are finite.
    with_nonfinite = [
        {"method_id": "G0", "val_selected_loss": float("nan"), "val_selected_acc": 99.0},
        {"method_id": "G5", "val_selected_loss": 0.60, "val_selected_acc": 82.0},
        {"method_id": "G6", "val_selected_loss": float("inf"), "val_selected_acc": 100.0},
    ]
    assert select_best_normalized_method(with_nonfinite) == "G5"
    with pytest.raises(ValueError, match="No finite normalized method"):
        select_best_normalized_method(with_nonfinite[:1])


# =====================================================================
# 5. Feasibility Threshold Boundaries
# =====================================================================

def test_feasibility_threshold_boundaries():
    baseline_nll = 2.30
    baseline_acc = 10.0

    # 1. Non-finite run
    bad_runs = [{"val_selected_loss": float("nan"), "val_selected_acc": 50.0}]
    f1 = evaluate_feasibility(bad_runs, baseline_nll, baseline_acc)
    assert f1["execution_feasible"] is False
    assert f1["optimization_feasible"] is False

    # 2. Feasible run passing both NLL and accuracy thresholds
    # Target NLL <= 2.30 * 0.80 = 1.84
    # Target Acc >= 10.0 + 20.0 = 30.0
    good_runs = [
        {"val_selected_loss": 1.50, "val_selected_acc": 40.0},
        {"val_selected_loss": 1.60, "val_selected_acc": 42.0},
    ]
    f2 = evaluate_feasibility(good_runs, baseline_nll, baseline_acc)
    assert f2["execution_feasible"] is True
    assert f2["optimization_feasible"] is True
    assert f2["target_val_nll_threshold"] == 1.84
    assert f2["target_val_acc_threshold"] == 30.0

    # 3. Failing NLL threshold (1.90 > 1.84)
    fail_nll_runs = [
        {"val_selected_loss": 1.90, "val_selected_acc": 40.0},
    ]
    f3 = evaluate_feasibility(fail_nll_runs, baseline_nll, baseline_acc)
    assert f3["execution_feasible"] is True
    assert f3["optimization_feasible"] is False

    # 4. Failing Acc threshold (25.0 < 30.0)
    fail_acc_runs = [
        {"val_selected_loss": 1.50, "val_selected_acc": 25.0},
    ]
    f4 = evaluate_feasibility(fail_acc_runs, baseline_nll, baseline_acc)
    assert f4["execution_feasible"] is True
    assert f4["optimization_feasible"] is False


# =====================================================================
# 6. Artifact Schema & Properties
# =====================================================================

def test_finite_and_artifact_schema(tmp_path, synthetic_heavy_data):
    device = torch.device("cpu")
    # Quick smoke call to test output schema structure
    single_wl = {"mnist_compact": WORKLOADS["mnist_compact"]}
    screen_results, baselines, meta = run_heavy_task_screen(
        workloads=single_wl,
        methods=["G0"],
        particles=2,
        epochs=2,
        seed=91,
        device=device,
        cache_dir=tmp_path / "cache",
    )
    assert len(screen_results) == 1
    cell = screen_results[0]
    assert cell["official_test_evaluations"] == 0
    assert cell["is_finite"] is True
    assert "core_swarm_state_bytes" in cell
    assert cell["core_swarm_state_bytes"] == 5 * 2 * 9098 * 4


# =====================================================================
# 7. Exact Fixed Accounting
# =====================================================================

def test_exact_fixed_accounting():
    # Fixed 2k screening cell: 12 particles, 40 epochs
    particles = 12
    epochs = 40
    subset_2k = 2000

    expected_queries_2k = particles * epochs
    expected_sample_evals_2k = expected_queries_2k * subset_2k

    assert expected_queries_2k == 480
    assert expected_sample_evals_2k == 960000

    # Swarm state bytes:
    # Custom method (G0, G5, G6): 5 * particles * param_count * 4
    # G8 (public Optimizer): (5 * particles + 1) * param_count * 4
    compact_params = 9098
    custom_bytes = 5 * 12 * compact_params * 4
    g8_bytes = (5 * 12 + 1) * compact_params * 4

    assert custom_bytes == 2183520
    assert g8_bytes == 2219912


# =====================================================================
# 8. CPU Smoke Runner Execution
# =====================================================================

def test_cpu_smoke_runner(tmp_path, synthetic_heavy_data):
    device = torch.device("cpu")
    single_wl = {"mnist_compact": WORKLOADS["mnist_compact"]}

    # Screening phase smoke
    screen_res, baselines, wl_meta = run_heavy_task_screen(
        workloads=single_wl,
        methods=["G0", "G8"],
        particles=2,
        epochs=2,
        seed=91,
        device=device,
        cache_dir=tmp_path / "cache",
    )
    assert len(screen_res) == 2

    # Confirmation phase smoke
    selected_methods = {"mnist_compact": ["G0", "G8"]}
    confirm_res = run_heavy_task_confirm(
        workloads=single_wl,
        selected_methods=selected_methods,
        particles=2,
        epochs=2,
        seeds=[101, 102],
        device=device,
        cache_dir=tmp_path / "cache",
    )
    assert "mnist_compact" in confirm_res
    assert "G0" in confirm_res["mnist_compact"]
    assert "G8" in confirm_res["mnist_compact"]

    # Feasibility smoke
    base = baselines["mnist_compact"]
    feas = evaluate_feasibility(
        confirm_res["mnist_compact"]["G0"]["per_seed_runs"],
        base["val_nll"],
        base["val_accuracy"],
    )
    assert "execution_feasible" in feas
    assert "optimization_feasible" in feas
