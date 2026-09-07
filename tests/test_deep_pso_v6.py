"""
Unit tests for MNIST PSO V6 Root-Cause Isolation (Phases A & B).

Covers:
1. Scale modes (per_tensor_sd, global_rms, identity)
2. G0 decode & default movement parity with V5 full-D
3. Exact antithetic & independent initialization invariants
4. Deterministic projection seeds
5. Equalized-dimension helper RMS behavior
6. Mutation moment reset
7. Transition full-pbest reevaluation & exact accounting
8. Validation checkpoints state neutrality
9. Geometry configuration table G0-G8
10. Deterministic confirmation selection logic
11. Guard proving Phase B loader never requests official test dataset (train=False)
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

from deep_pso_methods import LatentTransform, make_compact_cnn
from deep_pso_v6 import (
    V6GeometryConfig,
    V6LatentTransform,
    compute_equalized_subspace_radius,
    get_v6_geometry_table,
    prepare_mnist_v6_data,
    run_v6_pso,
    run_g8_optimizer,
    select_confirmation_configs,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        # Initialize deterministic weights
        nn.init.constant_(self.fc1.weight, 1.0)
        nn.init.constant_(self.fc1.bias, 0.5)
        nn.init.constant_(self.fc2.weight, -0.5)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# =====================================================================
# 1. Scale Modes Test
# =====================================================================

def test_scale_modes():
    device = torch.device("cpu")
    model = TinyModel()

    # Per-tensor SD
    cfg_per_tensor = V6GeometryConfig(config_id="T1", scale_type="per_tensor_sd")
    tf_per_tensor = V6LatentTransform(model, cfg_per_tensor, device)
    assert tf_per_tensor.scale_vec.shape[0] == tf_per_tensor.total_dim

    # Global RMS
    cfg_global_rms = V6GeometryConfig(config_id="T2", scale_type="global_rms")
    tf_global_rms = V6LatentTransform(model, cfg_global_rms, device)
    expected_rms = max(float(torch.sqrt(torch.mean(tf_global_rms.base_vec ** 2))), 1e-4)
    assert torch.allclose(tf_global_rms.scale_vec, torch.full_like(tf_global_rms.scale_vec, expected_rms))

    # Identity
    cfg_identity = V6GeometryConfig(config_id="T3", scale_type="identity")
    tf_identity = V6LatentTransform(model, cfg_identity, device)
    assert torch.allclose(tf_identity.scale_vec, torch.ones_like(tf_identity.scale_vec))


# =====================================================================
# 2. G0 Decode & Default Parity with V5 Full-D
# =====================================================================

def test_G0_decode_default_parity():
    device = torch.device("cpu")

    base_v5 = make_compact_cnn(seed=41).to(device)
    base_v6 = make_compact_cnn(seed=41).to(device)

    tf_v5 = LatentTransform(base_v5, latent_dim="full", device=device)

    cfg_g0 = get_v6_geometry_table()["G0"]
    tf_v6 = V6LatentTransform(base_v6, cfg_g0, device=device)

    # Verify scale_vec equality
    assert torch.allclose(tf_v5.scale_vec, tf_v6.scale_vec, atol=1e-6)
    assert torch.allclose(tf_v5.base_vec, tf_v6.base_vec, atol=1e-6)

    # Initial swarm parity with seed 91
    swarm_size = 30
    seed = 91
    Z_v5 = tf_v5.init_swarm(swarm_size=swarm_size, seed=seed)
    Z_v6 = tf_v6.init_swarm(swarm_size=swarm_size, seed=seed)

    assert torch.allclose(Z_v5, Z_v6, atol=1e-6)

    # Decode parity
    theta_v5 = tf_v5.decode(Z_v5)
    theta_v6 = tf_v6.decode(Z_v6)
    assert torch.allclose(theta_v5, theta_v6, atol=1e-6)

    # Single movement step parity
    c0 = c1 = 1.49618
    w = 0.7298
    latent_dim = tf_v6.latent_dim

    V_v5 = torch.zeros((swarm_size, latent_dim), dtype=torch.float32, device=device)
    V_v6 = torch.zeros((swarm_size, latent_dim), dtype=torch.float32, device=device)

    P_v5 = Z_v5.clone()
    P_v6 = Z_v6.clone()

    gbest_z_v5 = Z_v5[0].clone()
    gbest_z_v6 = Z_v6[0].clone()

    move_rng_v5 = torch.Generator(device=device)
    move_rng_v5.manual_seed(seed)

    move_rng_v6 = torch.Generator(device=device)
    move_rng_v6.manual_seed(seed)

    r1_v5 = torch.rand((swarm_size, latent_dim), generator=move_rng_v5, device=device)
    r2_v5 = torch.rand((swarm_size, latent_dim), generator=move_rng_v5, device=device)

    r1_v6 = torch.rand((swarm_size, latent_dim), generator=move_rng_v6, device=device)
    r2_v6 = torch.rand((swarm_size, latent_dim), generator=move_rng_v6, device=device)

    assert torch.allclose(r1_v5, r1_v6)
    assert torch.allclose(r2_v5, r2_v6)

    V_raw_v5 = w * V_v5 + c0 * r1_v5 * (P_v5 - Z_v5) + c1 * r2_v5 * (gbest_z_v5.unsqueeze(0) - Z_v5)
    V_raw_v6 = w * V_v6 + c0 * r1_v6 * (P_v6 - Z_v6) + c1 * r2_v6 * (gbest_z_v6.unsqueeze(0) - Z_v6)

    assert torch.allclose(V_raw_v5, V_raw_v6, atol=1e-6)


# =====================================================================
# 3. Exact Antithetic & Independent Initialization Invariants
# =====================================================================

def test_exact_independent_init_invariants():
    device = torch.device("cpu")
    model = TinyModel()
    swarm_size = 30
    seed = 123

    # 1. Antithetic mode
    cfg_anti = V6GeometryConfig(config_id="T_anti", init_position_mode="antithetic", position_radius=0.5)
    tf_anti = V6LatentTransform(model, cfg_anti, device)
    Z_anti = tf_anti.init_swarm(swarm_size=swarm_size, seed=seed)

    # Particle 0 is exact zero
    assert torch.norm(Z_anti[0]).item() == 0.0

    # For even swarm_size=30, particles 1..28 form exact pairs: (1, 2), (3, 4), ..., (27, 28)
    for idx in range(1, 28, 2):
        assert torch.allclose(Z_anti[idx], -Z_anti[idx + 1], atol=1e-6)

    # Particle 29 is zero filler
    assert torch.norm(Z_anti[29]).item() == 0.0

    # 2. Independent mode
    cfg_indep = V6GeometryConfig(config_id="T_indep", init_position_mode="independent", position_radius=0.5)
    tf_indep = V6LatentTransform(model, cfg_indep, device)
    Z_indep = tf_indep.init_swarm(swarm_size=swarm_size, seed=seed)

    # Particle 0 is exact zero
    assert torch.norm(Z_indep[0]).item() == 0.0

    # Particles 1..29 are non-zero and independent
    assert torch.norm(Z_indep[1]).item() > 0.0
    assert not torch.allclose(Z_indep[1], -Z_indep[2])


# =====================================================================
# 4. Deterministic Projection Seeds Test
# =====================================================================

def test_deterministic_projection_seeds():
    device = torch.device("cpu")
    model = TinyModel()

    cfg1 = V6GeometryConfig(config_id="P1", latent_dim=8, projection_seed=42)
    cfg2 = V6GeometryConfig(config_id="P2", latent_dim=8, projection_seed=42)
    cfg3 = V6GeometryConfig(config_id="P3", latent_dim=8, projection_seed=99)

    tf1 = V6LatentTransform(model, cfg1, device)
    tf2 = V6LatentTransform(model, cfg2, device)
    tf3 = V6LatentTransform(model, cfg3, device)

    # Same seed yields identical mapping
    assert torch.equal(tf1.k_indices, tf2.k_indices)
    assert torch.equal(tf1.weights, tf2.weights)

    # Different seed yields different mapping
    assert not torch.equal(tf1.k_indices, tf3.k_indices) or not torch.equal(tf1.weights, tf3.weights)


# =====================================================================
# 5. Equalized-Dimension Helper RMS Behavior Test
# =====================================================================

def test_equalized_dimension_helper_rms_behavior():
    r_290 = compute_equalized_subspace_radius(latent_dim=290, total_dim=9098, base_radius=0.5)
    r_1024 = compute_equalized_subspace_radius(latent_dim=1024, total_dim=9098, base_radius=0.5)
    r_4096 = compute_equalized_subspace_radius(latent_dim=4096, total_dim=9098, base_radius=0.5)
    r_full = compute_equalized_subspace_radius(latent_dim=9098, total_dim=9098, base_radius=0.5)

    assert r_290 == pytest.approx(0.5 * math.sqrt(9098 / 290), rel=1e-5)
    assert r_1024 == pytest.approx(0.5 * math.sqrt(9098 / 1024), rel=1e-5)
    assert r_4096 == pytest.approx(0.5 * math.sqrt(9098 / 4096), rel=1e-5)
    assert r_full == 0.5

    # Radii decrease monotonically as latent dimension increases toward full-D
    assert r_290 > r_1024 > r_4096 > r_full


# =====================================================================
# 6. Mutation Moment Reset Test
# =====================================================================

def test_mutation_moment_reset():
    device = torch.device("cpu")
    model = TinyModel()

    x_search = torch.randn(20, 10)
    y_search = torch.randint(0, 2, (20,))
    x_val = torch.randn(10, 10)
    y_val = torch.randint(0, 2, (10,))
    nested_subsets = {10: torch.arange(10), 20: torch.arange(20)}

    # Always-on mutation: mutation_prob = 1.0
    cfg = V6GeometryConfig(
        config_id="M1",
        mutation_prob=1.0,
        reset_velocity_radius=0.02,
    )
    transform = V6LatentTransform(model, cfg, device)

    res = run_v6_pso(
        transform=transform,
        base_model=model,
        x_search=x_search,
        y_search=y_search,
        x_val=x_val,
        y_val=y_val,
        nested_subsets=nested_subsets,
        schedule_str="10:2",
        epochs=2,
        swarm_size=5,
        seed=42,
        device=device,
        geom_config=cfg,
    )

    assert res["config_id"] == "M1"
    assert res["total_queries"] == 5 * 2
    assert res["mutation_events"] == 5 * 2
    assert res["final_moment_steps"] == [1] * 5


# =====================================================================
# 7. Transition Reevaluation & Exact Accounting Test
# =====================================================================

def test_transition_full_pbest_reevaluation_plus_exact_accounting():
    device = torch.device("cpu")
    model = TinyModel()

    x_search = torch.randn(50, 10)
    y_search = torch.randint(0, 2, (50,))
    x_val = torch.randn(10, 10)
    y_val = torch.randint(0, 2, (10,))

    nested_subsets = {
        5: torch.arange(5),
        20: torch.arange(20),
    }

    cfg = get_v6_geometry_table()["G0"]
    transform = V6LatentTransform(model, cfg, device)

    # Run 2-stage schedule: 5 samples for 2 epochs, then 20 samples for 2 epochs
    res = run_v6_pso(
        transform=transform,
        base_model=model,
        x_search=x_search,
        y_search=y_search,
        x_val=x_val,
        y_val=y_val,
        nested_subsets=nested_subsets,
        schedule_str="5:2,20:2",
        epochs=4,
        swarm_size=10,
        seed=42,
        device=device,
        geom_config=cfg,
        transition_reset_policy="reset_vm",
    )

    # Exact query accounting:
    # Stage 0: 10 particles x 2 epochs = 20 queries
    # Transition: 10 particles reevaluated on new subset = 10 queries
    # Stage 1: 10 particles x 2 epochs = 20 queries
    # Total queries = 50
    assert res["total_queries"] == 50
    assert res["transition_reevaluation_counts"] == 10

    # Sample evaluations:
    # Stage 0: 20 queries x 5 samples = 100
    # Transition: 10 queries x 20 samples = 200
    # Stage 1: 20 queries x 20 samples = 400
    # Total sample evals = 700
    assert res["total_sample_evaluations"] == 700
    assert res["final_moment_steps"] == [2] * 10


# =====================================================================
# 8. Validation Checkpoints State Neutrality Test
# =====================================================================

def test_validation_checkpoints_state_neutral():
    device = torch.device("cpu")
    model = TinyModel()

    x_search = torch.randn(20, 10)
    y_search = torch.randint(0, 2, (20,))
    x_val = torch.randn(10, 10)
    y_val = torch.randint(0, 2, (10,))
    nested_subsets = {20: torch.arange(20)}

    cfg = get_v6_geometry_table()["G0"]

    # Run with validation checkpoints every epoch (val_check_interval=1)
    tf1 = V6LatentTransform(model, cfg, device)
    res_chk = run_v6_pso(
        transform=tf1,
        base_model=model,
        x_search=x_search,
        y_search=y_search,
        x_val=x_val,
        y_val=y_val,
        nested_subsets=nested_subsets,
        schedule_str="20:5",
        epochs=5,
        swarm_size=6,
        seed=99,
        device=device,
        geom_config=cfg,
        val_check_interval=1,
    )

    # Run without intermediate validation checkpoints (val_check_interval=0)
    tf2 = V6LatentTransform(model, cfg, device)
    res_nochk = run_v6_pso(
        transform=tf2,
        base_model=model,
        x_search=x_search,
        y_search=y_search,
        x_val=x_val,
        y_val=y_val,
        nested_subsets=nested_subsets,
        schedule_str="20:5",
        epochs=5,
        swarm_size=6,
        seed=99,
        device=device,
        geom_config=cfg,
        val_check_interval=0,
    )

    # Final gbest positions, loss, and training accuracy must be IDENTICAL
    assert torch.allclose(res_chk["gbest_z"], res_nochk["gbest_z"], atol=1e-6)
    assert res_chk["gbest_loss"] == res_nochk["gbest_loss"]
    assert res_chk["gbest_acc"] == res_nochk["gbest_acc"]


# =====================================================================
# 9. Configuration Table G0-G8 Test
# =====================================================================

def test_configuration_table_G0_G8():
    table = get_v6_geometry_table()
    assert len(table) == 9
    for i in range(9):
        cid = f"G{i}"
        assert cid in table
        assert table[cid].config_id == cid

    assert table["G0"].scale_type == "per_tensor_sd"
    assert table["G0"].init_position_mode == "antithetic"
    assert table["G0"].initial_velocity_radius == 0.0

    assert table["G1"].scale_type == "global_rms"

    assert table["G2"].initial_velocity_radius == 0.5

    assert table["G3"].mutation_prob == 0.02

    assert table["G5"].reflective_bound == 6.0

    assert table["G6"].position_radius == 1.5

    assert table["G7"].init_position_mode == "independent"

    assert table["G8"].scale_type == "optimizer_default"


# =====================================================================
# 10. Deterministic Confirmation Selection Test
# =====================================================================

def test_deterministic_confirmation_selection():
    mock_screen = {
        "G0": {"val_selected_loss": 0.70, "val_selected_acc": 78.0},
        "G1": {"val_selected_loss": 0.65, "val_selected_acc": 80.0},
        "G2": {"val_selected_loss": 0.60, "val_selected_acc": 82.0},
        "G3": {"val_selected_loss": 0.58, "val_selected_acc": 83.0}, # Top 1 eligible
        "G4": {"val_selected_loss": 0.55, "val_selected_acc": 84.0}, # Top 0 eligible (best)
        "G5": {"val_selected_loss": 0.62, "val_selected_acc": 81.0},
        "G6": {"val_selected_loss": 0.64, "val_selected_acc": 80.5},
        "G7": {"val_selected_loss": 0.61, "val_selected_acc": 81.5},
        "G8": {"val_selected_loss": 0.48, "val_selected_acc": 85.0},
    }

    selected = select_confirmation_configs(mock_screen)
    assert len(selected) == 5
    assert selected[:3] == ["G0", "G1", "G8"]
    assert set(selected[3:]) == {"G4", "G3"}


# =====================================================================
# 11. Guard: Loader Never Requests Official Test Dataset (train=False)
# =====================================================================

def test_guard_loader_never_requests_official_test_dataset(monkeypatch):
    import torchvision.datasets

    called_train_flags = []

    original_mnist_init = torchvision.datasets.MNIST.__init__

    def mock_mnist_init(self, root, train=True, transform=None, target_transform=None, download=False):
        called_train_flags.append(train)
        if not train:
            raise AssertionError("CRITICAL VIOLATION: MNIST(train=False) requested during Phase B data loader!")
        # Perform mock initialization with synthetic data
        self.data = torch.randint(0, 256, (60000, 28, 28), dtype=torch.uint8)
        self.targets = torch.randint(0, 10, (60000,), dtype=torch.long)

    monkeypatch.setattr(torchvision.datasets.MNIST, "__init__", mock_mnist_init)

    x_search, y_search, x_val, y_val, nested_subsets, data_fp, provenance = prepare_mnist_v6_data()

    assert len(called_train_flags) > 0
    assert all(flag is True for flag in called_train_flags)
    assert provenance["official_test_evaluations"] == 0
    assert provenance["test_samples"] == 0
    assert x_search.shape == (50000, 1, 28, 28)
    assert x_val.shape == (10000, 1, 28, 28)


def test_g8_uses_exact_supplied_objective(monkeypatch):
    torch.manual_seed(7)
    device = torch.device("cpu")
    model = TinyModel()
    x_search = torch.randn(12, 10)
    y_search = torch.randint(0, 2, (12,))
    x_val = torch.randn(8, 10)
    y_val = torch.randint(0, 2, (8,))
    from pso.optimizer import Optimizer

    fit_args = {}
    original_fit = Optimizer.fit

    def recording_fit(self, *args, **kwargs):
        fit_args.update(kwargs)
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(Optimizer, "fit", recording_fit)

    result = run_g8_optimizer(
        base_model=model,
        x_2k=x_search,
        y_2k=y_search,
        x_val=x_val,
        y_val=y_val,
        epochs=2,
        swarm_size=4,
        seed=11,
        device=device,
    )

    assert result["total_queries"] == 8
    assert result["total_sample_evaluations"] == 8 * len(y_search)
    assert result["validation_evaluations"] == 5
    assert result["official_test_evaluations"] == 0
    assert fit_args["renewal"] == "loss"
