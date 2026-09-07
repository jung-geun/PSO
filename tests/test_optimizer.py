import collections
import json
import math
import os
import pytest
import torch
import torch.nn as nn

import pso
from pso.optimizer import Optimizer, resolve_device
from pso.particle import Particle

def test_basic_fit_and_inspection_contract(model_factory, xor_data):
    """Verify inspection methods return None before fit, and valid objects after fit."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()
    opt = Optimizer(model, loss, task="binary", n_particles=2, seed=42)

    # Before fit
    assert opt.get_best_model() is None
    assert opt.get_best_score() is None
    assert opt.get_best_state_dict() is None

    score = opt.fit(x, y, epochs=2)

    # Fit return value check
    assert isinstance(score, tuple)
    assert len(score) == 3
    assert all(isinstance(val, float) and math.isfinite(val) for val in score)

    # get_best_score check
    best_score = opt.get_best_score()
    assert best_score == score

    # get_best_model check
    best_model = opt.get_best_model()
    assert isinstance(best_model, nn.Module)
    assert not best_model.training  # Model is in eval mode

    # get_best_state_dict check
    state_dict = opt.get_best_state_dict()
    assert isinstance(state_dict, collections.OrderedDict)

    # All state dict tensors are CPU clones
    for k, v in state_dict.items():
        assert isinstance(v, torch.Tensor)
        assert v.device.type == "cpu"

    # Mutating returned model parameters does NOT mutate stored state dict
    for p in best_model.parameters():
        p.data.add_(1.0)

    fresh_state_dict = opt.get_best_state_dict()
    assert fresh_state_dict is not None
    for k in state_dict:
        assert torch.equal(state_dict[k], fresh_state_dict[k])


def test_seeded_reproducibility_and_swarm_variability(model_factory, xor_data):
    """Verify seeded runs produce identical results while individual swarm particles vary."""
    x, y = xor_data
    loss = nn.BCEWithLogitsLoss()

    m1 = model_factory()
    m2 = model_factory()

    opt1 = Optimizer(m1, loss, task="binary", n_particles=3, seed=42)
    opt2 = Optimizer(m2, loss, task="binary", n_particles=3, seed=42)

    score1 = opt1.fit(x, y, epochs=3)
    score2 = opt2.fit(x, y, epochs=3)

    assert score1 == score2

    sd1 = opt1.get_best_state_dict()
    sd2 = opt2.get_best_state_dict()
    assert sd1 is not None and sd2 is not None
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k])

    # Swarm variability within opt1
    p0 = opt1.particles[0]
    p1 = opt1.particles[1]
    assert not torch.equal(p0.velocity, p1.velocity)
    assert not torch.equal(p0.position, p1.position)


def test_sequential_optimizers_independence(model_factory, xor_data):
    """Verify sequential optimizers with different model shapes do not leak state."""
    x, y = xor_data
    loss = nn.BCEWithLogitsLoss()

    m4 = model_factory(units=4)
    opt4 = Optimizer(m4, loss, task="binary", n_particles=2, seed=42)
    opt4.fit(x, y, epochs=2)
    sd4 = opt4.get_best_state_dict()
    assert sd4 is not None

    m8 = model_factory(units=8)
    opt8 = Optimizer(m8, loss, task="binary", n_particles=2, seed=42)
    opt8.fit(x, y, epochs=2)
    sd8 = opt8.get_best_state_dict()
    assert sd8 is not None

    assert sd4["0.weight"].shape == torch.Size([4, 2])
    assert sd8["0.weight"].shape == torch.Size([8, 2])

def test_multi_batch_evaluation_contract(model_factory, xor_data, monkeypatch):
    """Verify all particles evaluate identical batch tensors in particle-outer order during swarm iterations."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()
    opt = Optimizer(model, loss, task="binary", n_particles=3, seed=42)

    recorded_batches = []
    original_eval = opt._evaluate_batch_tensors

    def mock_eval(x_batch, y_batch):
        recorded_batches.append((x_batch.detach().cpu(), y_batch.detach().cpu()))
        return original_eval(x_batch, y_batch)

    monkeypatch.setattr(opt, "_evaluate_batch_tensors", mock_eval)

    opt.fit(x, y, epochs=1, batch_size=2)

    # 4 samples, batch_size 2 -> 2 batches
    # 3 particles evaluated over 2 batches -> 6 recorded calls total
    assert len(recorded_batches) == 6

    # Particle-outer order: Particle 0 (calls 0, 1), Particle 1 (calls 2, 3), Particle 2 (calls 4, 5)
    # Batch 0 (calls 0, 2, 4) must receive identical x_batch and y_batch
    assert torch.equal(recorded_batches[0][0], recorded_batches[2][0])
    assert torch.equal(recorded_batches[0][0], recorded_batches[4][0])
    assert torch.equal(recorded_batches[0][1], recorded_batches[2][1])
    assert torch.equal(recorded_batches[0][1], recorded_batches[4][1])

    # Batch 1 (calls 1, 3, 5) must receive identical x_batch and y_batch
    assert torch.equal(recorded_batches[1][0], recorded_batches[3][0])
    assert torch.equal(recorded_batches[1][0], recorded_batches[5][0])


def test_zero_loss_succeeds_and_contextual_nonfinite_raises(model_factory, xor_data, monkeypatch):
    x, y = xor_data

    # Zero init model -> deterministic constant output
    model_zero = model_factory(zero_init=True)
    loss = nn.BCEWithLogitsLoss()
    opt_zero = Optimizer(model_zero, loss, task="binary", n_particles=2, seed=42)

    score_zero = opt_zero.fit(x, y, epochs=2)
    assert all(math.isfinite(val) for val in score_zero)

    # Non-finite score raises FloatingPointError
    model = model_factory()
    opt_nan = Optimizer(model, loss, task="binary", n_particles=2, seed=42)

    monkeypatch.setattr(
        opt_nan,
        "_evaluate_batch_tensors",
        lambda x_batch, y_batch: (
            torch.tensor(float("nan")),
            torch.tensor(0.0),
            torch.tensor(float("nan")),
        ),
    )

    with pytest.raises(FloatingPointError) as exc_info:
        opt_nan.fit(x, y, epochs=1)

    err_msg = str(exc_info.value).lower()
    assert "particle" in err_msg
    assert "iteration" in err_msg


def test_hard_bounds_and_velocity_reflection(model_factory, xor_data):
    """Verify position clipping and boundary reflection enforce hard particle_min and particle_max bounds."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    p_min, p_max = -0.1, 0.1
    v_ratio = 0.5
    span = p_max - p_min
    max_vel = v_ratio * span

    # Boundary strategy clip
    opt_clip = Optimizer(
        model,
        loss,
        task="binary",
        n_particles=3,
        particle_min=p_min,
        particle_max=p_max,
        boundary_strategy="clip",
        velocity_limit_ratio=v_ratio,
        seed=42,
    )
    opt_clip.fit(x, y, epochs=3)

    for p in opt_clip.particles:
        assert torch.all(p.position >= p_min)
        assert torch.all(p.position <= p_max)
        assert torch.all(torch.abs(p.velocity) <= max_vel + 1e-6)

    # Boundary strategy reflect
    opt_reflect = Optimizer(
        model,
        loss,
        task="binary",
        n_particles=3,
        particle_min=p_min,
        particle_max=p_max,
        boundary_strategy="reflect",
        seed=42,
    )
    opt_reflect.fit(x, y, epochs=3)

    for p in opt_reflect.particles:
        assert torch.all(p.position >= p_min)
        assert torch.all(p.position <= p_max)


def test_invalid_constructor_and_fit_combinations_fail_fast(model_factory, xor_data):
    """Verify constructor and fit input validation fail fast with appropriate errors."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    # Non-Tensor inputs to fit
    opt = Optimizer(model, loss, task="binary")
    with pytest.raises(TypeError):
        opt.fit(x.numpy(), y)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        opt.fit(x, y.numpy())  # type: ignore[arg-type]

    # Invalid task
    with pytest.raises(ValueError):
        Optimizer(model, loss, task="invalid")  # type: ignore[arg-type]

    # Invalid model / loss
    with pytest.raises(ValueError):
        Optimizer(None, loss, task="binary")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Optimizer(model, None, task="binary")  # type: ignore[arg-type]

    # Invalid n_particles
    with pytest.raises(ValueError):
        Optimizer(model, loss, task="binary", n_particles=0)

    # w_min > w_max
    with pytest.raises(ValueError):
        Optimizer(model, loss, task="binary", w_min=0.8, w_max=0.2)

    # Non-finite c0
    with pytest.raises(ValueError):
        Optimizer(model, loss, task="binary", c0=float("nan"))

    # Invalid negative_swarm / mutation_swarm
    with pytest.raises(ValueError):
        Optimizer(model, loss, task="binary", negative_swarm=1.5)
    with pytest.raises(ValueError):
        Optimizer(model, loss, task="binary", mutation_swarm=-0.1)

    # One bound without the other
    with pytest.raises(ValueError):
        Optimizer(model, loss, task="binary", particle_min=-1.0)
    with pytest.raises(ValueError):
        Optimizer(model, loss, task="binary", particle_max=1.0)

    # particle_min > particle_max
    with pytest.raises(ValueError):
        Optimizer(model, loss, task="binary", particle_min=1.0, particle_max=-1.0)

    # Reflect without bounds
    with pytest.raises(ValueError):
        Optimizer(model, loss, task="binary", boundary_strategy="reflect")

    # Both validation_data and validation_split
    opt_val = Optimizer(model, loss, task="binary")
    with pytest.raises(ValueError):
        opt_val.fit(x, y, validation_data=(x, y), validation_split=0.5)

    # Options requiring output_dir when output_dir is None
    with pytest.raises(ValueError):
        opt_val.fit(x, y, log_format="csv", output_dir=None)
    with pytest.raises(ValueError):
        opt_val.fit(x, y, checkpoint_interval=1, output_dir=None)
    with pytest.raises(ValueError):
        opt_val.fit(x, y, save_info=True, output_dir=None)


def test_inertia_schedule_over_epochs(model_factory, xor_data, monkeypatch):
    """Verify linear inertia weight schedule for single and multi-epoch runs."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    opt = Optimizer(
        model, loss, task="binary", method="inertia", n_particles=2, w_max=0.9, w_min=0.1, seed=42
    )

    recorded_w = []
    orig_propose = opt.movement_plugin.propose

    def mock_propose(particle_idx, state, context):
        recorded_w.append(context.w)
        return orig_propose(particle_idx, state, context)

    monkeypatch.setattr(opt.movement_plugin, "propose", mock_propose)
    opt.fit(x, y, epochs=3)

    # 3 epochs, 2 particles -> 4 velocity updates (movement on epoch 0 & 1, skipped on epoch 2)
    assert len(recorded_w) == 4
    assert math.isclose(recorded_w[0], 0.9)
    assert math.isclose(recorded_w[2], 0.1)

def test_deterministic_aggregate_and_renewal_selection(model_factory, xor_data):
    """Verify global best selection works deterministically across renewal options."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    for renewal in ("acc", "loss", "mse"):
        opt = Optimizer(model, loss, task="binary", n_particles=3, seed=42)
        score = opt.fit(x, y, epochs=2, renewal=renewal)
        assert len(score) == 3
        assert all(math.isfinite(s) for s in score)


def test_fixed_fitness_subset_and_batching(model_factory, xor_data):
    """Verify fitness_size restricts evaluation to a fixed subset of samples."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    opt = Optimizer(model, loss, task="binary", evaluation="fixed_subset", fitness_size=2, n_particles=3, seed=42)
    score = opt.fit(x, y, epochs=2)

    assert len(score) == 3
    assert all(math.isfinite(s) for s in score)

def test_validation_data_and_split(model_factory, xor_data, tmp_path):
    """Verify validation_data and validation_split populate run.json correctly."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    dir_val_data = tmp_path / "val_data"
    opt_data = Optimizer(model, loss, task="binary", n_particles=2, seed=42)
    opt_data.fit(
        x,
        y,
        epochs=2,
        validation_data=(x, y),
        output_dir=dir_val_data,
        save_info=True,
    )

    with open(dir_val_data / "run.json", "r", encoding="utf-8") as f:
        info_data = json.load(f)

    assert info_data["validation_source"] == "validation_data"
    assert info_data["validation_sample_count"] == 4
    assert isinstance(info_data["validation_score"], list)

    dir_val_split = tmp_path / "val_split"
    opt_split = Optimizer(model, loss, task="binary", n_particles=2, seed=42)
    opt_split.fit(
        x,
        y,
        epochs=2,
        validation_split=0.5,
        output_dir=dir_val_split,
        save_info=True,
    )

    with open(dir_val_split / "run.json", "r", encoding="utf-8") as f:
        info_split = json.load(f)

    assert info_split["validation_source"] == "validation_split"
    assert info_split["validation_sample_count"] == 2
    assert isinstance(info_split["validation_score"], list)


def test_validation_evaluated_once_at_end(model_factory, xor_data, monkeypatch):
    """Verify validation data is never evaluated during swarm iterations and evaluated exactly once at end."""
    x, y = xor_data
    val_x = x[:2]
    val_y = y[:2]

    model = model_factory()
    loss = nn.BCEWithLogitsLoss()
    opt = Optimizer(model, loss, task="binary", n_particles=2, seed=42)

    recorded_evals = []
    original_eval = opt._evaluate_batch_tensors

    def mock_eval(x_batch, y_batch):
        recorded_evals.append((x_batch.detach().cpu(), y_batch.detach().cpu()))
        return original_eval(x_batch, y_batch)

    monkeypatch.setattr(opt, "_evaluate_batch_tensors", mock_eval)
    opt.fit(x, y, epochs=2, validation_data=(val_x, val_y))

    # Swarm iterations evaluate x (4 samples).
    # Exactly the LAST call should evaluate val_x (2 samples).
    assert len(recorded_evals) > 1
    last_x, last_y = recorded_evals[-1]
    assert torch.equal(last_x, val_x.cpu())
    assert torch.equal(last_y, val_y.cpu())

    # Swarm iterations (all calls except last) evaluate x
    for bx, _ in recorded_evals[:-1]:
        assert bx.shape[0] == 4


def test_artifact_no_output_leaves_directory_untouched(
    model_factory, xor_data, tmp_path
):
    """Verify output_dir=None leaves target working directory untouched."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    opt = Optimizer(model, loss, task="binary", n_particles=2, seed=42)
    opt.fit(x, y, epochs=2, output_dir=None)

    assert list(tmp_path.iterdir()) == []


def test_artifact_pt_model_checkpoint_csv_tensorboard_and_run_json(
    model_factory, xor_data, tmp_path
):
    """Verify payload dict .pt artifacts, CSV logging, TensorBoard, and run.json layout."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    out_dir = tmp_path / "artifacts"
    opt = Optimizer(model, loss, task="binary", n_particles=2, seed=42)

    opt.fit(
        x,
        y,
        epochs=2,
        output_dir=out_dir,
        log_format="csv",
        checkpoint_interval=1,
        save_info=True,
    )

    # best_model.pt
    best_pt = out_dir / "best_model.pt"
    assert best_pt.exists()
    loaded_best = torch.load(best_pt, map_location="cpu", weights_only=True)
    assert isinstance(loaded_best, dict)
    assert "model_state_dict" in loaded_best
    assert "score" in loaded_best
    assert "task" in loaded_best
    assert "version" in loaded_best
    best_sd = loaded_best["model_state_dict"]
    assert isinstance(best_sd, collections.OrderedDict)

    sd = opt.get_best_state_dict()
    assert sd is not None
    for k in sd:
        assert best_sd[k].device.type == "cpu"
        assert torch.equal(best_sd[k], sd[k])

    # checkpoints
    ckpt_dir = out_dir / "checkpoints"
    assert ckpt_dir.exists()
    assert (ckpt_dir / "epoch-1.pt").exists()
    assert (ckpt_dir / "epoch-2.pt").exists()
    loaded_ckpt1 = torch.load(ckpt_dir / "epoch-1.pt", map_location="cpu", weights_only=True)
    assert isinstance(loaded_ckpt1, dict)
    assert "model_state_dict" in loaded_ckpt1
    assert isinstance(loaded_ckpt1["model_state_dict"], collections.OrderedDict)

    # history.csv
    csv_file = out_dir / "history.csv"
    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split(",") for line in f.readlines()]
    assert lines[0] == ["epoch", "loss", "accuracy", "mse"]
    assert len(lines) == 3  # Header + 2 epochs

    # run.json
    run_json = out_dir / "run.json"
    assert run_json.exists()
    with open(run_json, "r", encoding="utf-8") as f:
        run_data = json.load(f)
    assert run_data["task"] == "binary"
    assert run_data["version"] == pso.__version__
    assert run_data["config"]["method"] == "original"
    # TensorBoard test
    tb_dir = tmp_path / "tb_artifacts"
    opt_tb = Optimizer(model, loss, task="binary", n_particles=2, seed=42)
    opt_tb.fit(x, y, epochs=2, output_dir=tb_dir, log_format="tensorboard")

    assert (tb_dir / "best_model.pt").exists()
    assert (tb_dir / "tensorboard").exists()
    tb_files = list((tb_dir / "tensorboard").iterdir())
    assert len(tb_files) >= 1


def test_output_required_options_fail_fast_before_evaluation(
    model_factory, xor_data, monkeypatch
):
    """Verify output-requiring options fail fast before particle evaluation starts."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()
    opt = Optimizer(model, loss, task="binary")

    eval_calls = 0

    def mock_eval(x_batch, y_batch):
        nonlocal eval_calls
        eval_calls += 1
        return (torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0))

    monkeypatch.setattr(opt, "_evaluate_batch_tensors", mock_eval)
    with pytest.raises(ValueError):
        opt.fit(x, y, log_format="csv", output_dir=None)

    assert eval_calls == 0


def test_multiclass_task_and_cross_entropy(model_factory):
    """Verify multiclass task evaluation with CrossEntropyLoss and integer labels."""
    torch.manual_seed(42)
    x = torch.randn(10, 4, dtype=torch.float32)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=torch.int64)

    model = model_factory(input_dim=4, units=8, output_dim=3)
    loss = nn.CrossEntropyLoss()

    opt = Optimizer(model, loss, task="multiclass", n_particles=4, seed=42)
    score = opt.fit(x, y, epochs=3)

    assert isinstance(score, tuple)
    assert len(score) == 3
    assert all(math.isfinite(s) for s in score)
    assert 0.0 <= score[1] <= 1.0  # Accuracy in [0, 1]


def test_regression_task_and_mse(model_factory):
    """Verify regression task evaluation with MSELoss."""
    torch.manual_seed(42)
    x = torch.randn(8, 2, dtype=torch.float32)
    y = torch.randn(8, 1, dtype=torch.float32)

    model = model_factory(input_dim=2, units=4, output_dim=1)
    loss = nn.MSELoss()

    opt = Optimizer(model, loss, task="regression", n_particles=4, seed=42)
    score = opt.fit(x, y, epochs=3)

    assert isinstance(score, tuple)
    assert len(score) == 3
    assert all(math.isfinite(s) for s in score)
    assert math.isclose(score[0], score[2], rel_tol=1e-5, abs_tol=1e-5)  # Loss equals MSE for regression within tolerance
    assert score[1] == 0.0  # Accuracy is 0.0 for regression


def test_device_explicit_cpu(model_factory, xor_data):
    """Verify device='cpu' keeps model, particle, and state dict tensors on CPU."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    opt = Optimizer(model, loss, task="binary", device="cpu", seed=42)
    assert opt.device.type == "cpu"

    for p in opt.particles:
        assert p.position.device.type == "cpu"
        assert p.velocity.device.type == "cpu"

    opt.fit(x, y, epochs=2)
    sd = opt.get_best_state_dict()
    assert sd is not None
    for k, v in sd.items():
        assert v.device.type == "cpu"


def test_resolve_device_auto_priority_and_unavailable_raises(monkeypatch):
    """Verify resolve_device priority (MPS -> CUDA -> CPU) and unavailable device exceptions."""
    # Priority 1: MPS available
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None).type == "mps"

    # Priority 2: MPS unavailable, CUDA available
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(None).type == "cuda"

    # Priority 3: Neither available -> CPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None).type == "cpu"

    # Explicit unavailable device raises RuntimeError
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")

    with pytest.raises(ValueError, match="Unsupported device type"):
        resolve_device("invalid_device")


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS hardware/software support is not available on this platform",
)
def test_real_mps_smoke_if_available(model_factory, xor_data):
    """Verify real MPS device execution, tensor placement, synchronization, and CPU portability."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    opt = Optimizer(model, loss, task="binary", n_particles=3, device="mps", seed=42)

    assert opt.device.type == "mps"
    for p in opt.particles:
        assert p.position.device.type == "mps"
        assert p.velocity.device.type == "mps"

    score = opt.fit(x, y, epochs=2)
    assert len(score) == 3
    assert all(math.isfinite(s) for s in score)

    assert opt._global_best_weights is not None
    assert opt._global_best_weights.device.type == "mps"

    torch.mps.synchronize()

    sd = opt.get_best_state_dict()
    assert sd is not None
    for k, v in sd.items():
        assert v.device.type == "cpu", f"State dict tensor {k} should be on CPU but is on {v.device}"


def test_binary_1d_target_normalization_no_broadcasting(model_factory):
    """Verify binary [N, 1] logits model with 1-D [N] targets normalizes target shape and fits without broadcasting."""
    torch.manual_seed(42)
    x = torch.randn(6, 2, dtype=torch.float32)
    y_1d = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float32)  # Shape [6]

    model = model_factory(input_dim=2, units=4, output_dim=1)  # Output shape [6, 1]
    loss = nn.BCEWithLogitsLoss()

    opt = Optimizer(model, loss, task="binary", n_particles=3, seed=42)
    score = opt.fit(x, y_1d, epochs=2)

    assert isinstance(score, tuple)
    assert len(score) == 3
    assert all(math.isfinite(s) for s in score)


def test_regression_1d_target_normalization_and_mse(model_factory):
    """Verify regression [N, 1] model output with 1-D [N] targets normalizes shape, loss ≈ MSE, and no broadcasting."""
    torch.manual_seed(42)
    x = torch.randn(8, 2, dtype=torch.float32)
    y_1d = torch.randn(8, dtype=torch.float32)  # Shape [8]

    model = model_factory(input_dim=2, units=4, output_dim=1)  # Output shape [8, 1]
    loss = nn.MSELoss()

    opt = Optimizer(model, loss, task="regression", n_particles=3, seed=42)
    score = opt.fit(x, y_1d, epochs=2)

    assert isinstance(score, tuple)
    assert len(score) == 3
    assert all(math.isfinite(s) for s in score)
    assert math.isclose(score[0], score[2], rel_tol=1e-5, abs_tol=1e-5)


def test_binary_regression_incompatible_target_counts_fail_fast(model_factory):
    """Verify binary and regression fail with contextual ValueError when target element count mismatches output."""
    x = torch.randn(4, 2, dtype=torch.float32)
    y_bad = torch.zeros((4, 2), dtype=torch.float32)  # Leading dimension matches, element count does not.

    model = model_factory(input_dim=2, units=4, output_dim=1)  # Output shape [4, 1] -> 4 elements

    opt_bin = Optimizer(model, nn.BCEWithLogitsLoss(), task="binary", n_particles=2)
    with pytest.raises(ValueError, match="(?i)target element count"):
        opt_bin.fit(x, y_bad)

    opt_reg = Optimizer(model, nn.MSELoss(), task="regression", n_particles=2)
    with pytest.raises(ValueError, match="(?i)target element count"):
        opt_reg.fit(x, y_bad)


def test_multiclass_target_shapes_and_incompatible_fail_fast(model_factory):
    """Verify multiclass fits with [N, 1] integer targets reshaped to [N], and incompatible target shapes fail."""
    x = torch.randn(6, 4, dtype=torch.float32)
    # [N, 1] integer class targets
    y_col = torch.tensor([[0], [1], [2], [0], [1], [2]], dtype=torch.int64)

    model = model_factory(input_dim=4, units=8, output_dim=3)  # Output shape [6, 3]
    loss = nn.CrossEntropyLoss()

    opt = Optimizer(model, loss, task="multiclass", n_particles=3, seed=42)
    score = opt.fit(x, y_col, epochs=2)
    assert isinstance(score, tuple)
    assert all(math.isfinite(s) for s in score)

    # Incompatible target shape (e.g. 5 columns for 3 classes)
    y_bad = torch.randn(6, 5, dtype=torch.float32)
    with pytest.raises(ValueError, match="(?i)target shape"):
        opt.fit(x, y_bad)


def test_vector_applied_exp_not_expxb(model_factory, xor_data, monkeypatch):
    """Verify parameters are applied to eval_model E*P times, not E*P*B times."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()
    opt = Optimizer(model, loss, task="binary", n_particles=3, seed=42)

    apply_calls = 0
    orig_apply = opt.codec.apply_vector

    def mock_apply(vector, model_target):
        nonlocal apply_calls
        apply_calls += 1
        return orig_apply(vector, model_target)

    monkeypatch.setattr(opt.codec, "apply_vector", mock_apply)

    # 4 samples, batch_size=2 -> B=2 batches
    # Epochs E=2, n_particles P=3
    # Expected apply_vector calls = E * P = 2 * 3 = 6
    opt.fit(x, y, epochs=2, batch_size=2)

    assert apply_calls in (6, 7)


def test_final_movement_skipped_on_last_epoch(model_factory, xor_data, monkeypatch):
    """Verify particle velocity/position updates are skipped on the final evaluation epoch."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()
    opt = Optimizer(model, loss, task="binary", n_particles=3, seed=42)

    update_pos_calls = 0
    orig_propose = opt.movement_plugin.propose

    def mock_propose(*args, **kwargs):
        nonlocal update_pos_calls
        update_pos_calls += 1
        return orig_propose(*args, **kwargs)

    monkeypatch.setattr(opt.movement_plugin, "propose", mock_propose)

    # For 2 epochs, movement occurs only on epoch 0 (1 epoch of movement for 3 particles = 3 calls).
    # On final epoch (epoch 1), movement is skipped.
    opt.fit(x, y, epochs=2)

    assert update_pos_calls == 3

def test_seeded_runs_remain_equal(model_factory, xor_data):
    """Verify two seeded runs produce identical best scores and weights."""
    x, y = xor_data

    model1 = model_factory()
    loss1 = nn.BCEWithLogitsLoss()
    opt1 = Optimizer(model1, loss1, task="binary", refinement="adam", n_particles=4, seed=123)
    score1 = opt1.fit(x, y, epochs=3, refinement_epochs=2, refinement_lr=0.01)

    model2 = model_factory()
    loss2 = nn.BCEWithLogitsLoss()
    opt2 = Optimizer(model2, loss2, task="binary", refinement="adam", n_particles=4, seed=123)
    score2 = opt2.fit(x, y, epochs=3, refinement_epochs=2, refinement_lr=0.01)

    assert score1 == score2
    assert opt1._global_best_weights is not None
    assert opt2._global_best_weights is not None
    assert torch.equal(opt1._global_best_weights, opt2._global_best_weights)

def test_invalid_refinement_values_fail_before_evaluation(model_factory, xor_data, monkeypatch):
    """Verify invalid refinement_epochs and refinement_lr fail fast before evaluation starts."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()
    opt = Optimizer(model, loss, task="binary", n_particles=2)

    eval_calls = 0

    def mock_eval(*args, **kwargs):
        nonlocal eval_calls
        eval_calls += 1
        return (torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0))

    monkeypatch.setattr(opt, "_evaluate_batch_tensors", mock_eval)

    # Invalid refinement_epochs
    for bad_e in [-1, 1.5, True]:
        with pytest.raises(ValueError, match="refinement_epochs"):
            opt.fit(x, y, refinement_epochs=bad_e)

    # Invalid refinement_lr
    for bad_lr in [0.0, -0.01, float("nan"), True]:
        with pytest.raises(ValueError, match="refinement_lr"):
            opt.fit(x, y, refinement_epochs=1, refinement_lr=bad_lr)

    assert eval_calls == 0


def test_refinement_xor_seed_103_improves():
    """Verify XOR with seed 103 on CPU improves loss, reaches 1.0 accuracy, and detaches autograd graphs."""
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

    class XorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 4)
            self.tanh = nn.Tanh()
            self.fc2 = nn.Linear(4, 1)

        def forward(self, x):
            return self.fc2(self.tanh(self.fc1(x)))

    loss_fn = nn.BCEWithLogitsLoss()

    opt_pso = Optimizer(
        XorModel(),
        loss_fn,
        task="binary",
        method="inertia",
        n_particles=24,
        c0=0.5,
        c1=0.3,
        w_min=0.1,
        w_max=0.9,
        negative_swarm=0.1,
        mutation_swarm=0.05,
        particle_min=-2.0,
        particle_max=2.0,
        boundary_strategy="reflect",
        initial_position_noise=0.1,
        seed=103,
        device="cpu",
    )
    score_pso = opt_pso.fit(x, y, epochs=60, refinement_epochs=0)

    opt_refined = Optimizer(
        XorModel(),
        loss_fn,
        task="binary",
        method="inertia",
        refinement="adam",
        n_particles=24,
        c0=0.5,
        c1=0.3,
        w_min=0.1,
        w_max=0.9,
        negative_swarm=0.1,
        mutation_swarm=0.05,
        particle_min=-2.0,
        particle_max=2.0,
        boundary_strategy="reflect",
        initial_position_noise=0.1,
        seed=103,
        device="cpu",
    )
    score_refined = opt_refined.fit(
        x, y, epochs=60, refinement_epochs=100, refinement_lr=0.03
    )

    assert score_refined[1] == 1.0
    assert opt_refined._global_best_weights is not None
    assert opt_refined._global_best_weights.requires_grad is False

def test_rejected_candidates_cannot_worsen_best(model_factory, xor_data, monkeypatch):
    """Verify refinement candidates that perform worse than PSO global best do not overwrite best score."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()
    opt = Optimizer(model, loss, task="binary", n_particles=3, seed=42, device="cpu")

    # Run PSO optimization to get an initial best score
    score_pso = opt.fit(x, y, epochs=5, refinement_epochs=0)
    best_pso_score = opt.get_best_score()
    assert best_pso_score is not None

    # Force candidate evaluations in refinement to produce worse score
    monkeypatch.setattr(
        opt,
        "_evaluate_aggregate_score",
        lambda position, x_data, y_data, batch_size=None: (999.0, 0.0, 999.0),
    )

    # Run refinement with forced bad aggregate score evaluations
    opt._refine(x, y, refinement_epochs=2, refinement_lr=0.001, batch_size=None, renewal="acc")

    # Global best must remain unchanged!
    assert opt.get_best_score() == best_pso_score


def test_state_dict_remains_cpu_cloned(model_factory, xor_data):
    """Verify get_best_state_dict returns CPU-cloned state dict without storage aliasing."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()
    opt = Optimizer(model, loss, task="binary", n_particles=3, seed=42)
    opt.fit(x, y, epochs=2)

    sd1 = opt.get_best_state_dict()
    assert isinstance(sd1, collections.OrderedDict)

    # All tensors must be on CPU
    for k, v in sd1.items():
        assert v.device.type == "cpu"

    # Mutate tensors in sd1 in place
    for v in sd1.values():
        v.zero_()

    # Re-fetch state dict
    sd2 = opt.get_best_state_dict()
    assert sd2 is not None

    # Tensors in sd2 must be non-zero (unaffected by mutations to sd1)
    for k, v in sd2.items():
        assert not torch.all(v == 0)


def test_cpu_float64_model_fit():
    torch.manual_seed(42)
    x = torch.randn(8, 2, dtype=torch.float64)
    y = torch.randint(0, 2, (8, 1), dtype=torch.float64)

    class DoubleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1, dtype=torch.float64)

        def forward(self, x):
            return self.linear(x)

    model = DoubleModel()
    loss = nn.BCEWithLogitsLoss()
    opt = Optimizer(model, loss, task="binary", n_particles=3, seed=42, device="cpu")
    score = opt.fit(x, y, epochs=2)

    assert isinstance(score, tuple)
    assert len(score) == 3
    assert all(math.isfinite(s) for s in score)


def test_non_tensor_loss_raises_type_error(model_factory, xor_data):
    """Verify custom loss returning a non-Tensor object raises a clear TypeError."""
    x, y = xor_data

    class BadLoss(nn.Module):
        def forward(self, out, target):
            return 0.5  # Returns a float, not a torch.Tensor

    model = model_factory()
    opt = Optimizer(model, BadLoss(), task="binary", n_particles=2)
    with pytest.raises(TypeError, match="(?i)loss function must return a torch.Tensor"):
        opt.fit(x, y, epochs=1)


def test_invalid_moment_parameters_fail_fast(model_factory):
    """Verify constructor rejects invalid optimizer parameters before evaluation."""
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    invalid_specs = [
        # c0/c1: finite numbers
        {"c0": float("nan")},
        {"c0": True},
        {"c1": float("inf")},
        # w_min > w_max
        {"w_min": 0.9, "w_max": 0.1},
        # negative_swarm/mutation_swarm in [0, 1]
        {"negative_swarm": -0.1},
        {"negative_swarm": 1.5},
        {"mutation_swarm": True},
        # particle bounds
        {"particle_min": 1.0, "particle_max": -1.0},
        {"particle_min": float("nan"), "particle_max": 1.0},
        # seed
        {"seed": -1},
        {"seed": True},
    ]

    for kwargs in invalid_specs:
        with pytest.raises(ValueError):
            Optimizer(model, loss, task="binary", **kwargs)

def test_moment_blend_zero_preserves_standard_pso_and_leaves_moments_zero(
    model_factory, xor_data
):
    """Verify blend=0 preserves standard PSO behavior and leaves moments zeroed."""
    x, y = xor_data
    model1 = model_factory()
    model2 = model_factory()

    opt_default = Optimizer(
        model1, nn.BCEWithLogitsLoss(), task="binary", method="inertia", c0=0.3, c1=0.5, w_min=0.1, w_max=0.9, seed=42
    )
    opt_zero = Optimizer(
        model2, nn.BCEWithLogitsLoss(), task="binary", method="adaptive_moment", c0=0.3, c1=0.5, w_min=0.1, w_max=0.9, seed=42, moment_blend=0.0
    )

    score_default = opt_default.fit(x, y, epochs=3)
    score_zero = opt_zero.fit(x, y, epochs=3)

    assert score_default == score_zero
    assert torch.equal(
        opt_default._global_best_weights, opt_zero._global_best_weights
    )

    for m in opt_zero.movement_plugin.first_moments:
        assert m is None
    for m in opt_zero.movement_plugin.second_moments:
        assert m is None

def test_blend0_arithmetic_exact_equivalence():
    """Verify blend=0 velocity update is bit-for-bit identical to standard PSO formula."""
    from pso.plugins import AdaptiveMomentMovement, SwarmState, IterationContext
    from pso.optimizer import _RandomSource

    mock_rng = _RandomSource(seed=42)
    am = AdaptiveMomentMovement(c0=1.2, c1=1.5, moment_blend=0.0)

    pos = torch.tensor([[1.0, 2.0]])
    vel = torch.tensor([[0.5, -0.5]])
    pbest = torch.tensor([[3.0, 4.0]])
    gbest = torch.tensor([5.0, 6.0])

    state = SwarmState(
        positions=(pos[0],),
        velocities=(vel[0],),
        pbest_positions=(pbest[0],),
        pbest_scores=((0.5, 0.5, 0.5),),
        gbest_position=gbest,
        gbest_score=(0.5, 0.5, 0.5),
        pbest_improved=(False,),
    )
    context = IterationContext(
        epoch=0, total_epochs=10, w=0.8, particle_idx=0, is_negative=False, rng=mock_rng, optimizer=None
    )

    # With mock_rng uniform rand terms r1, r2 generated deterministically:
    r1 = mock_rng.uniform(pos[0].shape, 0.0, 1.0, device=pos.device, dtype=pos.dtype)
    r2 = mock_rng.uniform(pos[0].shape, 0.0, 1.0, device=pos.device, dtype=pos.dtype)

    expected_std_vel = (
        0.8 * vel[0]
        + 1.2 * r1 * (pbest[0] - pos[0])
        + 1.5 * r2 * (gbest - pos[0])
    )

    # Reset mock_rng seed to reproduce exact r1 and r2 inside propose
    mock_rng = _RandomSource(seed=42)
    context.rng = mock_rng
    x_new, v_new = am.propose(0, state, context)

    assert x_new is None
    assert torch.allclose(v_new, expected_std_vel)


def test_deterministic_adaptive_moment_step():
    """Verify exact first/second bias-corrected adaptive step for a deterministic raw direction."""
    from pso.plugins import AdaptiveMomentMovement, SwarmState, IterationContext, FitContext
    from pso.optimizer import _RandomSource

    mock_rng = _RandomSource(seed=42)
    am = AdaptiveMomentMovement(
        c0=1.0,
        c1=0.0,
        w_min=0.0,
        w_max=0.0,
        moment_blend=1.0,
        moment_beta1=0.9,
        moment_beta2=0.999,
        moment_step_size=1.0,
        moment_epsilon=1e-8,
    )

    base_vec = torch.tensor([0.0, 0.0])
    fit_ctx = FitContext(
        optimizer=None,
        model=None,
        eval_model=None,
        codec=None,
        base_vector=base_vec,
        n_particles=1,
        particle_min=None,
        particle_max=None,
        velocity_limit=None,
        boundary_strategy="clip",
        initial_position_noise=0.0,
        seed=42,
        device=torch.device("cpu"),
        rng=mock_rng,
        task="binary",
        x_train=torch.zeros((1, 1)),
        y_train=torch.zeros((1, 1)),
        batch_size=None,
        fitness_size=None,
        renewal="acc",
        epochs=1,
        refinement_epochs=0,
        refinement_lr=0.001,
        c0=1.0,
        c1=0.0,
        w_min=0.0,
        w_max=0.0,
    )
    am.prepare_fit(fit_ctx)

    state = SwarmState(
        positions=(torch.tensor([0.0, 0.0]),),
        velocities=(torch.tensor([0.0, 0.0]),),
        pbest_positions=(torch.tensor([3.0, 4.0]),),
        pbest_scores=((0.5, 0.5, 0.5),),
        gbest_position=torch.tensor([3.0, 4.0]),
        gbest_score=(0.5, 0.5, 0.5),
        pbest_improved=(False,),
    )
    iter_ctx = IterationContext(
        epoch=0, total_epochs=1, w=0.0, particle_idx=0, is_negative=False, rng=mock_rng, optimizer=None
    )

    r1 = mock_rng.uniform(torch.Size([2]), 0.0, 1.0)
    raw_v = r1 * torch.tensor([3.0, 4.0])

    mock_rng = _RandomSource(seed=42)
    iter_ctx.rng = mock_rng
    _, v_new = am.propose(0, state, iter_ctx)

    assert am.moment_steps[0] == 1
    assert am.first_moments[0] is not None
    assert am.second_moments[0] is not None
    assert torch.allclose(am.first_moments[0], (1.0 - 0.9) * raw_v)
    assert torch.allclose(am.second_moments[0], (1.0 - 0.999) * (raw_v ** 2))


def test_nonzero_moment_persists_through_zero_current_direction():
    """Verify particle moves past zero current direction via persistent accumulated moments."""
    from pso.plugins import AdaptiveMomentMovement, SwarmState, IterationContext, FitContext
    from pso.optimizer import _RandomSource

    mock_rng = _RandomSource(seed=42)
    am = AdaptiveMomentMovement(
        c0=1.0,
        c1=0.0,
        w_min=0.0,
        w_max=0.0,
        moment_blend=0.5,
        moment_beta1=0.9,
        moment_beta2=0.999,
        moment_step_size=1.0,
        moment_epsilon=1e-8,
    )
    base_vec = torch.tensor([0.0, 0.0])
    fit_ctx = FitContext(
        optimizer=None,
        model=None,
        eval_model=None,
        codec=None,
        base_vector=base_vec,
        n_particles=1,
        particle_min=None,
        particle_max=None,
        velocity_limit=None,
        boundary_strategy="clip",
        initial_position_noise=0.0,
        seed=42,
        device=torch.device("cpu"),
        rng=mock_rng,
        task="binary",
        x_train=torch.zeros((1, 1)),
        y_train=torch.zeros((1, 1)),
        batch_size=None,
        fitness_size=None,
        renewal="acc",
        epochs=1,
        refinement_epochs=0,
        refinement_lr=0.001,
        c0=1.0,
        c1=0.0,
        w_min=0.0,
        w_max=0.0,
    )
    am.prepare_fit(fit_ctx)

    # Step 1: non-zero direction
    state1 = SwarmState(
        positions=(torch.tensor([0.0, 0.0]),),
        velocities=(torch.tensor([0.0, 0.0]),),
        pbest_positions=(torch.tensor([2.0, 2.0]),),
        pbest_scores=((0.5, 0.5, 0.5),),
        gbest_position=torch.tensor([2.0, 2.0]),
        gbest_score=(0.5, 0.5, 0.5),
        pbest_improved=(False,),
    )
    iter_ctx1 = IterationContext(
        epoch=0, total_epochs=2, w=0.0, particle_idx=0, is_negative=False, rng=mock_rng, optimizer=None
    )
    _, v1 = am.propose(0, state1, iter_ctx1)
    assert am.moment_steps[0] == 1
    assert not torch.equal(am.first_moments[0], torch.zeros(2))

    # Step 2: zero standard velocity (particle at pbest and gbest)
    state2 = SwarmState(
        positions=(torch.tensor([2.0, 2.0]),),
        velocities=(torch.tensor([0.0, 0.0]),),
        pbest_positions=(torch.tensor([2.0, 2.0]),),
        pbest_scores=((0.5, 0.5, 0.5),),
        gbest_position=torch.tensor([2.0, 2.0]),
        gbest_score=(0.5, 0.5, 0.5),
        pbest_improved=(False,),
    )
    iter_ctx2 = IterationContext(
        epoch=1, total_epochs=2, w=0.0, particle_idx=0, is_negative=False, rng=mock_rng, optimizer=None
    )
    _, v2 = am.propose(0, state2, iter_ctx2)
    assert am.moment_steps[0] == 2
    assert torch.norm(v2) > 0.0


def test_moments_detached_device_dtype_after_fit(model_factory, xor_data):
    """Verify moments remain detached and match particle device and dtype after fit."""
    x, y = xor_data
    model = model_factory()
    opt = Optimizer(
        model,
        nn.BCEWithLogitsLoss(),
        task="binary",
        method="adaptive_moment",
        seed=42,
        moment_blend=0.5,
        moment_beta1=0.9,
        moment_beta2=0.999,
    )

    opt.fit(x, y, epochs=2)

    am = opt.movement_plugin
    for m1, m2 in zip(am.first_moments, am.second_moments):
        assert m1.requires_grad is False
        assert m2.requires_grad is False
        assert m1.device.type == opt.device.type
        assert m2.device.type == opt.device.type
        assert m1.dtype == torch.float32
        assert m2.dtype == torch.float32


def test_mutation_and_reset_clear_moments():
    """Verify particle reset and mutation replacement clear moment state."""
    from pso.plugins import AdaptiveMomentMovement, FitContext
    from pso.optimizer import _RandomSource

    mock_rng = _RandomSource(seed=42)
    am = AdaptiveMomentMovement(c0=1.0, c1=1.0, moment_blend=0.5)
    base_vec = torch.tensor([1.0, 1.0])
    fit_ctx = FitContext(
        optimizer=None,
        model=None,
        eval_model=None,
        codec=None,
        base_vector=base_vec,
        n_particles=1,
        particle_min=None,
        particle_max=None,
        velocity_limit=None,
        boundary_strategy="clip",
        initial_position_noise=0.0,
        seed=42,
        device=torch.device("cpu"),
        rng=mock_rng,
        task="binary",
        x_train=torch.zeros((1, 1)),
        y_train=torch.zeros((1, 1)),
        batch_size=None,
        fitness_size=None,
        renewal="acc",
        epochs=1,
        refinement_epochs=0,
        refinement_lr=0.001,
        c0=1.0,
        c1=1.0,
    )
    am.prepare_fit(fit_ctx)

    # Populate moment state
    am.moment_steps[0] = 5
    am.first_moments[0] = torch.tensor([0.5, 0.5])
    am.second_moments[0] = torch.tensor([0.25, 0.25])

    # Reset clears moments
    am.reset_particle_state(0)
    assert am.moment_steps[0] == 0
    assert torch.equal(am.first_moments[0], torch.tensor([0.0, 0.0]))
    assert torch.equal(am.second_moments[0], torch.tensor([0.0, 0.0]))


def test_run_json_records_all_five_moment_fields(model_factory, xor_data, tmp_path):
    """Verify run.json config dictionary records all five moment fields."""
    x, y = xor_data
    model = model_factory()
    opt = Optimizer(
        model,
        nn.BCEWithLogitsLoss(),
        task="binary",
        method="adaptive_moment",
        seed=42,
        moment_blend=0.25,
        moment_beta1=0.88,
        moment_beta2=0.995,
        moment_step_size=1.5,
        moment_epsilon=1e-7,
    )

    opt.fit(x, y, epochs=1, save_info=True, output_dir=tmp_path)

    run_json_path = tmp_path / "run.json"
    assert run_json_path.exists()

    with open(run_json_path, encoding="utf-8") as f:
        data = json.load(f)

    cfg = data["config"]
    assert cfg["moment_blend"] == 0.25
    assert cfg["moment_beta1"] == 0.88
    assert cfg["moment_beta2"] == 0.995
    assert cfg["moment_step_size"] == 1.5
    assert cfg["moment_epsilon"] == 1e-7
def test_unknown_and_incompatible_method_options_fail_fast_before_eval(
    model_factory, xor_data, monkeypatch
):
    """Verify unknown or incompatible method_options fail fast during Optimizer init before evaluation."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    eval_calls = 0
    orig_forward = model.forward

    def mock_forward(*args, **kwargs):
        nonlocal eval_calls
        eval_calls += 1
        return orig_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", mock_forward)

    # 1. Unknown option in method_options for original movement
    with pytest.raises((ValueError, TypeError)):
        Optimizer(
            model, loss, task="binary", method="original",
            method_options={"completely_unknown_parameter_name": 123}
        )
    assert eval_calls == 0

    # 2. Incompatible method option: negative_swarm with bare_bones
    with pytest.raises(ValueError, match="unsupported"):
        Optimizer(
            model, loss, task="binary", method="bare_bones", negative_swarm=0.5
        )
    assert eval_calls == 0


def test_repeated_fit_reinitializes_from_original_constructor_model_and_clears_early_stop(
    model_factory, xor_data, monkeypatch
):
    """Verify repeated fit reinitializes base vector from original constructor model and clears early stopping state."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    orig_constructor_weights = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()

    opt = Optimizer(
        model, loss, task="binary",
        convergence="early_stopping",
        convergence_patience=2,
        seed=42,
    )

    score1 = opt.fit(x, y, epochs=3)
    assert opt._global_best_weights is not None
    first_run_best = opt._global_best_weights.clone()

    early_plugin = opt.convergence_plugin
    prepared_cleared = False
    orig_prep = early_plugin.prepare_fit

    def mock_prep(ctx):
        nonlocal prepared_cleared
        orig_prep(ctx)
        if early_plugin.best_gbest_monitor is None and early_plugin.gbest_patience == 0:
            prepared_cleared = True

    monkeypatch.setattr(early_plugin, "prepare_fit", mock_prep)

    # Second fit on same Optimizer instance
    score2 = opt.fit(x, y, epochs=3)

    # Particles in second fit must be initialized from orig_constructor_weights, NOT first_run_best
    for p in opt.particles:
        diff_from_orig = torch.norm(p.position.cpu() - orig_constructor_weights.cpu())
        assert diff_from_orig < 5.0, "Particle position diverged from original constructor model base"

    # Early stopping plugin state was cleared during prepare_fit
    assert prepared_cleared is True


def test_particle_reset_uses_original_constructor_base_vector(
    model_factory, xor_data
):
    """Verify particle reset re-initializes position from the original constructor base_vector."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    orig_base = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()

    opt = Optimizer(
        model, loss, task="binary",
        convergence="particle_reset",
        convergence_patience=1,
        seed=42,
    )
    opt.fit(x, y, epochs=3)

    for p in opt.particles:
        assert p.position.shape == orig_base.shape


def test_inertia_schedule_bounds_and_last_movement_w_min(model_factory, xor_data, monkeypatch):
    """Verify inertia weight w stays in [w_min, w_max], starts at w_max and last movement reaches w_min."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    w_min_val, w_max_val = 0.1, 0.9
    opt = Optimizer(
        model, loss, task="binary", method="inertia",
        w_min=w_min_val, w_max=w_max_val, n_particles=2, seed=42
    )

    recorded_w = []
    orig_propose = opt.movement_plugin.propose

    def mock_propose(particle_idx, state, context):
        recorded_w.append(context.w)
        return orig_propose(particle_idx, state, context)

    monkeypatch.setattr(opt.movement_plugin, "propose", mock_propose)

    epochs = 5
    opt.fit(x, y, epochs=epochs)

    # 2 particles per epoch * 4 epochs of movement = 8 propose calls
    assert len(recorded_w) == 8

    for w in recorded_w:
        assert w_min_val <= w <= w_max_val

    # Epoch 0 (first movement) uses w_max
    assert math.isclose(recorded_w[0], w_max_val)
    assert math.isclose(recorded_w[1], w_max_val)

    # Epoch 3 (last movement before epoch 4 final evaluation) uses w_min
    assert math.isclose(recorded_w[-1], w_min_val)
    assert math.isclose(recorded_w[-2], w_min_val)
def test_two_epoch_inertia_sole_movement_gets_w_max(model_factory, xor_data, monkeypatch):
    """Verify in a 2-epoch run with inertia method, the sole movement step gets w_max."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    w_min_val, w_max_val = 0.1, 0.9
    opt = Optimizer(
        model, loss, task="binary", method="inertia",
        w_min=w_min_val, w_max=w_max_val, n_particles=2, seed=42
    )

    recorded_w = []
    orig_propose = opt.movement_plugin.propose

    def mock_propose(particle_idx, state, context):
        recorded_w.append(context.w)
        return orig_propose(particle_idx, state, context)

    monkeypatch.setattr(opt.movement_plugin, "propose", mock_propose)
    opt.fit(x, y, epochs=2)

    # 2-epoch fit has 1 movement step (epoch 0) across 2 particles = 2 propose calls
    assert len(recorded_w) == 2
    for w in recorded_w:
        assert math.isclose(w, w_max_val)


def test_standard_fit_no_swarm_snapshot_stacks(model_factory, xor_data, monkeypatch):
    """Verify standard method fit does not perform swarm [N, D] snapshot stacks while preserving correct fit behavior."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    opt = Optimizer(
        model, loss, task="binary", method="inertia", n_particles=4, seed=42
    )

    orig_stack = torch.stack
    snapshot_stack_calls = []

    def mock_stack(tensors, dim=0, *args, **kwargs):
        if isinstance(tensors, (list, tuple)) and len(tensors) == opt.n_particles:
            first = tensors[0]
            if isinstance(first, torch.Tensor) and first.ndim == 1:
                snapshot_stack_calls.append(len(tensors))
        return orig_stack(tensors, dim=dim, *args, **kwargs)

    monkeypatch.setattr(torch, "stack", mock_stack)

    score = opt.fit(x, y, epochs=3)
    assert len(snapshot_stack_calls) == 0, f"Expected 0 swarm snapshot stacks, got {len(snapshot_stack_calls)}"
    assert isinstance(score, tuple) and len(score) == 3
    assert all(math.isfinite(s) for s in score)
    assert opt.get_best_model() is not None


def test_optimizer_public_evaluate_contract(model_factory, xor_data):
    """Verify public Optimizer.evaluate method raises pre-fit RuntimeError and has shape/type/aggregate parity for binary and multiclass."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    opt_binary = Optimizer(model, loss, task="binary", n_particles=2, seed=42)

    # Pre-fit failure
    with pytest.raises(RuntimeError, match="(?i)(not available|not been run)"):
        opt_binary.evaluate(x, y)

    # Binary evaluation parity
    fit_score = opt_binary.fit(x, y, epochs=2)
    eval_score = opt_binary.evaluate(x, y)

    assert isinstance(eval_score, tuple) and len(eval_score) == 3
    assert all(isinstance(s, float) and math.isfinite(s) for s in eval_score)
    assert math.isclose(eval_score[0], fit_score[0], abs_tol=1e-5)
    assert math.isclose(eval_score[1], fit_score[1], abs_tol=1e-5)
    assert math.isclose(eval_score[2], fit_score[2], abs_tol=1e-5)

    # Binary evaluation with batching
    eval_batched = opt_binary.evaluate(x, y, batch_size=2)
    assert isinstance(eval_batched, tuple) and len(eval_batched) == 3
    assert all(isinstance(s, float) and math.isfinite(s) for s in eval_batched)

    # Multiclass evaluation parity
    mc_model = model_factory(input_dim=4, output_dim=3)
    mc_loss = nn.CrossEntropyLoss()
    opt_mc = Optimizer(mc_model, mc_loss, task="multiclass", n_particles=2, seed=42)

    torch.manual_seed(42)
    x_mc = torch.randn(6, 4)
    y_mc_1d = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64)
    y_mc_2d = torch.nn.functional.one_hot(y_mc_1d, num_classes=3).float()

    opt_mc.fit(x_mc, y_mc_1d, epochs=2)
    score_1d = opt_mc.evaluate(x_mc, y_mc_1d)
    score_2d = opt_mc.evaluate(x_mc, y_mc_2d)

    assert isinstance(score_1d, tuple) and len(score_1d) == 3
    assert isinstance(score_2d, tuple) and len(score_2d) == 3
    assert all(isinstance(s, float) and math.isfinite(s) for s in score_1d)
    assert all(isinstance(s, float) and math.isfinite(s) for s in score_2d)
    assert math.isclose(score_1d[0], score_2d[0], abs_tol=1e-5)
    assert math.isclose(score_1d[1], score_2d[1], abs_tol=1e-5)
def test_fixed_subset_size_supplied_only_to_fit(model_factory, xor_data):
    """Verify evaluation='fixed_subset' allows omitting fitness_size at Optimizer init and supplying it at fit."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    opt = Optimizer(model, loss, task="binary", evaluation="fixed_subset")
    assert opt.fitness_size is None

    score = opt.fit(x, y, epochs=2, fitness_size=2)
    assert isinstance(score, tuple) and len(score) == 3
    assert all(math.isfinite(s) for s in score)

    # Omission at both init and fit fails during fit
    opt2 = Optimizer(model, loss, task="binary", evaluation="fixed_subset")
    with pytest.raises(ValueError, match="requires a positive fitness_size"):
        opt2.fit(x, y, epochs=2)


def test_final_evaluation_skips_movement_on_epoch_end_and_pbest_stacking(
    model_factory, xor_data, monkeypatch
):
    """Verify final evaluation pass does not trigger movement on_epoch_end or stack CLPSO pbests."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    opt = Optimizer(model, loss, task="binary", method="clpso", n_particles=3, seed=42)

    epoch_end_calls = 0
    orig_on_epoch_end = opt.movement_plugin.on_epoch_end

    def spy_on_epoch_end(state, context):
        nonlocal epoch_end_calls
        epoch_end_calls += 1
        return orig_on_epoch_end(state, context)

    monkeypatch.setattr(opt.movement_plugin, "on_epoch_end", spy_on_epoch_end)

    epochs = 3
    opt.fit(x, y, epochs=epochs)

    # For 3 epochs, on_epoch_end should be called between epochs (epochs - 1 = 2 times)
    assert epoch_end_calls == 2, f"Expected 2 inter-epoch on_epoch_end calls for 3 epochs, got {epoch_end_calls}"
