import json
import math
import pytest
import torch
import torch.nn as nn

import pso
from pso import Optimizer
from pso.plugins import (
    BUILTIN_PLUGINS,
    BasePlugin,
    InitializationPlugin,
    EvaluationPlugin,
    MovementPlugin,
    ConvergencePlugin,
    RefinementPlugin,
    PluginMetadata,
    SwarmState,
    FitContext,
    IterationContext,
    OriginalMovement,
    InertiaMovement,
    ConstrictionMovement,
    FIPSMovement,
    CLPSOMovement,
    BareBonesMovement,
    AdaptiveMomentMovement,
    RingLocalBestMovement,
    QuantumMovement,
    ModelNoiseInitialization,
    UniformInitialization,
    FullEvaluation,
    FixedSubsetEvaluation,
    NoConvergence,
    ParticleResetConvergence,
    EarlyStoppingConvergence,
    NoRefinement,
    AdamRefinement,
    available_plugins,
    get_plugin,
)
from pso.optimizer import _RandomSource


def test_metadata_registry_and_public_exports():
    """Verify registry listing, stage filtering, and metadata properties across all plugins."""
    all_stages = available_plugins()
    assert set(all_stages.keys()) == {
        "movement",
        "initialization",
        "evaluation",
        "convergence",
        "refinement",
    }

    movement_plugins = available_plugins(stage="movement")
    expected_movement_keys = {
        "original",
        "inertia",
        "constriction",
        "fips",
        "clpso",
        "bare_bones",
        "adaptive_moment",
        "local_best",
        "quantum",
    }
    assert set(movement_plugins.keys()) == expected_movement_keys

    for stage, stage_dict in BUILTIN_PLUGINS.items():
        for name, cls in stage_dict.items():
            plugin = cls()
            meta = plugin.metadata
            assert isinstance(meta, PluginMetadata)
            assert meta.stage == stage
            assert isinstance(meta.title, str) and len(meta.title) > 0
            assert meta.fidelity in ("canonical", "experimental")
            assert isinstance(meta.gradient_required, bool)

    # Check specific provenance/fidelity invariants
    orig_meta = OriginalMovement.metadata
    assert orig_meta.title == "Original PSO"
    assert orig_meta.source == "10.1109/ICNN.1995.488968"
    assert orig_meta.gradient_required is False
    assert orig_meta.fidelity == "canonical"

    am_meta = AdaptiveMomentMovement.metadata
    assert am_meta.source is None
    assert am_meta.fidelity == "experimental"

    adam_meta = AdamRefinement.metadata
    assert adam_meta.gradient_required is True


def test_default_original_movement_formula():
    """Verify default movement ('original') has c0=c1=2.0 and no inertia velocity scaling."""
    rng = _RandomSource(seed=42)
    orig = OriginalMovement()
    assert orig.c0 == 2.0
    assert orig.c1 == 2.0

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
        epoch=0, total_epochs=10, w=0.5, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )

    r1 = rng.uniform(pos[0].shape, 0.0, 1.0)
    r2 = rng.uniform(pos[0].shape, 0.0, 1.0)

    # Reset seed to reproduce exact r1, r2
    rng = _RandomSource(seed=42)
    context.rng = rng
    x_new, v_new = orig.propose(0, state, context)

    expected_vel = vel[0] + 2.0 * r1 * (pbest[0] - pos[0]) + 2.0 * r2 * (gbest - pos[0])
    assert x_new is None
    assert torch.allclose(v_new, expected_vel)


def test_exact_one_step_movement_variants():
    """Verify deterministic single-step movement calculation for all movement plugins."""
    pos = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    vel = torch.tensor([[0.1, -0.1], [0.2, -0.2]])
    pbest = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    gbest = torch.tensor([5.0, 6.0])

    state = SwarmState(
        positions=tuple(pos),
        velocities=tuple(vel),
        pbest_positions=tuple(pbest),
        pbest_scores=((0.5, 0.5, 0.5), (0.4, 0.4, 0.4)),
        gbest_position=gbest,
        gbest_score=(0.4, 0.4, 0.4),
        pbest_improved=(False, False),
    )

    # 1. Inertia
    rng = _RandomSource(seed=42)
    inertia = InertiaMovement(c0=0.5, c1=0.5, w_min=0.1, w_max=0.9)
    ctx_inertia = IterationContext(
        epoch=0, total_epochs=5, w=0.9, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )
    r1 = rng.uniform(pos[0].shape, 0.0, 1.0)
    r2 = rng.uniform(pos[0].shape, 0.0, 1.0)
    rng = _RandomSource(seed=42)
    ctx_inertia.rng = rng
    _, v_inertia = inertia.propose(0, state, ctx_inertia)
    expected_inertia_v = 0.9 * vel[0] + 0.5 * r1 * (pbest[0] - pos[0]) + 0.5 * r2 * (gbest - pos[0])
    assert torch.allclose(v_inertia, expected_inertia_v)

    # 2. Constriction
    rng = _RandomSource(seed=42)
    const = ConstrictionMovement(c0=2.05, c1=2.05)
    ctx_const = IterationContext(
        epoch=0, total_epochs=5, w=1.0, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )
    r1 = rng.uniform(pos[0].shape, 0.0, 1.0)
    r2 = rng.uniform(pos[0].shape, 0.0, 1.0)
    rng = _RandomSource(seed=42)
    ctx_const.rng = rng
    _, v_const = const.propose(0, state, ctx_const)
    expected_const_v = const.chi * (vel[0] + 2.05 * r1 * (pbest[0] - pos[0]) + 2.05 * r2 * (gbest - pos[0]))
    assert torch.allclose(v_const, expected_const_v)

    # 3. FIPS
    rng = _RandomSource(seed=42)
    fips = FIPSMovement(c0=2.05, c1=2.05)
    ctx_fips = IterationContext(
        epoch=0, total_epochs=5, w=1.0, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )
    r1 = rng.uniform(pos[0].shape, 0.0, 4.1 / 2.0)
    r2 = rng.uniform(pos[0].shape, 0.0, 4.1 / 2.0)
    rng = _RandomSource(seed=42)
    ctx_fips.rng = rng
    _, v_fips = fips.propose(0, state, ctx_fips)
    expected_fips_v = fips.chi * (vel[0] + r1 * (pbest[0] - pos[0]) + r2 * (pbest[1] - pos[0]))
    assert torch.allclose(v_fips, expected_fips_v)

    # 4. CLPSO
    clpso = CLPSOMovement()
    fit_ctx = FitContext(
        optimizer=None, model=None, eval_model=None, codec=None,
        base_vector=pos[0], n_particles=2, particle_min=-5.0, particle_max=5.0,
        velocity_limit=None, boundary_strategy="clip", initial_position_noise=0.0,
        seed=42, device=torch.device("cpu"), rng=rng, task="binary",
        x_train=torch.zeros((1, 1)), y_train=torch.zeros((1, 1)), batch_size=None,
        fitness_size=None, renewal="acc", epochs=5, refinement_epochs=0, refinement_lr=0.001,
    )
    clpso.prepare_fit(fit_ctx)
    ctx_clpso = IterationContext(
        epoch=0, total_epochs=5, w=0.9, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )
    _, v_clpso = clpso.propose(0, state, ctx_clpso)
    assert v_clpso is not None and v_clpso.shape == pos[0].shape

    # 5. Bare Bones
    rng = _RandomSource(seed=42)
    bb = BareBonesMovement()
    ctx_bb = IterationContext(
        epoch=0, total_epochs=5, w=1.0, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )
    x_bb, v_bb = bb.propose(0, state, ctx_bb)
    assert x_bb is not None
    assert torch.equal(v_bb, torch.zeros_like(pos[0]))

    # 6. Adaptive Moment
    am = AdaptiveMomentMovement(moment_blend=0.5)
    am.prepare_fit(fit_ctx)
    ctx_am = IterationContext(
        epoch=0, total_epochs=5, w=0.5, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )
    _, v_am = am.propose(0, state, ctx_am)
    assert v_am is not None and v_am.shape == pos[0].shape
    # 7. Ring Local Best
    rng = _RandomSource(seed=42)
    ring_mov = RingLocalBestMovement(c0=1.49618, c1=1.49618, w_min=0.4, w_max=0.9, neighborhood_radius=1)
    ctx_ring = IterationContext(
        epoch=0, total_epochs=5, w=0.9, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )
    r1 = rng.uniform(pos[0].shape, 0.0, 1.0)
    r2 = rng.uniform(pos[0].shape, 0.0, 1.0)
    rng = _RandomSource(seed=42)
    ctx_ring.rng = rng
    _, v_ring = ring_mov.propose(0, state, ctx_ring)
    # For particle 0 with scores (0.5, 0.5, 0.5) vs particle 1 (0.4, 0.4, 0.4), under renewal="acc", particle 0 has higher acc (0.5 > 0.4), so lbest = pbest[0]
    expected_ring_v = 0.9 * vel[0] + 1.49618 * r1 * (pbest[0] - pos[0]) + 1.49618 * r2 * (pbest[0] - pos[0])
    assert torch.allclose(v_ring, expected_ring_v)

    # 8. Quantum
    rng = _RandomSource(seed=42)
    q_mov = QuantumMovement(beta_min=0.5, beta_max=1.0)
    ctx_q = IterationContext(
        epoch=1, total_epochs=5, w=1.0, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )
    phi = rng.uniform(pos[0].shape, 0.0, 1.0)
    u = rng.uniform(pos[0].shape, 0.0, 1.0)
    sign_rand = rng.uniform(pos[0].shape, 0.0, 1.0)
    rng = _RandomSource(seed=42)
    ctx_q.rng = rng
    x_q, v_q = q_mov.propose(0, state, ctx_q)
    mbest_expected = (pbest[0] + pbest[1]) / 2.0
    p_exp = phi * pbest[0] + (1.0 - phi) * gbest
    u_clamped = torch.clamp(u, min=1e-10, max=1.0)
    ln_u_inv = torch.log(1.0 / u_clamped)
    sign_exp = torch.where(sign_rand < 0.5, 1.0, -1.0)
    # epoch=1 in total_epochs=5 -> beta = beta_max = 1.0
    expected_x_q = p_exp + sign_exp * 1.0 * torch.abs(mbest_expected - pos[0]) * ln_u_inv
    assert torch.allclose(x_q, expected_x_q)
    assert torch.equal(v_q, torch.zeros_like(pos[0]))

    # The final applied move (epoch=T-1) reaches beta_min exactly.
    final_rng = _RandomSource(seed=42)
    ctx_q_final = IterationContext(
        epoch=4,
        total_epochs=5,
        w=1.0,
        particle_idx=0,
        is_negative=False,
        rng=final_rng,
        optimizer=None,
    )
    x_q_final, _ = q_mov.propose(0, state, ctx_q_final)
    expected_x_q_final = (
        p_exp
        + sign_exp
        * 0.5
        * torch.abs(mbest_expected - pos[0])
        * ln_u_inv
    )
    assert torch.allclose(x_q_final, expected_x_q_final)


def test_selector_and_incompatible_option_validation(model_factory, xor_data):
    """Verify fail-fast behavior for invalid plugin selectors and incompatible stage options."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    # Invalid stage selector strings
    with pytest.raises(ValueError, match="Unknown stage"):
        available_plugins(stage="nonexistent")

    with pytest.raises(ValueError, match="Unknown movement plugin"):
        Optimizer(model, loss, task="binary", method="nonexistent")

    with pytest.raises(ValueError, match="Unknown initialization plugin"):
        Optimizer(model, loss, task="binary", initialization="nonexistent")

    with pytest.raises(ValueError, match="Unknown evaluation plugin"):
        Optimizer(model, loss, task="binary", evaluation="nonexistent")

    with pytest.raises(ValueError, match="Unknown convergence plugin"):
        Optimizer(model, loss, task="binary", convergence="nonexistent")

    with pytest.raises(ValueError, match="Unknown refinement plugin"):
        Optimizer(model, loss, task="binary", refinement="nonexistent")

    # Constriction c0 + c1 <= 4.0
    with pytest.raises(ValueError, match=r"c0 \+ c1 > 4\.0"):
        ConstrictionMovement(c0=2.0, c1=2.0)

    # BareBones unsupported options
    bb = BareBonesMovement()
    rng = _RandomSource(seed=42)
    fit_ctx_bb_bad = FitContext(
        optimizer=None, model=None, eval_model=None, codec=None,
        base_vector=torch.tensor([0.0]), n_particles=1, particle_min=None, particle_max=None,
        velocity_limit=1.0, boundary_strategy="clip", initial_position_noise=0.0,
        seed=42, device=torch.device("cpu"), rng=rng, task="binary",
        x_train=torch.zeros((1, 1)), y_train=torch.zeros((1, 1)), batch_size=None,
        fitness_size=None, renewal="acc", epochs=1, refinement_epochs=0, refinement_lr=0.001,
    )
    with pytest.raises(ValueError, match="unsupported for Bare Bones"):
        bb.prepare_fit(fit_ctx_bb_bad)

    # Uniform initialization requires bounds
    uni = UniformInitialization()
    fit_ctx_uni_bad = FitContext(
        optimizer=None, model=None, eval_model=None, codec=None,
        base_vector=torch.tensor([0.0]), n_particles=1, particle_min=None, particle_max=None,
        velocity_limit=None, boundary_strategy="clip", initial_position_noise=0.0,
        seed=42, device=torch.device("cpu"), rng=rng, task="binary",
        x_train=torch.zeros((1, 1)), y_train=torch.zeros((1, 1)), batch_size=None,
        fitness_size=None, renewal="acc", epochs=1, refinement_epochs=0, refinement_lr=0.001,
    )
    with pytest.raises(ValueError, match="particle_min and particle_max"):
        uni.initialize(0, torch.tensor([0.0]), fit_ctx_uni_bad)

    # fitness_size with evaluation='full'
    with pytest.raises(ValueError, match="fitness_size is only valid with evaluation='fixed_subset'"):
        Optimizer(model, loss, task="binary", evaluation="full", fitness_size=2)

    # evaluation='fixed_subset' without fitness_size at constructor or fit time fails during fit
    opt_fs = Optimizer(model, loss, task="binary", evaluation="fixed_subset")
    with pytest.raises(ValueError, match="requires a positive fitness_size"):
        opt_fs.fit(x, y)

    # refinement_epochs > 0 with refinement='none'
    with pytest.raises(ValueError, match="refinement_epochs > 0 is valid only with refinement='adam'"):
        Optimizer(model, loss, task="binary", refinement="none", refinement_epochs=5)

    # Preconfigured custom MovementPlugin with conflicting kwarg
    custom_mov = InertiaMovement(c0=0.8, c1=0.8)
    with pytest.raises(ValueError, match="Conflicting parameter"):
        Optimizer(model, loss, task="binary", method=custom_mov, c0=0.5)


def test_evaluation_stage_semantics(model_factory):
    """Verify FullEvaluation and FixedSubsetEvaluation sample handling across epochs."""
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20, 1)).float()
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()
    rng = _RandomSource(seed=42)

    # Full evaluation
    full_eval = FullEvaluation()
    fit_ctx_full = FitContext(
        optimizer=None, model=None, eval_model=None, codec=None,
        base_vector=torch.tensor([0.0]), n_particles=2, particle_min=None, particle_max=None,
        velocity_limit=None, boundary_strategy="clip", initial_position_noise=0.0,
        seed=42, device=torch.device("cpu"), rng=rng, task="binary",
        x_train=x, y_train=y, batch_size=None, fitness_size=None, renewal="acc",
        epochs=3, refinement_epochs=0, refinement_lr=0.001,
    )
    full_eval.prepare_fit(fit_ctx_full)
    xf1, yf1 = full_eval.get_fitness_data(x, y, fit_ctx_full)
    xf2, yf2 = full_eval.get_fitness_data(x, y, fit_ctx_full)
    assert torch.equal(xf1, x) and torch.equal(xf2, x)

    # Fixed subset evaluation
    fixed_eval = FixedSubsetEvaluation(fitness_size=5)
    fit_ctx_fixed = FitContext(
        optimizer=None, model=None, eval_model=None, codec=None,
        base_vector=torch.tensor([0.0]), n_particles=2, particle_min=None, particle_max=None,
        velocity_limit=None, boundary_strategy="clip", initial_position_noise=0.0,
        seed=42, device=torch.device("cpu"), rng=rng, task="binary",
        x_train=x, y_train=y, batch_size=None, fitness_size=5, renewal="acc",
        epochs=3, refinement_epochs=0, refinement_lr=0.001,
    )
    fixed_eval.prepare_fit(fit_ctx_fixed)
    xs1, ys1 = fixed_eval.get_fitness_data(x, y, fit_ctx_fixed)
    xs2, ys2 = fixed_eval.get_fitness_data(x, y, fit_ctx_fixed)
    assert xs1.shape[0] == 5
    assert torch.equal(xs1, xs2) and torch.equal(ys1, ys2)


def test_every_fit_fresh_state_lifecycle(model_factory, xor_data):
    """Verify repeated fit() reinitializes all stage plugins and swarm state completely."""
    x, y = xor_data
    model1 = model_factory()
    model2 = model_factory()
    loss = nn.BCEWithLogitsLoss()

    opt1 = Optimizer(
        model1, loss, task="binary",
        method="adaptive_moment",
        convergence="particle_reset",
        refinement="adam",
        n_particles=4, seed=42,
        moment_blend=0.5,
        convergence_patience=2,
    )
    opt2 = Optimizer(
        model2, loss, task="binary",
        method="adaptive_moment",
        convergence="particle_reset",
        refinement="adam",
        n_particles=4, seed=42,
        moment_blend=0.5,
        convergence_patience=2,
    )

    # Independent seeded runs yield identical best weights and score
    score1 = opt1.fit(x, y, epochs=3, refinement_epochs=2)
    score2 = opt2.fit(x, y, epochs=3, refinement_epochs=2)
    assert score1 == score2
    assert torch.equal(opt1._global_best_weights, opt2._global_best_weights)

    # Second fit on opt1 clears global best and creates fresh particles & plugin state
    score1_run2 = opt1.fit(x, y, epochs=3, refinement_epochs=2)
    assert len(opt1.particles) == 4
    assert opt1._global_best_weights is not None
    assert all(math.isfinite(s) for s in score1_run2)


def test_convergence_stage_semantics(model_factory, xor_data):
    """Verify NoConvergence, ParticleResetConvergence, and EarlyStoppingConvergence behavior."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()
    rng = _RandomSource(seed=42)

    # 1. NoConvergence
    no_conv = NoConvergence()
    iter_ctx0 = IterationContext(
        epoch=0, total_epochs=10, w=0.5, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )
    assert no_conv.on_epoch_end((0.5, 0.5, 0.5), True, iter_ctx0) is False

    # 2. EarlyStoppingConvergence stops when patience is reached
    early_conv = EarlyStoppingConvergence(patience=2, min_delta=0.01, monitor="loss")
    fit_ctx = FitContext(
        optimizer=None, model=None, eval_model=None, codec=None,
        base_vector=torch.tensor([0.0]), n_particles=2, particle_min=None, particle_max=None,
        velocity_limit=None, boundary_strategy="clip", initial_position_noise=0.0,
        seed=42, device=torch.device("cpu"), rng=rng, task="binary",
        x_train=x, y_train=y, batch_size=None, fitness_size=None, renewal="acc",
        epochs=10, refinement_epochs=0, refinement_lr=0.001,
    )
    early_conv.prepare_fit(fit_ctx)

    # Epoch 0: initial gbest loss 0.5 (improved)
    iter_ctx1 = IterationContext(
        epoch=0, total_epochs=10, w=0.5, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )
    stop0 = early_conv.on_epoch_end((0.5, 0.5, 0.5), True, iter_ctx1)
    assert stop0 is False

    # Epoch 1: no improvement (gbest_improved=False) -> patience 1
    iter_ctx2 = IterationContext(
        epoch=1, total_epochs=10, w=0.5, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )
    stop1 = early_conv.on_epoch_end((0.5, 0.5, 0.5), False, iter_ctx2)
    assert stop1 is False

    # Epoch 2: no improvement (gbest_improved=False) -> patience 2 -> triggers stop
    iter_ctx3 = IterationContext(
        epoch=2, total_epochs=10, w=0.5, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )
    stop2 = early_conv.on_epoch_end((0.5, 0.5, 0.5), False, iter_ctx3)
    assert stop2 is True

    # 3. Integrated early stopping test
    opt_es = Optimizer(
        model, loss, task="binary", convergence="early_stopping", convergence_patience=2, seed=42
    )
    score_es = opt_es.fit(x, y, epochs=100)
    assert all(math.isfinite(s) for s in score_es)


def test_refinement_stage_semantics(model_factory, xor_data):
    """Verify NoRefinement and AdamRefinement contracts and gradient detachment."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    no_ref = NoRefinement()
    pos = torch.tensor([0.1, 0.2])
    score = (0.5, 0.5, 0.5)
    fit_ctx = FitContext(
        optimizer=None, model=None, eval_model=None, codec=None,
        base_vector=torch.tensor([0.0]), n_particles=2, particle_min=None, particle_max=None,
        velocity_limit=None, boundary_strategy="clip", initial_position_noise=0.0,
        seed=42, device=torch.device("cpu"), rng=_RandomSource(seed=42), task="binary",
        x_train=x, y_train=y, batch_size=None, fitness_size=None, renewal="acc",
        epochs=10, refinement_epochs=0, refinement_lr=0.001,
    )
    r_pos, r_score = no_ref.refine(pos, score, None, fit_ctx)
    assert torch.equal(r_pos, pos) and r_score == score

    adam_ref = AdamRefinement(epochs=5, lr=0.01)
    opt = Optimizer(model, loss, task="binary", refinement="adam", seed=42)
    score_res = opt.fit(x, y, epochs=2, refinement_epochs=5, refinement_lr=0.01)

    assert all(math.isfinite(s) for s in score_res)
    assert opt._global_best_weights is not None
    assert opt._global_best_weights.requires_grad is False

def test_adaptive_moment_allocation_guarantee(model_factory, xor_data):
    """Verify moments are NOT allocated when moment_blend=0.0 or standard methods used, and ARE allocated when moment_blend>0."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    # Standard method ('original') -> no moment plugin state
    opt_orig = Optimizer(model, loss, task="binary", method="original", seed=42)
    opt_orig.fit(x, y, epochs=2)
    assert not isinstance(opt_orig.movement_plugin, AdaptiveMomentMovement)

    # Adaptive moment with moment_blend=0.0 -> moment tensors remain None
    opt_am_zero = Optimizer(
        model, loss, task="binary", method="adaptive_moment", moment_blend=0.0, seed=42
    )
    opt_am_zero.fit(x, y, epochs=2)
    am_zero = opt_am_zero.movement_plugin
    assert isinstance(am_zero, AdaptiveMomentMovement)
    for m1, m2 in zip(am_zero.first_moments, am_zero.second_moments):
        assert m1 is None
        assert m2 is None

    # Adaptive moment with moment_blend=0.5 -> moment tensors allocated and initialized to zero
    opt_am_active = Optimizer(
        model, loss, task="binary", method="adaptive_moment", moment_blend=0.5, seed=42
    )
    opt_am_active.fit(x, y, epochs=2)
    am_active = opt_am_active.movement_plugin
    assert isinstance(am_active, AdaptiveMomentMovement)
    for m1, m2 in zip(am_active.first_moments, am_active.second_moments):
        assert isinstance(m1, torch.Tensor)
        assert isinstance(m2, torch.Tensor)
        assert m1.requires_grad is False
        assert m2.requires_grad is False


def test_run_json_provenance_and_selector_fields(model_factory, xor_data, tmp_path):
    """Verify run.json config dictionary records stage selectors and plugin metadata provenance."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    opt = Optimizer(
        model, loss, task="binary",
        method="inertia",
        initialization="model_noise",
        evaluation="fixed_subset",
        convergence="early_stopping",
        refinement="adam",
        fitness_size=2,
        convergence_patience=5,
        refinement_epochs=10,
        seed=42,
    )

    opt.fit(x, y, epochs=2, save_info=True, output_dir=tmp_path)

    run_file = tmp_path / "run.json"
    assert run_file.exists()

    with open(run_file, "r", encoding="utf-8") as f:
        info = json.load(f)

    assert info["version"] == pso.__version__
    cfg = info["config"]
    assert cfg["method"] == "inertia"
    assert cfg["initialization"] == "model_noise"
    assert cfg["evaluation"] == "fixed_subset"
    assert cfg["convergence"] == "early_stopping"
    assert cfg["refinement"] == "adam"

    plugins_meta = cfg["plugins"]
    assert plugins_meta["movement"]["title"] == "Inertia Weight PSO"
    assert plugins_meta["movement"]["source"] == "10.1109/ICEC.1998.699146"
    assert plugins_meta["movement"]["fidelity"] == "canonical"
    assert plugins_meta["movement"]["gradient_required"] is False

    assert plugins_meta["refinement"]["title"] == "Adam Post-Search Refinement"
    assert plugins_meta["refinement"]["gradient_required"] is True


def test_exact_doi_metadata_across_all_plugins():
    """Verify exact DOI source metadata across all builtin plugins."""
    expected_sources = {
        "original": "10.1109/ICNN.1995.488968",
        "inertia": "10.1109/ICEC.1998.699146",
        "constriction": "10.1109/4235.985692",
        "fips": "10.1109/TEVC.2004.826074",
        "clpso": "10.1109/TEVC.2005.857610",
        "bare_bones": "10.1109/SIS.2003.1202251",
        "adaptive_moment": None,
        "local_best": "10.1109/CEC.2002.1004493",
        "quantum": "10.1109/CEC.2004.1330875",
    }
    for name, expected_source in expected_sources.items():
        assert BUILTIN_PLUGINS["movement"][name]().metadata.source == expected_source

    for stage, stage_dict in BUILTIN_PLUGINS.items():
        if stage == "movement":
            continue
        for name, cls in stage_dict.items():
            if stage == "refinement" and name == "adam":
                assert cls().metadata.source == "10.1016/j.amc.2006.07.025"
            else:
                assert cls().metadata.source is None


def test_canonical_inertia_defaults_and_movement():
    """Verify canonical inertia movement defaults and exact one-step formula."""
    rng = _RandomSource(seed=42)
    inertia = InertiaMovement(c0=2.0, c1=2.0, w_min=0.4, w_max=0.9)
    assert inertia.c0 == 2.0
    assert inertia.c1 == 2.0
    assert inertia.w_min == 0.4
    assert inertia.w_max == 0.9

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
        epoch=0, total_epochs=5, w=0.9, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )

    r1 = rng.uniform(pos[0].shape, 0.0, 1.0)
    r2 = rng.uniform(pos[0].shape, 0.0, 1.0)

    rng = _RandomSource(seed=42)
    context.rng = rng
    _, v_new = inertia.propose(0, state, context)

    expected_vel = 0.9 * vel[0] + 2.0 * r1 * (pbest[0] - pos[0]) + 2.0 * r2 * (gbest - pos[0])
    assert torch.allclose(v_new, expected_vel)


def test_adaptive_moment_string_default_allocates_nonzero_moments(model_factory, xor_data):
    """Verify adaptive_moment string selector defaults to moment_blend=0.25 and allocates moments."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    opt = Optimizer(model, loss, task="binary", method="adaptive_moment", seed=42)
    am = opt.movement_plugin
    assert isinstance(am, AdaptiveMomentMovement)
    assert am.moment_blend == 0.25

    opt.fit(x, y, epochs=2)
    assert len(am.first_moments) == opt.n_particles
    assert len(am.second_moments) == opt.n_particles
    for m1, m2 in zip(am.first_moments, am.second_moments):
        assert isinstance(m1, torch.Tensor)
        assert isinstance(m2, torch.Tensor)
        assert m1.requires_grad is False
        assert m2.requires_grad is False

    has_nonzero = any(torch.norm(m1) > 0.0 for m1 in am.first_moments)
    assert has_nonzero


def test_clpso_tournament_uses_swarmstate_pbest_score_ordering():
    """Verify CLPSO exemplar tournaments order candidates via SwarmState.pbest_scores."""
    rng = _RandomSource(seed=42)
    clpso = CLPSOMovement(c=1.49445, refresh_gap=2)
    fit_ctx = FitContext(
        optimizer=None, model=None, eval_model=None, codec=None,
        base_vector=torch.tensor([0.0, 0.0]), n_particles=2, particle_min=-5.0, particle_max=5.0,
        velocity_limit=None, boundary_strategy="clip", initial_position_noise=0.0,
        seed=42, device=torch.device("cpu"), rng=rng, task="binary",
        x_train=torch.zeros((1, 2)), y_train=torch.zeros((1, 1)), batch_size=None,
        fitness_size=None, renewal="acc", epochs=5, refinement_epochs=0, refinement_lr=0.001,
    )
    clpso.prepare_fit(fit_ctx)

    pos = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    vel = torch.tensor([[0.1, 0.1], [0.1, 0.1]])
    pbest = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    gbest = torch.tensor([2.0, 2.0])

    state = SwarmState(
        positions=tuple(pos),
        velocities=tuple(vel),
        pbest_positions=tuple(pbest),
        pbest_scores=((0.8, 0.2, 0.8), (0.1, 0.9, 0.1)),
        gbest_position=gbest,
        gbest_score=(0.1, 0.9, 0.1),
        pbest_improved=(False, False),
    )

    clpso.stagnation[0] = 2
    iter_ctx = IterationContext(
        epoch=2, total_epochs=5, w=0.9, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )
    clpso.on_epoch_end(state, iter_ctx)
    assert clpso.stagnation[0] == 0

def test_clpso_stagnation_reset_and_vectorized_exemplar_gather():
    """Verify CLPSO improvement resets stagnation counter and vectorized exemplar gather selects expected pbest dimensions."""
    rng = _RandomSource(seed=42)
    clpso = CLPSOMovement(c=1.49445, refresh_gap=3)
    fit_ctx = FitContext(
        optimizer=None, model=None, eval_model=None, codec=None,
        base_vector=torch.tensor([0.0, 0.0]), n_particles=2, particle_min=-5.0, particle_max=5.0,
        velocity_limit=None, boundary_strategy="clip", initial_position_noise=0.0,
        seed=42, device=torch.device("cpu"), rng=rng, task="binary",
        x_train=torch.zeros((1, 2)), y_train=torch.zeros((1, 1)), batch_size=None,
        fitness_size=None, renewal="acc", epochs=5, refinement_epochs=0, refinement_lr=0.001,
    )
    clpso.prepare_fit(fit_ctx)

    pos = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    vel = torch.tensor([[0.1, 0.1], [0.1, 0.1]])
    pbest = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    gbest = torch.tensor([30.0, 40.0])

    # Test 1: Improvement resets stagnation
    clpso.stagnation = torch.tensor([2, 2], dtype=torch.int64)
    state_imp = SwarmState(
        positions=tuple(pos),
        velocities=tuple(vel),
        pbest_positions=tuple(pbest),
        pbest_scores=((0.8, 0.2, 0.8), (0.1, 0.9, 0.1)),
        gbest_position=gbest,
        gbest_score=(0.1, 0.9, 0.1),
        pbest_improved=(True, False),
    )
    iter_ctx = IterationContext(
        epoch=1, total_epochs=5, w=0.9, particle_idx=0, is_negative=False, rng=rng, optimizer=None
    )
    clpso.on_epoch_end(state_imp, iter_ctx)
    assert clpso.stagnation[0] == 0  # Particle 0 improved -> stagnation reset to 0
    # Test 2: Vectorized exemplar gather
    # Particle 0 exemplars [0, 1]: dim0 from particle 0 pbest (10.0), dim1 from particle 1 pbest (40.0)
    clpso.exemplars = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)

    rng_p0 = _RandomSource(seed=42)
    iter_ctx_p0 = IterationContext(
        epoch=1, total_epochs=5, w=0.9, particle_idx=0, is_negative=False, rng=rng_p0, optimizer=None
    )
    r_mock = _RandomSource(seed=42).uniform(pos[0].shape, 0.0, 1.0)

    _, v0_new = clpso.propose(0, state_imp, iter_ctx_p0)
    expected_e0 = torch.tensor([10.0, 40.0])
    expected_v0 = 0.9 * vel[0] + 1.49445 * r_mock * (expected_e0 - pos[0])
    assert torch.allclose(v0_new, expected_v0)
def test_fixed_subset_adam_refinement_receives_sampled_fitness_tensors(
    model_factory, xor_data, monkeypatch
):
    """Verify Adam refinement in fixed_subset mode operates on the exact sampled fitness tensors."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    opt = Optimizer(
        model, loss, task="binary",
        evaluation="fixed_subset",
        refinement="adam",
        fitness_size=2,
        refinement_epochs=2,
        seed=42,
    )

    captured_refine_args = []
    orig_refine = opt._refine

    def mock_refine(x_fit, y_fit, **kwargs):
        captured_refine_args.append((x_fit, y_fit))
        return orig_refine(x_fit, y_fit, **kwargs)

    monkeypatch.setattr(opt, "_refine", mock_refine)
    opt.fit(x, y, epochs=1, refinement_epochs=2)

    assert len(captured_refine_args) == 1
    x_fit_received, y_fit_received = captured_refine_args[0]
    assert x_fit_received.shape[0] == 2
    assert y_fit_received.shape[0] == 2


def test_fit_context_and_run_json_moment_field_signatures(model_factory, xor_data, tmp_path):
    """Verify FitContext and run.json signatures preserve all optional moment fields."""
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    opt = Optimizer(
        model, loss, task="binary",
        method="adaptive_moment",
        moment_blend=0.3,
        moment_beta1=0.85,
        moment_beta2=0.99,
        moment_step_size=0.8,
        moment_epsilon=1e-7,
        seed=42,
    )
    opt.fit(x, y, epochs=1, save_info=True, output_dir=tmp_path)

    with open(tmp_path / "run.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    cfg = data["config"]
    assert cfg["moment_blend"] == 0.3
    assert cfg["moment_beta1"] == 0.85
    assert cfg["moment_beta2"] == 0.99
    assert cfg["moment_step_size"] == 0.8
    assert cfg["moment_epsilon"] == 1e-7
def test_clpso_two_particle_all_own_fallback_chooses_external_exemplar():
    """Verify CLPSO two-particle (k=1) all-own fallback selects external candidate for at least one dimension."""
    clpso = CLPSOMovement(c=1.49, w_min=0.4, w_max=0.9, refresh_gap=7)
    rng = _RandomSource(seed=42)
    dim = 4

    fit_ctx = FitContext(
        optimizer=None,
        model=None,
        eval_model=None,
        codec=None,
        base_vector=torch.zeros(dim),
        n_particles=2,
        particle_min=None,
        particle_max=None,
        velocity_limit=None,
        boundary_strategy="clip",
        initial_position_noise=0.0,
        seed=42,
        device=torch.device("cpu"),
        rng=rng,
        task="binary",
        x_train=torch.zeros((4, 2)),
        y_train=torch.zeros((4, 1)),
        batch_size=None,
        fitness_size=None,
        renewal="acc",
        epochs=1,
        refinement_epochs=0,
        refinement_lr=0.001,
    )
    clpso.prepare_fit(fit_ctx)
    clpso.learning_probs = torch.zeros((2,), dtype=torch.float32)

    state = SwarmState(
        positions=tuple([torch.zeros(dim), torch.zeros(dim)]),
        velocities=tuple([torch.zeros(dim), torch.zeros(dim)]),
        pbest_positions=tuple([torch.zeros(dim), torch.zeros(dim)]),
        pbest_scores=((0.5, 0.5, 0.5), (0.4, 0.6, 0.4)),
        gbest_position=torch.zeros(dim),
        gbest_score=(0.4, 0.6, 0.4),
        pbest_improved=(False, False),
    )
    clpso.on_epoch_end(state, fit_ctx)

    clpso._sample_exemplars_for_particle(0, state, rng)
    ex = clpso.exemplars[0]

    assert not torch.all(ex == 0), "Particle 0 should not retain self-exemplar for all dimensions in fallback"
    assert torch.any(ex == 1), "Particle 0 must select particle 1 for at least one dimension in two-particle fallback"


def test_clpso_mse_tournament_ranking_order():
    """Verify CLPSO on_epoch_end ranks candidates by MSE ascending, loss ascending, then accuracy descending."""
    clpso = CLPSOMovement(c=1.49, w_min=0.4, w_max=0.9, refresh_gap=7)
    rng = _RandomSource(seed=42)
    dim = 2

    fit_ctx = FitContext(
        optimizer=None,
        model=None,
        eval_model=None,
        codec=None,
        base_vector=torch.zeros(dim),
        n_particles=3,
        particle_min=None,
        particle_max=None,
        velocity_limit=None,
        boundary_strategy="clip",
        initial_position_noise=0.0,
        seed=42,
        device=torch.device("cpu"),
        rng=rng,
        task="regression",
        x_train=torch.zeros((4, 2)),
        y_train=torch.zeros((4, 1)),
        batch_size=None,
        fitness_size=None,
        renewal="mse",
        epochs=1,
        refinement_epochs=0,
        refinement_lr=0.001,
    )
    clpso.prepare_fit(fit_ctx)

    state = SwarmState(
        positions=tuple([torch.zeros(dim)] * 3),
        velocities=tuple([torch.zeros(dim)] * 3),
        pbest_positions=tuple([torch.zeros(dim)] * 3),
        pbest_scores=(
            (0.5, 0.8, 0.1),  # Particle 0: mse=0.1, loss=0.5, acc=0.8
            (0.4, 0.8, 0.1),  # Particle 1: mse=0.1, loss=0.4, acc=0.8 (same mse, lower loss -> rank 1 < rank 0)
            (0.4, 0.9, 0.1),  # Particle 2: mse=0.1, loss=0.4, acc=0.9 (same mse, same loss, higher acc -> rank 2 < rank 1)
        ),
        gbest_position=torch.zeros(dim),
        gbest_score=(0.4, 0.9, 0.1),
        pbest_improved=(False, False, False),
    )
    clpso.on_epoch_end(state, fit_ctx)

    ranks = clpso._ranks
    assert ranks[2] < ranks[1] < ranks[0], f"Expected ranks[2] < ranks[1] < ranks[0], got {ranks}"
    assert ranks[2].item() == 0
    assert ranks[1].item() == 1
    assert ranks[0].item() == 2


def test_clpso_exemplar_gather_device_normalization():
    """Verify CLPSO propose handles CPU exemplar indices with normalized pbest matrix and particle device placement."""
    clpso = CLPSOMovement(c=1.49, w_min=0.4, w_max=0.9, refresh_gap=7)
    rng = _RandomSource(seed=42)
    dim = 3

    fit_ctx = FitContext(
        optimizer=None,
        model=None,
        eval_model=None,
        codec=None,
        base_vector=torch.zeros(dim),
        n_particles=2,
        particle_min=None,
        particle_max=None,
        velocity_limit=None,
        boundary_strategy="clip",
        initial_position_noise=0.0,
        seed=42,
        device=torch.device("cpu"),
        rng=rng,
        task="binary",
        x_train=torch.zeros((4, 2)),
        y_train=torch.zeros((4, 1)),
        batch_size=None,
        fitness_size=None,
        renewal="acc",
        epochs=1,
        refinement_epochs=0,
        refinement_lr=0.001,
    )
    clpso.prepare_fit(fit_ctx)

    clpso.exemplars = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.int64, device="cpu")
    clpso._pbest_matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, device="cpu")
    clpso._ranks = torch.tensor([0, 1], dtype=torch.int64, device="cpu")

    state = SwarmState(
        positions=tuple([torch.zeros(dim, dtype=torch.float32), torch.zeros(dim, dtype=torch.float32)]),
        velocities=tuple([torch.zeros(dim, dtype=torch.float32), torch.zeros(dim, dtype=torch.float32)]),
        pbest_positions=tuple([torch.zeros(dim), torch.zeros(dim)]),
        pbest_scores=((0.1, 0.9, 0.1), (0.2, 0.8, 0.2)),
        gbest_position=torch.zeros(dim),
        gbest_score=(0.1, 0.9, 0.1),
        pbest_improved=(False, False),
    )

    iter_ctx = IterationContext(
        epoch=0,
        total_epochs=1,
        w=0.5,
        particle_idx=0,
        is_negative=False,
        rng=rng,
        optimizer=None,
    )
    pos_opt, vel_opt = clpso.propose(0, state, iter_ctx)
    assert pos_opt is None
    assert vel_opt is not None
    assert vel_opt.device == state.positions[0].device
    assert vel_opt.dtype == state.positions[0].dtype
def test_ring_local_best_semantics(model_factory, xor_data):
    """Verify RingLocalBestMovement topology, wrapped ring neighbor deduplication, and optimization."""
    # Metadata assertions
    plugin_meta = available_plugins(stage="movement")["local_best"]
    assert plugin_meta.stage == "movement"
    assert plugin_meta.title == "Ring Local Best PSO"
    assert plugin_meta.source == "10.1109/CEC.2002.1004493"
    assert plugin_meta.fidelity == "canonical"
    assert plugin_meta.gradient_required is False

    # Option validation
    with pytest.raises(ValueError):
        RingLocalBestMovement(c0=float("nan"))
    with pytest.raises(ValueError):
        RingLocalBestMovement(w_min=0.9, w_max=0.4)
    with pytest.raises(ValueError):
        RingLocalBestMovement(neighborhood_radius=0)
    with pytest.raises(ValueError):
        RingLocalBestMovement(neighborhood_radius=1.5)

    # Wrapped ring deduplication for n=1 and n=2 swarms
    mov = RingLocalBestMovement(neighborhood_radius=2)
    pos = [torch.tensor([1.0, 1.0])]
    vel = [torch.tensor([0.0, 0.0])]
    pbest = [torch.tensor([2.0, 2.0])]
    scores = [(0.5, 0.5, 0.5)]
    state_n1 = SwarmState(
        positions=tuple(pos),
        velocities=tuple(vel),
        pbest_positions=tuple(pbest),
        pbest_scores=tuple(scores),
        gbest_position=pbest[0],
        gbest_score=scores[0],
        pbest_improved=(False,),
    )
    rng = _RandomSource(seed=42)
    ctx = IterationContext(epoch=1, total_epochs=5, w=0.5, particle_idx=0, is_negative=False, rng=rng, optimizer=None)
    pos_over, vel_over = mov.propose(0, state_n1, ctx)
    assert pos_over is None
    assert vel_over is not None

    # Full fit test with local_best
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()
    opt = Optimizer(
        model,
        loss,
        task="binary",
        method="local_best",
        c0=1.49618,
        c1=1.49618,
        w_min=0.4,
        w_max=0.9,
        seed=42,
    )
    res = opt.fit(x, y, epochs=5)
    assert len(res) == 3
    assert all(math.isfinite(s) for s in res)


def test_quantum_movement_semantics(model_factory, xor_data):
    """Verify QuantumMovement formula, beta scheduling, mbest caching, state resets, and error handling."""
    # Metadata assertions
    plugin_meta = available_plugins(stage="movement")["quantum"]
    assert plugin_meta.stage == "movement"
    assert plugin_meta.title == "Quantum PSO"
    assert plugin_meta.source == "10.1109/CEC.2004.1330875"
    assert plugin_meta.fidelity == "canonical"
    assert plugin_meta.gradient_required is False

    # Option validation
    with pytest.raises(ValueError):
        QuantumMovement(beta_min=1.2, beta_max=0.8)
    with pytest.raises(ValueError):
        QuantumMovement(beta_min=-0.1)

    # Rejection of unsupported controls
    q_mov = QuantumMovement()
    fit_ctx = FitContext(
        optimizer=None, model=None, eval_model=None, codec=None,
        base_vector=torch.zeros(2), n_particles=2, particle_min=None, particle_max=None,
        velocity_limit=None, boundary_strategy="clip", initial_position_noise=0.0,
        seed=42, device=torch.device("cpu"), rng=None, task="binary",
        x_train=torch.zeros((1, 1)), y_train=torch.zeros((1, 1)), batch_size=None,
        fitness_size=None, renewal="acc", epochs=5, refinement_epochs=0, refinement_lr=0.001,
        negative_swarm=0.1,
    )
    with pytest.raises(ValueError, match="negative_swarm is unsupported"):
        q_mov.prepare_fit(fit_ctx)

    fit_ctx_mut = FitContext(
        optimizer=None, model=None, eval_model=None, codec=None,
        base_vector=torch.zeros(2), n_particles=2, particle_min=None, particle_max=None,
        velocity_limit=None, boundary_strategy="clip", initial_position_noise=0.0,
        seed=42, device=torch.device("cpu"), rng=None, task="binary",
        x_train=torch.zeros((1, 1)), y_train=torch.zeros((1, 1)), batch_size=None,
        fitness_size=None, renewal="acc", epochs=5, refinement_epochs=0, refinement_lr=0.001,
        mutation_swarm=0.1,
    )
    with pytest.raises(ValueError, match="mutation_swarm is unsupported"):
        q_mov.prepare_fit(fit_ctx_mut)

    fit_ctx_vlim = FitContext(
        optimizer=None, model=None, eval_model=None, codec=None,
        base_vector=torch.zeros(2), n_particles=2, particle_min=None, particle_max=None,
        velocity_limit=0.5, boundary_strategy="clip", initial_position_noise=0.0,
        seed=42, device=torch.device("cpu"), rng=None, task="binary",
        x_train=torch.zeros((1, 1)), y_train=torch.zeros((1, 1)), batch_size=None,
        fitness_size=None, renewal="acc", epochs=5, refinement_epochs=0, refinement_lr=0.001,
    )
    with pytest.raises(ValueError, match="velocity_limit is unsupported"):
        q_mov.prepare_fit(fit_ctx_vlim)

    # Optimizer fail fast for quantum with unsupported parameters
    x, y = xor_data
    model = model_factory()
    loss = nn.BCEWithLogitsLoss()

    with pytest.raises(ValueError, match="negative_swarm is unsupported"):
        Optimizer(model, loss, task="binary", method="quantum", negative_swarm=0.1)

    with pytest.raises(ValueError, match="mutation_swarm is unsupported"):
        Optimizer(model, loss, task="binary", method="quantum", mutation_swarm=0.1)

    with pytest.raises(ValueError, match="velocity_limit is unsupported"):
        Optimizer(model, loss, task="binary", method="quantum", particle_min=-5.0, particle_max=5.0, velocity_limit_ratio=0.1)

    # Repeated fit state reset check
    opt = Optimizer(model, loss, task="binary", method="quantum", method_options={"beta_min": 0.5, "beta_max": 1.0}, seed=42)
    res1 = opt.fit(x, y, epochs=3)
    assert opt.movement_plugin._mbest is not None
    res2 = opt.fit(x, y, epochs=3)
    assert len(res2) == 3
    assert all(math.isfinite(s) for s in res2)
