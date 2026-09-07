import inspect
import subprocess
import sys
import pytest
import torch
import pso
from pso import Optimizer, Particle, __version__


def test_canonical_exports_and_all():
    """Verify pso exports Optimizer, Particle, __version__, stage plugins and defines __all__ correctly."""
    expected_all = [
        "Optimizer",
        "Particle",
        "__version__",
        "BasePlugin",
        "InitializationPlugin",
        "EvaluationPlugin",
        "MovementPlugin",
        "ConvergencePlugin",
        "RefinementPlugin",
        "PluginMetadata",
        "SwarmState",
        "available_plugins",
    ]
    assert pso.__all__ == expected_all
    assert pso.Optimizer is Optimizer
    assert pso.Particle is Particle
    assert pso.__version__ == "4.0.0"
    assert __version__ == "4.0.0"

    from pso.plugins import (
        BasePlugin,
        InitializationPlugin,
        EvaluationPlugin,
        MovementPlugin,
        ConvergencePlugin,
        RefinementPlugin,
        PluginMetadata,
        SwarmState,
        available_plugins,
    )
    assert pso.BasePlugin is BasePlugin
    assert pso.InitializationPlugin is InitializationPlugin
    assert pso.EvaluationPlugin is EvaluationPlugin
    assert pso.MovementPlugin is MovementPlugin
    assert pso.ConvergencePlugin is ConvergencePlugin
    assert pso.RefinementPlugin is RefinementPlugin
    assert pso.PluginMetadata is PluginMetadata
    assert pso.SwarmState is SwarmState
    assert pso.available_plugins is available_plugins


def test_lowercase_aliases_and_legacy_api_absent():
    """Verify lowercase names and legacy get_best_weights are excluded/absent."""
    assert "optimizer" not in pso.__all__
    assert "particle" not in pso.__all__
    assert not hasattr(Optimizer, "get_best_weights")
    assert hasattr(Optimizer, "get_best_state_dict")

    if hasattr(pso, "optimizer"):
        obj = getattr(pso, "optimizer")
        assert not isinstance(obj, type)

    if hasattr(pso, "particle"):
        obj = getattr(pso, "particle")
        assert not isinstance(obj, type)


def test_optimizer_init_signature_and_kwonly():
    """Verify Optimizer.__init__ parameter names and keyword-only positions."""
    sig = inspect.signature(Optimizer.__init__)
    params = sig.parameters

    assert "model" in params
    assert "loss" in params

    # Positional parameters (excluding self)
    assert params["model"].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    )
    assert params["loss"].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    )

    kwonly_expected = [
        "method",
        "initialization",
        "evaluation",
        "convergence",
        "refinement",
        "method_options",
        "n_particles",
        "c0",
        "c1",
        "w_min",
        "w_max",
        "negative_swarm",
        "mutation_swarm",
        "particle_min",
        "particle_max",
        "velocity_limit_ratio",
        "boundary_strategy",
        "initial_position_noise",
        "seed",
        "device",
        "fitness_size",
        "convergence_patience",
        "convergence_min_delta",
        "convergence_monitor",
        "refinement_epochs",
        "refinement_lr",
        "moment_blend",
        "moment_beta1",
        "moment_beta2",
        "moment_step_size",
        "moment_epsilon",
    ]

    for name in kwonly_expected:
        assert name in params, f"Missing parameter {name} in Optimizer.__init__"
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"Parameter {name} must be KEYWORD_ONLY"
        )


def test_optimizer_fit_signature_and_kwonly():
    """Verify Optimizer.fit parameter names and keyword-only positions."""
    sig = inspect.signature(Optimizer.fit)
    params = sig.parameters

    assert "x" in params
    assert "y" in params

    kwonly_expected = [
        "epochs",
        "batch_size",
        "fitness_size",
        "renewal",
        "validation_data",
        "validation_split",
        "output_dir",
        "log_format",
        "checkpoint_interval",
        "save_info",
    ]

    for name in kwonly_expected:
        assert name in params, f"Missing parameter {name} in Optimizer.fit"
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"Parameter {name} must be KEYWORD_ONLY"
        )


def test_kwonly_positional_and_unknown_kwargs(model_factory, xor_data):
    """Verify passing keyword-only arguments positionally or unknown kwargs raises TypeError."""
    x, y = xor_data
    model = model_factory()
    loss = torch.nn.BCEWithLogitsLoss()

    with pytest.raises(TypeError):
        Optimizer(model, loss, "binary")  # type: ignore[call-arg]

    opt = Optimizer(model, loss, task="binary")
    with pytest.raises(TypeError):
        opt.fit(x, y, invalid_unknown_arg=123)  # type: ignore[call-arg]


def test_subprocess_import_quiet_stdout():
    """Verify importing pso in a fresh subprocess produces exit code 0 and empty stdout."""
    res = subprocess.run(
        [sys.executable, "-c", "import pso"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Import failed with stderr: {res.stderr}"
    assert res.stdout == "", f"Expected empty stdout from import pso, got: {res.stdout!r}"
