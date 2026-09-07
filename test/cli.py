"""
Command-Line Interface Helpers for PSO Experiments.

Provides standard argparse argument groups and helper functions for PSO stage selectors
(method, initialization, evaluation, convergence, refinement) and common execution parameters.
"""

import argparse
from typing import Any

# Supported stage plugin selector options
STAGE_SELECTORS: dict[str, list[str]] = {
    "method": [
        "original",
        "inertia",
        "constriction",
        "fips",
        "clpso",
        "bare_bones",
        "adaptive_moment",
    ],
    "initialization": ["model_noise", "uniform"],
    "evaluation": ["full", "fixed_subset"],
    "convergence": ["none", "particle_reset", "early_stopping"],
    "refinement": ["none", "adam"],
}


def add_stage_selector_args(
    parser: argparse.ArgumentParser, defaults: dict[str, Any] | None = None
) -> argparse.ArgumentParser:
    """Adds the 5 explicit stage selector arguments to an argparse parser."""
    defaults = defaults or {}

    parser.add_argument(
        "--method",
        type=str,
        choices=STAGE_SELECTORS["method"],
        default=defaults.get("method", "original"),
        help="PSO movement stage method (default: %(default)s)",
    )
    parser.add_argument(
        "--initialization",
        type=str,
        choices=STAGE_SELECTORS["initialization"],
        default=defaults.get("initialization", "model_noise"),
        help="PSO particle initialization stage (default: %(default)s)",
    )
    parser.add_argument(
        "--evaluation",
        type=str,
        choices=STAGE_SELECTORS["evaluation"],
        default=defaults.get("evaluation", "full"),
        help="PSO objective evaluation stage (default: %(default)s)",
    )
    parser.add_argument(
        "--convergence",
        type=str,
        choices=STAGE_SELECTORS["convergence"],
        default=defaults.get("convergence", "none"),
        help="PSO convergence behavior stage (default: %(default)s)",
    )
    parser.add_argument(
        "--refinement",
        type=str,
        choices=STAGE_SELECTORS["refinement"],
        default=defaults.get("refinement", "none"),
        help="PSO post-search refinement stage (default: %(default)s)",
    )
    return parser


def add_pso_args(
    parser: argparse.ArgumentParser, defaults: dict[str, Any] | None = None
) -> argparse.ArgumentParser:
    """Adds stage selectors and common PSO hyperparameter arguments to an argparse parser."""
    defaults = defaults or {}

    # Add 5 stage selectors
    add_stage_selector_args(parser, defaults)

    # Core execution and hyperparameter options
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.get("seed", 42),
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.get("device", None),
        help="Execution target device (cpu, cuda, mps) (default: auto)",
    )
    parser.add_argument(
        "--n-particles",
        "--particles",
        dest="n_particles",
        type=int,
        default=defaults.get("n_particles", 30),
        help="Number of swarm particles (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=defaults.get("epochs", 80),
        help="Number of PSO optimization epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        "--batch",
        dest="batch_size",
        type=int,
        default=defaults.get("batch_size", None),
        help="Batch size for objective evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--fitness-size",
        type=int,
        default=defaults.get("fitness_size", None),
        help="Fixed subset sample count (required when evaluation='fixed_subset')",
    )
    parser.add_argument(
        "--refinement-epochs",
        type=int,
        default=defaults.get("refinement_epochs", 0),
        help="Refinement epoch count (required when refinement='adam')",
    )
    parser.add_argument(
        "--refinement-lr",
        type=float,
        default=defaults.get("refinement_lr", 0.001),
        help="Refinement learning rate for Adam optimizer (default: %(default)s)",
    )
    parser.add_argument(
        "--renewal",
        type=str,
        choices=["acc", "loss", "mse"],
        default=defaults.get("renewal", "loss"),
        help="Primary metric for global best selection (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=defaults.get("output_dir", None),
        help="Directory to save model checkpoints and logs",
    )

    # Optional coefficient overrides
    parser.add_argument(
        "--c0",
        type=float,
        default=defaults.get("c0", None),
        help="Cognitive acceleration coefficient override",
    )
    parser.add_argument(
        "--c1",
        type=float,
        default=defaults.get("c1", None),
        help="Social acceleration coefficient override",
    )
    parser.add_argument(
        "--w-min",
        dest="w_min",
        type=float,
        default=defaults.get("w_min", None),
        help="Minimum inertia weight override",
    )
    parser.add_argument(
        "--w-max",
        dest="w_max",
        type=float,
        default=defaults.get("w_max", None),
        help="Maximum inertia weight override",
    )
    parser.add_argument(
        "--negative-swarm",
        type=float,
        default=defaults.get("negative_swarm", 0.0),
        help="Negative swarm velocity coefficient (default: %(default)s)",
    )
    parser.add_argument(
        "--mutation-swarm",
        type=float,
        default=defaults.get("mutation_swarm", 0.0),
        help="Swarm mutation probability (default: %(default)s)",
    )
    parser.add_argument(
        "--particle-min",
        type=float,
        default=defaults.get("particle_min", None),
        help="Lower bound for particle position clamping",
    )
    parser.add_argument(
        "--particle-max",
        type=float,
        default=defaults.get("particle_max", None),
        help="Upper bound for particle position clamping",
    )
    parser.add_argument(
        "--velocity-limit-ratio",
        type=float,
        default=defaults.get("velocity_limit_ratio", None),
        help="Maximum velocity limit ratio relative to search domain",
    )
    parser.add_argument(
        "--boundary-strategy",
        type=str,
        choices=["clip", "reflect"],
        default=defaults.get("boundary_strategy", "clip"),
        help="Position boundary handling strategy (default: %(default)s)",
    )
    parser.add_argument(
        "--initial-position-noise",
        type=float,
        default=defaults.get("initial_position_noise", 0.05),
        help="Initial position noise scale (default: %(default)s)",
    )
    # Convergence stage options
    parser.add_argument(
        "--convergence-patience",
        dest="convergence_patience",
        type=int,
        default=defaults.get("convergence_patience", 10),
        help="Convergence reset/early stopping patience epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--convergence-min-delta",
        dest="convergence_min_delta",
        type=float,
        default=defaults.get("convergence_min_delta", 0.0001),
        help="Minimum improvement delta for convergence (default: %(default)s)",
    )
    parser.add_argument(
        "--convergence-monitor",
        dest="convergence_monitor",
        type=str,
        choices=["loss", "acc", "accuracy", "mse"],
        default=defaults.get("convergence_monitor", "loss"),
        help="Metric monitored for convergence (default: %(default)s)",
    )

    # Adaptive moment options
    parser.add_argument(
        "--moment-blend",
        dest="moment_blend",
        type=float,
        default=defaults.get("moment_blend", None),
        help="Adaptive moment blend factor (default: 0.25 when method='adaptive_moment', else 0.0)",
    )
    parser.add_argument(
        "--moment-beta1",
        dest="moment_beta1",
        type=float,
        default=defaults.get("moment_beta1", None),
        help="Adaptive moment beta1 parameter (default: 0.9 when method='adaptive_moment')",
    )
    parser.add_argument(
        "--moment-beta2",
        dest="moment_beta2",
        type=float,
        default=defaults.get("moment_beta2", None),
        help="Adaptive moment beta2 parameter (default: 0.999 when method='adaptive_moment')",
    )
    parser.add_argument(
        "--moment-step-size",
        dest="moment_step_size",
        type=float,
        default=defaults.get("moment_step_size", None),
        help="Adaptive moment step size (default: 1.0 when method='adaptive_moment')",
    )
    parser.add_argument(
        "--moment-epsilon",
        dest="moment_epsilon",
        type=float,
        default=defaults.get("moment_epsilon", None),
        help="Adaptive moment epsilon parameter (default: 1e-8 when method='adaptive_moment')",
    )

    # Repeatable method options
    parser.add_argument(
        "--method-option",
        dest="method_options",
        action="append",
        metavar="KEY=VALUE",
        help="Additional key=value option for movement method (repeatable)",
    )
    return parser


def parse_method_options(options: list[str] | None) -> dict[str, Any]:
    """Parses a list of 'KEY=VALUE' strings into a dictionary with typed values."""
    res: dict[str, Any] = {}
    if not options:
        return res
    for opt in options:
        if "=" not in opt:
            raise ValueError(f"Invalid --method-option format '{opt}', expected 'KEY=VALUE'")
        key, val = opt.split("=", 1)
        key = key.strip()
        val = val.strip()
        val_lower = val.lower()
        if val_lower == "true":
            parsed_val: Any = True
        elif val_lower == "false":
            parsed_val = False
        else:
            try:
                parsed_val = int(val)
            except ValueError:
                try:
                    parsed_val = float(val)
                except ValueError:
                    parsed_val = val
        res[key] = parsed_val
    return res


def build_optimizer_kwargs(
    args: argparse.Namespace,
    *,
    model: Any = None,
    loss: Any = None,
    task: str | None = None,
    inertia_profile: dict[str, float] | None = None,
    **extra_kwargs: Any,
) -> dict[str, Any]:
    """Builds Optimizer constructor keyword arguments from parsed CLI arguments.

    Applies method-specific parameter compatibility rules, workload inertia profiles
    (only when method='inertia'), repeatable method options (--method-option), and
    adaptive moment flags (only when method='adaptive_moment').
    """
    method = getattr(args, "method", "original")
    parsed_method_opts = parse_method_options(getattr(args, "method_options", None))

    if method == "inertia":
        profile = inertia_profile or {}
        c0 = (
            parsed_method_opts["c0"]
            if "c0" in parsed_method_opts
            else (args.c0 if getattr(args, "c0", None) is not None else profile.get("c0"))
        )
        c1 = (
            parsed_method_opts["c1"]
            if "c1" in parsed_method_opts
            else (args.c1 if getattr(args, "c1", None) is not None else profile.get("c1"))
        )
        w_min = (
            parsed_method_opts["w_min"]
            if "w_min" in parsed_method_opts
            else (args.w_min if getattr(args, "w_min", None) is not None else profile.get("w_min"))
        )
        w_max = (
            parsed_method_opts["w_max"]
            if "w_max" in parsed_method_opts
            else (args.w_max if getattr(args, "w_max", None) is not None else profile.get("w_max"))
        )
    elif method in ("original", "constriction", "fips"):
        c0 = (
            parsed_method_opts["c0"]
            if "c0" in parsed_method_opts
            else getattr(args, "c0", None)
        )
        c1 = (
            parsed_method_opts["c1"]
            if "c1" in parsed_method_opts
            else getattr(args, "c1", None)
        )
        w_min = None
        w_max = None
    elif method == "bare_bones":
        c0 = None
        c1 = None
        w_min = None
        w_max = None
    else:
        c0 = (
            parsed_method_opts["c0"]
            if "c0" in parsed_method_opts
            else getattr(args, "c0", None)
        )
        c1 = (
            parsed_method_opts["c1"]
            if "c1" in parsed_method_opts
            else getattr(args, "c1", None)
        )
        w_min = (
            parsed_method_opts["w_min"]
            if "w_min" in parsed_method_opts
            else getattr(args, "w_min", None)
        )
        w_max = (
            parsed_method_opts["w_max"]
            if "w_max" in parsed_method_opts
            else getattr(args, "w_max", None)
        )

    if method in ("fips", "clpso", "bare_bones"):
        neg_swarm = 0.0
    else:
        neg_swarm = (
            float(parsed_method_opts["negative_swarm"])
            if "negative_swarm" in parsed_method_opts
            else float(getattr(args, "negative_swarm", 0.0))
        )

    if method == "bare_bones":
        mut_swarm = 0.0
    else:
        mut_swarm = (
            float(parsed_method_opts["mutation_swarm"])
            if "mutation_swarm" in parsed_method_opts
            else float(getattr(args, "mutation_swarm", 0.0))
        )

    if method == "bare_bones":
        vel_ratio = None
    else:
        vel_ratio = (
            parsed_method_opts["velocity_limit_ratio"]
            if "velocity_limit_ratio" in parsed_method_opts
            else getattr(args, "velocity_limit_ratio", None)
        )

    fitness_size = (
        getattr(args, "fitness_size", None)
        if getattr(args, "evaluation", None) == "fixed_subset"
        else None
    )
    refinement_epochs = (
        getattr(args, "refinement_epochs", 0)
        if getattr(args, "refinement", None) == "adam"
        else 0
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "loss": loss,
        "task": task,
        "method": method,
        "initialization": getattr(args, "initialization", "model_noise"),
        "evaluation": getattr(args, "evaluation", "full"),
        "convergence": getattr(args, "convergence", "none"),
        "refinement": getattr(args, "refinement", "none"),
        "method_options": parsed_method_opts,
        "n_particles": getattr(args, "n_particles", 30),
        "c0": c0,
        "c1": c1,
        "w_min": w_min,
        "w_max": w_max,
        "negative_swarm": neg_swarm,
        "mutation_swarm": mut_swarm,
        "particle_min": getattr(args, "particle_min", None),
        "particle_max": getattr(args, "particle_max", None),
        "velocity_limit_ratio": vel_ratio,
        "boundary_strategy": getattr(args, "boundary_strategy", "clip"),
        "initial_position_noise": getattr(args, "initial_position_noise", 0.05),
        "seed": getattr(args, "seed", None),
        "device": getattr(args, "device", None),
        "fitness_size": fitness_size,
        "convergence_patience": getattr(args, "convergence_patience", 10),
        "convergence_min_delta": getattr(args, "convergence_min_delta", 0.0001),
        "convergence_monitor": getattr(args, "convergence_monitor", "loss"),
        "refinement_epochs": refinement_epochs,
        "refinement_lr": getattr(args, "refinement_lr", 0.001),
    }

    if method == "adaptive_moment":
        if "moment_blend" in parsed_method_opts:
            m_blend = float(parsed_method_opts["moment_blend"])
        elif getattr(args, "moment_blend", None) is not None:
            m_blend = float(getattr(args, "moment_blend"))
        else:
            m_blend = 0.25
        kwargs["moment_blend"] = m_blend

        for param_name in (
            "moment_beta1",
            "moment_beta2",
            "moment_step_size",
            "moment_epsilon",
        ):
            if param_name in parsed_method_opts:
                kwargs[param_name] = float(parsed_method_opts[param_name])
            elif getattr(args, param_name, None) is not None:
                kwargs[param_name] = float(getattr(args, param_name))

    kwargs.update(extra_kwargs)
    return kwargs
