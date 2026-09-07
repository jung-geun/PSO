import collections
import copy
import csv
import json
import math
import os
from typing import Any, Literal, Sequence
import torch
import torch.nn as nn

from ._version import __version__
from ._weights import ParameterCodec
from .particle import Particle
from .plugins import (
    BasePlugin,
    ConvergencePlugin,
    EvaluationPlugin,
    FitContext,
    InitializationPlugin,
    IterationContext,
    MovementPlugin,
    PluginMetadata,
    RefinementPlugin,
    SwarmState,
    _is_at_least_delta,
    _is_better_score,
    get_plugin,
)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """
    Resolves execution device.
    If device is None, auto-selects mps if available & built, else cuda, else cpu.
    Explicit device requirement ('mps', 'cuda', 'cpu') checks availability or raises RuntimeError.
    """
    if device is None:
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    if isinstance(device, str):
        try:
            dev = torch.device(device)
        except RuntimeError as e:
            raise ValueError(f"Unsupported device type: '{device}'") from e
    elif isinstance(device, torch.device):
        dev = device
    else:
        raise TypeError(
            f"device must be a string, torch.device, or None, got {type(device)}"
        )

    if dev.type == "mps":
        built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not (built and avail):
            raise RuntimeError(
                f"Explicit MPS device requested ('{device}'), but PyTorch MPS backend is not available (built={built}, available={avail})."
            )
    elif dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Explicit CUDA device requested ('{device}'), but CUDA is not available."
            )
    elif dev.type == "cpu":
        pass
    else:
        raise ValueError(
            f"Unsupported device type: '{dev.type}'. Only 'cpu', 'cuda', and 'mps' are supported."
        )

    return dev


def _is_better_score(
    new_score: tuple[float, float, float],
    best_score: tuple[float, float, float] | None,
    renewal: str = "acc",
) -> bool:
    """
    Returns True if new_score is strictly better than best_score according to renewal
    metric and deterministic tie breaks (loss asc, accuracy desc, mse asc).
    """
    if best_score is None:
        return True

    n_loss, n_acc, n_mse = new_score
    b_loss, b_acc, b_mse = best_score

    if renewal in ("acc", "accuracy"):
        if n_acc != b_acc:
            return n_acc > b_acc
    elif renewal == "loss":
        if n_loss != b_loss:
            return n_loss < b_loss
    elif renewal == "mse":
        if n_mse != b_mse:
            return n_mse < b_mse
    else:
        raise ValueError(f"Unknown renewal metric: {renewal}")

    if n_loss != b_loss:
        return n_loss < b_loss
    if n_acc != b_acc:
        return n_acc > b_acc
    if n_mse != b_mse:
        return n_mse < b_mse

    return False


def _is_at_least_delta(delta: float, min_delta: float) -> bool:
    """
    Returns True if directional improvement delta meets min_delta, handling floating-point boundaries.
    """
    if min_delta > 0:
        return delta > min_delta or math.isclose(
            delta, min_delta, rel_tol=1e-12, abs_tol=1e-15
        )
    return delta > 0


def _validate_score(
    score: Sequence[Any], particle_idx: int, iteration: int
) -> tuple[float, float, float]:
    """
    Validates that a score sequence contains 3 finite numbers.
    Raises FloatingPointError naming particle index and iteration if non-finite.
    """
    try:
        parsed = (float(score[0]), float(score[1]), float(score[2]))
    except (IndexError, TypeError, ValueError) as e:
        raise FloatingPointError(
            f"Invalid score format {score} for particle {particle_idx} at iteration {iteration}"
        ) from e

    if not all(math.isfinite(x) for x in parsed):
        raise FloatingPointError(
            f"Non-finite score {parsed} encountered for particle {particle_idx} at iteration {iteration}"
        )
    return parsed


class _RandomSource:
    """
    Private CPU/device random generator wrapper for drawing stochastic tensors and events.
    Transfers generated tensors to reference device and dtype if needed.
    """

    def __init__(self, seed: int | None = None, device: torch.device | str = "cpu"):
        self.cpu_generator = torch.Generator(device="cpu")
        dev = resolve_device(device) if not isinstance(device, torch.device) else device
        self.search_generator = (
            self.cpu_generator
            if dev.type == "cpu"
            else torch.Generator(device=dev)
        )

        if seed is not None:
            self.cpu_generator.manual_seed(seed)
            if self.search_generator is not self.cpu_generator:
                self.search_generator.manual_seed(seed)
        else:
            self.cpu_generator.seed()
            if self.search_generator is not self.cpu_generator:
                self.search_generator.seed()

    def uniform(
        self,
        shape: torch.Size | tuple[int, ...],
        low: float = 0.0,
        high: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if device is None:
            target_dev = self.search_generator.device
        else:
            target_dev = (
                resolve_device(device) if not isinstance(device, torch.device) else device
            )

        if self.search_generator.device.type == target_dev.type:
            r = torch.rand(
                shape, generator=self.search_generator, device=target_dev, dtype=dtype
            )
        else:
            r = torch.rand(
                shape, generator=self.cpu_generator, device="cpu", dtype=dtype
            ).to(device=target_dev, dtype=dtype)
        return low + (high - low) * r
    def bernoulli_event(self, p: float) -> bool:
        r = torch.rand((1,), generator=self.cpu_generator, device="cpu").item()
        return r < p

    def permutation(self, n: int) -> torch.Tensor:
        return torch.randperm(n, generator=self.cpu_generator, device="cpu")

    def choice(self, n: int, size: int) -> torch.Tensor:
        perm = torch.randperm(n, generator=self.cpu_generator, device="cpu")
        return perm[:size]

    def randint(
        self,
        low: int,
        high: int,
        size: tuple[int, ...] | int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if size is None:
            shape = (1,)
        elif isinstance(size, int):
            shape = (size,)
        else:
            shape = size
        r = torch.randint(
            low=low,
            high=high,
            size=shape,
            generator=self.cpu_generator,
            device="cpu",
        )
        if device is not None and str(device) != "cpu":
            return r.to(device)
        return r

class Optimizer:
    """
    Particle Swarm Optimizer for PyTorch nn.Module models with stage-plugin architecture.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        *,
        task: Literal["binary", "multiclass", "regression"],
        method: str | MovementPlugin = "original",
        initialization: str | InitializationPlugin = "model_noise",
        evaluation: str | EvaluationPlugin = "full",
        convergence: str | ConvergencePlugin = "none",
        refinement: str | RefinementPlugin = "none",
        method_options: dict[str, Any] | None = None,
        n_particles: int = 10,
        c0: float | None = None,
        c1: float | None = None,
        w_min: float | None = None,
        w_max: float | None = None,
        negative_swarm: float = 0.0,
        mutation_swarm: float = 0.0,
        particle_min: float | None = None,
        particle_max: float | None = None,
        velocity_limit_ratio: float | None = None,
        boundary_strategy: Literal["clip", "reflect"] = "clip",
        initial_position_noise: float = 0.05,
        seed: int | None = None,
        device: str | torch.device | None = None,
        fitness_size: int | None = None,
        convergence_patience: int = 10,
        convergence_min_delta: float = 0.0001,
        convergence_monitor: str = "loss",
        refinement_epochs: int = 0,
        refinement_lr: float = 0.001,
        moment_blend: float | None = None,
        moment_beta1: float | None = None,
        moment_beta2: float | None = None,
        moment_step_size: float | None = None,
        moment_epsilon: float | None = None,
    ):
        if model is None or not isinstance(model, nn.Module):
            raise ValueError("model must be an instance of torch.nn.Module")

        if loss is None or not isinstance(loss, nn.Module):
            raise ValueError("loss must be an instance of torch.nn.Module")

        if task not in ("binary", "multiclass", "regression"):
            raise ValueError(
                "task must be one of 'binary', 'multiclass', 'regression'"
            )

        if (
            isinstance(n_particles, bool)
            or not isinstance(n_particles, int)
            or n_particles < 1
        ):
            raise ValueError("n_particles must be an integer >= 1")

        for name, val in [
            ("c0", c0),
            ("c1", c1),
            ("w_min", w_min),
            ("w_max", w_max),
            ("moment_blend", moment_blend),
            ("moment_beta1", moment_beta1),
            ("moment_beta2", moment_beta2),
            ("moment_step_size", moment_step_size),
            ("moment_epsilon", moment_epsilon),
        ]:
            if val is not None and (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
            ):
                raise ValueError(f"{name} must be a finite number")

        if w_min is not None and w_max is not None and float(w_min) > float(w_max):
            raise ValueError("w_min must be <= w_max")

        for name, val in [
            ("negative_swarm", negative_swarm),
            ("mutation_swarm", mutation_swarm),
        ]:
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
                or not (0.0 <= float(val) <= 1.0)
            ):
                raise ValueError(f"{name} must be a finite float in range [0, 1]")

        if (particle_min is None) != (particle_max is None):
            raise ValueError("particle_min and particle_max must be provided together")

        if particle_min is not None and particle_max is not None:
            if (
                isinstance(particle_min, bool)
                or not isinstance(particle_min, (int, float))
                or not math.isfinite(particle_min)
            ):
                raise ValueError("particle_min must be a finite number")
            if (
                isinstance(particle_max, bool)
                or not isinstance(particle_max, (int, float))
                or not math.isfinite(particle_max)
            ):
                raise ValueError("particle_max must be a finite number")
            if particle_min > particle_max:
                raise ValueError("particle_min must be <= particle_max")

        if velocity_limit_ratio is not None:
            if (
                isinstance(velocity_limit_ratio, bool)
                or not isinstance(velocity_limit_ratio, (int, float))
                or not math.isfinite(velocity_limit_ratio)
                or not (0.0 < float(velocity_limit_ratio) <= 1.0)
            ):
                raise ValueError(
                    "velocity_limit_ratio must be a finite float in range (0, 1]"
                )
            if particle_min is None or particle_max is None:
                raise ValueError(
                    "velocity_limit_ratio requires paired particle_min and particle_max bounds"
                )
            if particle_min >= particle_max:
                raise ValueError(
                    "velocity_limit_ratio requires particle_min < particle_max"
                )

        if boundary_strategy not in ("clip", "reflect"):
            raise ValueError("boundary_strategy must be one of 'clip', 'reflect'")

        if boundary_strategy == "reflect":
            if particle_min is None or particle_max is None:
                raise ValueError(
                    "boundary_strategy 'reflect' requires paired particle_min and particle_max bounds"
                )
            if particle_min >= particle_max:
                raise ValueError(
                    "boundary_strategy 'reflect' requires particle_min < particle_max"
                )

        if (
            isinstance(initial_position_noise, bool)
            or not isinstance(initial_position_noise, (int, float))
            or not math.isfinite(initial_position_noise)
            or initial_position_noise < 0.0
        ):
            raise ValueError(
                "initial_position_noise must be a finite nonnegative number"
            )

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a non-negative integer")

        if (
            isinstance(convergence_patience, bool)
            or not isinstance(convergence_patience, int)
            or convergence_patience <= 0
        ):
            raise ValueError("convergence_patience must be a positive integer")

        if (
            isinstance(convergence_min_delta, bool)
            or not isinstance(convergence_min_delta, (int, float))
            or not math.isfinite(convergence_min_delta)
            or convergence_min_delta < 0
        ):
            raise ValueError(
                "convergence_min_delta must be a finite nonnegative number"
            )

        if convergence_monitor not in ("loss", "acc", "accuracy", "mse"):
            raise ValueError(
                "convergence_monitor must be one of 'loss', 'acc', 'accuracy', 'mse'"
            )

        if (
            isinstance(refinement_epochs, bool)
            or not isinstance(refinement_epochs, int)
            or refinement_epochs < 0
        ):
            raise ValueError("refinement_epochs must be a non-negative integer")

        if (
            isinstance(refinement_lr, bool)
            or not isinstance(refinement_lr, (int, float))
            or not math.isfinite(refinement_lr)
            or float(refinement_lr) <= 0.0
        ):
            raise ValueError("refinement_lr must be a positive finite float")

        self.device = resolve_device(device)
        self.task = task

        self.model = copy.deepcopy(model)
        self.eval_model = copy.deepcopy(model).to(self.device)
        self.eval_loss = copy.deepcopy(loss).to(self.device)

        if self.device.type == "mps":
            self.eval_model = self.eval_model.to(dtype=torch.float32)
            if hasattr(self.eval_loss, "to"):
                self.eval_loss = self.eval_loss.to(dtype=torch.float32)

        self.eval_model.eval()
        if hasattr(self.eval_loss, "eval"):
            self.eval_loss.eval()

        self.codec = ParameterCodec(self.eval_model)
        self._base_vector = self.codec.encode(self.eval_model).clone().detach()
        base_vector = self._base_vector.clone()

        self.n_particles = n_particles
        self.negative_swarm = float(negative_swarm)
        self.mutation_swarm = float(mutation_swarm)
        self.particle_min = float(particle_min) if particle_min is not None else None
        self.particle_max = float(particle_max) if particle_max is not None else None
        self.velocity_limit_ratio = (
            float(velocity_limit_ratio) if velocity_limit_ratio is not None else None
        )
        self.boundary_strategy = boundary_strategy
        self.initial_position_noise = float(initial_position_noise)

        if (
            self.velocity_limit_ratio is not None
            and self.particle_min is not None
            and self.particle_max is not None
        ):
            self.velocity_limit = float(
                self.velocity_limit_ratio * (self.particle_max - self.particle_min)
            )
        else:
            self.velocity_limit = None

        self.seed = seed
        self.renewal = "acc"
        self.fitness_size = fitness_size
        self.convergence_patience = convergence_patience
        self.convergence_min_delta = float(convergence_min_delta)
        self.convergence_monitor = convergence_monitor
        self.refinement_epochs = refinement_epochs
        self.refinement_lr = float(refinement_lr)

        self._method_selector = method if isinstance(method, str) else method.metadata.title
        self._initialization_selector = initialization if isinstance(initialization, str) else initialization.metadata.title
        self._evaluation_selector = evaluation if isinstance(evaluation, str) else evaluation.metadata.title
        self._convergence_selector = convergence if isinstance(convergence, str) else convergence.metadata.title
        self._refinement_selector = refinement if isinstance(refinement, str) else refinement.metadata.title

        # Split stage options cleanly to prevent cross-stage consumption
        movement_opts = dict(method_options or {})
        if c0 is not None:
            movement_opts["c0"] = c0
        if c1 is not None:
            movement_opts["c1"] = c1
        if w_min is not None:
            movement_opts["w_min"] = w_min
        if w_max is not None:
            movement_opts["w_max"] = w_max
        if moment_blend is not None:
            movement_opts["moment_blend"] = moment_blend
        if moment_beta1 is not None:
            movement_opts["moment_beta1"] = moment_beta1
        if moment_beta2 is not None:
            movement_opts["moment_beta2"] = moment_beta2
        if moment_step_size is not None:
            movement_opts["moment_step_size"] = moment_step_size
        if moment_epsilon is not None:
            movement_opts["moment_epsilon"] = moment_epsilon

        init_opts = {}
        if self._initialization_selector not in ("uniform", "Uniform Bounded Space Initialization"):
            if initial_position_noise != 0.05:
                init_opts["noise"] = initial_position_noise

        eval_opts = {}
        if self._evaluation_selector not in ("full", "Full Dataset Evaluation"):
            if fitness_size is not None:
                eval_opts["fitness_size"] = fitness_size

        conv_opts = {}
        if self._convergence_selector not in ("none", "No Convergence Action"):
            conv_opts = {
                "patience": convergence_patience,
                "min_delta": convergence_min_delta,
                "monitor": convergence_monitor,
            }

        refine_opts = {}
        if self._refinement_selector not in ("none", "No Refinement"):
            refine_opts = {
                "epochs": refinement_epochs,
                "lr": refinement_lr,
            }
        # Handle custom movement plugin conflict validation
        if isinstance(method, MovementPlugin):
            for k in ("c0", "c1", "w_min", "w_max"):
                if getattr(method, k, None) is not None and movement_opts.get(k) is not None:
                    if float(getattr(method, k)) != float(movement_opts[k]):
                        raise ValueError(
                            f"Conflicting parameter '{k}' provided for preconfigured custom movement instance"
                        )

        self.movement_plugin: MovementPlugin = get_plugin("movement", method, movement_opts)  # type: ignore
        self.initialization_plugin: InitializationPlugin = get_plugin(
            "initialization", initialization, init_opts
        )  # type: ignore
        self.evaluation_plugin: EvaluationPlugin = get_plugin(
            "evaluation", evaluation, eval_opts
        )  # type: ignore
        self.convergence_plugin: ConvergencePlugin = get_plugin(
            "convergence", convergence, conv_opts
        )  # type: ignore
        self.refinement_plugin: RefinementPlugin = get_plugin(
            "refinement", refinement, refine_opts
        )  # type: ignore

        # Resolve scalar parameter attributes from movement plugin (or None if irrelevant)
        self.c0 = (
            float(getattr(self.movement_plugin, "c0"))
            if hasattr(self.movement_plugin, "c0") and getattr(self.movement_plugin, "c0", None) is not None
            else None
        )
        self.c1 = (
            float(getattr(self.movement_plugin, "c1"))
            if hasattr(self.movement_plugin, "c1") and getattr(self.movement_plugin, "c1", None) is not None
            else None
        )
        self.w_min = (
            float(getattr(self.movement_plugin, "w_min"))
            if hasattr(self.movement_plugin, "w_min") and getattr(self.movement_plugin, "w_min", None) is not None
            else None
        )
        self.w_max = (
            float(getattr(self.movement_plugin, "w_max"))
            if hasattr(self.movement_plugin, "w_max") and getattr(self.movement_plugin, "w_max", None) is not None
            else None
        )

        self.moment_blend = (
            float(getattr(self.movement_plugin, "moment_blend"))
            if hasattr(self.movement_plugin, "moment_blend")
            else (float(moment_blend) if moment_blend is not None else 0.0)
        )
        self.moment_beta1 = (
            float(getattr(self.movement_plugin, "moment_beta1"))
            if hasattr(self.movement_plugin, "moment_beta1")
            else (float(moment_beta1) if moment_beta1 is not None else 0.9)
        )
        self.moment_beta2 = (
            float(getattr(self.movement_plugin, "moment_beta2"))
            if hasattr(self.movement_plugin, "moment_beta2")
            else (float(moment_beta2) if moment_beta2 is not None else 0.999)
        )
        self.moment_step_size = (
            float(getattr(self.movement_plugin, "moment_step_size"))
            if hasattr(self.movement_plugin, "moment_step_size")
            else (float(moment_step_size) if moment_step_size is not None else 1.0)
        )
        self.moment_epsilon = (
            float(getattr(self.movement_plugin, "moment_epsilon"))
            if hasattr(self.movement_plugin, "moment_epsilon")
            else (float(moment_epsilon) if moment_epsilon is not None else 1e-8)
        )

        # Validate stage/option combinations
        eval_title = self.evaluation_plugin.metadata.title
        if self.fitness_size is not None and eval_title != "Fixed Subset Evaluation":
            raise ValueError("fitness_size is only valid with evaluation='fixed_subset'")

        refine_title = self.refinement_plugin.metadata.title
        if self.refinement_epochs > 0 and refine_title == "No Refinement":
            raise ValueError("refinement_epochs > 0 is valid only with refinement='adam'")

        mv_title = self.movement_plugin.metadata.title
        if self.negative_swarm != 0.0 and mv_title in (
            "Fully Informed Particle Swarm (FIPS)",
            "Comprehensive Learning PSO (CLPSO)",
            "Bare Bones PSO",
            "Quantum PSO",
        ):
            raise ValueError(
                f"negative_swarm is unsupported for movement method '{mv_title}'"
            )

        if self.mutation_swarm != 0.0 and mv_title in (
            "Bare Bones PSO",
            "Quantum PSO",
        ):
            raise ValueError(f"mutation_swarm is unsupported for {mv_title}")

        if self.velocity_limit is not None and mv_title in (
            "Bare Bones PSO",
            "Quantum PSO",
        ):
            raise ValueError(f"velocity_limit is unsupported for {mv_title}")

        self._random_source = _RandomSource(seed=self.seed, device=self.device)
        self.generator = self._random_source.cpu_generator

        self._global_best_score: tuple[float, float, float] | None = None
        self._global_best_weights: torch.Tensor | None = None
        self.particles: list[Particle] = []

    def get_best_model(self) -> nn.Module | None:
        """
        Returns a fresh deepcopied eval-mode nn.Module on selected device with best parameters,
        or None if optimization has not been run.
        """
        if self._global_best_weights is None:
            return None
        best_model = copy.deepcopy(self.model).to(self.device)
        if self.device.type == "mps":
            best_model = best_model.to(dtype=torch.float32)
        self.codec.apply_vector(self._global_best_weights, best_model)
        best_model.eval()
        return best_model

    def get_best_score(self) -> tuple[float, float, float] | None:
        """
        Returns the best score as an immutable 3-float tuple (loss, acc, mse),
        or None if optimization has not been run.
        """
        if self._global_best_score is None:
            return None
        return (
            float(self._global_best_score[0]),
            float(self._global_best_score[1]),
            float(self._global_best_score[2]),
        )

    def get_best_state_dict(self) -> collections.OrderedDict[str, torch.Tensor] | None:
        """
        Returns a defensive CPU-cloned state dict of the best model,
        or None if optimization has not been run.
        """
        if self._global_best_weights is None:
            return None
        return self.codec.to_state_dict(self._global_best_weights, self.eval_model)
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> tuple[float, float, float]:
        """
        Evaluates the current best model weights on input data (x, y) with aggregate metric semantics.
        Validates input tensor shapes and types. Mutates no state and produces no artifacts.
        """
        if self._global_best_weights is None:
            raise RuntimeError("Optimization has not been run or best weights are unavailable")

        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("x and y must be torch.Tensor instances")

        if x.ndim == 0 or y.ndim == 0:
            raise ValueError("x and y must have a leading dimension (ndim >= 1)")

        len_x = x.shape[0]
        len_y = y.shape[0]
        if len_x != len_y:
            raise ValueError(f"x and y leading dimensions must match: {len_x} != {len_y}")
        if len_x == 0:
            raise ValueError("x and y leading dimensions must be nonzero")

        if batch_size is not None:
            if (
                isinstance(batch_size, bool)
                or not isinstance(batch_size, int)
                or batch_size <= 0
            ):
                raise ValueError("batch_size must be a positive integer")

        dtype_model = next(self.eval_model.parameters()).dtype
        x_dev = x.to(device=self.device, dtype=dtype_model)

        if self.task == "multiclass":
            if y.ndim > 1 and y.shape[-1] > 1:
                y_dev = y.to(device=self.device, dtype=dtype_model)
            else:
                y_dev = y.to(device=self.device, dtype=torch.int64)
        elif self.task in ("binary", "regression"):
            y_dev = y.to(device=self.device, dtype=dtype_model)

        raw_score = self._evaluate_aggregate_score(
            self._global_best_weights, x_dev, y_dev, batch_size=batch_size
        )
        return _validate_score(raw_score, particle_idx=-1, iteration=-1)

    def _compute_batch_loss(
        self, out: torch.Tensor, y_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalizes targets and computes (raw_loss, loss_tensor).
        """
        batch_size = out.shape[0]
        if self.task in ("binary", "regression"):
            if y_batch.numel() != out.numel():
                raise ValueError(
                    f"Target element count ({y_batch.numel()}) does not match model output element count ({out.numel()}) for task '{self.task}'."
                )
            target_norm = y_batch.to(device=out.device, dtype=out.dtype).reshape_as(out)
            raw_loss = self.eval_loss(out, target_norm)
        elif self.task == "multiclass":
            if out.ndim < 2:
                raise ValueError(
                    f"Multiclass model output must have rank >= 2 (logits of shape [batch_size, num_classes]), got shape {tuple(out.shape)}."
                )
            if tuple(y_batch.shape) == tuple(out.shape):
                target_norm = y_batch.to(device=out.device, dtype=out.dtype)
            elif y_batch.numel() == batch_size:
                target_norm = y_batch.to(
                    device=out.device, dtype=torch.int64
                ).reshape(-1)
            else:
                raise ValueError(
                    f"Multiclass target shape {tuple(y_batch.shape)} (numel={y_batch.numel()}) is incompatible with model output shape {tuple(out.shape)} (batch_size={batch_size})."
                )
            raw_loss = self.eval_loss(out, target_norm)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        if not isinstance(raw_loss, torch.Tensor):
            raise TypeError(
                f"Loss function must return a torch.Tensor, got {type(raw_loss).__name__}"
            )

        loss_tensor = (
            raw_loss.mean() if raw_loss.numel() > 1 else raw_loss.reshape(())
        )
        return raw_loss, loss_tensor

    def _compute_batch_metrics(
        self, out: torch.Tensor, y_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes (raw_loss, loss_tensor, acc_tensor, mse_tensor) for model output and target batch.
        """
        batch_size = out.shape[0]
        raw_loss, loss_tensor = self._compute_batch_loss(out, y_batch)

        if self.task in ("binary", "regression"):
            target_norm = y_batch.to(device=out.device, dtype=out.dtype).reshape_as(out)
            if self.task == "binary":
                if isinstance(self.eval_loss, nn.BCEWithLogitsLoss):
                    probs = torch.sigmoid(out)
                else:
                    probs = out
                preds = (probs >= 0.5).to(out.dtype)
                acc_tensor = (preds == target_norm).to(dtype=out.dtype).mean()
                mse_tensor = torch.mean((probs - target_norm) ** 2)
            else:
                acc_tensor = torch.tensor(0.0, device=out.device, dtype=out.dtype)
                mse_tensor = torch.mean((out - target_norm) ** 2)

        elif self.task == "multiclass":
            if tuple(y_batch.shape) == tuple(out.shape):
                target_norm = y_batch.to(device=out.device, dtype=out.dtype)
                target_classes = torch.argmax(target_norm, dim=-1)
                y_one_hot = target_norm
            else:
                target_norm = y_batch.to(
                    device=out.device, dtype=torch.int64
                ).reshape(-1)
                target_classes = target_norm
                num_classes = out.shape[-1]
                y_one_hot = torch.nn.functional.one_hot(
                    target_classes, num_classes=num_classes
                ).to(dtype=out.dtype)

            probs = torch.softmax(out, dim=-1)
            preds = torch.argmax(out, dim=-1)
            acc_tensor = (preds == target_classes).to(dtype=out.dtype).mean()
            mse_tensor = torch.mean((probs - y_one_hot) ** 2)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return raw_loss, loss_tensor, acc_tensor, mse_tensor

    def _evaluate_batch_tensors(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates batch metrics on current installed model parameters.
        Returns 0D device tensors (loss, acc, mse).
        """
        out = self.eval_model(x_batch)
        if not isinstance(out, torch.Tensor) or out.ndim < 1:
            raise ValueError("Model output must be a torch.Tensor with rank >= 1.")

        batch_size = x_batch.shape[0]
        if out.shape[0] != batch_size:
            raise ValueError(
                f"Model output leading dimension ({out.shape[0]}) does not match batch size ({batch_size})."
            )

        _, loss_t, acc_t, mse_t = self._compute_batch_metrics(out, y_batch)
        return loss_t, acc_t, mse_t

    def _evaluate_batch(
        self, position: torch.Tensor, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> tuple[float, float, float]:
        """
        Scalar tuple evaluation wrapper.
        """
        self.codec.apply_vector(position, self.eval_model)
        with torch.inference_mode():
            t_loss, t_acc, t_mse = self._evaluate_batch_tensors(x_batch, y_batch)
        return (t_loss.item(), t_acc.item(), t_mse.item())

    def _evaluate_aggregate_score(
        self,
        position: torch.Tensor,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        batch_size: int | None = None,
    ) -> tuple[float, float, float]:
        """
        Evaluates aggregate score across all batches of x_data, y_data for a given position.
        """
        self.codec.apply_vector(position, self.eval_model)
        n_samples = x_data.shape[0]
        effective_batch_size = (
            n_samples
            if batch_size is None or batch_size >= n_samples
            else batch_size
        )

        acc_dtype = self.codec.dtype
        batch_loss_sum = torch.tensor(0.0, device=self.device, dtype=acc_dtype)
        batch_acc_sum = torch.tensor(0.0, device=self.device, dtype=acc_dtype)
        batch_mse_sum = torch.tensor(0.0, device=self.device, dtype=acc_dtype)
        total_samples = 0

        with torch.inference_mode():
            for start in range(0, n_samples, effective_batch_size):
                end = min(start + effective_batch_size, n_samples)
                batch_len = end - start
                x_b = x_data[start:end]
                y_b = y_data[start:end]
                t_loss, t_acc, t_mse = self._evaluate_batch_tensors(x_b, y_b)
                batch_loss_sum += t_loss * batch_len
                batch_acc_sum += t_acc * batch_len
                batch_mse_sum += t_mse * batch_len
                total_samples += batch_len

        agg_tensor = torch.stack([
            batch_loss_sum / total_samples,
            batch_acc_sum / total_samples,
            batch_mse_sum / total_samples,
        ])
        agg_cpu = agg_tensor.detach().cpu().tolist()
        return (float(agg_cpu[0]), float(agg_cpu[1]), float(agg_cpu[2]))

    def _optimize(
        self,
        x_fitness: torch.Tensor,
        y_fitness: torch.Tensor,
        fit_context: FitContext,
        *,
        epochs: int = 10,
        batch_size: int | None = None,
        renewal: str = "acc",
        checkpoint_interval: int | None = None,
    ) -> tuple[
        tuple[float, float, float],
        list[dict[str, Any]],
        dict[int, torch.Tensor],
    ]:
        n_samples = x_fitness.shape[0]
        effective_batch_size = (
            n_samples
            if batch_size is None or batch_size >= n_samples
            else batch_size
        )

        history: list[dict[str, Any]] = []
        checkpoint_snapshots: dict[int, torch.Tensor] = {}

        for epoch in range(epochs):
            epoch_num = epoch + 1
            if self.w_min is not None and self.w_max is not None:
                if epochs <= 2:
                    w = self.w_max
                else:
                    w_raw = self.w_max - (self.w_max - self.w_min) * (epoch / (epochs - 2))
                    w = max(self.w_min, min(self.w_max, w_raw))
            else:
                w = 0.0

            epoch_scores = torch.zeros(
                (self.n_particles, 3), device=self.device, dtype=self.codec.dtype
            )

            for i, p in enumerate(self.particles):
                self.codec.apply_vector(p.position, self.eval_model)

                batch_loss_sum = torch.tensor(
                    0.0, device=self.device, dtype=self.codec.dtype
                )
                batch_acc_sum = torch.tensor(
                    0.0, device=self.device, dtype=self.codec.dtype
                )
                batch_mse_sum = torch.tensor(
                    0.0, device=self.device, dtype=self.codec.dtype
                )
                total_samples = 0

                with torch.inference_mode():
                    for start in range(0, n_samples, effective_batch_size):
                        end = min(start + effective_batch_size, n_samples)
                        batch_len = end - start

                        x_batch = x_fitness[start:end]
                        y_batch = y_fitness[start:end]

                        t_loss, t_acc, t_mse = self._evaluate_batch_tensors(
                            x_batch, y_batch
                        )
                        batch_loss_sum += t_loss * batch_len
                        batch_acc_sum += t_acc * batch_len
                        batch_mse_sum += t_mse * batch_len
                        total_samples += batch_len

                epoch_scores[i, 0] = batch_loss_sum / total_samples
                epoch_scores[i, 1] = batch_acc_sum / total_samples
                epoch_scores[i, 2] = batch_mse_sum / total_samples

            cpu_scores = epoch_scores.detach().cpu().tolist()
            pbest_improved = [False] * self.n_particles
            pending_resets = [False] * self.n_particles

            for i, p in enumerate(self.particles):
                score = _validate_score(
                    cpu_scores[i],
                    particle_idx=i,
                    iteration=epoch_num,
                )
                if p.personal_best_score is None or _is_better_score(
                    score, p.personal_best_score, renewal
                ):
                    p.personal_best_score = score
                    p.personal_best_weights = p.position.clone()
                    pbest_improved[i] = True

                if _is_better_score(score, self._global_best_score, renewal):
                    self._global_best_score = score
                    self._global_best_weights = p.position.clone()

                iter_ctx = IterationContext(
                    epoch=epoch_num,
                    total_epochs=epochs,
                    w=w,
                    particle_idx=i,
                    is_negative=p.negative,
                    rng=self._random_source,
                    optimizer=self,
                )
                should_reset = self.convergence_plugin.on_particle_evaluated(
                    i, score, pbest_improved[i], iter_ctx
                )
                if should_reset:
                    pending_resets[i] = True

            if self._global_best_weights is None:
                raise RuntimeError(
                    "Global best weights not set before velocity calculation"
                )

            gbest_improved = _is_better_score(
                self._global_best_score, getattr(self, "_prev_gbest_score", None), renewal
            )
            self._prev_gbest_score = self._global_best_score

            epoch_iter_ctx = IterationContext(
                epoch=epoch_num,
                total_epochs=epochs,
                w=w,
                particle_idx=-1,
                is_negative=False,
                rng=self._random_source,
                optimizer=self,
            )
            stop_early = self.convergence_plugin.on_epoch_end(
                self._global_best_score, gbest_improved, epoch_iter_ctx
            )

            if epoch < epochs - 1 and not stop_early:
                # Build SwarmState snapshot BEFORE movement calculation
                swarm_positions = tuple(p.position for p in self.particles)
                swarm_velocities = tuple(p.velocity for p in self.particles)
                swarm_pbests = tuple(
                    p.personal_best_weights if p.personal_best_weights is not None else p.position
                    for p in self.particles
                )
                pbest_scores_tuple = tuple(
                    p.personal_best_score
                    if p.personal_best_score is not None
                    else (float("inf"), float("-inf"), float("inf"))
                    for p in self.particles
                )
                swarm_state = SwarmState(
                    positions=swarm_positions,
                    velocities=swarm_velocities,
                    pbest_positions=swarm_pbests,
                    pbest_scores=pbest_scores_tuple,
                    gbest_position=self._global_best_weights.detach(),
                    gbest_score=self._global_best_score,
                    pbest_improved=tuple(pbest_improved),
                )

                self.movement_plugin.on_epoch_end(swarm_state, epoch_iter_ctx)

                for i, p in enumerate(self.particles):
                    p_iter_ctx = IterationContext(
                        epoch=epoch_num,
                        total_epochs=epochs,
                        w=w,
                        particle_idx=i,
                        is_negative=p.negative,
                        rng=self._random_source,
                        optimizer=self,
                    )
                    pos_override, vel_override = self.movement_plugin.propose(
                        i, swarm_state, p_iter_ctx
                    )
                    if pos_override is not None:
                        p.velocity = torch.zeros_like(p.velocity)
                        p.position = pos_override
                    elif vel_override is not None:
                        proposed_v = vel_override
                        if (
                            self.mutation_swarm > 0.0
                            and self._random_source.bernoulli_event(
                                self.mutation_swarm
                            )
                        ):
                            proposed_v = self._random_source.uniform(
                                p.position.shape,
                                -0.2,
                                0.2,
                                device=self.device,
                                dtype=self.codec.dtype,
                            )
                            self.movement_plugin.reset_particle_state(i)
                        if self.velocity_limit is not None:
                            proposed_v = torch.clamp(
                                proposed_v, -self.velocity_limit, self.velocity_limit
                            )
                        p.velocity = proposed_v
                        p.position = p.position + p.velocity

                    p.apply_boundary_strategy(
                        self.particle_min, self.particle_max, self.boundary_strategy
                    )

                for i, p in enumerate(self.particles):
                    if pending_resets[i]:
                        base_vec = fit_context.base_vector
                        p.reset(base_vec, fit_context, self.initialization_plugin)
                        self.movement_plugin.reset_particle_state(i)
                        self.convergence_plugin.reset_particle(i)
            best = self.get_best_score()
            if best is None:
                raise RuntimeError(
                    "Optimization epoch completed without recording a best score"
                )

            history.append(
                {
                    "epoch": epoch_num,
                    "loss": float(best[0]),
                    "accuracy": float(best[1]),
                    "mse": float(best[2]),
                }
            )

            if (
                checkpoint_interval is not None
                and epoch_num % checkpoint_interval == 0
            ):
                if self._global_best_weights is not None:
                    checkpoint_snapshots[epoch_num] = (
                        self._global_best_weights.detach().cpu().clone()
                    )

            if stop_early:
                break

        best = self.get_best_score()
        if best is None:
            raise RuntimeError(
                "Optimization completed without recording a best score"
            )

        return best, history, checkpoint_snapshots

    def _refine(
        self,
        x_fitness: torch.Tensor,
        y_fitness: torch.Tensor,
        *,
        refinement_epochs: int,
        refinement_lr: float,
        batch_size: int | None,
        renewal: str,
    ) -> None:
        """
        Adam local search on fitness tensors/batching.
        """
        if refinement_epochs <= 0 or self._global_best_weights is None:
            return

        candidate_weights = self._global_best_weights.clone()
        self.codec.apply_vector(candidate_weights, self.eval_model)
        self.eval_model.eval()

        optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=refinement_lr)

        n_samples = x_fitness.shape[0]
        effective_batch_size = (
            n_samples
            if batch_size is None or batch_size >= n_samples
            else batch_size
        )

        for epoch in range(refinement_epochs):
            for start in range(0, n_samples, effective_batch_size):
                end = min(start + effective_batch_size, n_samples)
                x_b = x_fitness[start:end]
                y_b = y_fitness[start:end]

                optimizer.zero_grad()
                out = self.eval_model(x_b)
                if not isinstance(out, torch.Tensor) or out.ndim < 1:
                    raise ValueError("Model output must be a torch.Tensor with rank >= 1.")
                if out.shape[0] != x_b.shape[0]:
                    raise ValueError(
                        f"Model output leading dimension ({out.shape[0]}) does not match batch size ({x_b.shape[0]})."
                    )

                _, batch_loss = self._compute_batch_loss(out, y_b)
                batch_loss.backward()
                optimizer.step()

                if self.particle_min is not None and self.particle_max is not None:
                    with torch.no_grad():
                        for p in self.eval_model.parameters():
                            p.clamp_(self.particle_min, self.particle_max)

            candidate_w = self.codec.encode(self.eval_model)
            candidate_score = self._evaluate_aggregate_score(
                candidate_w, x_fitness, y_fitness, batch_size
            )
            validated_score = _validate_score(
                candidate_score, particle_idx=-1, iteration=epoch + 1
            )
            if _is_better_score(validated_score, self._global_best_score, renewal):
                self._global_best_score = validated_score
                self._global_best_weights = candidate_w.clone()

    def _save_artifacts(
        self,
        *,
        output_dir: str | os.PathLike,
        epochs: int,
        batch_size: int | None,
        fitness_size: int | None,
        renewal: str,
        refinement_epochs: int = 0,
        refinement_lr: float = 0.001,
        validation_split: float | None,
        val_source: str | None,
        log_format: str,
        checkpoint_interval: int | None,
        save_info: bool,
        best_score: tuple[float, float, float],
        val_score: tuple[float, float, float] | None,
        val_sample_count: int | None,
        history: list[dict[str, Any]],
        checkpoint_snapshots: dict[int, torch.Tensor],
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)

        best_state_dict = self.get_best_state_dict()
        if best_state_dict is None:
            raise RuntimeError(
                "Cannot save best model because optimization state is missing"
            )

        best_checkpoint = {
            "model_state_dict": best_state_dict,
            "score": best_score,
            "task": self.task,
            "device": self.device.type,
            "version": __version__,
        }
        torch.save(best_checkpoint, os.path.join(output_dir, "best_model.pt"))

        if checkpoint_snapshots:
            checkpoints_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            for epoch_num, weights_vector in sorted(checkpoint_snapshots.items()):
                ckpt_path = os.path.join(checkpoints_dir, f"epoch-{epoch_num}.pt")
                ckpt_state_dict = self.codec.to_state_dict(
                    weights_vector, self.eval_model
                )
                ckpt_payload = {
                    "epoch": epoch_num,
                    "model_state_dict": ckpt_state_dict,
                    "score": (
                        history[epoch_num - 1]["loss"],
                        history[epoch_num - 1]["accuracy"],
                        history[epoch_num - 1]["mse"],
                    ),
                    "task": self.task,
                    "device": self.device.type,
                    "version": __version__,
                }
                torch.save(ckpt_payload, ckpt_path)

        if log_format == "csv":
            csv_path = os.path.join(output_dir, "history.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["epoch", "loss", "accuracy", "mse"]
                )
                writer.writeheader()
                writer.writerows(history)
        elif log_format == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = os.path.join(output_dir, "tensorboard")
            writer = SummaryWriter(log_dir=tb_dir)
            for row in history:
                ep = row["epoch"]
                writer.add_scalar("loss/train", row["loss"], ep)
                writer.add_scalar("accuracy/train", row["accuracy"], ep)
                writer.add_scalar("mse/train", row["mse"], ep)
            writer.close()

        if save_info:
            loss_name = getattr(
                self.eval_loss, "__class__", type(self.eval_loss)
            ).__name__
            run_info = {
                "version": __version__,
                "task": self.task,
                "device": self.device.type,
                "loss_function": loss_name,
                "config": {
                    "method": self._method_selector,
                    "initialization": self._initialization_selector,
                    "evaluation": self._evaluation_selector,
                    "convergence": self._convergence_selector,
                    "refinement": self._refinement_selector,
                    "plugins": {
                        "movement": {
                            "title": self.movement_plugin.metadata.title,
                            "source": self.movement_plugin.metadata.source,
                            "gradient_required": self.movement_plugin.metadata.gradient_required,
                            "fidelity": self.movement_plugin.metadata.fidelity,
                            "options": self.movement_plugin.get_options(),
                        },
                        "initialization": {
                            "title": self.initialization_plugin.metadata.title,
                            "source": self.initialization_plugin.metadata.source,
                            "gradient_required": self.initialization_plugin.metadata.gradient_required,
                            "fidelity": self.initialization_plugin.metadata.fidelity,
                            "options": self.initialization_plugin.get_options(),
                        },
                        "evaluation": {
                            "title": self.evaluation_plugin.metadata.title,
                            "source": self.evaluation_plugin.metadata.source,
                            "gradient_required": self.evaluation_plugin.metadata.gradient_required,
                            "fidelity": self.evaluation_plugin.metadata.fidelity,
                            "options": self.evaluation_plugin.get_options(),
                        },
                        "convergence": {
                            "title": self.convergence_plugin.metadata.title,
                            "source": self.convergence_plugin.metadata.source,
                            "gradient_required": self.convergence_plugin.metadata.gradient_required,
                            "fidelity": self.convergence_plugin.metadata.fidelity,
                            "options": self.convergence_plugin.get_options(),
                        },
                        "refinement": {
                            "title": self.refinement_plugin.metadata.title,
                            "source": self.refinement_plugin.metadata.source,
                            "gradient_required": self.refinement_plugin.metadata.gradient_required,
                            "fidelity": self.refinement_plugin.metadata.fidelity,
                            "options": self.refinement_plugin.get_options(),
                        },
                    },
                    "n_particles": self.n_particles,
                    "c0": self.c0,
                    "c1": self.c1,
                    "w_min": self.w_min,
                    "w_max": self.w_max,
                    "negative_swarm": self.negative_swarm,
                    "mutation_swarm": self.mutation_swarm,
                    "particle_min": self.particle_min,
                    "particle_max": self.particle_max,
                    "velocity_limit_ratio": self.velocity_limit_ratio,
                    "boundary_strategy": self.boundary_strategy,
                    "initial_position_noise": self.initial_position_noise,
                    "seed": self.seed,
                    "fitness_size": fitness_size,
                    "convergence_patience": self.convergence_patience,
                    "convergence_min_delta": self.convergence_min_delta,
                    "convergence_monitor": self.convergence_monitor,
                    "moment_blend": self.moment_blend,
                    "moment_beta1": self.moment_beta1,
                    "moment_beta2": self.moment_beta2,
                    "moment_step_size": self.moment_step_size,
                    "moment_epsilon": self.moment_epsilon,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "renewal": renewal,
                    "refinement_epochs": refinement_epochs,
                    "refinement_lr": refinement_lr,
                    "validation_source": val_source,
                    "validation_split": validation_split,
                    "output_dir": str(output_dir),
                    "log_format": log_format,
                    "checkpoint_interval": checkpoint_interval,
                    "save_info": save_info,
                },
                "best_training_score": [float(x) for x in best_score],
                "validation_score": (
                    [float(x) for x in val_score] if val_score is not None else None
                ),
                "validation_source": val_source,
                "validation_sample_count": val_sample_count,
            }
            with open(os.path.join(output_dir, "run.json"), "w", encoding="utf-8") as f:
                json.dump(run_info, f, indent=2)

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        epochs: int = 10,
        batch_size: int | None = None,
        fitness_size: int | None = None,
        renewal: str = "acc",
        refinement_epochs: int = 0,
        refinement_lr: float = 0.001,
        validation_data: tuple[torch.Tensor, torch.Tensor] | None = None,
        validation_split: float | None = None,
        output_dir: str | os.PathLike | None = None,
        log_format: Literal["none", "csv", "tensorboard"] = "none",
        checkpoint_interval: int | None = None,
        save_info: bool = False,
    ) -> tuple[float, float, float]:
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("x and y must be torch.Tensor instances")

        if x.ndim == 0 or y.ndim == 0:
            raise ValueError("x and y must have a leading dimension (ndim >= 1)")

        len_x = x.shape[0]
        len_y = y.shape[0]
        if len_x != len_y:
            raise ValueError(f"x and y leading dimensions must match: {len_x} != {len_y}")
        if len_x == 0:
            raise ValueError("x and y leading dimensions must be nonzero")

        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")

        if batch_size is not None:
            if (
                isinstance(batch_size, bool)
                or not isinstance(batch_size, int)
                or batch_size <= 0
            ):
                raise ValueError("batch_size must be a positive integer")

        if renewal not in ("acc", "loss", "mse"):
            raise ValueError("renewal must be one of 'acc', 'loss', 'mse'")
        self.renewal = renewal

        if (
            isinstance(refinement_epochs, bool)
            or not isinstance(refinement_epochs, int)
            or refinement_epochs < 0
        ):
            raise ValueError("refinement_epochs must be a non-negative integer")

        if (
            isinstance(refinement_lr, bool)
            or not isinstance(refinement_lr, (int, float))
            or not math.isfinite(refinement_lr)
            or float(refinement_lr) <= 0.0
        ):
            raise ValueError("refinement_lr must be a positive finite float")

        if log_format not in ("none", "csv", "tensorboard"):
            raise ValueError("log_format must be one of 'none', 'csv', 'tensorboard'")

        if checkpoint_interval is not None:
            if (
                isinstance(checkpoint_interval, bool)
                or not isinstance(checkpoint_interval, int)
                or checkpoint_interval <= 0
            ):
                raise ValueError("checkpoint_interval must be a positive integer")

        if not isinstance(save_info, bool):
            raise ValueError("save_info must be a boolean")

        if validation_data is not None and validation_split is not None:
            raise ValueError(
                "validation_data and validation_split are mutually exclusive"
            )

        eval_title = self.evaluation_plugin.metadata.title
        if fitness_size is not None and eval_title != "Fixed Subset Evaluation":
            raise ValueError("fitness_size is only valid with evaluation='fixed_subset'")

        refine_title = self.refinement_plugin.metadata.title
        if refinement_epochs > 0 and refine_title == "No Refinement":
            raise ValueError("refinement_epochs > 0 is valid only with refinement='adam'")

        # Restore eval model parameters from constructor-time base vector and reset run state
        self.codec.apply_vector(self._base_vector.clone(), self.eval_model)
        self._global_best_score = None
        self._global_best_weights = None
        if hasattr(self, "_prev_gbest_score"):
            del self._prev_gbest_score

        val_x, val_y = None, None
        val_source = None

        if validation_data is not None:
            if not isinstance(validation_data, tuple) or len(validation_data) != 2:
                raise ValueError("validation_data must be a tuple of (val_x, val_y)")
            v_x, v_y = validation_data
            if not isinstance(v_x, torch.Tensor) or not isinstance(v_y, torch.Tensor):
                raise TypeError(
                    "validation_data elements must be torch.Tensor instances"
                )
            if v_x.ndim == 0 or v_y.ndim == 0:
                raise ValueError(
                    "validation_data elements must have a leading dimension"
                )
            len_val_x = v_x.shape[0]
            len_val_y = v_y.shape[0]
            if len_val_x != len_val_y:
                raise ValueError(
                    f"validation_data leading dimensions must match: {len_val_x} != {len_val_y}"
                )
            if len_val_x == 0:
                raise ValueError("validation_data leading dimensions must be nonzero")
            val_x, val_y = v_x, v_y
            val_source = "validation_data"

        if validation_split is not None:
            if (
                isinstance(validation_split, bool)
                or not isinstance(validation_split, (int, float))
                or not math.isfinite(validation_split)
                or not (0.0 < float(validation_split) < 1.0)
            ):
                raise ValueError(
                    "validation_split must be a finite numeric strictly between 0 and 1"
                )
            val_source = "validation_split"

        if output_dir is not None:
            if isinstance(output_dir, bool) or not isinstance(
                output_dir, (str, os.PathLike)
            ):
                raise ValueError("output_dir must be a valid path-like string or Path")

        if output_dir is None and (
            log_format != "none" or checkpoint_interval is not None or save_info
        ):
            raise ValueError(
                "output_dir is required when log_format != 'none', checkpoint_interval is set, or save_info is True"
            )

        if validation_split is not None:
            n_samples = len_x
            n_val = int(math.floor(n_samples * float(validation_split)))
            if n_val == 0 or n_val >= n_samples:
                raise ValueError(
                    "validation_split results in an empty training or validation set"
                )
            perm = self._random_source.permutation(n_samples)
            val_indices = perm[:n_val]
            train_indices = perm[n_val:]

            x_train = x[train_indices]
            val_x = x[val_indices]
            y_train = y[train_indices]
            val_y = y[val_indices]
        else:
            x_train = x
            y_train = y

        dtype_model = next(self.eval_model.parameters()).dtype
        x_train_dev = x_train.to(device=self.device, dtype=dtype_model)

        if self.task == "multiclass":
            if y_train.ndim > 1 and y_train.shape[-1] > 1:
                y_train_dev = y_train.to(device=self.device, dtype=dtype_model)
            else:
                y_train_dev = y_train.to(device=self.device, dtype=torch.int64)
        elif self.task in ("binary", "regression"):
            y_train_dev = y_train.to(device=self.device, dtype=dtype_model)

        if val_x is not None and val_y is not None:
            val_x = val_x.to(device=self.device, dtype=dtype_model)
            if self.task == "multiclass":
                if val_y.ndim > 1 and val_y.shape[-1] > 1:
                    val_y = val_y.to(device=self.device, dtype=dtype_model)
                else:
                    val_y = val_y.to(device=self.device, dtype=torch.int64)
            elif self.task in ("binary", "regression"):
                val_y = val_y.to(device=self.device, dtype=dtype_model)

        base_vector = self.codec.encode(self.eval_model)
        effective_fitness_size = fitness_size if fitness_size is not None else self.fitness_size
        effective_refine_epochs = refinement_epochs if refinement_epochs > 0 else self.refinement_epochs
        effective_refine_lr = refinement_lr if refinement_lr != 0.001 else self.refinement_lr

        fit_ctx = FitContext(
            optimizer=self,
            model=self.model,
            eval_model=self.eval_model,
            codec=self.codec,
            base_vector=base_vector,
            n_particles=self.n_particles,
            particle_min=self.particle_min,
            particle_max=self.particle_max,
            velocity_limit=self.velocity_limit,
            boundary_strategy=self.boundary_strategy,
            initial_position_noise=self.initial_position_noise,
            seed=self.seed,
            device=self.device,
            rng=self._random_source,
            task=self.task,
            x_train=x_train_dev,
            y_train=y_train_dev,
            batch_size=batch_size,
            fitness_size=effective_fitness_size,
            renewal=renewal,
            epochs=epochs,
            refinement_epochs=effective_refine_epochs,
            refinement_lr=effective_refine_lr,
            c0=self.c0,
            c1=self.c1,
            w_min=self.w_min,
            w_max=self.w_max,
            negative_swarm=self.negative_swarm,
            mutation_swarm=self.mutation_swarm,
        )

        # Prepare fit for all 5 stage plugins
        self.initialization_plugin.prepare_fit(fit_ctx)
        self.evaluation_plugin.prepare_fit(fit_ctx)
        self.movement_plugin.prepare_fit(fit_ctx)
        self.convergence_plugin.prepare_fit(fit_ctx)
        self.refinement_plugin.prepare_fit(fit_ctx)

        # Create fresh swarm for each fit
        num_negative = int(round(self.negative_swarm * self.n_particles))
        self.particles = []
        for i in range(self.n_particles):
            p = Particle(
                index=i,
                base_vector=base_vector,
                context=fit_ctx,
                init_plugin=self.initialization_plugin,
                negative=(i < num_negative),
            )
            self.particles.append(p)

        x_fitness, y_fitness = self.evaluation_plugin.get_fitness_data(
            x_train_dev, y_train_dev, fit_ctx
        )

        best_score, history, checkpoint_snapshots = self._optimize(
            x_fitness,
            y_fitness,
            fit_ctx,
            epochs=epochs,
            batch_size=batch_size,
            renewal=renewal,
            checkpoint_interval=checkpoint_interval,
        )

        if effective_refine_epochs > 0:
            refined_w, refined_s = self.refinement_plugin.refine(
                self._global_best_weights
                if self._global_best_weights is not None
                else base_vector,
                best_score,
                self._evaluate_batch,
                fit_ctx,
            )
            best_score = refined_s

        val_score = None
        val_sample_count = None
        if val_x is not None and val_y is not None:
            if self._global_best_weights is None:
                raise RuntimeError("Best model not available after optimization")
            raw_val = self._evaluate_aggregate_score(
                self._global_best_weights, val_x, val_y, batch_size=batch_size
            )
            val_score = _validate_score(raw_val, particle_idx=-1, iteration=-1)
            val_sample_count = val_x.shape[0]

        if output_dir is not None:
            self._save_artifacts(
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                fitness_size=effective_fitness_size,
                renewal=renewal,
                refinement_epochs=effective_refine_epochs,
                refinement_lr=effective_refine_lr,
                validation_split=validation_split,
                val_source=val_source,
                log_format=log_format,
                checkpoint_interval=checkpoint_interval,
                save_info=save_info,
                best_score=best_score,
                val_score=val_score,
                val_sample_count=val_sample_count,
                history=history,
                checkpoint_snapshots=checkpoint_snapshots,
            )

        return best_score
