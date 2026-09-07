import copy
from dataclasses import dataclass
import inspect
import math
from typing import Any, Literal, Sequence
import torch
import torch.nn as nn


@dataclass(frozen=True)
class PluginMetadata:
    stage: Literal["movement", "initialization", "evaluation", "convergence", "refinement"]
    title: str
    source: str | None = None
    gradient_required: bool = False
    fidelity: Literal["canonical", "experimental"] = "canonical"


@dataclass(frozen=True)
class SwarmState:
    positions: tuple[torch.Tensor, ...]
    velocities: tuple[torch.Tensor, ...]
    pbest_positions: tuple[torch.Tensor, ...]
    pbest_scores: tuple[tuple[float, float, float], ...]
    gbest_position: torch.Tensor
    gbest_score: tuple[float, float, float]
    pbest_improved: tuple[bool, ...]


@dataclass
class FitContext:
    optimizer: Any
    model: nn.Module
    eval_model: nn.Module
    codec: Any
    base_vector: torch.Tensor
    n_particles: int
    particle_min: float | None
    particle_max: float | None
    velocity_limit: float | None
    boundary_strategy: str
    initial_position_noise: float
    seed: int | None
    device: torch.device
    rng: Any
    task: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    batch_size: int | None
    fitness_size: int | None
    renewal: str
    epochs: int
    refinement_epochs: int
    refinement_lr: float
    c0: float | None = None
    c1: float | None = None
    w_min: float | None = None
    w_max: float | None = None
    negative_swarm: float = 0.0
    mutation_swarm: float = 0.0


@dataclass
class IterationContext:
    epoch: int
    total_epochs: int
    w: float
    particle_idx: int
    is_negative: bool
    rng: Any
    optimizer: Any


def _is_better_score(
    score_a: Sequence[float],
    score_b: Sequence[float] | None,
    renewal: str = "acc",
) -> bool:
    if score_b is None:
        return True
    if renewal == "acc":
        if score_a[1] != score_b[1]:
            return score_a[1] > score_b[1]
        if score_a[0] != score_b[0]:
            return score_a[0] < score_b[0]
        return score_a[2] < score_b[2]
    elif renewal == "loss":
        if score_a[0] != score_b[0]:
            return score_a[0] < score_b[0]
        if score_a[1] != score_b[1]:
            return score_a[1] > score_b[1]
        return score_a[2] < score_b[2]
    elif renewal == "mse":
        if score_a[2] != score_b[2]:
            return score_a[2] < score_b[2]
        if score_a[0] != score_b[0]:
            return score_a[0] < score_b[0]
        return score_a[1] > score_b[1]
    return False


def _is_at_least_delta(delta: float, min_delta: float) -> bool:
    if min_delta > 0:
        return delta >= min_delta
    return delta > 0


class BasePlugin:
    metadata: PluginMetadata

    def prepare_fit(self, context: FitContext) -> None:
        pass

    def get_options(self) -> dict[str, Any]:
        return {}
class InitializationPlugin(BasePlugin):
    def initialize(
        self, index: int, base_vector: torch.Tensor, context: FitContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class ModelNoiseInitialization(InitializationPlugin):
    metadata = PluginMetadata(
        stage="initialization",
        title="Model Weight + Uniform Noise Initialization",
        source=None,
        gradient_required=False,
        fidelity="canonical",
    )

    def __init__(self, noise: float = 0.05):
        if (
            isinstance(noise, bool)
            or not isinstance(noise, (int, float))
            or not math.isfinite(noise)
            or float(noise) < 0.0
        ):
            raise ValueError("noise must be a finite nonnegative number")
        self.noise = float(noise)

    def get_options(self) -> dict[str, Any]:
        return {"noise": self.noise}

    def initialize(
        self, index: int, base_vector: torch.Tensor, context: FitContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = base_vector.device
        dtype = base_vector.dtype
        noise_val = (
            context.initial_position_noise
            if context.initial_position_noise is not None
            else self.noise
        )
        pos = base_vector.clone()
        if noise_val > 0.0:
            n = context.rng.uniform(
                pos.shape, -noise_val, noise_val, device=device, dtype=dtype
            )
            pos = pos + n
        if context.particle_min is not None and context.particle_max is not None:
            pos = torch.clamp(pos, context.particle_min, context.particle_max)
        vel = context.rng.uniform(pos.shape, -0.2, 0.2, device=device, dtype=dtype)
        if context.velocity_limit is not None:
            vel = torch.clamp(vel, -context.velocity_limit, context.velocity_limit)
        return pos, vel


class UniformInitialization(InitializationPlugin):
    metadata = PluginMetadata(
        stage="initialization",
        title="Uniform Bounded Space Initialization",
        source=None,
        gradient_required=False,
        fidelity="canonical",
    )

    def initialize(
        self, index: int, base_vector: torch.Tensor, context: FitContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.particle_min is None or context.particle_max is None:
            raise ValueError(
                "uniform initialization requires finite particle_min and particle_max bounds"
            )
        device = base_vector.device
        dtype = base_vector.dtype
        pos = context.rng.uniform(
            base_vector.shape,
            context.particle_min,
            context.particle_max,
            device=device,
            dtype=dtype,
        )
        vel = context.rng.uniform(pos.shape, -0.2, 0.2, device=device, dtype=dtype)
        if context.velocity_limit is not None:
            vel = torch.clamp(vel, -context.velocity_limit, context.velocity_limit)
        return pos, vel


class EvaluationPlugin(BasePlugin):
    def get_fitness_data(
        self, x_train: torch.Tensor, y_train: torch.Tensor, context: FitContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class FullEvaluation(EvaluationPlugin):
    metadata = PluginMetadata(
        stage="evaluation",
        title="Full Dataset Evaluation",
        source=None,
        gradient_required=False,
        fidelity="canonical",
    )

    def get_fitness_data(
        self, x_train: torch.Tensor, y_train: torch.Tensor, context: FitContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x_train, y_train


class FixedSubsetEvaluation(EvaluationPlugin):
    metadata = PluginMetadata(
        stage="evaluation",
        title="Fixed Subset Evaluation",
        source=None,
        gradient_required=False,
        fidelity="experimental",
    )

    def __init__(self, fitness_size: int | None = None):
        if fitness_size is not None:
            if (
                isinstance(fitness_size, bool)
                or not isinstance(fitness_size, int)
                or fitness_size < 1
            ):
                raise ValueError("fitness_size must be an integer >= 1")
        self.fitness_size = fitness_size
        self._x_sub: torch.Tensor | None = None
        self._y_sub: torch.Tensor | None = None

    def get_options(self) -> dict[str, Any]:
        return {"fitness_size": self.fitness_size}

    def prepare_fit(self, context: FitContext) -> None:
        f_size = (
            context.fitness_size
            if context.fitness_size is not None
            else self.fitness_size
        )
        if f_size is None or f_size <= 0:
            raise ValueError(
                "evaluation='fixed_subset' requires a positive fitness_size"
            )
        n_train = context.x_train.shape[0]
        if f_size > n_train:
            raise ValueError(
                f"fitness_size ({f_size}) cannot exceed post-validation-split training size ({n_train})"
            )
        fitness_idx = context.rng.choice(n_train, size=f_size)
        self._x_sub = context.x_train[fitness_idx]
        self._y_sub = context.y_train[fitness_idx]

    def get_fitness_data(
        self, x_train: torch.Tensor, y_train: torch.Tensor, context: FitContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._x_sub is None or self._y_sub is None:
            raise RuntimeError(
                "FixedSubsetEvaluation was not prepared before get_fitness_data"
            )
        return self._x_sub, self._y_sub


class MovementPlugin(BasePlugin):
    def propose(
        self, particle_idx: int, state: SwarmState, context: IterationContext
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        raise NotImplementedError

    def on_epoch_end(self, state: SwarmState, context: IterationContext) -> None:
        pass

    def reset_particle_state(self, particle_idx: int) -> None:
        pass


class OriginalMovement(MovementPlugin):
    metadata = PluginMetadata(
        stage="movement",
        title="Original PSO",
        source="10.1109/ICNN.1995.488968",
        gradient_required=False,
        fidelity="canonical",
    )

    def __init__(self, c0: float = 2.0, c1: float = 2.0):
        for name, val in [("c0", c0), ("c1", c1)]:
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
            ):
                raise ValueError(f"{name} must be a finite number")
        self.c0 = float(c0)
        self.c1 = float(c1)

    def get_options(self) -> dict[str, Any]:
        return {"c0": self.c0, "c1": self.c1}

    def propose(
        self, particle_idx: int, state: SwarmState, context: IterationContext
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x = state.positions[particle_idx]
        v = state.velocities[particle_idx]
        pbest = state.pbest_positions[particle_idx]
        gbest = state.gbest_position
        device = x.device
        dtype = x.dtype

        r1 = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)
        r2 = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)

        cog = self.c0 * r1 * (pbest - x)
        if not context.is_negative:
            soc = self.c1 * r2 * (gbest - x)
        else:
            soc = -self.c1 * r2 * (gbest - x)

        v_new = v + cog + soc
        return None, v_new


class InertiaMovement(MovementPlugin):
    metadata = PluginMetadata(
        stage="movement",
        title="Inertia Weight PSO",
        source="10.1109/ICEC.1998.699146",
        gradient_required=False,
        fidelity="canonical",
    )

    def __init__(
        self,
        c0: float = 2.0,
        c1: float = 2.0,
        w_min: float = 0.4,
        w_max: float = 0.9,
    ):
        for name, val in [
            ("c0", c0),
            ("c1", c1),
            ("w_min", w_min),
            ("w_max", w_max),
        ]:
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
            ):
                raise ValueError(f"{name} must be a finite number")
        if float(w_min) > float(w_max):
            raise ValueError("w_min must be <= w_max")
        self.c0 = float(c0)
        self.c1 = float(c1)
        self.w_min = float(w_min)
        self.w_max = float(w_max)

    def get_options(self) -> dict[str, Any]:
        return {"c0": self.c0, "c1": self.c1, "w_min": self.w_min, "w_max": self.w_max}

    def propose(
        self, particle_idx: int, state: SwarmState, context: IterationContext
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x = state.positions[particle_idx]
        v = state.velocities[particle_idx]
        pbest = state.pbest_positions[particle_idx]
        gbest = state.gbest_position
        device = x.device
        dtype = x.dtype

        r1 = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)
        r2 = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)

        cog = self.c0 * r1 * (pbest - x)
        if not context.is_negative:
            soc = self.c1 * r2 * (gbest - x)
        else:
            soc = -self.c1 * r2 * (gbest - x)

        v_new = context.w * v + cog + soc
        return None, v_new


class ConstrictionMovement(MovementPlugin):
    metadata = PluginMetadata(
        stage="movement",
        title="Constriction Coefficient PSO",
        source="10.1109/4235.985692",
        gradient_required=False,
        fidelity="canonical",
    )

    def __init__(self, c0: float = 2.05, c1: float = 2.05):
        for name, val in [("c0", c0), ("c1", c1)]:
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
            ):
                raise ValueError(f"{name} must be a finite number")
        self.c0 = float(c0)
        self.c1 = float(c1)
        phi = self.c0 + self.c1
        if phi <= 4.0:
            raise ValueError(f"Constriction PSO requires c0 + c1 > 4.0, got phi={phi}")
        self.chi = 2.0 / abs(2.0 - phi - math.sqrt(phi * phi - 4.0 * phi))

    def get_options(self) -> dict[str, Any]:
        return {"c0": self.c0, "c1": self.c1, "chi": float(self.chi)}

    def propose(
        self, particle_idx: int, state: SwarmState, context: IterationContext
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x = state.positions[particle_idx]
        v = state.velocities[particle_idx]
        pbest = state.pbest_positions[particle_idx]
        gbest = state.gbest_position
        device = x.device
        dtype = x.dtype

        r1 = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)
        r2 = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)

        cog = self.c0 * r1 * (pbest - x)
        if not context.is_negative:
            soc = self.c1 * r2 * (gbest - x)
        else:
            soc = -self.c1 * r2 * (gbest - x)

        v_new = self.chi * (v + cog + soc)
        return None, v_new


class FIPSMovement(MovementPlugin):
    metadata = PluginMetadata(
        stage="movement",
        title="Fully Informed Particle Swarm (FIPS)",
        source="10.1109/TEVC.2004.826074",
        gradient_required=False,
        fidelity="canonical",
    )

    def __init__(self, c0: float = 2.05, c1: float = 2.05):
        for name, val in [("c0", c0), ("c1", c1)]:
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
            ):
                raise ValueError(f"{name} must be a finite number")
        self.c0 = float(c0)
        self.c1 = float(c1)
        phi = self.c0 + self.c1
        if phi <= 4.0:
            raise ValueError(f"FIPS requires c0 + c1 > 4.0, got phi={phi}")
        self.phi = phi
        self.chi = 2.0 / abs(2.0 - phi - math.sqrt(phi * phi - 4.0 * phi))

    def get_options(self) -> dict[str, Any]:
        return {"c0": self.c0, "c1": self.c1, "phi": float(self.phi), "chi": float(self.chi)}

    def prepare_fit(self, context: FitContext) -> None:
        if context.negative_swarm != 0.0:
            raise ValueError("negative_swarm is unsupported for FIPS")

    def propose(
        self, particle_idx: int, state: SwarmState, context: IterationContext
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if context.is_negative:
            raise ValueError("negative_swarm is unsupported for FIPS")
        x = state.positions[particle_idx]
        v = state.velocities[particle_idx]
        device = x.device
        dtype = x.dtype
        n_particles = len(state.positions)

        scale = self.phi / float(n_particles)
        social_term = torch.zeros_like(x)
        for j in range(n_particles):
            pbest_j = state.pbest_positions[j]
            r_j = context.rng.uniform(x.shape, 0.0, scale, device=device, dtype=dtype)
            social_term = social_term + r_j * (pbest_j - x)

        v_new = self.chi * (v + social_term)
        return None, v_new


class CLPSOMovement(MovementPlugin):
    metadata = PluginMetadata(
        stage="movement",
        title="Comprehensive Learning PSO (CLPSO)",
        source="10.1109/TEVC.2005.857610",
        gradient_required=False,
        fidelity="canonical",
    )

    def __init__(
        self,
        c: float = 1.49445,
        w_min: float = 0.4,
        w_max: float = 0.9,
        refresh_gap: int = 7,
        c0: float | None = None,
    ):
        c_val = float(c0) if c0 is not None else float(c)
        if not math.isfinite(c_val) or c_val <= 0.0:
            raise ValueError("CLPSO acceleration coefficient must be positive")
        if (
            isinstance(w_min, bool)
            or not isinstance(w_min, (int, float))
            or not math.isfinite(w_min)
            or isinstance(w_max, bool)
            or not isinstance(w_max, (int, float))
            or not math.isfinite(w_max)
        ):
            raise ValueError("w_min and w_max must be finite numbers")
        if float(w_min) > float(w_max):
            raise ValueError("w_min must be <= w_max")
        if (
            isinstance(refresh_gap, bool)
            or not isinstance(refresh_gap, int)
            or refresh_gap < 1
        ):
            raise ValueError("refresh_gap must be an integer >= 1")

        self.c = c_val
        self.c0 = c_val
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.refresh_gap = int(refresh_gap)
        self.exemplars: torch.Tensor | None = None
        self.stagnation: torch.Tensor | None = None
        self.learning_probs: torch.Tensor | None = None
        self._pbest_matrix: torch.Tensor | None = None
        self._dim_indices: torch.Tensor | None = None
        self._ranks: torch.Tensor | None = None
        self.n_particles: int = 0
        self.dim: int = 0
        self.renewal: str = "acc"

    def get_options(self) -> dict[str, Any]:
        return {
            "c": self.c,
            "c0": self.c,
            "w_min": self.w_min,
            "w_max": self.w_max,
            "refresh_gap": self.refresh_gap,
        }

    def prepare_fit(self, context: FitContext) -> None:
        if context.negative_swarm != 0.0:
            raise ValueError("negative_swarm is unsupported for CLPSO")
        self.n_particles = context.n_particles
        self.dim = context.base_vector.numel()
        self.renewal = context.renewal

        probs = torch.zeros((self.n_particles,), dtype=torch.float32)
        if self.n_particles == 1:
            probs[0] = 0.05
        else:
            denom = math.exp(10.0) - 1.0
            for i in range(self.n_particles):
                probs[i] = (
                    0.05
                    + 0.45
                    * (math.exp(10.0 * i / (self.n_particles - 1)) - 1.0)
                    / denom
                )

        self.learning_probs = probs
        self.stagnation = torch.zeros((self.n_particles,), dtype=torch.int64)
        self.exemplars = None
        self._pbest_matrix = None
        self._dim_indices = None
        self._ranks = None

    def _sample_exemplars_for_particle(
        self,
        particle_idx: int,
        state: SwarmState,
        rng: Any,
    ) -> None:
        assert self.exemplars is not None
        assert self.learning_probs is not None
        assert self._ranks is not None

        pc = float(self.learning_probs[particle_idx])
        n_particles = self.n_particles
        dim = self.dim

        ex = torch.full((dim,), particle_idx, dtype=torch.int64)

        if n_particles == 1:
            self.exemplars[particle_idx] = ex
            return

        candidates = [p for p in range(n_particles) if p != particle_idx]
        k = len(candidates)
        candidates_t = torch.tensor(candidates, dtype=torch.int64)

        if k == 1:
            r = rng.uniform((dim,), 0.0, 1.0, device="cpu")
            mask = (r < pc)
            if mask.any():
                ex[mask] = candidates_t[0]
            else:
                d_pick = int(rng.randint(0, dim, size=1, device="cpu").item())
                ex[d_pick] = candidates_t[0]
            self.exemplars[particle_idx] = ex
            return

        r = rng.uniform((dim,), 0.0, 1.0, device="cpu")
        mask = (r < pc)
        m = int(mask.sum().item())

        if m > 0:
            c_idx1 = rng.randint(0, k, size=m, device="cpu")
            c_idx2_raw = rng.randint(0, k - 1, size=m, device="cpu")
            c_idx2 = torch.where(c_idx2_raw >= c_idx1, c_idx2_raw + 1, c_idx2_raw)

            p1 = candidates_t[c_idx1]
            p2 = candidates_t[c_idx2]

            rank1 = self._ranks[p1]
            rank2 = self._ranks[p2]

            winners = torch.where(rank1 < rank2, p1, p2)
            ex[mask] = winners
            all_own = False
        else:
            all_own = True

        if all_own:
            d_pick = int(rng.randint(0, dim, size=1, device="cpu").item())
            c_idx1 = int(rng.randint(0, k, size=1, device="cpu").item())
            c_idx2_raw = int(rng.randint(0, k - 1, size=1, device="cpu").item())
            c_idx2 = c_idx2_raw + 1 if c_idx2_raw >= c_idx1 else c_idx2_raw

            p1 = candidates_t[c_idx1].item()
            p2 = candidates_t[c_idx2].item()

            winner = p1 if self._ranks[p1] < self._ranks[p2] else p2
            ex[d_pick] = winner

        self.exemplars[particle_idx] = ex

    def propose(
        self, particle_idx: int, state: SwarmState, context: IterationContext
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if context.is_negative:
            raise ValueError("negative_swarm is unsupported for CLPSO")
        if self.exemplars is None or self._pbest_matrix is None or self._ranks is None:
            self.on_epoch_end(state, context)

        assert self.exemplars is not None
        assert self._pbest_matrix is not None
        x = state.positions[particle_idx]
        v = state.velocities[particle_idx]
        device = x.device
        dtype = x.dtype
        dim = x.numel()

        pbest_device = self._pbest_matrix.device
        if (
            self._dim_indices is None
            or self._dim_indices.device != pbest_device
            or self._dim_indices.numel() != dim
        ):
            self._dim_indices = torch.arange(dim, device=pbest_device)

        ex_indices = self.exemplars[particle_idx].to(device=pbest_device)
        e_i = self._pbest_matrix[ex_indices, self._dim_indices].to(
            device=device, dtype=dtype
        )

        r = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)
        v_new = context.w * v + self.c * r * (e_i - x)
        return None, v_new

    def on_epoch_end(self, state: SwarmState, context: IterationContext) -> None:
        if self.n_particles == 0:
            return

        self._pbest_matrix = torch.stack(state.pbest_positions)

        scores_keys = []
        for score in state.pbest_scores:
            if score is None:
                scores_keys.append((float("inf"), float("inf"), float("inf")))
            else:
                loss, acc, mse = score
                if self.renewal in ("acc", "accuracy"):
                    scores_keys.append((-acc, loss, mse))
                elif self.renewal == "loss":
                    scores_keys.append((loss, -acc, mse))
                else:
                    scores_keys.append((mse, loss, -acc))
        sorted_indices = sorted(range(self.n_particles), key=lambda idx: scores_keys[idx])
        ranks = torch.zeros(self.n_particles, dtype=torch.int64)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank
        self._ranks = ranks

        if self.exemplars is None:
            self.exemplars = torch.zeros((self.n_particles, self.dim), dtype=torch.int64)
            for i in range(self.n_particles):
                self._sample_exemplars_for_particle(i, state, context.rng)
            if self.stagnation is not None:
                self.stagnation.zero_()
            return

        if self.stagnation is None:
            return
        n_particles = len(state.positions)
        for i in range(n_particles):
            if state.pbest_improved[i]:
                self.stagnation[i] = 0
            else:
                self.stagnation[i] += 1
                if self.stagnation[i] >= self.refresh_gap:
                    self._sample_exemplars_for_particle(i, state, context.rng)
                    self.stagnation[i] = 0

    def reset_particle_state(self, particle_idx: int) -> None:
        if self.stagnation is not None and particle_idx < len(self.stagnation):
            self.stagnation[particle_idx] = 0
class BareBonesMovement(MovementPlugin):
    metadata = PluginMetadata(
        stage="movement",
        title="Bare Bones PSO",
        source="10.1109/SIS.2003.1202251",
        gradient_required=False,
        fidelity="canonical",
    )

    def prepare_fit(self, context: FitContext) -> None:
        if context.negative_swarm != 0.0:
            raise ValueError("negative_swarm is unsupported for Bare Bones PSO")
        if context.mutation_swarm != 0.0:
            raise ValueError("mutation_swarm is unsupported for Bare Bones PSO")
        if context.velocity_limit is not None:
            raise ValueError("velocity_limit is unsupported for Bare Bones PSO")

    def propose(
        self, particle_idx: int, state: SwarmState, context: IterationContext
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if context.is_negative:
            raise ValueError("negative_swarm is unsupported for Bare Bones PSO")
        x = state.positions[particle_idx]
        pbest = state.pbest_positions[particle_idx]
        gbest = state.gbest_position
        device = x.device
        dtype = x.dtype

        mu = 0.5 * (pbest + gbest)
        sigma = torch.abs(pbest - gbest)

        r_norm = torch.randn(
            x.shape,
            generator=context.rng.cpu_generator,
            device="cpu",
            dtype=dtype,
        ).to(device)
        gauss_sample = mu + sigma * r_norm

        u = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)
        x_new = torch.where(u < 0.5, pbest, gauss_sample)
        v_new = torch.zeros_like(x)
        return x_new, v_new


class AdaptiveMomentMovement(MovementPlugin):
    metadata = PluginMetadata(
        stage="movement",
        title="Adaptive Path-Moment PSO",
        source=None,
        gradient_required=False,
        fidelity="experimental",
    )

    def __init__(
        self,
        c0: float = 0.5,
        c1: float = 0.3,
        w_min: float = 0.1,
        w_max: float = 0.9,
        moment_blend: float = 0.25,
        moment_beta1: float = 0.9,
        moment_beta2: float = 0.999,
        moment_step_size: float = 1.0,
        moment_epsilon: float = 1e-8,
    ):
        for name, val in [
            ("c0", c0),
            ("c1", c1),
            ("w_min", w_min),
            ("w_max", w_max),
        ]:
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
            ):
                raise ValueError(f"{name} must be a finite number")
        if float(w_min) > float(w_max):
            raise ValueError("w_min must be <= w_max")

        if (
            isinstance(moment_blend, bool)
            or not isinstance(moment_blend, (int, float))
            or not math.isfinite(moment_blend)
            or not (0.0 <= float(moment_blend) <= 1.0)
        ):
            raise ValueError("moment_blend must be a finite float in range [0, 1]")

        for name, val in [
            ("moment_beta1", moment_beta1),
            ("moment_beta2", moment_beta2),
        ]:
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
                or not (0.0 < float(val) < 1.0)
            ):
                raise ValueError(f"{name} must be a finite float strictly in range (0, 1)")

        if (
            isinstance(moment_step_size, bool)
            or not isinstance(moment_step_size, (int, float))
            or not math.isfinite(moment_step_size)
            or float(moment_step_size) <= 0.0
        ):
            raise ValueError("moment_step_size must be a positive finite float")

        if (
            isinstance(moment_epsilon, bool)
            or not isinstance(moment_epsilon, (int, float))
            or not math.isfinite(moment_epsilon)
            or float(moment_epsilon) <= 0.0
        ):
            raise ValueError("moment_epsilon must be a positive finite float")

        self.c0 = float(c0)
        self.c1 = float(c1)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.moment_blend = float(moment_blend)
        self.moment_beta1 = float(moment_beta1)
        self.moment_beta2 = float(moment_beta2)
        self.moment_step_size = float(moment_step_size)
        self.moment_epsilon = float(moment_epsilon)

        self.first_moments: list[torch.Tensor | None] = []
        self.second_moments: list[torch.Tensor | None] = []
        self.moment_steps: list[int] = []

    def get_options(self) -> dict[str, Any]:
        return {
            "c0": self.c0,
            "c1": self.c1,
            "w_min": self.w_min,
            "w_max": self.w_max,
            "moment_blend": self.moment_blend,
            "moment_beta1": self.moment_beta1,
            "moment_beta2": self.moment_beta2,
            "moment_step_size": self.moment_step_size,
            "moment_epsilon": self.moment_epsilon,
        }

    def prepare_fit(self, context: FitContext) -> None:
        dim = context.base_vector.numel()
        device = context.device
        dtype = context.base_vector.dtype
        n_particles = context.n_particles

        self.first_moments = []
        self.second_moments = []
        self.moment_steps = [0] * n_particles

        if self.moment_blend > 0.0:
            for _ in range(n_particles):
                self.first_moments.append(
                    torch.zeros(dim, device=device, dtype=dtype)
                )
                self.second_moments.append(
                    torch.zeros(dim, device=device, dtype=dtype)
                )
        else:
            for _ in range(n_particles):
                self.first_moments.append(None)
                self.second_moments.append(None)

    def reset_particle_state(self, particle_idx: int) -> None:
        if (
            particle_idx < len(self.first_moments)
            and self.first_moments[particle_idx] is not None
        ):
            self.first_moments[particle_idx].zero_()
        if (
            particle_idx < len(self.second_moments)
            and self.second_moments[particle_idx] is not None
        ):
            self.second_moments[particle_idx].zero_()
        if particle_idx < len(self.moment_steps):
            self.moment_steps[particle_idx] = 0

    def propose(
        self, particle_idx: int, state: SwarmState, context: IterationContext
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x = state.positions[particle_idx]
        v = state.velocities[particle_idx]
        pbest = state.pbest_positions[particle_idx]
        gbest = state.gbest_position
        device = x.device
        dtype = x.dtype

        r1 = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)
        r2 = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)

        cog = self.c0 * r1 * (pbest - x)
        if not context.is_negative:
            soc = self.c1 * r2 * (gbest - x)
        else:
            soc = -self.c1 * r2 * (gbest - x)

        standard_velocity = context.w * v + cog + soc

        if self.moment_blend > 0.0:
            fm = self.first_moments[particle_idx]
            sm = self.second_moments[particle_idx]
            if fm is None or sm is None:
                fm = torch.zeros_like(x)
                sm = torch.zeros_like(x)
                self.first_moments[particle_idx] = fm
                self.second_moments[particle_idx] = sm

            self.moment_steps[particle_idx] += 1
            step = self.moment_steps[particle_idx]

            fm.mul_(self.moment_beta1).add_(
                standard_velocity, alpha=1.0 - self.moment_beta1
            )
            sm.mul_(self.moment_beta2).addcmul_(
                standard_velocity,
                standard_velocity,
                value=1.0 - self.moment_beta2,
            )

            bias_corr1 = 1.0 - (self.moment_beta1**step)
            bias_corr2 = 1.0 - (self.moment_beta2**step)

            first_hat = fm / bias_corr1
            second_hat = sm / bias_corr2

            historical_scale = torch.sqrt(torch.mean(second_hat))
            adaptive_direction = (
                self.moment_step_size
                * historical_scale
                * first_hat
                / (torch.sqrt(second_hat) + self.moment_epsilon)
            )

            v_new = (
                (1.0 - self.moment_blend) * standard_velocity
                + self.moment_blend * adaptive_direction
            )
        else:
            v_new = standard_velocity

        return None, v_new



class RingLocalBestMovement(MovementPlugin):
    metadata = PluginMetadata(
        stage="movement",
        title="Ring Local Best PSO",
        source="10.1109/CEC.2002.1004493",
        gradient_required=False,
        fidelity="canonical",
    )

    def __init__(
        self,
        c0: float = 1.49618,
        c1: float = 1.49618,
        w_min: float = 0.4,
        w_max: float = 0.9,
        neighborhood_radius: int = 1,
    ):
        for name, val in [
            ("c0", c0),
            ("c1", c1),
            ("w_min", w_min),
            ("w_max", w_max),
        ]:
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
            ):
                raise ValueError(f"{name} must be a finite number")
        if float(w_min) > float(w_max):
            raise ValueError("w_min must be <= w_max")

        if (
            isinstance(neighborhood_radius, bool)
            or not isinstance(neighborhood_radius, int)
            or neighborhood_radius < 1
        ):
            raise ValueError("neighborhood_radius must be an integer >= 1")

        self.c0 = float(c0)
        self.c1 = float(c1)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.neighborhood_radius = int(neighborhood_radius)
        self.renewal: str = "acc"

    def prepare_fit(self, context: FitContext) -> None:
        self.renewal = context.renewal
    def get_options(self) -> dict[str, Any]:
        return {
            "c0": self.c0,
            "c1": self.c1,
            "w_min": self.w_min,
            "w_max": self.w_max,
            "neighborhood_radius": self.neighborhood_radius,
        }

    def propose(
        self, particle_idx: int, state: SwarmState, context: IterationContext
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x = state.positions[particle_idx]
        v = state.velocities[particle_idx]
        pbest = state.pbest_positions[particle_idx]
        device = x.device
        dtype = x.dtype
        n_particles = len(state.positions)

        neighbors = []
        for r in range(-self.neighborhood_radius, self.neighborhood_radius + 1):
            idx = (particle_idx + r) % n_particles
            if idx not in neighbors:
                neighbors.append(idx)

        renewal = (
            context.optimizer.renewal
            if getattr(context, "optimizer", None) is not None
            else getattr(self, "renewal", "acc")
        )

        lbest_idx = None
        lbest_score = None
        for j in neighbors:
            score_j = state.pbest_scores[j]
            if _is_better_score(score_j, lbest_score, renewal):
                lbest_score = score_j
                lbest_idx = j

        lbest = state.pbest_positions[lbest_idx]

        r1 = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)
        r2 = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)

        cog = self.c0 * r1 * (pbest - x)
        if not context.is_negative:
            soc = self.c1 * r2 * (lbest - x)
        else:
            soc = -self.c1 * r2 * (lbest - x)

        v_new = context.w * v + cog + soc
        return None, v_new


class QuantumMovement(MovementPlugin):
    metadata = PluginMetadata(
        stage="movement",
        title="Quantum PSO",
        source="10.1109/CEC.2004.1330875",
        gradient_required=False,
        fidelity="canonical",
    )

    def __init__(
        self,
        beta_min: float = 0.5,
        beta_max: float = 1.0,
    ):
        for name, val in [("beta_min", beta_min), ("beta_max", beta_max)]:
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
            ):
                raise ValueError(f"{name} must be a finite number")
        if float(beta_min) > float(beta_max):
            raise ValueError("beta_min must be <= beta_max")
        if float(beta_min) < 0.0:
            raise ValueError("beta_min must be >= 0.0")

        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self._mbest: torch.Tensor | None = None

    def get_options(self) -> dict[str, Any]:
        return {"beta_min": self.beta_min, "beta_max": self.beta_max}

    def prepare_fit(self, context: FitContext) -> None:
        if context.negative_swarm != 0.0:
            raise ValueError("negative_swarm is unsupported for Quantum PSO")
        if context.mutation_swarm != 0.0:
            raise ValueError("mutation_swarm is unsupported for Quantum PSO")
        if context.velocity_limit is not None:
            raise ValueError("velocity_limit is unsupported for Quantum PSO")
        self._mbest = None

    def on_epoch_end(self, state: SwarmState, context: IterationContext) -> None:
        if state.pbest_positions:
            self._mbest = (
                torch.stack(state.pbest_positions, dim=0).mean(dim=0).detach().clone()
            )

    def reset_particle_state(self, particle_idx: int) -> None:
        pass

    def propose(
        self, particle_idx: int, state: SwarmState, context: IterationContext
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if context.is_negative:
            raise ValueError("negative_swarm is unsupported for Quantum PSO")

        x = state.positions[particle_idx]
        pbest = state.pbest_positions[particle_idx]
        gbest = state.gbest_position
        device = x.device
        dtype = x.dtype

        if self._mbest is None:
            mbest = (
                torch.stack(state.pbest_positions, dim=0).mean(dim=0).detach().clone()
            )
        else:
            mbest = self._mbest.to(device=device, dtype=dtype)

        if context.total_epochs <= 2:
            beta = self.beta_max
        else:
            frac = (context.epoch - 1) / float(context.total_epochs - 2)
            frac = max(0.0, min(1.0, frac))
            beta = self.beta_max - (self.beta_max - self.beta_min) * frac

        phi = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)
        p = phi * pbest + (1.0 - phi) * gbest

        u = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)
        u = torch.clamp(u, min=1e-10, max=1.0)
        ln_u_inv = torch.log(1.0 / u)

        sign_rand = context.rng.uniform(x.shape, 0.0, 1.0, device=device, dtype=dtype)
        sign = torch.where(sign_rand < 0.5, 1.0, -1.0)

        delta = beta * torch.abs(mbest - x) * ln_u_inv
        x_new = p + sign * delta

        v_new = torch.zeros_like(x)
        return x_new, v_new

class ConvergencePlugin(BasePlugin):
    def on_particle_evaluated(
        self,
        particle_idx: int,
        score: tuple[float, float, float],
        pbest_improved: bool,
        context: IterationContext,
    ) -> bool:
        return False

    def on_epoch_end(
        self,
        gbest_score: tuple[float, float, float],
        gbest_improved: bool,
        context: IterationContext,
    ) -> bool:
        return False

    def reset_particle(self, particle_idx: int) -> None:
        pass


class NoConvergence(ConvergencePlugin):
    metadata = PluginMetadata(
        stage="convergence",
        title="No Convergence Action",
        source=None,
        gradient_required=False,
        fidelity="canonical",
    )


class ParticleResetConvergence(ConvergencePlugin):
    metadata = PluginMetadata(
        stage="convergence",
        title="Particle Stagnation Reset",
        source=None,
        gradient_required=False,
        fidelity="experimental",
    )

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        monitor: str = "loss",
    ):
        if (
            isinstance(patience, bool)
            or not isinstance(patience, int)
            or patience < 1
        ):
            raise ValueError("patience must be an integer >= 1")
        if (
            isinstance(min_delta, bool)
            or not isinstance(min_delta, (int, float))
            or not math.isfinite(min_delta)
            or float(min_delta) < 0.0
        ):
            raise ValueError("min_delta must be a finite nonnegative number")
        if monitor not in ("loss", "acc", "accuracy", "mse"):
            raise ValueError("monitor must be one of 'loss', 'acc', 'accuracy', 'mse'")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = monitor
        self.patience_counters: list[int] = []
        self.best_monitor_values: list[float | None] = []

    def get_options(self) -> dict[str, Any]:
        return {"patience": self.patience, "min_delta": self.min_delta, "monitor": self.monitor}

    def prepare_fit(self, context: FitContext) -> None:
        self.patience_counters = [0] * context.n_particles
        self.best_monitor_values = [None] * context.n_particles

    def reset_particle(self, particle_idx: int) -> None:
        if particle_idx < len(self.patience_counters):
            self.patience_counters[particle_idx] = 0
            self.best_monitor_values[particle_idx] = None

    def on_particle_evaluated(
        self,
        particle_idx: int,
        score: tuple[float, float, float],
        pbest_improved: bool,
        context: IterationContext,
    ) -> bool:
        if self.monitor in ("acc", "accuracy"):
            current_val = score[1]
        elif self.monitor == "loss":
            current_val = score[0]
        elif self.monitor == "mse":
            current_val = score[2]
        else:
            current_val = score[0]

        if self.best_monitor_values[particle_idx] is None:
            self.best_monitor_values[particle_idx] = current_val
            self.patience_counters[particle_idx] = 0
            return False

        prev_val = self.best_monitor_values[particle_idx]
        assert prev_val is not None
        improved = False
        if self.monitor in ("acc", "accuracy"):
            delta = current_val - prev_val
            improved = _is_at_least_delta(delta, self.min_delta)
        else:
            delta = prev_val - current_val
            improved = _is_at_least_delta(delta, self.min_delta)

        if improved:
            self.best_monitor_values[particle_idx] = current_val
            self.patience_counters[particle_idx] = 0
            return False
        else:
            self.patience_counters[particle_idx] += 1
            if self.patience_counters[particle_idx] >= self.patience:
                return True
            return False


class EarlyStoppingConvergence(ConvergencePlugin):
    metadata = PluginMetadata(
        stage="convergence",
        title="Global Best Early Stopping",
        source=None,
        gradient_required=False,
        fidelity="canonical",
    )

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        monitor: str = "loss",
    ):
        if (
            isinstance(patience, bool)
            or not isinstance(patience, int)
            or patience < 1
        ):
            raise ValueError("patience must be an integer >= 1")
        if (
            isinstance(min_delta, bool)
            or not isinstance(min_delta, (int, float))
            or not math.isfinite(min_delta)
            or float(min_delta) < 0.0
        ):
            raise ValueError("min_delta must be a finite nonnegative number")
        if monitor not in ("loss", "acc", "accuracy", "mse"):
            raise ValueError("monitor must be one of 'loss', 'acc', 'accuracy', 'mse'")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = monitor
        self.gbest_patience = 0
        self.best_gbest_monitor: float | None = None

    def get_options(self) -> dict[str, Any]:
        return {"patience": self.patience, "min_delta": self.min_delta, "monitor": self.monitor}

    def prepare_fit(self, context: FitContext) -> None:
        self.gbest_patience = 0
        self.best_gbest_monitor = None

    def on_epoch_end(
        self,
        gbest_score: tuple[float, float, float],
        gbest_improved: bool,
        context: IterationContext,
    ) -> bool:
        if self.monitor in ("acc", "accuracy"):
            current_val = gbest_score[1]
        elif self.monitor == "loss":
            current_val = gbest_score[0]
        elif self.monitor == "mse":
            current_val = gbest_score[2]
        else:
            current_val = gbest_score[0]

        if self.best_gbest_monitor is None:
            self.best_gbest_monitor = current_val
            self.gbest_patience = 0
            return False

        if self.monitor in ("acc", "accuracy"):
            delta = current_val - self.best_gbest_monitor
            improved = _is_at_least_delta(delta, self.min_delta)
        else:
            delta = self.best_gbest_monitor - current_val
            improved = _is_at_least_delta(delta, self.min_delta)

        if improved:
            self.best_gbest_monitor = current_val
            self.gbest_patience = 0
            return False
        else:
            self.gbest_patience += 1
            if self.gbest_patience >= self.patience:
                return True
            return False


class RefinementPlugin(BasePlugin):
    def refine(
        self,
        gbest_position: torch.Tensor,
        gbest_score: tuple[float, float, float],
        eval_fn: Any,
        context: FitContext,
    ) -> tuple[torch.Tensor, tuple[float, float, float]]:
        raise NotImplementedError


class NoRefinement(RefinementPlugin):
    metadata = PluginMetadata(
        stage="refinement",
        title="No Refinement",
        source=None,
        gradient_required=False,
        fidelity="canonical",
    )

    def refine(
        self,
        gbest_position: torch.Tensor,
        gbest_score: tuple[float, float, float],
        eval_fn: Any,
        context: FitContext,
    ) -> tuple[torch.Tensor, tuple[float, float, float]]:
        return gbest_position, gbest_score


class AdamRefinement(RefinementPlugin):
    metadata = PluginMetadata(
        stage="refinement",
        title="Adam Post-Search Refinement",
        source="10.1016/j.amc.2006.07.025",
        gradient_required=True,
        fidelity="experimental",
    )

    def __init__(self, epochs: int = 10, lr: float = 0.001):
        if (
            isinstance(epochs, bool)
            or not isinstance(epochs, int)
            or epochs < 0
        ):
            raise ValueError("epochs must be an integer >= 0")
        if (
            isinstance(lr, bool)
            or not isinstance(lr, (int, float))
            or not math.isfinite(lr)
            or float(lr) <= 0.0
        ):
            raise ValueError("lr must be a positive finite float")

        self.epochs = int(epochs)
        self.lr = float(lr)

    def get_options(self) -> dict[str, Any]:
        return {"epochs": self.epochs, "lr": self.lr}

    def prepare_fit(self, context: FitContext) -> None:
        if context.refinement_epochs > 0:
            self.epochs = context.refinement_epochs
        if context.refinement_lr > 0:
            self.lr = context.refinement_lr

    def refine(
        self,
        gbest_position: torch.Tensor,
        gbest_score: tuple[float, float, float],
        eval_fn: Any,
        context: FitContext,
    ) -> tuple[torch.Tensor, tuple[float, float, float]]:
        if self.epochs <= 0:
            return gbest_position, gbest_score

        x_fit, y_fit = context.optimizer.evaluation_plugin.get_fitness_data(
            context.x_train, context.y_train, context
        )
        optimizer = context.optimizer
        optimizer._refine(
            x_fit,
            y_fit,
            refinement_epochs=self.epochs,
            refinement_lr=self.lr,
            batch_size=context.batch_size,
            renewal=context.renewal,
        )
        refined_score = optimizer.get_best_score()
        if refined_score is not None and optimizer._global_best_weights is not None:
            return optimizer._global_best_weights, refined_score
        return gbest_position, gbest_score


BUILTIN_PLUGINS: dict[str, dict[str, type[BasePlugin]]] = {
    "movement": {
        "original": OriginalMovement,
        "inertia": InertiaMovement,
        "constriction": ConstrictionMovement,
        "fips": FIPSMovement,
        "clpso": CLPSOMovement,
        "bare_bones": BareBonesMovement,
        "adaptive_moment": AdaptiveMomentMovement,
        "local_best": RingLocalBestMovement,
        "quantum": QuantumMovement,
    },
    "initialization": {
        "model_noise": ModelNoiseInitialization,
        "uniform": UniformInitialization,
    },
    "evaluation": {
        "full": FullEvaluation,
        "fixed_subset": FixedSubsetEvaluation,
    },
    "convergence": {
        "none": NoConvergence,
        "particle_reset": ParticleResetConvergence,
        "early_stopping": EarlyStoppingConvergence,
    },
    "refinement": {
        "none": NoRefinement,
        "adam": AdamRefinement,
    },
}


def available_plugins(stage: str | None = None) -> dict[str, Any]:
    if stage is not None:
        if stage not in BUILTIN_PLUGINS:
            raise ValueError(
                f"Unknown stage '{stage}'. Must be one of {list(BUILTIN_PLUGINS.keys())}"
            )
        return {
            name: cls().metadata for name, cls in BUILTIN_PLUGINS[stage].items()
        }
    return {
        s: {name: cls().metadata for name, cls in BUILTIN_PLUGINS[s].items()}
        for s in BUILTIN_PLUGINS
    }


def get_plugin(
    stage: str,
    selector: str | BasePlugin,
    options: dict[str, Any] | None = None,
) -> BasePlugin:
    if stage not in BUILTIN_PLUGINS:
        raise ValueError(f"Unknown stage '{stage}'")
    if isinstance(selector, BasePlugin):
        plugin_copy = copy.deepcopy(selector)
        if plugin_copy.metadata.stage != stage:
            raise ValueError(
                f"Plugin stage '{plugin_copy.metadata.stage}' does not match expected stage '{stage}'"
            )
        if options:
            cls = type(plugin_copy)
            sig = inspect.signature(cls.__init__)
            valid_params = set(sig.parameters.keys()) - {"self"}
            unknown_params = set(options.keys()) - valid_params
            if unknown_params:
                raise ValueError(
                    f"Unknown or incompatible option(s) {sorted(unknown_params)} for {stage} plugin '{plugin_copy.metadata.title}'"
                )
        return plugin_copy
    if isinstance(selector, str):
        if selector not in BUILTIN_PLUGINS[stage]:
            raise ValueError(
                f"Unknown {stage} plugin '{selector}'. Options: {list(BUILTIN_PLUGINS[stage].keys())}"
            )
        cls = BUILTIN_PLUGINS[stage][selector]
        opts = options or {}
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        unknown_params = set(opts.keys()) - valid_params
        if unknown_params:
            raise ValueError(
                f"Unknown or incompatible option(s) {sorted(unknown_params)} for {stage} plugin '{selector}'. Valid options: {sorted(valid_params)}"
            )
        kwargs = {k: v for k, v in opts.items() if v is not None}
        return cls(**kwargs)
    raise TypeError(f"Invalid selector type {type(selector)} for stage '{stage}'")
