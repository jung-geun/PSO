from typing import Any, Literal
import torch
from .plugins import FitContext, InitializationPlugin


class Particle:
    """
    Particle Swarm Optimization particle (engine-owned state, plugin-driven movement).
    Each particle owns position, velocity, personal best score, personal best weights,
    and monitor state on device.
    """

    def __init__(
        self,
        index: int,
        base_vector: torch.Tensor,
        context: FitContext,
        init_plugin: InitializationPlugin,
        negative: bool = False,
    ):
        self.index = index
        self.negative = negative

        pos, vel = init_plugin.initialize(index, base_vector, context)
        self.position: torch.Tensor = pos
        self.velocity: torch.Tensor = vel

        self.personal_best_score: tuple[float, float, float] | None = None
        self.personal_best_weights: torch.Tensor | None = None
        self.personal_best_monitor_value: float | None = None

    def reset(
        self,
        base_vector: torch.Tensor,
        context: FitContext,
        init_plugin: InitializationPlugin,
    ) -> None:
        """
        Resets particle position, velocity, and personal best state using the initialization plugin.
        """
        pos, vel = init_plugin.initialize(self.index, base_vector, context)
        self.position = pos
        self.velocity = vel
        self.personal_best_score = None
        self.personal_best_weights = None
        self.personal_best_monitor_value = None

    def apply_boundary_strategy(
        self,
        particle_min: float | None,
        particle_max: float | None,
        boundary_strategy: str = "clip",
    ) -> None:
        """
        Applies boundary constraints (clip or reflect) to particle position and velocity.
        """
        if particle_min is not None and particle_max is not None:
            if boundary_strategy == "reflect":
                span = float(particle_max - particle_min)
                shift = self.position - particle_min
                q = torch.floor(shift / span).to(torch.int64)
                m = torch.remainder(shift, 2.0 * span)
                bounded_pos = torch.where(
                    m <= span,
                    particle_min + m,
                    particle_min + (2.0 * span - m),
                )
                self.velocity = torch.where(q % 2 != 0, -self.velocity, self.velocity)
                self.position = bounded_pos
            else:
                self.position = torch.clamp(
                    self.position, particle_min, particle_max
                )
