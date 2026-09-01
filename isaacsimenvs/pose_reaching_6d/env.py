"""DirectRLEnv for 6D pose reaching, one hand or many.

Set ``assets.robot_population_seed`` or ``robot_population_path`` and every env
gets its own design sharing one articulation view; leave them unset and
``assets.robot_spec`` names one hand for all of them. Task math lives in the
modules called from each hook.
"""

from __future__ import annotations

import torch
from isaaclab.envs import DirectRLEnv

from .env_cfg import PoseReachEnvCfg
from .obs_utils import (
    apply_action_pipeline, apply_wrench_dr, build_observations,
    compute_intermediate_values, derive_spaces,
)
from .reset_utils import allocate_state_buffers, log_step_metrics, reset_env_state
from .reward_utils import (
    compute_rewards, compute_terminations, update_tolerance_curriculum,
)
from .scene_utils import (
    apply_physx_material_properties, finalize_population, resolve_spec,
    setup_scene,
)

__all__ = ["PoseReachEnv", "PoseReachEnvCfg"]


class PoseReachEnv(DirectRLEnv):
    """Grasp a tool-like object off a table and drive it through SE(3) goals."""

    cfg: PoseReachEnvCfg

    def __init__(
        self, cfg: PoseReachEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        """cfg is IN/OUT: callers read the derived spaces back to size the policy."""
        self.robot_spec = spec = resolve_spec(self, cfg)
        derive_spaces(cfg, spec)
        super().__init__(cfg, render_mode, **kwargs)   # runs _setup_scene
        apply_physx_material_properties(self)          # needs the built stage
        allocate_state_buffers(self)
        finalize_population(self)                      # needs both of the above

    # --- Isaac Lab hooks -----------------------------------------------------

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        reset_env_state(
            self, torch.as_tensor(env_ids, device=self.device, dtype=torch.long))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        apply_action_pipeline(self, actions)
        apply_wrench_dr(self)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._cur_targets)  # decimation x / step

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)
        compute_intermediate_values(self)
        return compute_terminations(self)

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)
        log_step_metrics(self)
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return build_observations(self)
