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
    compute_intermediate_values,
)
from .reset_utils import allocate_state_buffers, log_step_metrics, reset_env_state
from .reward_utils import (
    compute_rewards, compute_terminations, update_tolerance_curriculum,
)
from .scene_utils import (
    finalize_scene, setup_scene,
)

__all__ = ["PoseReachEnv", "PoseReachEnvCfg"]


class PoseReachEnv(DirectRLEnv):
    """Grasp a tool-like object off a table and drive it through SE(3) goals."""

    cfg: PoseReachEnvCfg

    def __init__(
        self, cfg: PoseReachEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        """cfg is IN/OUT: setup_scene writes the derived spaces onto it."""
        super().__init__(cfg, render_mode, **kwargs)   # runs _setup_scene
        # Need the started sim; must land before the first _reset_idx.
        allocate_state_buffers(self)
        finalize_scene(self)

    # --- Isaac Lab hooks -----------------------------------------------------

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        # isaaclab always passes an int64 tensor on device: reset() sends
        # arange(num_envs), step() sends reset_buf.nonzero(). The None guard
        # every reference task carries is unreachable.
        assert env_ids is not None and env_ids.dtype == torch.long, f"env_ids={env_ids}"
        super()._reset_idx(env_ids)
        reset_env_state(self, env_ids)

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
