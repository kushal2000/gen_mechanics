"""DirectRLEnv for stationary-hand in-hand reorientation.

Subclasses ``PoseReachEnv`` so ``coevolution/utils/rlgames_utils.py``'s
``isinstance(inner, PoseReachEnv)`` check keeps working -- that is the gate
that ferries curriculum state through an rl_games checkpoint on resume (see
``reward_utils.py``'s module docstring). Every hook is overridden, though:
this task has no arm, no table, and a different scene/reward/reset/obs
pipeline end to end, so nothing of ``PoseReachEnv``'s own hook BODIES is
reused, only its identity and its ``env_cfg.TerminationCfg``.
"""

from __future__ import annotations

import torch
from isaaclab.envs import DirectRLEnv

from isaacsimenvs.pose_reaching_6d.env import PoseReachEnv

from .env_cfg import InHandReorientEnvCfg
from .obs_utils import build_observations, compute_intermediate_values, pre_physics_step
from .reset_utils import allocate_state_buffers, log_step_metrics, reset_env_state
from .reward_utils import (
    compute_rewards, compute_terminations, update_goal_curriculum, update_tolerance_curriculum,
)
from .reward_utils import extra_curriculum_state as _extra_curriculum_state
from .reward_utils import restore_extra_curriculum_state as _restore_extra_curriculum_state
from .scene_utils import finalize_scene, setup_scene

__all__ = ["InHandReorientEnv", "InHandReorientEnvCfg"]


class InHandReorientEnv(PoseReachEnv):
    """Reorient a cube in a fixed, palm-up hand towards a target orientation."""

    cfg: InHandReorientEnvCfg

    def __init__(
        self, cfg: InHandReorientEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        # Deliberately DirectRLEnv.__init__, not PoseReachEnv.__init__: the
        # latter calls pose_reaching_6d's own allocate_state_buffers/
        # finalize_scene after the super().__init__() that boots the sim.
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)
        allocate_state_buffers(self)
        finalize_scene(self)

    # --- Isaac Lab hooks -----------------------------------------------------

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        assert env_ids is not None and env_ids.dtype == torch.long, f"env_ids={env_ids}"
        DirectRLEnv._reset_idx(self, env_ids)
        reset_env_state(self, env_ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        pre_physics_step(self, actions)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._cur_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)  # increments self._frame_counter
        update_goal_curriculum(self)  # reads it right after -- see its docstring
        compute_intermediate_values(self)
        return compute_terminations(self)

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)
        log_step_metrics(self)
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return build_observations(self)

    # --- generic curriculum-checkpoint hook -----------------------------
    # pose_reaching_6d.reward_utils.curriculum.get_curriculum_state/
    # set_curriculum_state (shared with PoseReachEnv) call these by name,
    # guarded with callable(...), to ferry this env's EXTRA (goal-difficulty)
    # curriculum state through the same rl_games checkpoint the tolerance
    # curriculum already uses, without that shared module knowing this env's
    # own state names.

    def extra_curriculum_state(self) -> dict:
        return _extra_curriculum_state(self)

    def restore_extra_curriculum_state(self, state: dict) -> None:
        _restore_extra_curriculum_state(self, state)
