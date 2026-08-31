"""Thin DirectRLEnv wrapper for the PoseReach task.

The env owns Isaac Lab hook wiring and state buffers. Task math lives in the
utility modules called from each hook, and everything hardware-specific comes
from ``self.robot_spec`` (see isaacsimenvs/pose_reaching_6d/scene_utils/robots/), so swapping hands is a config
change rather than an edit here.
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv

from isaacsimenvs.pose_reaching_6d.scene_utils.robots import get_robot_spec

from .env_cfg import PoseReachEnvCfg
from .obs_utils import apply_action_pipeline, apply_wrench_dr
from .reset_utils import log_step_metrics
from .obs_utils import (
    build_observations,
    compute_intermediate_values,
    compute_obs_dim,
)
from .reset_utils import allocate_state_buffers, reset_env_state
from .reward_utils import compute_rewards
from .scene_utils import apply_physx_material_properties, setup_scene
from .reward_utils import compute_terminations, update_tolerance_curriculum


__all__ = ["PoseReachEnv", "PoseReachEnvCfg"]


class PoseReachEnv(DirectRLEnv):
    cfg: PoseReachEnvCfg

    def __init__(
        self, cfg: PoseReachEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        # Resolve the robot BEFORE super().__init__: DirectRLEnv and rl_games
        # read action_space / observation_space off the configclass, and all
        # three are derived from the spec. _setup_scene also needs self.robot_spec.
        spec = self._resolve_spec(cfg)
        self.robot_spec = spec

        # action_space defaults to 0 ("derive"). A non-zero value is a caller
        # assertion about the robot; disagreeing with the spec means a stale
        # config, which would otherwise silently truncate the action vector.
        if cfg.action_space not in (0, spec.num_joints):
            raise ValueError(
                f"cfg.action_space={cfg.action_space} disagrees with robot_spec "
                f"{spec.name!r} ({spec.num_joints} joints). Leave it 0 to derive."
            )
        cfg.action_space = spec.num_joints

        # Obs/state widths follow the configured field lists and the hand's
        # joint and fingertip counts.
        cfg.observation_space = compute_obs_dim(cfg.obs.obs_list, spec)
        cfg.state_space = compute_obs_dim(cfg.obs.state_list, spec)

        super().__init__(cfg, render_mode, **kwargs)  # runs _setup_scene
        apply_physx_material_properties(self)
        allocate_state_buffers(self)
        self._post_init_hook()

    # --- extension points for PoseReachMultiEnv -----------------------------
    # Two hooks rather than `if population is not None` scattered through this
    # class: the single-embodiment env is the one the pretrained policy, the
    # parity test and the running training jobs depend on, and it should not
    # carry branches for a mode it never takes.

    def _resolve_spec(self, cfg: PoseReachEnvCfg):
        """The RobotSpec defining the action and observation layout."""
        return get_robot_spec(cfg.assets.robot_spec)

    def _post_init_hook(self) -> None:
        """Runs after the scene, materials and state buffers exist."""
        return None

    # --- checkpointable env state -------------------------------------------
    # rl_games saves whatever vec_env.get_env_state() returns into the
    # checkpoint and hands it back on resume. Nothing implemented it, so
    # env_state was None in every checkpoint and a resumed run silently
    # restarted its curriculum -- an easier task on a different reward scale
    # than the checkpoint was written under, since the tolerance also scales the
    # keypoint reward. The state is defined HERE, on the env that owns it,
    # rather than in the rl_games adapter that merely ferries it.

    def get_curriculum_state(self) -> dict:
        """Curriculum state for the checkpoint. See utils/curriculum.py."""
        from .reward_utils import get_curriculum_state
        return get_curriculum_state(self)

    def set_curriculum_state(self, state: dict | None) -> None:
        """Restore curriculum state from a checkpoint."""
        from .reward_utils import set_curriculum_state
        set_curriculum_state(self, state)

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        reset_env_state(
            self,
            torch.as_tensor(env_ids, device=self.device, dtype=torch.long),
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        apply_action_pipeline(self, actions)
        apply_wrench_dr(self)

    def _apply_action(self) -> None:
        # Called decimation times per policy step; idempotent.
        self.robot.set_joint_position_target(self._cur_targets)

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

