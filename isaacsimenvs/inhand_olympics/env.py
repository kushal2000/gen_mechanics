"""InHandOlympicsEnv: skeleton. The scene and joint-target actions work; observations are joint state only, the
reward is zero and episodes only time out. Tasks fill these in later.

Actions: one per joint in the articulation's (Isaac Lab) joint order, ``env.robot.joint_names``, each in [-1, 1]
mapped onto that joint's limits as an absolute position target.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import DirectRLEnv

from .env_cfg import InHandOlympicsEnvCfg
from .scene import setup_scene
from .wrist import WRIST_JOINTS


class InHandOlympicsEnv(DirectRLEnv):
    cfg: InHandOlympicsEnvCfg

    def __init__(self, cfg: InHandOlympicsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        limits = self.robot.data.soft_joint_pos_limits[0]
        self.joint_lower, self.joint_upper = limits[:, 0].clone(), limits[:, 1].clone()
        self.wrist_ids = [self.robot.joint_names.index(n) for n in WRIST_JOINTS]
        self.hand_ids = [i for i in range(self.robot.num_joints) if i not in self.wrist_ids]
        self.targets = self.robot.data.default_joint_pos.clone()

    def _setup_scene(self) -> None:
        setup_scene(self)

    def action_to_target(self, actions: torch.Tensor) -> torch.Tensor:
        a = actions.clamp(-1.0, 1.0)
        return self.joint_lower + 0.5 * (a + 1.0) * (self.joint_upper - self.joint_lower)

    def target_to_action(self, targets: torch.Tensor) -> torch.Tensor:
        return 2.0 * (targets - self.joint_lower) / (self.joint_upper - self.joint_lower) - 1.0

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.targets = self.action_to_target(actions)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.targets)

    def _get_observations(self) -> dict:
        q = self.target_to_action(self.robot.data.joint_pos)
        return {"policy": torch.cat([q, 0.1 * self.robot.data.joint_vel], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        q = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.targets[env_ids] = q
