"""InHandOlympicsEnv: skeleton. The scene and the action pipeline work; observations are joint state only, the
reward is zero and episodes only time out. Tasks fill these in later.

Actions: one per joint in the articulation's (Isaac Lab) joint order, ``env.robot.joint_names``, each in [-1, 1].
Wrist joints are delta commands, finger joints absolute targets, both through an EMA and an optional random
action latency; see env_cfg.ActionCfg.
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
        a = self.cfg.action
        self.wrist_speed = torch.tensor([a.wrist_max_speed_lin] * 3 + [a.wrist_max_speed_ang] * 3,
                                        device=self.device)
        # PD targets (all joints) and the wrist's integrated, un-smoothed target.
        self.targets = self.robot.data.default_joint_pos.clone()
        self.wrist_command = self.targets[:, self.wrist_ids].clone()
        # The last action_delay_max + 1 actions, newest first; and the action applied this step (after the delay).
        self.action_queue = torch.zeros(self.num_envs, a.action_delay_max + 1, self.robot.num_joints,
                                        device=self.device)
        self.applied_actions = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)

    def _setup_scene(self) -> None:
        setup_scene(self)

    # --- absolute <-> [-1, 1] over the joint ranges (the finger action; also the observation's scaling) ---
    def target_to_action(self, targets: torch.Tensor) -> torch.Tensor:
        return 2.0 * (targets - self.joint_lower) / (self.joint_upper - self.joint_lower) - 1.0

    def action_to_target(self, actions: torch.Tensor) -> torch.Tensor:
        return self.joint_lower + 0.5 * (actions.clamp(-1.0, 1.0) + 1.0) * (self.joint_upper - self.joint_lower)

    def wrist_action_toward(self, pose: torch.Tensor) -> torch.Tensor:
        """The wrist action that moves the integrated command toward ``pose`` (N, 6) as fast as allowed."""
        return ((pose - self.wrist_command) / (self.wrist_speed * self.step_dt)).clamp(-1.0, 1.0)

    def action_toward(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """The action steering every joint to ``joint_pos`` (N, J): absolute for the fingers, the fastest allowed
        delta for the wrist. For viewers and tests."""
        act = self.target_to_action(joint_pos)
        act[:, self.wrist_ids] = self.wrist_action_toward(joint_pos[:, self.wrist_ids])
        return act

    def _delay(self, a: torch.Tensor) -> torch.Tensor:
        """Action latency: one of the env's last action_delay_max + 1 actions, picked uniformly per env."""
        start = self.episode_length_buf == 0
        if start.any():
            self.action_queue[start] = a[start].unsqueeze(1)
        self.action_queue = torch.roll(self.action_queue, shifts=1, dims=1)
        self.action_queue[:, 0] = a
        k = torch.randint(0, self.action_queue.shape[1], (self.num_envs,), device=self.device)
        return self.action_queue[torch.arange(self.num_envs, device=self.device), k]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        a = actions.clamp(-1.0, 1.0)
        cfg = self.cfg.action
        if cfg.use_action_delay and cfg.action_delay_max > 0:
            a = self._delay(a)
        self.applied_actions = a
        w, h = self.wrist_ids, self.hand_ids
        lo, hi = self.joint_lower[w], self.joint_upper[w]
        self.wrist_command = torch.minimum(torch.maximum(
            self.wrist_command + a[:, w] * self.wrist_speed * self.step_dt, lo), hi)
        self.targets[:, w] = cfg.wrist_ema * self.wrist_command + (1.0 - cfg.wrist_ema) * self.targets[:, w]
        hand_raw = self.action_to_target(a)[:, h]
        self.targets[:, h] = cfg.hand_ema * hand_raw + (1.0 - cfg.hand_ema) * self.targets[:, h]

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
        self.wrist_command[env_ids] = q[:, self.wrist_ids]
        pen = self.pen.data.default_root_state[env_ids].clone()
        pen[:, :3] += self.scene.env_origins[env_ids]
        self.pen.write_root_pose_to_sim(pen[:, :7], env_ids)
        self.pen.write_root_velocity_to_sim(torch.zeros_like(pen[:, 7:]), env_ids)
