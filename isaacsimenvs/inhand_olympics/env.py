"""InHandOlympicsEnv: one hand on a 6-DOF dummy wrist above a table, one pen, several tasks.

Every env runs one task per episode, drawn uniformly from ``cfg.task.names`` at reset and shown to the policy as a
one-hot. A task is a sequence of 6D goal poses for the pen (tasks/); everything after that is shared here:

    success      every pen keypoint (tip, centre, tail) within goal.success_tolerance of its goal position;
                 a hit sets the task's next goal and restarts the episode clock (goal.seconds_per_goal per goal)
    reward       keypoint progress toward the goal (best-so-far, in-hand style) + goal bonus + fingertip approach
                 progress - joint-velocity penalties
    terminated   the pen left the goal's neighbourhood (termination.max_goal_distance) or fell off the table
    truncated    out of time for this goal, or goal.max_successes hit

Actions: one per joint in canonical order (``env.canon_joint_names``: the 6 wrist joints, then the hand's in its
spec's order, as in-hand reorientation puts arm before hand), each in [-1, 1]. Wrist joints are delta commands,
finger joints absolute targets, both through an EMA and an optional random action latency; see env_cfg.ActionCfg.
Internally everything is in Isaac Lab's joint order.

Observations: in-hand reorientation's format, observations.py ("policy" from obs.obs_list, "critic" from
obs.state_list).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_matrix

from .env_cfg import InHandOlympicsEnvCfg
from .goals import keypoints, pen_keypoint_offsets
from .observations import build_observations, field_sizes, width
from .scene import setup_scene
from .tasks import TASK_NAMES, make_tasks
from .wrist import WRIST_JOINTS


class InHandOlympicsEnv(DirectRLEnv):
    cfg: InHandOlympicsEnvCfg

    def __init__(self, cfg: InHandOlympicsEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.episode_length_s = cfg.goal.seconds_per_goal
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        limits = self.robot.data.soft_joint_pos_limits[0]
        self.joint_lower, self.joint_upper = limits[:, 0].clone(), limits[:, 1].clone()
        self.wrist_ids = [self.robot.joint_names.index(j) for j in WRIST_JOINTS]
        self.hand_ids = [i for i in range(self.robot.num_joints) if i not in self.wrist_ids]
        spec, lab = self.robot_spec, list(self.robot.joint_names)
        self.canon_joint_names = list(WRIST_JOINTS) + list(spec.hand_joint_names)
        assert sorted(self.canon_joint_names) == sorted(lab), (self.canon_joint_names, lab)
        self.perm_lab_to_canon = torch.tensor([lab.index(j) for j in self.canon_joint_names], device=dev)
        self.perm_canon_to_lab = torch.tensor([self.canon_joint_names.index(j) for j in lab], device=dev)
        hand = slice(len(WRIST_JOINTS), None)
        self.joint_lower_canon = self.joint_lower[self.perm_lab_to_canon].expand(n, -1).clone()
        self.joint_upper_canon = self.joint_upper[self.perm_lab_to_canon].expand(n, -1).clone()
        self.joint_lower_hand = self.joint_lower_canon[:, hand]
        self.joint_upper_hand = self.joint_upper_canon[:, hand]
        self.joint_enabled = (self.joint_upper_hand - self.joint_lower_hand > 1e-6).float()
        self.prev_joint_pos_canon = self.robot.data.joint_pos[:, self.perm_lab_to_canon].clone()
        self.prev_joint_vel_canon = torch.zeros_like(self.prev_joint_pos_canon)
        a = self.cfg.action
        self.wrist_speed = torch.tensor([a.wrist_max_speed_lin] * 3 + [a.wrist_max_speed_ang] * 3, device=dev)
        # PD targets (all joints) and the wrist's integrated, un-smoothed target.
        self.targets = self.robot.data.default_joint_pos.clone()
        self.wrist_command = self.targets[:, self.wrist_ids].clone()
        # The last action_delay_max + 1 actions, newest first; and the action applied this step (after the delay).
        self.action_queue = torch.zeros(n, a.action_delay_max + 1, self.robot.num_joints, device=dev)
        self.applied_actions = torch.zeros(n, self.robot.num_joints, device=dev)

        self.palm_body = self.robot.body_names.index(spec.palm_body_name)
        self.tip_bodies = [self.robot.body_names.index(b) for b in spec.fingertip_body_names]
        self.tip_offsets = torch.tensor(spec.fingertip_offsets, device=dev)
        self.pen_offsets = pen_keypoint_offsets(self.cfg.pen.length, dev)
        self._init_hand_geometry(spec)

        # Tasks and goals. Goal poses are in the env frame.
        self.tasks = make_tasks(self)
        self.task_id = torch.zeros(n, dtype=torch.long, device=dev)          # index into self.tasks
        self.task_onehot_index = torch.tensor([TASK_NAMES.index(t.name) for t in self.tasks], device=dev)
        self.goal_pos = torch.zeros(n, 3, device=dev)
        self.goal_quat = torch.zeros(n, 4, device=dev)
        self.goal_quat[:, 0] = 1.0
        self.successes = torch.zeros(n, device=dev)
        # Best-so-far distances for the progress rewards; NaN = not measured yet (taken from the next step).
        self.closest_goal_dist = torch.full((n,), math.nan, device=dev)
        self.closest_tip_dist = torch.full((n, len(self.tip_bodies)), math.nan, device=dev)
        self.goal_dist = torch.zeros(n, device=dev)
        self.is_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.termination_reasons: dict[str, torch.Tensor] = {}
        self.reward_terms: dict[str, torch.Tensor] = {}
        self.reward_buf = torch.zeros(n, device=dev)       # DirectRLEnv sets it in step(); the obs reads it at reset

    def _setup_scene(self) -> None:
        setup_scene(self)
        spec, o = self.robot_spec, self.cfg.obs
        sizes = field_sizes(self.cfg.action_space, spec.num_hand_joints, spec.num_fingertips, len(TASK_NAMES))
        self.cfg.observation_space = width(o.obs_list, sizes)
        self.cfg.state_space = width(o.state_list, sizes)

    def _init_hand_geometry(self, spec) -> None:
        """The buffers in-hand's observation helpers read, built as pose_reaching_6d reset.py builds them for one
        fixed hand: link boxes per joint, the palm centre, and the canonical palm frame from the palm slab."""
        n, dev = self.num_envs, self.device
        self._joint_link_body_ids = self.robot.find_bodies(list(spec.joint_link_bodies), preserve_order=True)[0]
        assert len(self._joint_link_body_ids) == spec.num_hand_joints
        self._joint_link_bbox_local = torch.tensor(spec.joint_link_boxes, device=dev).expand(n, -1, -1, -1)
        self._joint_geometry_valid = torch.tensor(spec.joint_geometry_valid, device=dev).expand(n, -1)
        self.palm_center_offset = torch.tensor(spec.palm_center_offset, device=dev).expand(n, -1)
        kp = torch.tensor(spec.palm_keypoints, device=dev).expand(n, -1, -1)       # (N, 4, 3), palm body frame
        edges = kp[:, 1:] - kp[:, :1]                                                 # grasp normal, width, length
        self.palm_extents = edges.norm(dim=-1)
        axes = (edges / self.palm_extents.unsqueeze(-1)).transpose(1, 2)
        self.palm_frame_quat = quat_from_matrix(axes)
        self.palm_keypoints_canon = quat_apply_inverse(
            self.palm_frame_quat.unsqueeze(1).expand(-1, 4, -1), kp - self.palm_center_offset.unsqueeze(1))

    def task_onehot(self) -> torch.Tensor:
        onehot = torch.zeros(self.num_envs, len(TASK_NAMES), device=self.device)
        onehot[torch.arange(self.num_envs, device=self.device), self.task_onehot_index[self.task_id]] = 1.0
        return onehot

    # --- used by the tasks --------------------------------------------------------------------------------
    def place_pen(self, ids: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor) -> None:
        """Put the pen at ``pos`` (env frame) / ``quat`` (wxyz), at rest."""
        pose = torch.cat([pos + self.scene.env_origins[ids], quat], dim=-1)
        self.pen.write_root_pose_to_sim(pose, ids)
        self.pen.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), ids)

    def set_goal(self, ids: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor) -> None:
        self.goal_pos[ids] = pos
        self.goal_quat[ids] = quat

    # --- actions --------------------------------------------------------------------------------------------
    def target_to_action(self, targets: torch.Tensor) -> torch.Tensor:
        """Absolute joint positions -> [-1, 1] over the joint ranges (the finger action; the obs scaling)."""
        return 2.0 * (targets - self.joint_lower) / (self.joint_upper - self.joint_lower) - 1.0

    def action_to_target(self, actions: torch.Tensor) -> torch.Tensor:
        return self.joint_lower + 0.5 * (actions.clamp(-1.0, 1.0) + 1.0) * (self.joint_upper - self.joint_lower)

    def wrist_action_toward(self, pose: torch.Tensor) -> torch.Tensor:
        """The wrist action that moves the integrated command toward ``pose`` (N, 6) as fast as allowed."""
        return ((pose - self.wrist_command) / (self.wrist_speed * self.step_dt)).clamp(-1.0, 1.0)

    def action_toward(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """The (canonical-order) action steering every joint to ``joint_pos`` (N, J, Lab order): absolute for the
        fingers, the fastest allowed delta for the wrist. For viewers and tests."""
        act = self.target_to_action(joint_pos)
        act[:, self.wrist_ids] = self.wrist_action_toward(joint_pos[:, self.wrist_ids])
        return act[:, self.perm_lab_to_canon]

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
        self.prev_joint_pos_canon.copy_(self.robot.data.joint_pos[:, self.perm_lab_to_canon])
        self.prev_joint_vel_canon.copy_(self.robot.data.joint_vel[:, self.perm_lab_to_canon])
        a = actions.clamp(-1.0, 1.0)
        cfg = self.cfg.action
        if cfg.use_action_delay and cfg.action_delay_max > 0:
            a = self._delay(a)
        self.applied_actions = a                       # canonical order, after the latency
        a = a[:, self.perm_canon_to_lab]
        w, h = self.wrist_ids, self.hand_ids
        lo, hi = self.joint_lower[w], self.joint_upper[w]
        self.wrist_command = torch.minimum(torch.maximum(
            self.wrist_command + a[:, w] * self.wrist_speed * self.step_dt, lo), hi)
        self.targets[:, w] = cfg.wrist_ema * self.wrist_command + (1.0 - cfg.wrist_ema) * self.targets[:, w]
        hand_raw = self.action_to_target(a)[:, h]
        self.targets[:, h] = cfg.hand_ema * hand_raw + (1.0 - cfg.hand_ema) * self.targets[:, h]

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.targets)

    # --- state ------------------------------------------------------------------------------------------------
    def pen_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The pen's position (env frame) and orientation (wxyz)."""
        return self.pen.data.root_pos_w - self.scene.env_origins, self.pen.data.root_quat_w

    def palm_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.robot.data.body_pos_w[:, self.palm_body] - self.scene.env_origins,
                self.robot.data.body_quat_w[:, self.palm_body])

    def fingertips(self) -> torch.Tensor:
        """(N, F, 3) fingertip pad points, env frame."""
        pos = self.robot.data.body_pos_w[:, self.tip_bodies] - self.scene.env_origins.unsqueeze(1)
        quat = self.robot.data.body_quat_w[:, self.tip_bodies]
        n, f = pos.shape[:2]
        off = quat_apply(quat.reshape(-1, 4), self.tip_offsets.unsqueeze(0).expand(n, f, 3).reshape(-1, 3))
        return pos + off.view(n, f, 3)

    def _goal_distance(self) -> torch.Tensor:
        pos, quat = self.pen_pose()
        kp = keypoints(pos, quat, self.pen_offsets)
        kp_goal = keypoints(self.goal_pos, self.goal_quat, self.pen_offsets)
        return (kp - kp_goal).norm(dim=-1).amax(dim=-1)

    # --- RL hooks ---------------------------------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.goal_dist = self._goal_distance()
        self.is_success = self.goal_dist < self.cfg.goal.success_tolerance
        pos, _ = self.pen_pose()
        t = self.cfg.termination
        away = (pos - self.goal_pos).norm(dim=-1) > t.max_goal_distance
        fell = pos[:, 2] < self.cfg.table.top_height - t.fall_below_table
        out_of_time = (self.episode_length_buf >= self.max_episode_length - 1) & ~self.is_success
        all_goals = (self.successes + self.is_success.float()) >= self.cfg.goal.max_successes
        self.termination_reasons = {"away": away, "fell": fell, "timeout": out_of_time,
                                    "max_successes": all_goals}
        return away | fell, out_of_time | all_goals

    def _get_rewards(self) -> torch.Tensor:
        r = self.cfg.reward
        d = self.goal_dist
        self.closest_goal_dist = torch.where(torch.isnan(self.closest_goal_dist), d, self.closest_goal_dist)
        progress = (self.closest_goal_dist - d).clamp(min=0.0)
        self.closest_goal_dist = torch.minimum(self.closest_goal_dist, d)

        pos, _ = self.pen_pose()
        tip = (self.fingertips() - pos.unsqueeze(1)).norm(dim=-1)                     # (N, F)
        self.closest_tip_dist = torch.where(torch.isnan(self.closest_tip_dist), tip, self.closest_tip_dist)
        approach = (self.closest_tip_dist - tip).clamp(min=0.0).mean(dim=-1)
        self.closest_tip_dist = torch.minimum(self.closest_tip_dist, tip)

        vel = self.robot.data.joint_vel
        self.reward_terms = {
            "keypoint": r.keypoint_rew_scale * progress,
            "goal_bonus": r.goal_bonus * self.is_success.float(),
            "approach": r.approach_rew_scale * approach,
            "wrist_penalty": -r.wrist_action_penalty * vel[:, self.wrist_ids].abs().sum(-1),
            "hand_penalty": -r.hand_action_penalty * vel[:, self.hand_ids].abs().sum(-1),
        }
        reward = sum(self.reward_terms.values())

        hit = self.is_success.nonzero(as_tuple=False).squeeze(-1)
        if len(hit):
            self.successes[hit] += 1
            self._next_goal(hit)
            self.episode_length_buf[hit] = 0
            self.closest_goal_dist[hit] = math.nan
        self.extras["log"] = {f"reward/{k}": v.mean() for k, v in self.reward_terms.items()}
        self.extras["log"]["successes"] = self.successes.mean()
        return reward

    def _next_goal(self, ids: torch.Tensor) -> None:
        for k, task in enumerate(self.tasks):
            sel = ids[self.task_id[ids] == k]
            if len(sel):
                task.next_goal(sel)

    def _get_observations(self) -> dict:
        return build_observations(self)

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(ids)
        root = self.robot.data.default_root_state[ids].clone()
        root[:, :3] += self.scene.env_origins[ids]
        self.robot.write_root_pose_to_sim(root[:, :7], ids)
        q = self.robot.data.default_joint_pos[ids]
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        self.targets[ids] = q
        self.wrist_command[ids] = q[:, self.wrist_ids]
        self.prev_joint_pos_canon[ids] = q[:, self.perm_lab_to_canon]
        self.prev_joint_vel_canon[ids] = 0.0

        self.task_id[ids] = torch.randint(0, len(self.tasks), (len(ids),), device=self.device)
        for k, task in enumerate(self.tasks):
            sel = ids[self.task_id[ids] == k]
            if len(sel):
                task.reset(sel)
        self.successes[ids] = 0.0
        self.closest_goal_dist[ids] = math.nan
        self.closest_tip_dist[ids] = math.nan
