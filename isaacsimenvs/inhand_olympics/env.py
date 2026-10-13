"""InHandOlympicsEnv: one hand on a 6-DOF dummy wrist above a table, one pen, several tasks (tasks/).

Every env runs one task per episode, drawn uniformly from ``cfg.task.names`` at reset and shown to the policy as a
one-hot. A task is a sequence of 6D goal poses for the pen (tasks/); everything after that is shared here:

    success      every pen keypoint (4 along its axis) within goal.success_tolerance of its goal position for the
                 task's success_steps consecutive steps; a hit sets the task's next goal and restarts the episode
                 clock (goal.seconds_per_goal per goal)
    perturbation random pushes and twists on the pen in envs whose task asks for them (cfg.perturb), from the
                 episode's first hit on, while the pen is lifted
    reward       keypoint progress toward the goal (best-so-far, in-hand style) + goal bonus (spread over a goal's
                 success_steps: bonus / success_steps per step within tolerance) + fingertip approach progress
                 - joint-velocity penalties - a penalty on the step the pen is lost (away or fell)
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
from .tasks import NUM_TASK_SLOTS, TASK_NAMES, make_tasks
from .wrist import WRIST_JOINTS


class InHandOlympicsEnv(DirectRLEnv):
    cfg: InHandOlympicsEnvCfg

    @property
    def POSE_VIEWER(self):  # noqa: N802 -- train.py's hook for an env-specific wandb pose viewer
        from .pose_viewer import OlympicsPoseViewer
        return OlympicsPoseViewer

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
        self.pen_rest_height = self.cfg.table.top_height + self.cfg.pen.diameter / 2   # centre, lying on the table
        self._init_hand_geometry(spec)

        # Tasks and goals. Goal poses are in the env frame.
        self.tasks = make_tasks(self)
        self.task_id = torch.zeros(n, dtype=torch.long, device=dev)          # index into self.tasks
        self.task_onehot_index = torch.tensor([TASK_NAMES.index(t.name) for t in self.tasks], device=dev)
        self.task_success_steps = torch.tensor([getattr(t, "success_steps", 1) for t in self.tasks], device=dev)
        self.task_perturbed = torch.tensor([bool(getattr(t, "perturbed", False)) for t in self.tasks], device=dev)
        self.near_goal_steps = torch.zeros(n, dtype=torch.long, device=dev)
        # Perturbations (cfg.perturb): the current push and twist, and each env's chance per step of a new one.
        self.pen_forces = torch.zeros(n, 1, 3, device=dev)
        self.pen_torques = torch.zeros(n, 1, 3, device=dev)
        self.force_prob = torch.zeros(n, device=dev)
        self.torque_prob = torch.zeros(n, device=dev)
        self.goal_pos = torch.zeros(n, 3, device=dev)
        self.goal_quat = torch.zeros(n, 4, device=dev)
        self.goal_quat[:, 0] = 1.0
        self.successes = torch.zeros(n, device=dev)
        # Best-so-far distances for the progress rewards; NaN = not measured yet (taken from the next step).
        self.closest_goal_dist = torch.full((n,), math.nan, device=dev)
        self.closest_tip_dist = torch.full((n, len(self.tip_bodies)), math.nan, device=dev)
        self.goal_dist = torch.zeros(n, device=dev)
        self.is_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.near_goal = torch.zeros(n, dtype=torch.bool, device=dev)
        self.termination_reasons: dict[str, torch.Tensor] = {}
        self.reward_terms: dict[str, torch.Tensor] = {}
        self.reward_buf = torch.zeros(n, device=dev)       # DirectRLEnv sets it in step(); the obs reads it at reset

    def _setup_scene(self) -> None:
        setup_scene(self)
        spec, o = self.robot_spec, self.cfg.obs
        sizes = field_sizes(self.cfg.action_space, spec.num_hand_joints, spec.num_fingertips, NUM_TASK_SLOTS)
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
        onehot = torch.zeros(self.num_envs, NUM_TASK_SLOTS, device=self.device)
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

    def _perturb(self) -> None:
        """Pushes and twists on the pen (cfg.perturb), in envs whose task asks for them, once the episode's first
        goal is hit and while the pen is lifted; in-hand's wrench DR (pose_reaching_6d/obs_utils/actions.apply_wrench_dr)."""
        p, dt, n = self.cfg.perturb, self.step_dt, self.num_envs
        self.pen_forces *= p.force_decay ** (dt / p.decay_interval) if p.force_decay > 0 else 0.0
        self.pen_torques *= p.torque_decay ** (dt / p.decay_interval) if p.torque_decay > 0 else 0.0
        mass = self.pen.data.default_mass.to(self.device).view(n, 1, 1)
        f_new = torch.randn(n, 1, 3, device=self.device) * mass * p.force_scale
        t_new = torch.randn(n, 1, 3, device=self.device) * mass * p.torque_scale
        fire_f = (torch.rand(n, device=self.device) < self.force_prob).view(n, 1, 1)
        fire_t = (torch.rand(n, device=self.device) < self.torque_prob).view(n, 1, 1)
        self.pen_forces = torch.where(fire_f, f_new, self.pen_forces)
        self.pen_torques = torch.where(fire_t, t_new, self.pen_torques)
        pos, _ = self.pen_pose()
        # Only once the goal has been hit this episode (the pen picked up and held), and while the pen is up.
        active = (self.task_perturbed[self.task_id] & (self.successes > 0)
                  & (pos[:, 2] > self.pen_rest_height + p.lifted_height))
        a = active.float().view(n, 1, 1)
        self.pen_forces *= a
        self.pen_torques *= a
        self.pen.set_external_force_and_torque(forces=self.pen_forces, torques=self.pen_torques, is_global=True)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if bool(self.task_perturbed.any()):
            self._perturb()
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
        near = self.goal_dist < self.cfg.goal.success_tolerance
        self.near_goal = near
        self.near_goal_steps = torch.where(near, self.near_goal_steps + 1, torch.zeros_like(self.near_goal_steps))
        self.is_success = self.near_goal_steps >= self.task_success_steps[self.task_id]
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
            # Spread over the steps a goal needs: goal_bonus / success_steps on every step within tolerance, so a
            # full hold earns goal_bonus and a hold cut short keeps what it earned (a one-step goal: unchanged).
            "goal_bonus": r.goal_bonus * self.near_goal.float() / self.task_success_steps[self.task_id],
            "approach": r.approach_rew_scale * approach,
            "lost_pen": -r.lost_pen_penalty * (self.termination_reasons["away"]
                                               | self.termination_reasons["fell"]).float(),
            "wrist_penalty": -r.wrist_action_penalty * vel[:, self.wrist_ids].abs().sum(-1),
            "hand_penalty": -r.hand_action_penalty * vel[:, self.hand_ids].abs().sum(-1),
        }
        reward = sum(self.reward_terms.values())

        hit = self.is_success.nonzero(as_tuple=False).squeeze(-1)
        if len(hit):
            self.successes[hit] += 1
            self.near_goal_steps[hit] = 0
            self._next_goal(hit)
            self.episode_length_buf[hit] = 0
            self.closest_goal_dist[hit] = math.nan
        self.extras["log"] = {f"reward/{k}": v.mean() for k, v in self.reward_terms.items()}
        self.extras["log"]["successes"] = self.successes.mean()
        # Per env; the training observer averages them over the envs whose episode ends this step (as in-hand's
        # episode_final/*).
        final = {"successes": self.successes.clone(),
                 **{f"done_{k}": v.float() for k, v in self.termination_reasons.items()}}
        # The same per task, as "task_<task>/<metric>" (NaN outside the task's envs): one wandb section per task
        # (coevolution/utils/rlgames_utils.py PER_TASK_PREFIX).
        nan = torch.full_like(self.successes, math.nan)
        overall = list(final.items())
        for k, task in enumerate(self.tasks):
            mine = self.task_id == k
            for name, value in overall:
                final[f"task_{task.name}/{name}"] = torch.where(mine, value, nan)
        self.extras["episode_final"] = final
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
        # Fingers: towards a uniform draw over each joint's range (cfg.reset); the wrist at its default pose.
        q = self.robot.data.default_joint_pos[ids].clone()
        qd = torch.zeros_like(q)
        h, r = self.hand_ids, self.cfg.reset
        lo, hi = self.joint_lower[h], self.joint_upper[h]
        draw = lo + (hi - lo) * torch.rand(len(ids), len(h), device=self.device)
        q[:, h] = torch.lerp(q[:, h], draw, r.finger_pos_fraction).clamp(lo, hi)
        qd[:, h] = torch.empty(len(ids), len(h), device=self.device).uniform_(-r.finger_vel, r.finger_vel)
        self.robot.write_joint_state_to_sim(q, qd, env_ids=ids)
        self.targets[ids] = q
        self.wrist_command[ids] = q[:, self.wrist_ids]
        self.prev_joint_pos_canon[ids] = q[:, self.perm_lab_to_canon]
        self.prev_joint_vel_canon[ids] = qd[:, self.perm_lab_to_canon]

        self.task_id[ids] = torch.randint(0, len(self.tasks), (len(ids),), device=self.device)
        for k, task in enumerate(self.tasks):
            sel = ids[self.task_id[ids] == k]
            if len(sel):
                task.reset(sel)
        self.successes[ids] = 0.0
        self.near_goal_steps[ids] = 0
        lo, hi = (math.log(v) for v in self.cfg.perturb.prob_range)
        for prob in (self.force_prob, self.torque_prob):
            prob[ids] = torch.exp(torch.empty(len(ids), device=self.device).uniform_(lo, hi))
        self.pen_forces[ids] = 0.0
        self.pen_torques[ids] = 0.0
        self.closest_goal_dist[ids] = math.nan
        self.closest_tip_dist[ids] = math.nan
