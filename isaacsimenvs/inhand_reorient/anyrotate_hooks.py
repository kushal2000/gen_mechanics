"""The ``anyrotate`` profile's env hooks (Kit side).

Glue between ``InHandReorientEnv`` and ``anyrotate_profile.py`` (the
arithmetic, CPU-tested). Per control step (20 Hz):

- ``pre_physics_step``: relative joint targets q-bar_t = q-bar_{t-1} +
  eta a_t + (1 - eta) a_{t-1} (Sec. 3.1), held over the 3 physics substeps.
- ``_get_dones``: palm-frame geometry, keypoint distance and rotation
  distance to the goal, rotation about the commanded axis this step; drop
  (kp_dist > 0.1) or off-axis (> 45 degrees) terminates, 600 steps time out.
- ``_get_rewards``: simulated touch (App. F), Eq. 2's reward, goal update
  (a reached goal is replaced by the current orientation rotated 30 degrees
  about k), the reward curriculum lambda_rew and the optional gravity
  curriculum, both from finished episodes.
- ``_reset_idx``: canonical hand pose plus noise, the object at the hand's
  palm-up spawn point in a random orientation, a new axis k and goal; with
  ``anyrotate.grasp_cache`` set, a cached stable grasp of the env's design
  instead (``grasp_cache.py``; HORA's reset), when the design has one.

The legacy buffers (``_successes``, ``_rot_error``, ``_termination_reasons``,
``_reward_terms``, ...) are filled with this profile's values, so
``nan_guard``, ``design_scoring`` and ``reset_utils.log_step_metrics`` work
unchanged; ``design_scoring`` banks this profile's own metrics (rotation
about k and time to terminate) when ``env._anyrotate`` is set.
"""

from __future__ import annotations

import math

import torch

from . import anyrotate_profile as ar
from . import design_scoring
from . import hora_profile as hp
from . import repose_profile as rp
from .obs_utils import _fingertip_valid_mask, _joint_valid_mask, update_palm_frame_geometry
from .reset_utils import _object_spawn_offset, _population_default_joint_pos

__all__ = [
    "allocate_buffers", "pre_physics_step", "apply_action", "get_dones", "get_rewards", "reset_env_state",
    "build_observations", "randomize_object_physics",
]

GRAVITY = 9.81


def _cfg(env):
    return env.cfg.anyrotate


def _limits(env):
    limits = env.robot.data.soft_joint_pos_limits
    return limits[..., 0], limits[..., 1]


def _palm_quat(env):
    return env.robot.data.body_quat_w[:, env.palm_body_idx]


def allocate_buffers(env) -> None:
    a = _cfg(env)
    n, device = env.num_envs, env.device
    j = env.hand_spec.num_hand_joints
    k = len(env.fingertip_body_idx)
    env._ar_offsets = ar.keypoint_offsets(a.keypoint_distance_m, device=device)
    env._ar_axis = torch.zeros(n, 3, device=device)
    env._ar_axis[:, 2] = 1.0
    env._ar_target = env.robot.data.default_joint_pos.clone()
    env._ar_action = torch.zeros(n, j, device=device)
    env._ar_prev_action = torch.zeros(n, j, device=device)
    env._ar_prev_joint_pos = env.robot.data.default_joint_pos.clone()
    env._ar_goal_pos_palm = torch.zeros(n, 3, device=device)
    env._ar_goal_age = torch.zeros(n, dtype=torch.long, device=device)
    env._ar_timer_goals = torch.zeros(n, device=device)
    env._ar_axis_obj = env._ar_axis.clone()
    env._ar_axis_tilt = torch.zeros(n, device=device)
    env._ar_prev_obj_quat_palm = torch.zeros(n, 4, device=device)
    env._ar_prev_obj_quat_palm[:, 0] = 1.0
    env._ar_rot_step = torch.zeros(n, device=device)
    env._ar_rotation_rad = torch.zeros(n, device=device)
    env._ar_kp_dist = torch.zeros(n, device=device)
    env._ar_rot_dist = torch.zeros(n, device=device)
    env._ar_force_smooth = torch.zeros(n, k, 3, device=device)
    env._ar_contact = torch.zeros(n, k, dtype=torch.bool, device=device)
    env._ar_contact_pose = torch.zeros(n, k, 2, device=device)
    env._ar_contact_force = torch.zeros(n, k, device=device)
    env._ar_nontip = torch.zeros(n, device=device)
    env._fingertip_quat_palm = torch.zeros(n, 4 * k, device=device)
    env._ar_g_eval = 0.0
    env._ar_rot_eval = 0.0
    env._ar_lambda = 0.0 if a.reward_curriculum else 1.0
    env._ar_axis_stage = 0 if a.axis_curriculum_z_first else 1
    env._ar_difficulty = torch.zeros(n, device=device)
    env._ar_gravity = torch.tensor([0.0, 0.0, -GRAVITY], device=device)
    env._current_success_tolerance = float(a.d_tol)
    env._hora = hp.is_hora(env.cfg)
    if env._hora:
        env._hora_hist = torch.zeros(n, hp.HISTORY_LEN, 2 * j, device=device)
        env._hora_q_init = env.robot.data.default_joint_pos.clone()
        env._hora_z0 = torch.zeros(n, device=device)
        env._hora_obj_pos_prev = torch.zeros(n, 3, device=device)
        env._hora_rotate = torch.zeros(n, device=device)
    _setup_contact_indices(env)
    randomize_object_physics(env)
    _apply_population_actuator(env)  # before the grasp search: grasps under the training actuator
    _setup_grasp_cache(env)  # needs full gravity: before the curriculum zeroes it
    if a.gravity_curriculum:
        _set_gravity(env, 0.0)


def _setup_contact_indices(env) -> None:
    """The per-body contact sensors are created in fingertip order
    (``scene_utils._add_anyrotate_contact_sensor``). Population: finger f's
    tip contact comes from the sensor on its last real link
    (``anyrotate_profile.tip_sources``; the f*_link5 markers have no
    collider), and those links leave the non-tip count."""
    if len(env.ar_tip_sensors) != len(env.fingertip_body_idx):
        raise RuntimeError("one fingertip contact sensor per fingertip body expected")
    env._ar_tip_src = None
    env._ar_nontip_mask = None
    tables = getattr(env, "hand_tables", None)
    if tables is None:
        return
    valid = torch.as_tensor(tables.joint_valid, dtype=torch.bool, device=env.device)
    distal = ar.distal_slots(valid)[env.scene_record["design_idx"]]
    env._ar_tip_src, env._ar_nontip_mask = ar.tip_sources(distal, env.ar_tip_names, env.ar_nontip_names)
    points = [True] * len(env.ar_tip_names) + list(env.ar_nontip_points)
    used = sorted(set(env._ar_tip_src.flatten().tolist()))
    if not all(points[i] for i in used):
        raise RuntimeError("a population fingertip source sensor does not track contact points")
    print(f"[anyrotate] population fingertip contact from each finger's last real link "
          f"({len(used)} distinct sensor bodies)", flush=True)


def contact_forces(env):
    """``(tip_force_w (n, k, 3), tip_contact_pos_w (n, k, 3), nontip_force
    (n, m))``: object contact force on each fingertip, its contact point
    (NaN where none) and the force magnitude on every body Eq. 7 counts."""
    tip_f = torch.stack([torch.nan_to_num(s.data.force_matrix_w[:, 0, 0, :]) for s in env.ar_tip_sensors], dim=1)
    tip_p = torch.stack([s.data.contact_pos_w[:, 0, 0, :] for s in env.ar_tip_sensors], dim=1)
    if env.ar_nontip_sensors:
        non_f = torch.stack([torch.nan_to_num(s.data.force_matrix_w[:, 0, 0, :])
                             for s in env.ar_nontip_sensors], dim=1)
    else:
        non_f = torch.zeros(env.num_envs, 0, 3, device=env.device)
    if env._ar_tip_src is None:
        return tip_f, tip_p, non_f.norm(dim=-1)
    nan = torch.full_like(tip_p[:, :1], float("nan"))
    non_p = torch.stack([s.data.contact_pos_w[:, 0, 0, :] if pts else nan[:, 0]
                         for s, pts in zip(env.ar_nontip_sensors, env.ar_nontip_points)], dim=1)
    all_f = torch.cat([tip_f, non_f], dim=1)
    all_p = torch.cat([tip_p, non_p], dim=1)
    idx = env._ar_tip_src.unsqueeze(-1).expand(-1, -1, 3)
    return (torch.gather(all_f, 1, idx), torch.gather(all_p, 1, idx),
            all_f.norm(dim=-1) * env._ar_nontip_mask)


def randomize_object_physics(env) -> None:
    """Table 4: object mass U(mass_range) and centre of mass U(-com, com)^3,
    once per env (the inertia is scaled with the mass)."""
    a = _cfg(env)
    view = env.object.root_physx_view
    idx = torch.arange(env.num_envs, dtype=torch.int32)
    masses = view.get_masses().clone()
    lo, hi = a.mass_range
    new_mass = torch.rand(env.num_envs) * (hi - lo) + lo
    ratio = new_mass / masses.reshape(env.num_envs).clamp(min=1e-9)
    inertias = view.get_inertias().clone()
    inertias = inertias * ratio.reshape(-1, *([1] * (inertias.dim() - 1)))
    view.set_masses(new_mass.reshape(masses.shape), idx)
    view.set_inertias(inertias, idx)
    coms = view.get_coms().clone()
    offset = (torch.rand(env.num_envs, 3) * 2.0 - 1.0) * a.com_range
    if coms.dim() == 3:
        coms[:, 0, 0:3] = coms[:, 0, 0:3] + offset
    else:
        coms[:, 0:3] = coms[:, 0:3] + offset
    view.set_coms(coms, idx)
    env._ar_object_mass = new_mass.to(env.device)
    env._ar_object_com = offset.to(env.device)
    dims = (a.capsule_radius, a.capsule_width) if a.object_shape == "capsule" else (a.box_size, a.box_size)
    env._ar_object_dims = torch.tensor(dims, device=env.device).expand(env.num_envs, 2).clone()


def _apply_population_actuator(env) -> None:
    """``population_hand_actuator``: every real joint (``_joint_valid_mask``)
    of every design gets hand_stiffness / hand_damping / hand_effort_limit,
    in the simulation and in Isaac Lab's actuator model (which computes the
    applied torque the penalties read). Ghost joints keep theirs."""
    a = _cfg(env)
    if not a.population_hand_actuator or getattr(env, "hand_tables", None) is None:
        return
    mask = _joint_valid_mask(env)
    values = {"stiffness": float(a.hand_stiffness), "damping": float(a.hand_damping),
              "effort": float(a.hand_effort_limit)}
    data = env.robot.data
    k = torch.where(mask, torch.full_like(data.joint_stiffness, values["stiffness"]), data.joint_stiffness)
    d = torch.where(mask, torch.full_like(data.joint_damping, values["damping"]), data.joint_damping)
    e = torch.where(mask, torch.full_like(data.joint_effort_limits, values["effort"]), data.joint_effort_limits)
    env.robot.write_joint_stiffness_to_sim(k)
    env.robot.write_joint_damping_to_sim(d)
    env.robot.write_joint_effort_limit_to_sim(e)
    for actuator in env.robot.actuators.values():
        ids = actuator.joint_indices
        cols = list(range(mask.shape[1]))[ids] if isinstance(ids, slice) else list(ids)
        m = mask[:, cols]
        for attr, v in (("stiffness", values["stiffness"]), ("damping", values["damping"]),
                        ("effort_limit", values["effort"]), ("effort_limit_sim", values["effort"])):
            t = getattr(actuator, attr, None)
            if isinstance(t, torch.Tensor) and t.shape == m.shape:
                t[:] = torch.where(m, torch.full_like(t, v), t)
    print(f"[anyrotate] population actuator: {int(mask.sum())} real joint(s) set to stiffness "
          f"{values['stiffness']}, damping {values['damping']}, effort {values['effort']} N m", flush=True)


def _setup_grasp_cache(env) -> None:
    """``anyrotate.grasp_cache`` set: load (and, with
    ``grasp_cache_generate``, complete) the stable-grasp cache. Per env:
    ``_ar_grace`` is the settle phase (``grasp_settle_steps`` for an env
    whose design has a cached grasp, ``axis_check_grace_steps``
    otherwise) and ``_ar_q0`` the pose penalty's reference (HORA: the
    episode's initial grasp pose). Without a cache every env keeps
    ``axis_check_grace_steps`` and the canonical q0, as before."""
    a = _cfg(env)
    n = env.num_envs
    env._ar_grasps = None
    env._ar_grasp_has = None
    env._ar_grasp_counts = None
    env._ar_q0 = None
    env._ar_grace = torch.full((n,), int(a.axis_check_grace_steps), dtype=torch.long, device=env.device)
    if not a.grasp_cache:
        return
    from .grasp_cache_gen import ensure_grasp_table

    table, design_idx, _sources, report = ensure_grasp_table(env)
    env._ar_grasps = table
    env._ar_grasp_design_idx = design_idx
    env._ar_grasp_has = table.has_grasp
    env._ar_grasp_counts = table.counts
    env._ar_grasp_report = report
    env._ar_grasp_env_ok = table.has_grasp[design_idx]
    env._ar_grace = torch.where(env._ar_grasp_env_ok, torch.full_like(env._ar_grace, int(a.grasp_settle_steps)),
                                env._ar_grace)
    ids = torch.arange(n, device=env.device)
    env._ar_q0 = _population_default_joint_pos(env, ids).clone()


def _set_gravity(env, frac: float) -> None:
    """Scene-wide gravity frac x (0, 0, -9.81) (Dexsuite's variable_gravity
    event: PhysX gravity is global, not per env)."""
    import carb
    import isaaclab.sim as sim_utils

    g = (0.0, 0.0, -GRAVITY * float(frac))
    sim_utils.SimulationContext.instance().physics_sim_view.set_gravity(carb.Float3(*g))
    env._ar_gravity = torch.tensor(g, device=env.device)


# --------------------------------------------------------------------------
# Control
# --------------------------------------------------------------------------


def pre_physics_step(env, actions: torch.Tensor) -> None:
    a = _cfg(env)
    act = actions.clone().clamp(-1.0, 1.0)
    joint_mask = _joint_valid_mask(env)
    if joint_mask is not None:
        act = torch.where(joint_mask, act, torch.zeros_like(act))
    env._ar_prev_joint_pos = env.robot.data.joint_pos.clone()
    lower, upper = _limits(env)
    if env._hora:
        # HORA: targets = prev_targets + 1/24 * actions, clamped; no smoothing.
        env._ar_target = torch.max(torch.min(env._ar_target + act * a.action_scale, upper), lower)
        env._hora_obj_pos_prev = env.object.data.root_pos_w.clone()
    else:
        env._ar_target, _ = ar.relative_joint_targets(
            env._ar_target, act, env._ar_prev_action, a.action_eta, a.action_scale, lower, upper)
    env._prev_actions_this_step = env._prev_actions
    env._prev_actions = act
    env._ar_prev_action = act


def apply_action(env) -> None:
    env.robot.set_joint_position_target(env._ar_target)


# --------------------------------------------------------------------------
# Geometry, dones
# --------------------------------------------------------------------------


def refresh_geometry(env) -> None:
    update_palm_frame_geometry(env)
    palm_q = _palm_quat(env)
    env._goal_quat_palm = rp.quat_mul(rp.quat_conjugate(palm_q), env._goal_quat_w)
    env._ar_kp_dist = ar.keypoint_distance(
        env._obj_pos_palm, env._obj_quat_palm, env._ar_goal_pos_palm, env._goal_quat_palm, env._ar_offsets)
    env._ar_rot_dist = rp.rotation_distance(env._obj_quat_palm, env._goal_quat_palm)
    env._rot_error = env._ar_rot_dist
    data = env.robot.data
    tip_idx = env.fingertip_body_idx
    pos, quat, _vel = rp.palm_frame_fingertips(
        data.body_pos_w[:, env.palm_body_idx], palm_q, data.body_pos_w[:, tip_idx], data.body_quat_w[:, tip_idx],
        data.body_vel_w[:, tip_idx])
    env._fingertip_pos_palm = pos
    env._fingertip_quat_palm = quat


def get_dones(env, nan_guard) -> tuple[torch.Tensor, torch.Tensor]:
    a = _cfg(env)
    env._frame_counter += 1
    env._prev_rot_error = env._rot_error
    refresh_geometry(env)
    nan_guard.guard_step_state(env)
    design_scoring.step_scoring_state(env)
    env._ar_rot_step = ar.rotation_about_axis(env._ar_prev_obj_quat_palm, env._obj_quat_palm, env._ar_axis)
    env._ar_rot_step = torch.where(env._nonfinite_mask, torch.zeros_like(env._ar_rot_step), env._ar_rot_step)
    env._ar_prev_obj_quat_palm = env._obj_quat_palm.clone()
    if env._hora:
        return _hora_dones(env, nan_guard)
    # The episode's rotation about k (AnyRotate's Rot) counts from the end of
    # the settle phase: a dropped object tumbles while it lands or falls off.
    manipulating = env.episode_length_buf > env._ar_grace
    env._ar_rotation_rad = env._ar_rotation_rad + torch.where(
        manipulating, env._ar_rot_step, torch.zeros_like(env._ar_rot_step))
    settled = (env.episode_length_buf == env._ar_grace).nonzero(as_tuple=False).squeeze(-1)
    if settled.numel() > 0:
        # End of the settle phase: goal and axis reference from the settled object.
        _new_goal(env, settled, env._obj_quat_palm[settled], env._obj_pos_palm[settled])
        env._ar_kp_dist = ar.keypoint_distance(
            env._obj_pos_palm, env._obj_quat_palm, env._ar_goal_pos_palm, env._goal_quat_palm, env._ar_offsets)
        env._ar_rot_dist = rp.rotation_distance(env._obj_quat_palm, env._goal_quat_palm)
        env._rot_error = env._ar_rot_dist
    tilt = ar.axis_tilt(env._obj_quat_palm, env._ar_axis_obj, env._ar_axis)
    env._ar_axis_tilt = tilt
    valid = env.episode_length_buf > env._ar_grace
    dropped, off_axis = ar.terminations(env._ar_kp_dist, a.d_max, tilt, valid, math.radians(a.axis_dev_max_deg))
    time_out = env.episode_length_buf >= env.max_episode_length
    env._termination_reasons = {"drop": dropped, "off_axis": off_axis & ~dropped, "timeout": time_out}
    terminated = nan_guard.add_nonfinite_termination(env, dropped | off_axis)
    return terminated, time_out


def _hora_dones(env, nan_guard):
    """HORA's check_termination: the object below its start height, or the
    episode length. Rotation about k counts from the first step."""
    h = env.cfg.hora
    env._ar_rotation_rad = env._ar_rotation_rad + env._ar_rot_step
    env._hora_rotate = (env._ar_rot_step / float(env.step_dt)).clamp(h.angvel_clip_min, h.angvel_clip_max)
    dropped = hp.dropped(env.object.data.root_pos_w[:, 2], env._hora_z0, h.drop_dz)
    time_out = env.episode_length_buf >= env.max_episode_length
    env._ar_axis_tilt = ar.axis_tilt(env._obj_quat_palm, env._ar_axis_obj, env._ar_axis)
    env._termination_reasons = {"drop": dropped, "off_axis": torch.zeros_like(dropped), "timeout": time_out}
    terminated = nan_guard.add_nonfinite_termination(env, dropped)
    return terminated, time_out


def _hora_rewards(env) -> torch.Tensor:
    """HORA's compute_reward (``hora_profile``)."""
    h = env.cfg.hora
    mask = _joint_valid_mask(env)
    tau = env.robot.data.applied_torque
    terms = {
        "rotate": env._hora_rotate,
        "linvel": hp.linvel_penalty(env._hora_obj_pos_prev, env.object.data.root_pos_w, float(env.step_dt)),
        "pose": hp.pose_diff_penalty(env.robot.data.joint_pos, env._hora_q_init, mask),
        "torque": hp.torque_penalty(tau, mask),
        "work": hp.work_penalty(tau, env.robot.data.joint_vel, mask),
    }
    scales = dict(rotate=h.rotate_reward_scale, linvel=h.obj_linvel_penalty_scale, pose=h.pose_diff_penalty_scale,
                  torque=h.torque_penalty_scale, work=h.work_penalty_scale)
    terms = {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in terms.items()}
    reward = hp.combine_reward(terms, scales)
    env._reward_terms = {
        "rotation_rew": scales["rotate"] * terms["rotate"], "linvel_penalty": scales["linvel"] * terms["linvel"],
        "pose_penalty": scales["pose"] * terms["pose"], "torque_penalty": scales["torque"] * terms["torque"],
        "work_penalty": scales["work"] * terms["work"], "total_reward": reward,
    }
    env._is_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    _update_curricula(env)
    return reward


# --------------------------------------------------------------------------
# Touch, reward, goals, curricula
# --------------------------------------------------------------------------


def _update_touch(env) -> None:
    a = _cfg(env)
    n = env.num_envs
    tip_force_w, cpos_w, nontip_force = contact_forces(env)  # (n, k, 3), (n, k, 3), (n, m)
    k = tip_force_w.shape[1]
    tip_q = env.robot.data.body_quat_w[:, env.fingertip_body_idx].reshape(n * k, 4)
    tip_p = env.robot.data.body_pos_w[:, env.fingertip_body_idx].reshape(n * k, 3)
    f_tip = rp.quat_apply_inverse(tip_q, tip_force_w.reshape(n * k, 3)).reshape(n, k, 3)
    cpos = cpos_w.reshape(n * k, 3)
    has = torch.isfinite(cpos).all(dim=-1, keepdim=True)
    d = rp.quat_apply_inverse(tip_q, torch.nan_to_num(cpos) - tip_p)
    d = torch.where(has, torch.nn.functional.normalize(d, dim=-1), torch.zeros_like(d)).reshape(n, k, 3)
    c, pose, force, fs = ar.simulated_tactile(
        f_tip, d, env._ar_force_smooth, alpha=a.force_alpha, threshold=a.contact_threshold,
        beta_f=a.force_beta, f_max=a.force_max, beta_p=a.pose_beta, p_max=a.pose_max)
    tip_mask = _fingertip_valid_mask(env)
    if tip_mask is not None:
        c = c & tip_mask
        pose = pose * tip_mask.unsqueeze(-1)
        force = force * tip_mask
    env._ar_force_smooth = fs
    env._ar_contact, env._ar_contact_pose, env._ar_contact_force = c, pose, force
    env._ar_nontip = (nontip_force > a.contact_threshold).float().sum(dim=-1)


def _weights(a) -> dict:
    return dict(kp=a.w_kp, rot=a.w_rot, goal=a.w_goal, gc=a.w_gc, bc=a.w_bc, omega=a.w_omega, pose=a.w_pose,
                work=a.w_work, torque=a.w_torque, penalty=a.w_penalty)


def get_rewards(env) -> torch.Tensor:
    a = _cfg(env)
    _update_touch(env)
    if env._hora:
        return _hora_rewards(env)
    joint_mask = _joint_valid_mask(env)
    q = env.robot.data.joint_pos
    q0 = (env.scene_record.get("default_joint_pos") if getattr(env, "hand_tables", None) is not None
          else None)
    q0 = env.robot.data.default_joint_pos if q0 is None else q0
    if env._ar_q0 is not None:
        q0 = env._ar_q0  # grasp cache: the episode's initial pose (HORA)
    tau = env.robot.data.applied_torque
    nonfinite = env._nonfinite_mask
    goal_hit = ar.goal_reached(env._ar_kp_dist, env._ar_rot_dist, a.d_tol, a.goal_tol_metric) & ~nonfinite
    r_gc, r_bc = ar.contact_rewards(env._ar_contact, env._ar_nontip)
    reasons = env._termination_reasons
    terms = {
        "kp": ar.keypoint_reward(env._ar_kp_dist, a.kp_a, a.kp_b, a.kp_scale),
        "rot": ar.rotation_reward(env._ar_rot_step, a.rot_clip),
        "goal": goal_hit.float(),
        "gc": r_gc,
        "bc": r_bc,
        "omega": ar.angular_velocity_penalty(env._obj_ang_vel_palm, a.omega_max),
        "pose": ar.pose_penalty(q, q0, joint_mask),
        "work": ar.work_penalty(tau, q - env._ar_prev_joint_pos, joint_mask),
        "torque": ar.torque_penalty(tau, joint_mask),
        "penalty": (reasons["drop"] | reasons["off_axis"]).float(),
    }
    w = _weights(a)
    reward = ar.combine_rewards(terms, w, env._ar_lambda)
    lam = env._ar_lambda
    env._reward_terms = {
        "keypoint_rew": w["kp"] * terms["kp"], "rotation_rew": w["rot"] * terms["rot"],
        "goal_bonus_rew": w["goal"] * terms["goal"], "good_contact_rew": lam * w["gc"] * terms["gc"],
        "bad_contact_penalty": -lam * w["bc"] * terms["bc"],
        "ang_vel_penalty": lam * w["omega"] * terms["omega"], "pose_penalty": lam * w["pose"] * terms["pose"],
        "work_penalty": lam * w["work"] * terms["work"], "torque_penalty": lam * w["torque"] * terms["torque"],
        "termination_penalty": -w["penalty"] * terms["penalty"], "total_reward": reward,
    }
    env._is_success = goal_hit
    env._successes = env._successes + goal_hit.long()
    within = env._ar_rot_dist <= a.d_tol
    env._consec_success_steps = torch.where(
        within, env._consec_success_steps + 1, torch.zeros_like(env._consec_success_steps))
    ids = goal_hit.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() > 0:
        _new_goal(env, ids, env._obj_quat_palm[ids], env._obj_pos_palm[ids])
    env._ar_goal_age += 1
    if a.goal_advance == "timer":
        due = ar.goal_timer_due(env._ar_goal_age, a.goal_timer_s, float(env.step_dt)) & ~goal_hit
        due = due & (env.episode_length_buf > env._ar_grace)
        tids = due.nonzero(as_tuple=False).squeeze(-1)
        if tids.numel() > 0:
            _advance_goal(env, tids)
    elif a.goal_advance != "reach":
        raise ValueError(f"anyrotate.goal_advance={a.goal_advance!r}; expected 'reach' or 'timer'")
    _update_curricula(env)
    return reward


def _new_goal(env, env_ids, obj_quat_palm, obj_pos_palm) -> None:
    """Sec. 3.1: the current object orientation rotated by the goal increment
    about k (palm frame); the goal position is the object's position now."""
    a = _cfg(env)
    goal_palm = ar.next_goal(obj_quat_palm, env._ar_axis[env_ids], math.radians(a.goal_increment_deg))
    env._goal_quat_w[env_ids] = rp.quat_mul(_palm_quat(env)[env_ids], goal_palm)
    env._ar_goal_pos_palm[env_ids] = obj_pos_palm
    env._goal_quat_palm = rp.quat_mul(rp.quat_conjugate(_palm_quat(env)), env._goal_quat_w)
    env._ar_goal_age[env_ids] = 0
    # Re-anchor the axis-tilt reference: the object's body axis along k now.
    env._ar_axis_obj[env_ids] = ar.axis_in_object_frame(obj_quat_palm, env._ar_axis[env_ids])


def _advance_goal(env, env_ids) -> None:
    """``goal_advance=timer``: the pending goal moves on by the increment
    about k from where it was (not from the object), so a stalled object
    falls behind. The axis-tilt reference is left alone."""
    a = _cfg(env)
    palm_q = _palm_quat(env)[env_ids]
    goal_palm = rp.quat_mul(rp.quat_conjugate(palm_q), env._goal_quat_w[env_ids])
    goal_palm = ar.next_goal(goal_palm, env._ar_axis[env_ids], math.radians(a.goal_increment_deg))
    env._goal_quat_w[env_ids] = rp.quat_mul(palm_q, goal_palm)
    env._goal_quat_palm = rp.quat_mul(rp.quat_conjugate(_palm_quat(env)), env._goal_quat_w)
    env._ar_goal_age[env_ids] = 0
    env._ar_timer_goals[env_ids] += 1.0


def _update_curricula(env) -> None:
    """Episode-end bookkeeping, before ``_reset_idx`` clears the counters:
    g_eval (running mean of goals per episode) for lambda_rew (Eq. 15), the
    z-axis-first axis stage, and the Dexsuite gravity difficulty."""
    a = _cfg(env)
    done = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if done.numel() == 0:
        return
    goals = env._successes[done].float()
    rotations = env._ar_rotation_rad[done] / (2.0 * math.pi)
    beta = float(a.curriculum_ema)
    m = min(1.0, beta * done.numel())
    env._ar_g_eval = (1.0 - m) * env._ar_g_eval + m * float(goals.mean())
    env._ar_rot_eval = (1.0 - m) * env._ar_rot_eval + m * float(rotations.mean())
    if a.reward_curriculum:
        env._ar_lambda = ar.reward_curriculum_lambda(env._ar_g_eval, a.curriculum_g_min, a.curriculum_g_max)
    if env._ar_axis_stage == 0 and env._ar_rot_eval >= a.axis_curriculum_rotations:
        env._ar_axis_stage = 1
        print(f"[anyrotate] axis curriculum: z only -> {a.axis_sampling} at frame {env._frame_counter} "
              f"(rotations/episode {env._ar_rot_eval:.2f})", flush=True)
    if a.gravity_curriculum:
        promote = ar.gravity_promoted(goals, rotations, a.gravity_promote_metric, a.gravity_promote_goals,
                                      a.gravity_promote_rotations)
        cur = env._ar_difficulty[done]
        down = cur if a.gravity_promotion_only else cur - 1
        env._ar_difficulty[done] = torch.where(promote, cur + 1, down).clamp(0, a.gravity_max_difficulty)
        frac = float(env._ar_difficulty.mean()) / max(a.gravity_max_difficulty, 1)
        target = frac if frac >= 0.1 else 0.0  # Dexsuite: no change below a 0.1 fraction
        if abs(target * -GRAVITY - float(env._ar_gravity[2])) > 0.05:
            _set_gravity(env, target)


# --------------------------------------------------------------------------
# Reset
# --------------------------------------------------------------------------


def reset_env_state(env, env_ids: torch.Tensor) -> None:
    a = _cfg(env)
    n, device = env_ids.numel(), env.device
    default_pos = _population_default_joint_pos(env, env_ids)
    lower, upper = _limits(env)
    lo, hi = lower[env_ids], upper[env_ids]
    noise = (torch.rand_like(default_pos) * 2.0 - 1.0) * a.reset_joint_noise
    q = torch.max(torch.min(default_pos + noise, hi), lo)
    q_target = q

    palm_pos_w = env.robot.data.body_pos_w[env_ids, env.palm_body_idx]
    palm_q = env.robot.data.body_quat_w[env_ids, env.palm_body_idx]
    obj_pos_palm = _object_spawn_offset(env, env_ids)
    obj_pos_w = palm_pos_w + rp.quat_apply(palm_q, obj_pos_palm)
    obj_q_w = torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1)
    obj_q_palm = rp.quat_mul(rp.quat_conjugate(palm_q), obj_q_w)
    if env._ar_grasps is not None:
        q, q_target, obj_pos_palm, obj_q_palm, obj_pos_w, obj_q_w = _grasp_reset(
            env, env_ids, q, palm_pos_w, palm_q, obj_pos_palm, obj_q_palm, obj_pos_w, obj_q_w, lo, hi)
        env._ar_q0[env_ids] = torch.where(env._ar_grasp_env_ok[env_ids].unsqueeze(-1), q, default_pos)

    env.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    env.robot.set_joint_position_target(q_target, env_ids=env_ids)
    env._ar_target[env_ids] = q_target
    env._ar_prev_joint_pos[env_ids] = q
    env._ar_prev_action[env_ids] = 0.0
    env._prev_actions[env_ids] = 0.0

    env.object.write_root_state_to_sim(
        torch.cat([obj_pos_w, obj_q_w, torch.zeros(n, 6, device=device)], dim=-1), env_ids=env_ids)
    env._spawn_obj_pos_palm[env_ids] = obj_pos_palm

    mode = a.axis_sampling if env._ar_axis_stage >= 1 else "z"
    if mode == "z" and a.z_axis_frame == "world_up":
        env._ar_axis[env_ids] = ar.world_up_in_palm(palm_q)
    elif mode == "z" and a.z_axis_frame != "palm":
        raise ValueError(f"anyrotate.z_axis_frame={a.z_axis_frame!r}; expected 'palm' or 'world_up'")
    else:
        env._ar_axis[env_ids] = ar.sample_axes(n, mode, device=device)
    _new_goal(env, env_ids, obj_q_palm, obj_pos_palm)
    env.goal_viz.write_root_state_to_sim(
        torch.cat([obj_pos_w, env._goal_quat_w[env_ids], torch.zeros(n, 6, device=device)], dim=-1),
        env_ids=env_ids)

    env._prev_episode_successes[env_ids] = env._successes[env_ids]
    env._successes[env_ids] = 0
    env._consec_success_steps[env_ids] = 0
    env._ar_rotation_rad[env_ids] = 0.0
    env._ar_timer_goals[env_ids] = 0.0
    env._ar_force_smooth[env_ids] = 0.0
    env._ar_prev_obj_quat_palm[env_ids] = obj_q_palm
    env._prev_rot_error[env_ids] = 0.0
    if env._hora:
        env._hora_q_init[env_ids] = q
        env._hora_z0[env_ids] = obj_pos_w[:, 2]
        env._hora_obj_pos_prev[env_ids] = obj_pos_w
        frame = torch.cat([hp.unscale(q, lo, hi), q_target], dim=-1)
        env._hora_hist = hp.fill_history(env._hora_hist, env_ids, frame)
    refresh_geometry(env)
    design_scoring.reset_scoring_state(env, env_ids)


def _grasp_reset(env, env_ids, q, palm_pos_w, palm_q, obj_pos_palm, obj_q_palm, obj_pos_w, obj_q_w, lo, hi):
    """Envs whose design has cached grasps start from one, drawn uniformly
    (HORA): joints at its settled positions, PD targets at the targets that
    held it, the object at rest at its palm-frame pose. Optional U(-noise,
    noise) on joints and object position (0 by default; neither paper adds
    any). Other envs keep the drop reset."""
    a = _cfg(env)
    t = env._ar_grasps
    rows, ok = t.sample(env._ar_grasp_design_idx[env_ids])
    if t.total == 0 or not bool(ok.any()):
        return q, q, obj_pos_palm, obj_q_palm, obj_pos_w, obj_q_w
    okc = ok.unsqueeze(-1)
    gq, gqt = t.q[rows], t.q_target[rows]
    if a.grasp_reset_joint_noise > 0:
        jn = (torch.rand_like(gq) * 2.0 - 1.0) * a.grasp_reset_joint_noise
        mask = _joint_valid_mask(env)
        if mask is not None:
            jn = jn * mask[env_ids]
        gq = torch.max(torch.min(gq + jn, hi), lo)
        gqt = torch.max(torch.min(gqt + jn, hi), lo)
    gp = t.obj_pose[rows, :3]
    if a.grasp_reset_obj_pos_noise > 0:
        gp = gp + (torch.rand_like(gp) * 2.0 - 1.0) * a.grasp_reset_obj_pos_noise
    gr = t.obj_pose[rows, 3:]
    q_new = torch.where(okc, gq, q)
    qt_new = torch.where(okc, gqt, q)
    pos_palm = torch.where(okc, gp, obj_pos_palm)
    quat_palm = torch.where(okc, gr, obj_q_palm)
    pos_w = torch.where(okc, palm_pos_w + rp.quat_apply(palm_q, gp), obj_pos_w)
    quat_w = torch.where(okc, rp.quat_mul(palm_q, gr), obj_q_w)
    return q_new, qt_new, pos_palm, quat_palm, pos_w, quat_w


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------


def _noise(x: torch.Tensor, std: float, on: bool) -> torch.Tensor:
    return x + torch.randn_like(x) * std if on and std > 0 else x


def _hora_observations(env) -> dict[str, torch.Tensor]:
    """HORA's compute_observations: the last 3 frames of [unscaled joint
    positions + U(-1, 1) x jointNoiseScale, joint targets], plus its 9
    privileged values (object position, scale, mass, friction, CoM)."""
    h = env.cfg.hora
    a = _cfg(env)
    lower, upper = _limits(env)
    q = env.robot.data.joint_pos
    noise = (torch.rand_like(q) * 2.0 - 1.0) * h.joint_noise_scale
    frame = torch.cat([hp.unscale(q + noise, lower, upper), env._ar_target], dim=-1)
    mask = _joint_valid_mask(env)
    if mask is not None:
        frame = frame * torch.cat([mask, mask], dim=-1)
    env._hora_hist = hp.push_history(env._hora_hist, frame)
    n = env.num_envs
    priv = torch.cat([
        env._obj_pos_palm, env._ar_object_dims[:, :1], env._ar_object_mass.unsqueeze(-1),
        torch.full((n, 1), float(a.static_friction), device=env.device), env._ar_object_com], dim=-1)
    obs = torch.cat([env._hora_hist.reshape(n, -1), priv], dim=-1)
    return {"policy": obs, "critic": obs}


def build_observations(env) -> dict[str, torch.Tensor]:
    a = _cfg(env)
    if env._hora:
        return _hora_observations(env)
    noisy = bool(a.obs_noise)
    joint_mask = _joint_valid_mask(env)
    tip_mask = _fingertip_valid_mask(env)
    n = env.num_envs
    k = len(env.fingertip_body_idx)

    def jm(x):
        return x if joint_mask is None else torch.where(joint_mask, x, torch.zeros_like(x))

    def tm(x, width):
        if tip_mask is None:
            return x
        return (x.reshape(n, k, width) * tip_mask.unsqueeze(-1)).reshape(n, k * width)

    palm_q = _palm_quat(env)
    c = env._ar_contact.float()
    values = {
        "joint_pos": jm(_noise(env.robot.data.joint_pos, a.joint_noise, noisy)),
        "fingertip_pos_palm": tm(_noise(env._fingertip_pos_palm, a.tip_pos_noise, noisy), 3),
        "fingertip_quat_palm": tm(_noise(env._fingertip_quat_palm, a.tip_quat_noise, noisy), 4),
        "prev_actions": jm(env._ar_prev_action),
        "target_joint_pos": jm(env._ar_target),
        "tip_contact": c,
        "tip_contact_pose": (_noise(env._ar_contact_pose, a.contact_pose_noise, noisy)
                             * c.unsqueeze(-1)).reshape(n, 2 * k),
        "tip_contact_force": _noise(env._ar_contact_force, a.contact_force_noise, noisy) * c,
        "rotation_axis": env._ar_axis,
        "object_pos_palm": env._obj_pos_palm,
        "object_quat_palm": env._obj_quat_palm,
        "object_ang_vel_palm": env._obj_ang_vel_palm,
        "object_dims": env._ar_object_dims,
        "object_com": env._ar_object_com,
        "object_mass": env._ar_object_mass.unsqueeze(-1),
        "gravity_palm": rp.quat_apply_inverse(palm_q, env._ar_gravity.expand(n, 3)) / GRAVITY,
        "goal_pos_palm": env._ar_goal_pos_palm,
        "goal_quat_palm": rp.quat_mul(rp.quat_conjugate(palm_q), env._goal_quat_w),
    }
    obs = torch.cat([values[f] for f in env.cfg.obs.obs_list], dim=-1)
    return {"policy": obs, "critic": obs}


def log_metrics(env) -> None:
    """Scalars and per-episode values next to reset_utils.log_step_metrics'."""
    a = _cfg(env)
    dt = float(env.step_dt)
    ep = env.extras.setdefault("episode_final", {})
    ep["rotation_rad"] = env._ar_rotation_rad.clone()
    ep["rotations"] = env._ar_rotation_rad / (2.0 * math.pi)
    ep["ttt_s"] = env.episode_length_buf.float() * dt
    if a.goal_advance == "timer":
        ep["timer_goal_advances"] = env._ar_timer_goals.clone()
    env.extras["lambda_rew"] = float(env._ar_lambda)
    env.extras["g_eval"] = float(env._ar_g_eval)
    env.extras["rotations_eval"] = float(env._ar_rot_eval)
    env.extras["gravity_z"] = env._ar_gravity[2]
    env.extras["kp_dist_mean"] = env._ar_kp_dist.mean()
    env.extras["tip_contacts_mean"] = env._ar_contact.float().sum(dim=-1).mean()
    env.extras["nontip_contacts_mean"] = env._ar_nontip.mean()
    env.extras["axis_stage"] = float(env._ar_axis_stage)
    env.extras["axis_tilt_mean"] = env._ar_axis_tilt.mean()
    env.extras["rot_rate_rad_s"] = env._ar_rot_step.mean() / dt
    if a.gravity_curriculum:
        env.extras["gravity_difficulty_mean"] = env._ar_difficulty.mean()
