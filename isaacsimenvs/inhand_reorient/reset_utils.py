"""Reset-time state: object spawn, goal sampling, and the buffers the
curriculum and reward need across steps.

Free functions over a duck-typed ``env`` (no Isaac import at module scope
beyond what every caller already has booted), mirroring
``pose_reaching_6d.reset_utils``.
"""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, random_orientation

from isaacsimenvs.pose_reaching_6d.reward_utils.curriculum import initial_success_tolerance

from .goal_curriculum import goal_curriculum_mode as _goal_curriculum_mode
from .goal_curriculum import goal_mode_code as _goal_mode_code

__all__ = [
    "allocate_state_buffers", "reset_env_state", "reset_goal_trackers", "log_step_metrics",
]


def allocate_state_buffers(env) -> None:
    """Run once, after the scene is up (mirrors ``PoseReachEnv.__init__``)."""
    n, device = env.num_envs, env.device
    j = env.hand_spec.num_hand_joints

    env._prev_actions = torch.zeros(n, j, device=device)
    env._goal_quat_w = torch.zeros(n, 4, device=device)
    env._goal_quat_w[:, 0] = 1.0
    env._spawn_obj_pos_palm = torch.zeros(n, 3, device=device)
    env._consec_success_steps = torch.zeros(n, device=device, dtype=torch.long)
    env._successes = torch.zeros(n, device=device, dtype=torch.long)
    env._prev_episode_successes = torch.zeros(n, device=device, dtype=torch.long)
    env._prev_rot_error = torch.zeros(n, device=device)
    env._rot_error = torch.zeros(n, device=device)
    env._is_success = torch.zeros(n, device=device, dtype=torch.bool)
    env._termination_reasons: dict[str, torch.Tensor] = {}

    # Curriculum state rlgames_utils.py ferries through the rl_games
    # checkpoint (get_env_state/set_env_state); names match
    # pose_reaching_6d.reward_utils.curriculum exactly on purpose.
    env._frame_counter = 0
    env._last_curriculum_update = 0
    env._current_success_tolerance = initial_success_tolerance(env)

    # Goal-difficulty curriculum (I24, Phase 1c): which sampling mode is
    # active, and when it last advanced -- ferried through the same
    # checkpoint hooks as the tolerance curriculum (see reward_utils.py).
    env._goal_curriculum_stage = 0
    env._last_goal_curriculum_update = 0


def _sample_goal(env, env_ids: torch.Tensor, ref_quat: torch.Tensor) -> torch.Tensor:
    """A new goal orientation for ``env_ids``, relative to ``ref_quat`` (the
    object's CURRENT world orientation: its just-written spawn pose on a full
    reset, or wherever it was when it hit the previous goal on a mid-episode
    resample -- see the two call sites below).

    Modes (``_goal_curriculum_mode``):
      - "axis": rotate ``ref_quat`` by a random angle about world +z only --
        the calibrated palm-normal direction for every hand in
        ``hand_calibration.json`` (Phase 1b), so this is a 1-DOF spin task
        regardless of ``ref_quat``'s own tilt.
      - "delta": rotate ``ref_quat`` by a random angle up to
        ``delta_rotation_degrees`` about a random axis.
      - "full" / "absolute": ``ref_quat`` is ignored; a fully random goal.
    """
    n = env_ids.numel()
    mode = _goal_curriculum_mode(env)
    if mode in ("full", "absolute"):
        return random_orientation(n, device=env.device)
    if mode == "axis":
        axis = torch.zeros(n, 3, device=env.device)
        axis[:, 2] = 1.0
        angle = torch.rand(n, device=env.device) * 2.0 * math.pi
    elif mode == "delta":
        axis = torch.nn.functional.normalize(torch.randn(n, 3, device=env.device), dim=-1)
        angle = (torch.rand(n, device=env.device) * 2.0 - 1.0) * math.radians(
            env.cfg.reset.delta_rotation_degrees)
    else:
        raise ValueError(f"unknown goal sampling mode {mode!r}")
    dq = quat_from_angle_axis(angle, axis)
    return quat_mul(dq, ref_quat)


def reset_goal_trackers(env, env_ids: torch.Tensor) -> None:
    """A goal was hit mid-episode: resample it, without touching anything
    else (episode_length_buf reset is the caller's job, as in
    ``pose_reaching_6d.reward_utils.termination``)."""
    ref_quat = env.object.data.root_quat_w[env_ids]
    env._goal_quat_w[env_ids] = _sample_goal(env, env_ids, ref_quat)
    env._consec_success_steps[env_ids] = 0


def _object_spawn_offset(env, env_ids: torch.Tensor) -> torch.Tensor:
    """`(len(env_ids), 3)`, in the palm/root's own LOCAL frame (matching
    `env.cfg.reset.object_spawn_offset`'s existing "above the palm centre,
    in the palm's own frame" convention). Single-hand path: the one global
    cfg value, byte-identical to before this function existed. Population
    path (`env.hand_tables` set): each env's OWN design's `palm_up`-
    calibrated spawn point (over its fingertip workspace, not a fixed
    offset -- every design's geometry/orientation differs)."""
    tables = getattr(env, "hand_tables", None)
    if tables is None:
        offset = torch.as_tensor(env.cfg.reset.object_spawn_offset, device=env.device, dtype=torch.float32)
        return offset.expand(env_ids.numel(), 3)
    design_idx = env.scene_record["design_idx"][env_ids]
    spawn = torch.as_tensor(tables.spawn_offset, device=env.device, dtype=torch.float32)
    return spawn[design_idx]


def _population_default_joint_pos(env, env_ids: torch.Tensor) -> torch.Tensor:
    """`(len(env_ids), num_joints)` -- each of these envs' OWN design's
    default pose, in the articulation view's phys-column order. Review risk
    11 ("per-env defaults would revert to the template if the articulation
    re-initializes"): `env.robot.data.default_joint_pos` is Isaac Lab's OWN
    mutable buffer, which `scene_utils._resolve_population_joint_permutation`
    overwrites once per env-construction -- but re-derives from the scene-
    wide CFG template (env 0's design) if Isaac Lab's articulation
    re-initializes (e.g. a timeline stop/play cycle re-running
    `Articulation._initialize_impl`), silently reverting every OTHER env
    back to the wrong design's defaults with no error. `env.scene_record
    ["default_joint_pos"]` is instead a plain tensor this package owns, set
    once from the same per-env computation and never touched by Isaac Lab's
    own re-init path -- reading FROM it here (and in `obs_utils.
    pre_physics_step`'s action-centering) makes both immune to that,
    regardless of when/why a re-init happens. `None` on the single-hand
    path (`env.hand_tables` unset) -- unaffected, falls through to Isaac
    Lab's own buffer exactly as before this existed."""
    cached = env.scene_record.get("default_joint_pos") if getattr(env, "hand_tables", None) is not None else None
    if cached is not None:
        return cached[env_ids]
    return env.robot.data.default_joint_pos[env_ids]


def reset_env_state(env, env_ids: torch.Tensor) -> None:
    """Full reset: hand joints, object spawn, goal, and per-episode buffers."""
    n = env_ids.numel()
    spec = env.hand_spec

    # Hand joints to their (0) default pose plus uniform noise, clamped inside
    # each joint's own limits.
    default_pos = _population_default_joint_pos(env, env_ids)
    lower = env.robot.data.soft_joint_pos_limits[env_ids, :, 0]
    upper = env.robot.data.soft_joint_pos_limits[env_ids, :, 1]
    noise = (torch.rand_like(default_pos) * 2.0 - 1.0) * env.cfg.reset.joint_reset_noise
    joint_pos = torch.clamp(default_pos + noise, lower, upper)
    joint_vel = torch.zeros_like(joint_pos)
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    # ``_cur_targets`` is pre_physics_step's EMA state, not part of the
    # articulation view -- DirectRLEnv._reset_idx doesn't touch it, so
    # without this line a mid-training partial reset would resume tracking
    # whatever target was in flight for THIS env_id right before its
    # previous episode ended (e.g. a dropped/thrown configuration), fighting
    # the freshly-written joint_pos above for the first few steps of every
    # episode after the first. Doesn't exist yet on the very first reset
    # (pre_physics_step creates it lazily, after this env has been reset at
    # least once) -- nothing to fix up there.
    if hasattr(env, "_cur_targets"):
        env._cur_targets[env_ids] = joint_pos

    # Object: spawned just above the palm centre, random orientation, small
    # position jitter -- all in the palm's own (world) frame at reset time.
    palm_pos_w = env.robot.data.body_pos_w[env_ids, env.palm_body_idx]
    palm_quat_w = env.robot.data.body_quat_w[env_ids, env.palm_body_idx]
    offset = _object_spawn_offset(env, env_ids)
    jitter = (torch.rand(n, 3, device=env.device) * 2.0 - 1.0) * env.cfg.reset.object_position_noise
    local_pos = offset + jitter
    env._spawn_obj_pos_palm[env_ids] = local_pos
    obj_pos_w = palm_pos_w + quat_apply(palm_quat_w, local_pos)
    obj_quat_w = random_orientation(n, device=env.device)

    obj_state = torch.cat([obj_pos_w, obj_quat_w, torch.zeros(n, 6, device=env.device)], dim=-1)
    env.object.write_root_state_to_sim(obj_state, env_ids=env_ids)

    # Goal: curriculum-staged, relative to the object's just-written spawn
    # orientation (see ``_sample_goal``'s docstring). This used to always be
    # ``random_orientation`` here regardless of ``goal_sampling_type`` /
    # the curriculum -- ``_sample_goal`` (this module's other caller of it,
    # ``reset_goal_trackers``) was reachable only by first REACHING a goal,
    # so under the old fully-random-at-reset code the curriculum's easier
    # stages never actually applied to an episode's first (and, given how
    # rarely a random goal was reached, effectively only) goal.
    env._goal_quat_w[env_ids] = _sample_goal(env, env_ids, obj_quat_w)
    goal_pos_w = palm_pos_w + quat_apply(palm_quat_w, offset)
    goal_state = torch.cat(
        [goal_pos_w, env._goal_quat_w[env_ids], torch.zeros(n, 6, device=env.device)], dim=-1)
    env.goal_viz.write_root_state_to_sim(goal_state, env_ids=env_ids)

    # Per-episode buffers.
    env._prev_actions[env_ids] = 0.0
    env._consec_success_steps[env_ids] = 0
    env._prev_episode_successes[env_ids] = env._successes[env_ids]
    env._successes[env_ids] = 0
    env._prev_rot_error[env_ids] = 0.0  # overwritten by compute_intermediate_values just below

    # DirectRLEnv.reset() calls _get_observations() straight after _reset_idx(),
    # with no _get_dones() in between -- so the palm-frame cache has to be
    # fresh here too, not only from _get_dones() on a normal step. Geometry
    # only (not the success-streak bookkeeping): this can run on a PARTIAL
    # env_ids reset mid-training, and _get_dones() will still run this step
    # for every env, so a second success-streak update here would double
    # count it for the envs NOT being reset.
    from .obs_utils import update_palm_frame_geometry

    update_palm_frame_geometry(env)

    # Part C (graded per-design scoring): this episode's STARTING rot error
    # is env._rot_error[env_ids] as of the update_palm_frame_geometry call
    # just above (the just-written goal/object state) -- must run AFTER it,
    # same reasoning as the success-streak note above (this can run on a
    # partial env_ids reset; design_scoring.reset_scoring_state only touches
    # env_ids, matching the other per-episode buffers in this function).
    from . import design_scoring

    design_scoring.reset_scoring_state(env, env_ids)


def log_step_metrics(env) -> None:
    """Publish step-level extras consumed by RL-Games' ``EnvStatsAlgoObserver``
    (``coevolution/train.py``'s ``observers``), mirroring
    ``pose_reaching_6d.reset_utils.logging_utils.log_step_metrics`` exactly so
    the two tasks' TensorBoard/W&B panels line up: ``episode_cumulative``
    (summed per episode, then averaged over episodes finishing this step) and
    ``episode_final`` (read once, at the same episodes' last step) are the two
    dict shapes that observer understands; anything else has to be a scalar
    (or 0-dim tensor) to be picked up as ``direct_info``.

    ``env.extras["successes"]`` is read directly by
    ``coevolution/design_rewards.py`` (per-design reward banking) -- kept as
    ``_prev_episode_successes``, unchanged, for that reason.
    """
    term_cfg = env.cfg.termination
    if term_cfg.max_consecutive_successes > 0:
        all_goals_hit = env._successes >= term_cfg.max_consecutive_successes
    else:
        all_goals_hit = torch.zeros_like(env._successes, dtype=torch.bool)

    episode_final = {
        "successes": env._successes.float(),
        "all_goals_hit": all_goals_hit.float(),
    }
    episode_final.update({
        f"done_{name}": value.float() for name, value in env._termination_reasons.items()
    })

    env.extras["episode_cumulative"] = env._reward_terms
    env.extras["episode_final"] = episode_final
    env.extras["successes"] = env._prev_episode_successes.float()
    env.extras["current_success_tolerance"] = float(env._current_success_tolerance)
    # Goal-difficulty curriculum (I26 decoupling): the stage index and a
    # numeric mode code (see goal_curriculum.GOAL_MODE_CODES: axis=0,
    # delta=1, full/absolute=2) so TensorBoard shows exactly when the goal
    # curriculum advanced, next to current_success_tolerance above.
    env.extras["goal_curriculum_stage"] = float(env._goal_curriculum_stage)
    env.extras["goal_mode_code"] = _goal_mode_code(_goal_curriculum_mode(env))

    # Batch-level scalars refreshed every step (same "direct_info" mechanism
    # current_success_tolerance uses above): rotation-error mean/median and
    # mean hold-streak length are continuous, per-step quantities with no
    # natural "episode_final" reading, unlike success/drop/timeout which are
    # per-EPISODE outcomes already covered by episode_final above.
    env.extras["rot_error_mean"] = float(env._rot_error.mean())
    env.extras["rot_error_median"] = float(env._rot_error.median())
    env.extras["consec_success_steps_mean"] = float(env._consec_success_steps.float().mean())
