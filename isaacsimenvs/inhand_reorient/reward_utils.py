"""Reward and termination for in-hand reorientation.

The tolerance curriculum itself (``update_tolerance_curriculum``,
``get_curriculum_state``/``set_curriculum_state``) is reused directly from
``pose_reaching_6d`` rather than reimplemented: it is a pure function of
``env._current_success_tolerance`` / ``env._frame_counter`` /
``env._last_curriculum_update`` / ``env._prev_episode_successes`` /
``env.cfg.termination.*``, all of which this package's ``reset_utils`` and
``env_cfg`` already provide under the same names -- and
``coevolution/utils/rlgames_utils.py`` hardcodes an import of
``get_curriculum_state``/``set_curriculum_state`` from
``isaacsimenvs.pose_reaching_6d.reward_utils`` for EVERY env's checkpoint
save/restore, so reusing the same function objects is also the only way this
env's curriculum resumes at all.
"""

from __future__ import annotations

import torch

from isaacsimenvs.pose_reaching_6d.reward_utils.curriculum import (  # noqa: F401
    get_curriculum_state, set_curriculum_state,
)
from isaacsimenvs.pose_reaching_6d.reward_utils.termination import (  # noqa: F401
    update_tolerance_curriculum,
)

__all__ = [
    "compute_rewards", "compute_terminations", "update_tolerance_curriculum",
    "update_goal_curriculum", "extra_curriculum_state", "restore_extra_curriculum_state",
    "get_curriculum_state", "set_curriculum_state",
]


def update_goal_curriculum(env) -> None:
    """Widen the goal-sampling curriculum (``reset_utils._sample_goal``'s
    modes) once completed episodes average enough goals, mirroring
    ``update_tolerance_curriculum`` (I24, Phase 1c). Reads
    ``env._frame_counter`` AFTER ``update_tolerance_curriculum`` has already
    advanced it this step -- call this right after that function, not before,
    to avoid a double-increment (see ``env.py::_get_dones``)."""
    cfg = env.cfg.reset
    if not cfg.goal_curriculum_enabled:
        return
    if env._frame_counter - env._last_goal_curriculum_update < cfg.goal_curriculum_interval:
        return
    successes = env._prev_episode_successes.float()
    if successes.numel() == 0 or successes.mean().item() < cfg.goal_curriculum_success_threshold:
        return
    env._last_goal_curriculum_update = env._frame_counter
    if env._goal_curriculum_stage >= len(cfg.goal_curriculum_stages) - 1:
        return
    env._goal_curriculum_stage += 1
    stage_name = cfg.goal_curriculum_stages[env._goal_curriculum_stage]
    print(f"[inhand_reorient] goal curriculum -> stage {env._goal_curriculum_stage} "
          f"({stage_name!r}) at frame {env._frame_counter}", flush=True)


def extra_curriculum_state(env) -> dict:
    """This env's half of the generic ``extra_curriculum_state`` hook
    ``pose_reaching_6d.reward_utils.curriculum.get_curriculum_state`` calls
    (guarded there with ``callable(...)``, so it stays ignorant of these
    names) -- the goal-difficulty curriculum's own state, additive to the
    tolerance curriculum's 4-key dict. ``InHandReorientEnv.extra_curriculum_
    state`` is a thin bound-method wrapper around this free function, kept
    here with the rest of this package's curriculum logic."""
    return {
        "goal_curriculum_stage": int(env._goal_curriculum_stage),
        "last_goal_curriculum_update": int(env._last_goal_curriculum_update),
    }


def restore_extra_curriculum_state(env, state: dict) -> None:
    """This env's half of the generic ``restore_extra_curriculum_state``
    hook ``pose_reaching_6d.reward_utils.curriculum.set_curriculum_state``
    calls. Tolerant of missing keys, same as the tolerance curriculum's own
    restore -- a checkpoint written before the goal curriculum existed has
    neither key and should still load."""
    env._goal_curriculum_stage = int(
        state.get("goal_curriculum_stage", env._goal_curriculum_stage))
    env._last_goal_curriculum_update = int(
        state.get("last_goal_curriculum_update", env._last_goal_curriculum_update))
    if "goal_curriculum_stage" in state:
        print(f"[curriculum] restored goal curriculum stage "
              f"{env._goal_curriculum_stage} from the checkpoint", flush=True)


def _joint_valid_mask(env) -> torch.Tensor | None:
    """`(num_envs, 32)` bool, `True` where that env's own design has a REAL
    (non-ghost) joint in that envelope slot -- `None` for the single-hand
    path (`env.hand_tables` is only set by `scene/author_grammar.py`'s
    population path), so every existing single-hand call site is
    byte-identical to before this function existed (Phase 2's hard
    requirement)."""
    tables = getattr(env, "hand_tables", None)
    if tables is None:
        return None
    design_idx = env.scene_record["design_idx"]
    valid = torch.as_tensor(tables.joint_valid, device=env.device, dtype=torch.bool)
    return valid[design_idx]


def compute_rewards(env) -> torch.Tensor:
    """Reward adopted from IsaacGymEnvs' AllegroHand/ShadowHand (Makoviychuk
    et al., 2021): a dense inverse-distance rotation term plus a position
    term keeping the object near its spawn point, a goal bonus, action and
    fall penalties. The previous version's only dense signal was
    ``rotation_progress_scale * (prev_rot_error - rot_error)`` -- a per-step
    DELTA that is zero-mean under noise and gives no gradient toward
    "closer" independent of whether the error is shrinking THIS step; the
    inverse-distance term below is nonzero-mean and grows as the error
    shrinks, which is what the literature recipe actually relies on. Kept
    (default OFF, ``rotation_progress_scale=0.0``) for an A/B toggle.
    """
    cfg = env.cfg.reward
    # Goal position == the object's own spawn point in the palm frame: it
    # never moves within an episode (reset_utils.reset_goal_trackers
    # resamples the goal ORIENTATION only), so no separate goal-position
    # buffer is needed.
    dist = (env._obj_pos_palm - env._spawn_obj_pos_palm).norm(dim=-1)
    rot_rew = cfg.rot_reward_scale / (env._rot_error + cfg.rot_eps)
    dist_rew = -cfg.dist_reward_scale * dist
    progress = env._prev_rot_error - env._rot_error
    joint_mask = _joint_valid_mask(env)  # None on the single-hand path (see its docstring)
    if joint_mask is None:
        action_penalty = (env._prev_actions ** 2).sum(dim=-1)
        action_delta_penalty = ((env._prev_actions - env._prev_actions_this_step) ** 2).sum(dim=-1)
        hand_vel_penalty = (env.robot.data.joint_vel ** 2).sum(dim=-1)
    else:
        action_penalty = ((env._prev_actions ** 2) * joint_mask).sum(dim=-1)
        action_delta_penalty = (
            ((env._prev_actions - env._prev_actions_this_step) ** 2) * joint_mask
        ).sum(dim=-1)
        hand_vel_penalty = ((env.robot.data.joint_vel ** 2) * joint_mask).sum(dim=-1)

    goal_bonus_term = cfg.goal_bonus * env._is_success.float()
    action_penalty_term = -cfg.action_penalty_scale * action_penalty
    action_delta_penalty_term = -cfg.action_delta_penalty_scale * action_delta_penalty
    hand_vel_penalty_term = -cfg.hand_velocity_penalty_scale * hand_vel_penalty
    drop_penalty_term = -cfg.drop_penalty * env._termination_reasons.get(
        "drop", torch.zeros_like(env._is_success)).float()
    progress_term = cfg.rotation_progress_scale * progress

    reward = (
        rot_rew + dist_rew + progress_term + goal_bonus_term + action_penalty_term
        + action_delta_penalty_term + hand_vel_penalty_term + drop_penalty_term
    )
    # Per-term breakdown, for TensorBoard (logging_utils.log_step_metrics
    # publishes this as env.extras["episode_cumulative"], mirroring
    # pose_reaching_6d's reward_utils.rewards -- see that module for why the
    # SAME dict shape matters (EnvStatsAlgoObserver sums it over each
    # episode, then averages over episodes that just finished).
    env._reward_terms = {
        "rotation_rew": rot_rew,
        "distance_rew": dist_rew,
        "rotation_progress_rew": progress_term,
        "goal_bonus_rew": goal_bonus_term,
        "action_penalty": action_penalty_term,
        "action_delta_penalty": action_delta_penalty_term,
        "hand_velocity_penalty": hand_vel_penalty_term,
        "drop_penalty": drop_penalty_term,
        "total_reward": reward,
    }
    return reward


def compute_terminations(env) -> tuple[torch.Tensor, torch.Tensor]:
    term_cfg = env.cfg.termination

    env._successes = env._successes + env._is_success.long()
    goal_reset_ids = env._is_success.nonzero(as_tuple=False).squeeze(-1)
    if goal_reset_ids.numel() > 0:
        from .reset_utils import reset_goal_trackers

        reset_goal_trackers(env, goal_reset_ids)
        env.episode_length_buf[goal_reset_ids] = 0

    displacement = (env._obj_pos_palm - env._spawn_obj_pos_palm).norm(dim=-1)
    below_palm = env._obj_pos_palm[:, 2] < -0.5 * env.cfg.reset.drop_distance_m
    drop = (displacement > env.cfg.reset.drop_distance_m) | below_palm

    if term_cfg.max_consecutive_successes > 0:
        max_successes_reached = env._successes >= term_cfg.max_consecutive_successes
    else:
        max_successes_reached = torch.zeros_like(drop)

    terminated = drop | max_successes_reached
    truncated = env.episode_length_buf >= env.max_episode_length
    env._termination_reasons = {
        "drop": drop, "max_successes": max_successes_reached, "timeout": truncated,
    }
    return terminated, truncated
