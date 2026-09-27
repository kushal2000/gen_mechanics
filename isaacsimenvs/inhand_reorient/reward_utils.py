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
    "get_curriculum_state", "set_curriculum_state",
]


def compute_rewards(env) -> torch.Tensor:
    cfg = env.cfg.reward
    progress = env._prev_rot_error - env._rot_error
    action_penalty = (env._prev_actions ** 2).sum(dim=-1)
    action_delta_penalty = ((env._prev_actions - env._prev_actions_this_step) ** 2).sum(dim=-1)
    hand_vel_penalty = (env.robot.data.joint_vel ** 2).sum(dim=-1)

    reward = (
        cfg.rotation_progress_scale * progress
        + cfg.goal_bonus * env._is_success.float()
        - cfg.action_penalty_scale * action_penalty
        - cfg.action_delta_penalty_scale * action_delta_penalty
        - cfg.hand_velocity_penalty_scale * hand_vel_penalty
    )
    reward = reward - cfg.drop_penalty * env._termination_reasons.get(
        "drop", torch.zeros_like(env._is_success)).float()
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
