"""Termination and success-tolerance curriculum helpers."""

from __future__ import annotations

import torch

from ..reset_utils import reset_goal_trackers


def update_tolerance_curriculum(env) -> None:
    """Shrink success tolerance when completed episodes average enough goals."""
    env._frame_counter += 1
    term = env.cfg.termination
    if env._frame_counter - env._last_curriculum_update >= term.tolerance_curriculum_interval:
        successes = env._prev_episode_successes.float()
        eligible_mask = None
        if hasattr(env, "_curriculum_eligible_mask"):
            eligible_mask = env._curriculum_eligible_mask()
        if eligible_mask is not None:
            successes = successes[eligible_mask]

        threshold = term.tolerance_curriculum_success_threshold
        if hasattr(env, "_curriculum_success_threshold"):
            custom_threshold = env._curriculum_success_threshold()
            if custom_threshold is not None:
                threshold = float(custom_threshold)

        if successes.numel() > 0 and successes.mean().item() >= threshold:
            new_tol = env._current_success_tolerance * term.tolerance_curriculum_increment
            new_tol = max(min(new_tol, term.success_tolerance), term.target_success_tolerance)
            env._current_success_tolerance = new_tol
            env._last_curriculum_update = env._frame_counter

    # Eval pins the success criterion.
    if term.eval_success_tolerance is not None:
        env._current_success_tolerance = float(term.eval_success_tolerance)


def compute_terminations(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Update goal-hit state and return ``(terminated, truncated)``."""
    term_cfg = env.cfg.termination
    env_origins = env.scene.env_origins
    is_success = env._is_success

    # Authoritative updates on goal-hit.
    env._successes = env._successes + is_success.long()
    goal_reset_ids = is_success.nonzero(as_tuple=False).squeeze(-1)
    if goal_reset_ids.numel() > 0:
        reset_goal_trackers(env, goal_reset_ids)
        # zero the length buf so truncation doesn't fire
        env.episode_length_buf[goal_reset_ids] = 0

    # Termination causes.
    if term_cfg.drop_distance_m is not None:
        # A fixed palm-up hand has no floor to fall below, so "dropped" is a
        # distance from the hand. Measured to the palm CENTRE, not link_7's
        # origin: the two differ by 0.095 + palm_length/2, which VARIES PER
        # DESIGN, so a body-relative threshold would give a long-palmed hand
        # less slack than a short one and bias the per-design return that
        # co-evolution selects on. _palm_center_pos_w is set by
        # compute_intermediate_values, which runs first in _get_dones.
        fall = torch.norm(
            env.object.data.root_pos_w - env._palm_center_pos_w, dim=-1
        ) > term_cfg.drop_distance_m
    else:
        object_z_local = env.object.data.root_pos_w[:, 2] - env_origins[:, 2]
        fall = object_z_local < 0.1

    if term_cfg.max_consecutive_successes > 0:
        max_successes_reached = env._successes >= term_cfg.max_consecutive_successes
    else:
        max_successes_reached = torch.zeros_like(fall)

    hand_far = env._curr_fingertip_distances.max(dim=-1).values > 1.5

    terminated = fall | max_successes_reached | hand_far
    truncated = env.episode_length_buf >= env.max_episode_length
    env._termination_reasons = {
        "fall": fall,
        "max_successes": max_successes_reached,
        "hand_far": hand_far,
        "timeout": truncated,
    }
    return terminated, truncated


__all__ = ["update_tolerance_curriculum", "compute_terminations"]
