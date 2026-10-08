"""Task metric publishing for PoseReach."""

from __future__ import annotations

import torch


def log_step_metrics(env) -> None:
    """Publish step-level extras consumed by RL-Games observers."""
    term_cfg = env.cfg.termination
    if term_cfg.max_consecutive_successes > 0:
        all_goals_hit = env._successes >= term_cfg.max_consecutive_successes
    else:
        all_goals_hit = torch.zeros_like(env._successes, dtype=torch.bool)

    episode_final = {
        "successes": env._successes.float(),
        "all_goals_hit": all_goals_hit.float(),
    }
    episode_final.update(
        {
            f"done_{name}": value.float()
            for name, value in env._termination_reasons.items()
        }
    )

    episode_final.update(_per_hand(env, episode_final))

    env.extras["episode_cumulative"] = env._reward_terms
    env.extras["episode_final"] = episode_final
    env.extras["successes"] = env._prev_episode_successes.float()
    env.extras["current_success_tolerance"] = float(env._current_success_tolerance)


# Per-hand copies of these episode-final metrics in a multi-hand scene (robots/multi_hand.py), keyed
# "per_hand_<metric>/<label>" with HandSet.labels (L5_full, L4_no_index, ... R2_index_middle): one wandb section
# per metric, its panels sorted left before right, then by finger count. Successes only -- falls and timeouts per
# hand were 2/3 of the panels and rarely read. NaN outside the hand's envs; the observer keeps each key's finite
# values only.
PER_HAND_PREFIX = "per_hand_"
PER_HAND_METRICS = ("successes",)


def _per_hand(env, final: dict) -> dict:
    hs = getattr(env.scene_record, "hand_set", None)
    if hs is None:
        return {}
    masks = getattr(env, "_per_hand_masks", None)
    if masks is None:
        idx = env.scene_record.robot_design_index
        names = hs.labels or tuple(s.hand_name for s in hs.specs)
        masks = env._per_hand_masks = [(names[h], idx == h) for h in range(hs.n_hands)]
    nan = torch.tensor(float("nan"), device=env.device)
    return {f"{PER_HAND_PREFIX}{k}/{name}": torch.where(m, final[k], nan)
            for name, m in masks for k in PER_HAND_METRICS if k in final}


__all__ = ["PER_HAND_METRICS", "PER_HAND_PREFIX", "log_step_metrics"]
