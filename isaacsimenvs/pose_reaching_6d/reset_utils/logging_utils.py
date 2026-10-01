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
# "per_hand/<hand>_<metric>" -- wandb files a key under the text before its first "/", so they land in
# their own "per_hand" tab. NaN outside the hand's envs; the observer keeps each key's finite values only.
PER_HAND_METRICS = ("successes", "done_fall", "done_timeout")


def _per_hand(env, final: dict) -> dict:
    hs = getattr(env.scene_record, "hand_set", None)
    if hs is None:
        return {}
    masks = getattr(env, "_per_hand_masks", None)
    if masks is None:
        idx = env.scene_record.robot_design_index
        masks = env._per_hand_masks = [(s.hand_name, idx == h) for h, s in enumerate(hs.specs)]
    nan = torch.tensor(float("nan"), device=env.device)
    return {f"per_hand/{name}_{k}": torch.where(m, final[k], nan)
            for name, m in masks for k in PER_HAND_METRICS if k in final}


__all__ = ["PER_HAND_METRICS", "log_step_metrics"]
