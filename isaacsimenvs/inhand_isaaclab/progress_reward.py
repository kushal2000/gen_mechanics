"""Our reward's SHAPE, as a reward term for the reference env.

THE HYPOTHESIS. IsaacLab's task reward is ``1/(dtheta + 0.1)``, paid every step, bounded
in [0.0103, 0.333] per step after the dt scaling -- always positive, never exhausted, and
maximal when aligned. Ours is ``keypoint_rew_scale * max(0, best_ever - current)``, paid
only when the object beats its own record for this goal. Measured on the live runs, the
consequences are: 95% of their per-step income is the guaranteed dense term, against 53%
of ours; one goal is 29% of a whole episode's return for them and 752% for us; and the
random-init policy already collects 82% of the return our policy reaches after 890M
frames. This rung swaps the shape and nothing else.

WHAT IS HELD FIXED, deliberately. The quantity is still ``quat_error_magnitude`` against
their sampled goal -- NOT our keypoint residual. Changing the metric at the same time
would confound "progress instead of proximity" with "keypoints instead of an angle", and
the keypoint set has its own anisotropy (5.74-9.94 deg for one threshold) that deserves
its own rung. ``success_bonus`` stays at 250 and the three penalties are untouched.

THE WEIGHT IS NOT ARBITRARY. RewardManager multiplies every term by ``weight * dt``
(reward_manager.py:150), and a progress delta is not a rate, so the dt still applies: the
episode total is ``weight * dt * sum(deltas)``, and the deltas over one goal sum to at
most the initial error, ~pi rad. Matching the ~28 per episode their dense term currently
earns needs ``weight = 28 * 30 / pi ~= 267``. Magnitude is therefore held comparable to
what it replaces, so what changes is the shape alone.

WHY THE BASELINE RESETS ON A NEW GOAL. Their command term resamples the goal the moment
it is reached, without resetting the object. A progress reward measured against a stale
baseline would be dead from the first success onward, so the record is cleared whenever
the goal quaternion changes -- which is what ``_clear_goal_trackers`` does in our env
(reset.py:586). Detecting it from the command itself rather than from a success flag also
covers a resample this term never sees.
"""

from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils


class track_orientation_progress(ManagerTermBase):
    """Reward only IMPROVEMENT on the best orientation error seen for the current goal."""

    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(cfg, env)
        dev = env.device
        # inf is the sentinel for "no record yet", so the first step of a goal pays 0 and
        # sets the baseline -- the same contract as our -1.0 sentinel in observations.py:257.
        self._best = torch.full((env.num_envs,), float("inf"), device=dev)
        self._goal = torch.zeros((env.num_envs, 4), device=dev)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._best[env_ids] = float("inf")
        self._goal[env_ids] = 0.0

    def __call__(self, env, command_name: str,
                 object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
        asset = env.scene[object_cfg.name]
        goal = env.command_manager.get_term(command_name).command[:, 3:7]
        dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal)

        fresh = torch.isinf(self._best) | ((goal - self._goal).abs().sum(-1) > 1e-6)
        self._best = torch.where(fresh, dtheta, self._best)
        self._goal = goal.clone()

        delta = (self._best - dtheta).clamp(min=0.0)
        self._best = torch.minimum(self._best, dtheta)
        return delta


__all__ = ["track_orientation_progress"]
