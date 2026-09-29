"""Our keypoint residual, as the quantity inside THEIR dense reward.

THE HYPOTHESIS. Our task measures orientation error as a keypoint residual rather than an
angle: four offsets o_i on the cube, residual r = max_i |(R_obj - R_goal) o_i|, taken about
each set's own centre (orientation_only_goal). IsaacLab measures quat_error_magnitude, the
true geodesic angle. The two differ in three ways that could matter to a learner:
  * the shipped corner set {+-(1,1,1), +-(1,1,-1)} is two antipodal pairs -- rank 2,
    coplanar -- so r = 2 sin(theta/2) * K(axis) with K spanning a factor sqrt(3) across
    rotation axes: the same angle scores up to 1.73x differently depending on the axis;
  * r is chordal (2 sin(theta/2)), so it saturates near 180 deg where the angle does not;
  * it is a max over keypoints, which gives a gradient only through whichever one is
    currently worst.

ONE DELTA. The reward keeps their exact shape, 1/(x + 0.1), and only x changes: x = r/|o|
instead of theta. Dividing by |o| (the keypoint radius, K_max) makes x agree with theta for
small rotations about the worst axis, so the scale matches what it replaces. The success
bonus and termination still use their quaternion angle, so this changes the SIGNAL the policy
climbs and leaves what counts as a goal alone.

KEYPOINTS are ours exactly: KEYPOINT_CORNERS scaled by 0.5 * keypoint_scale(1.5) * edge,
with the edge taken from this env's own cube so the rung stays correct on the 45 mm variant.
"""

from __future__ import annotations

import math

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils

# observations.py:24, verbatim. Two antipodal pairs; see the module docstring.
KEYPOINT_CORNERS = ((1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1))
KEYPOINT_SCALE = 1.5          # env_cfg.py:134 / InHandReorient.yaml:134
REF_EDGE_M = 0.06             # dex_cube_instanceable.usd at scale 1


def keypoint_residual_normalised(env, goal: torch.Tensor, object_cfg: SceneEntityCfg
                                 ) -> torch.Tensor:
    """r / |o|: our orientation residual, scaled so it equals theta for small rotations
    about the worst axis. Shared by the dense and the progress variants."""
    asset = env.scene[object_cfg.name]

    scale = getattr(env.cfg.scene.object.spawn, "scale", None)
    edge = REF_EDGE_M * (float(scale[0]) if scale else 1.0)
    offs = torch.tensor(KEYPOINT_CORNERS, device=env.device, dtype=torch.float32)
    offs = offs * (0.5 * KEYPOINT_SCALE * edge)                        # (4, 3)
    n = asset.data.root_quat_w.shape[0]
    o = offs.unsqueeze(0).expand(n, -1, -1)                            # (N, 4, 3)

    q_obj = asset.data.root_quat_w.unsqueeze(1).expand(-1, 4, -1)
    q_goal = goal.unsqueeze(1).expand(-1, 4, -1)
    # About each set's own centre: translation cancels, the rotation residual remains.
    r = torch.norm(math_utils.quat_apply(q_obj, o) - math_utils.quat_apply(q_goal, o),
                   dim=-1).max(dim=-1).values                          # (N,) metres

    k_max = 0.5 * KEYPOINT_SCALE * edge * math.sqrt(3.0)               # |o|
    return r / k_max


def track_orientation_keypoint_inv(env, command_name: str, rot_eps: float = 0.1,
                                   object_cfg: SceneEntityCfg = SceneEntityCfg("object")
                                   ) -> torch.Tensor:
    """Their dense shape on our metric (not launched; kept for a later rung)."""
    goal = env.command_manager.get_term(command_name).command[:, 3:7]
    return 1.0 / (keypoint_residual_normalised(env, goal, object_cfg) + rot_eps)


class track_orientation_keypoint_progress(ManagerTermBase):
    """OUR REWARD on their env: progress against a record, on the keypoint residual.

    Identical to progress_reward.track_orientation_progress in every respect -- record
    cleared on reset and on a goal resample, inf sentinel so a goal's first step pays 0,
    weight 267 -- except that the quantity is keypoint_residual_normalised instead of the
    quaternion angle. Normalising by |o| keeps the small-angle scale equal to theta, so the
    weight carries over and this rung differs from progressrew in the metric alone.
    """

    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._best = torch.full((env.num_envs,), float("inf"), device=env.device)
        self._goal = torch.zeros((env.num_envs, 4), device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._best[env_ids] = float("inf")
        self._goal[env_ids] = 0.0

    def __call__(self, env, command_name: str,
                 object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
        goal = env.command_manager.get_term(command_name).command[:, 3:7]
        x = keypoint_residual_normalised(env, goal, object_cfg)
        fresh = torch.isinf(self._best) | ((goal - self._goal).abs().sum(-1) > 1e-6)
        self._best = torch.where(fresh, x, self._best)
        self._goal = goal.clone()
        delta = (self._best - x).clamp(min=0.0)
        self._best = torch.minimum(self._best, x)
        return delta


__all__ = ["keypoint_residual_normalised", "track_orientation_keypoint_inv",
           "track_orientation_keypoint_progress"]
