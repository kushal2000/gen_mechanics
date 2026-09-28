"""Per-env guard against non-finite (NaN/Inf) physics state.

A grammar population holds many designs PhysX has never seen before; under
large, saturated actions an occasional articulation or object can blow up to
NaN/Inf. Before this guard, one such env was enough to end a whole training
run: a NaN observation row makes the policy's ``mu`` NaN, rl_games' own
``mu * 0 + sigma`` turns that into a NaN std, and ``torch.normal`` raises
``normal expects all elements of std >= 0.0`` (evolution pilot
G_V2S_s0, gen 7). NaN also compares False against every threshold, so the
drop check never ends such an episode by itself.

What this module does, per step, in the env hooks (see ``env.py``):

- ``guard_step_state`` (``_get_dones``, right after the palm-frame geometry
  refresh): flags every env whose object state, joint state or derived
  geometry is non-finite, and overwrites that env's cached geometry with
  finite placeholders so NaN cannot reach the reward, the success streak or
  ``design_scoring``'s accumulators.
- ``add_nonfinite_termination``: terminates the flagged envs (reason
  ``"nonfinite"``, logged as ``episode_final/done_nonfinite``), so the normal
  ``_reset_idx`` path rewrites their joint and object state.
  ``design_scoring.bank_done_episodes`` counts these per design.
- ``sanitize_reward`` / ``sanitize_observations``: last-line clean-up of
  what goes back to the learner (a flagged env's reward is the drop penalty;
  a non-finite observation entry becomes 0, or the clamp bound for +-Inf).

Kit-free (torch only), same discipline as ``drop_detection.py``, so it is
CPU-testable under plain pytest.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "GEOMETRY_CACHE_FIELDS", "allocate_guard_buffers", "nonfinite_rows", "detect_nonfinite_envs",
    "guard_step_state", "add_nonfinite_termination", "sanitize_reward", "sanitize_observations",
]

# The per-step geometry cache obs_utils.update_palm_frame_geometry writes, in
# the order it writes them. Every field here is sanitised for flagged envs.
GEOMETRY_CACHE_FIELDS = (
    "_obj_pos_palm", "_obj_quat_palm", "_obj_lin_vel_palm", "_obj_ang_vel_palm",
    "_goal_quat_palm", "_obj_quat_rel_goal", "_rot_error", "_fingertip_pos_palm",
)
_QUAT_FIELDS = ("_obj_quat_palm", "_goal_quat_palm", "_obj_quat_rel_goal")


def allocate_guard_buffers(env) -> None:
    """Run once, after the scene is up."""
    env._nonfinite_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._nonfinite_env_steps_total = 0
    env._nonfinite_obs_rows_total = 0


def nonfinite_rows(*tensors) -> torch.Tensor:
    """`(n,)` bool: True where ANY of `tensors` (each `(n, ...)`, `None`
    entries skipped) has a NaN/Inf anywhere in that row."""
    mask = None
    for t in tensors:
        if t is None:
            continue
        bad = ~torch.isfinite(t)
        if bad.dim() > 1:
            bad = bad.reshape(bad.shape[0], -1).any(dim=1)
        mask = bad if mask is None else (mask | bad)
    if mask is None:
        raise ValueError("nonfinite_rows needs at least one tensor")
    return mask


def detect_nonfinite_envs(env) -> torch.Tensor:
    """`(num_envs,)` bool: the object's root state, the hand's joint state,
    or this step's derived geometry is non-finite."""
    obj = env.object.data
    rob = env.robot.data
    return nonfinite_rows(
        obj.root_pos_w, obj.root_quat_w, obj.root_lin_vel_w, obj.root_ang_vel_w,
        rob.joint_pos, rob.joint_vel,
        *(getattr(env, name, None) for name in GEOMETRY_CACHE_FIELDS),
    )


def _where_rows(mask: torch.Tensor, fill: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    m = mask.reshape(-1, *([1] * (value.dim() - 1)))
    return torch.where(m, fill.to(dtype=value.dtype).expand_as(value), value)


def guard_step_state(env) -> torch.Tensor:
    """Detect this step's non-finite envs (stored as `env._nonfinite_mask`)
    and give their cached geometry finite placeholders: rotation error pi
    (the worst possible, so never a spurious success), object position at
    its own spawn point, identity quaternions, zero velocities and fingertip
    positions. Returns the mask."""
    mask = detect_nonfinite_envs(env)
    env._nonfinite_mask = mask
    if not bool(mask.any()):
        return mask
    env._nonfinite_env_steps_total += int(mask.sum())
    for name in ("_rot_error", "_prev_rot_error"):
        value = getattr(env, name, None)
        if value is not None:
            setattr(env, name, torch.where(mask, torch.full_like(value, math.pi), value))
    env._obj_pos_palm = _where_rows(mask, env._spawn_obj_pos_palm, env._obj_pos_palm)
    for name in _QUAT_FIELDS:
        value = getattr(env, name, None)
        if value is not None:
            identity = torch.zeros_like(value)
            identity[..., 0] = 1.0
            setattr(env, name, _where_rows(mask, identity, value))
    for name in ("_obj_lin_vel_palm", "_obj_ang_vel_palm", "_fingertip_pos_palm"):
        value = getattr(env, name, None)
        if value is not None:
            setattr(env, name, _where_rows(mask, torch.zeros_like(value), value))
    return mask


def add_nonfinite_termination(env, terminated: torch.Tensor) -> torch.Tensor:
    """OR this step's flagged envs into `terminated` and publish them as
    termination reason ``"nonfinite"`` (always present, all-False on a
    healthy step, so its TensorBoard curve exists from the start)."""
    mask = getattr(env, "_nonfinite_mask", None)
    if mask is None:
        mask = torch.zeros_like(terminated, dtype=torch.bool)
    env._termination_reasons["nonfinite"] = mask
    return terminated | mask


def sanitize_reward(env, reward: torch.Tensor) -> torch.Tensor:
    """A flagged env's reward is `-drop_penalty` (the object is as good as
    lost); any other non-finite reward entry is zeroed. The per-term
    breakdown (`env._reward_terms`, summed per episode for TensorBoard) is
    cleaned the same way so a single blow-up cannot turn a whole epoch's
    logged averages into NaN."""
    mask = getattr(env, "_nonfinite_mask", None)
    if mask is None:
        mask = torch.zeros(reward.shape[0], dtype=torch.bool, device=reward.device)
    penalty = torch.full_like(reward, -float(env.cfg.reward.drop_penalty))
    out = torch.where(mask, penalty, reward)
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    terms = getattr(env, "_reward_terms", None)
    if terms:
        for key, value in list(terms.items()):
            if key == "total_reward":
                terms[key] = out
                continue
            clean = torch.where(mask, torch.zeros_like(value), value)
            terms[key] = torch.where(torch.isfinite(clean), clean, torch.zeros_like(clean))
    return out


def sanitize_observations(env, obs: dict) -> dict:
    """Replace NaN with 0 and +-Inf with the observation clamp bound in every
    observation tensor, counting (in `env._nonfinite_obs_rows_total`) the
    policy rows that needed it. A non-zero count after the reset path has
    run means PhysX did not recover that env from the state write."""
    clamp = float(env.cfg.obs.clamp_abs_observations)
    out = {}
    for key, value in obs.items():
        finite = torch.isfinite(value)
        if bool(finite.all()):
            out[key] = value
            continue
        if key == "policy":
            env._nonfinite_obs_rows_total += int((~finite).reshape(value.shape[0], -1).any(dim=1).sum())
        out[key] = torch.nan_to_num(value, nan=0.0, posinf=clamp, neginf=-clamp)
    return out
