"""The ``hora`` task profile: z-axis in-hand rotation after HORA.

Source: H. Qi, A. Kumar, R. Calandra, Y. Ma, J. Malik, "In-Hand Object
Rotation via Rapid Motor Adaptation", CoRL 2022; code github.com/HaozhiQi/
hora (``hora/tasks/allegro_hand_hora.py``, ``configs/task/AllegroHandHora.yaml``,
``configs/train/AllegroHandHora.yaml``, ``scripts/train_s1.sh``). This module
holds the Kit-free arithmetic; ``anyrotate_hooks`` runs it when
``task_profile == "hora"``.

The profile is a member of the anyrotate family (``anyrotate_profile.
is_anyrotate``): the scene, object, hand placement and actuator, contact
sensors, grasp cache and design scoring are the anyrotate ones; HORA's
reward, termination, observation, control and timing replace AnyRotate's.

Ported (HORA's numbers in ``HoraCfg``): reward = clip(omega . k, -0.5, 0.5)
(finite-difference angular velocity over the control step) - 0.3 ||v||_1
(finite-difference object velocity) - 0.3 ||q - q_init||^2 - 0.1 ||tau||^2
- 2 (tau . qdot)^2; termination when the object falls below its start
height (``reset_height_threshold``; here relative, ``drop_dz``) or after
400 steps; control q-bar += a / 24 at 20 Hz (1/120 s physics x 6), no
smoothing; observation = the last 3 of [unscaled joint positions + U(+-0.02)
noise, joint targets], plus the 9 privileged values (object position,
scale, mass, friction, centre of mass) that HORA's stage-1 policy embeds;
resets from the grasp cache, no noise, q_init = the cached pose; PPO as
``InHandAnyRotatePPO.yaml`` (identical to HORA's PPO except that one MLP
takes [o, x] instead of a separate 8-d privileged embedding).

Not ported: HORA's object set and scale randomisation (one 5.25 cm box per
run, the anyrotate object), friction randomisation U(0.3, 3) (one value,
``HoraCfg.friction``), PD-gain randomisation, random disturbance forces
(forceScale 2, probability 0.25), and stage 2 (proprioceptive adaptation).
"""

from __future__ import annotations

import torch

from .anyrotate_profile import quat_to_rotvec
from .repose_profile import quat_conjugate, quat_mul

__all__ = [
    "DesignRewardScale", "morphology_table", "MORPH_PER_SLOT",
    "PROFILE_HORA", "is_hora", "rotate_reward", "linvel_penalty", "pose_diff_penalty", "torque_penalty",
    "work_penalty", "combine_reward", "unscale", "push_history", "fill_history", "dropped", "HORA_OBS_FIELDS",
]

PROFILE_HORA = "hora"
HORA_OBS_FIELDS = ("hora_proprio_hist", "hora_priv")
HISTORY_LEN = 3  # compute_observations: obs_buf_lag_history[:, -3:]


def is_hora(cfg) -> bool:
    return getattr(cfg, "task_profile", "legacy") == PROFILE_HORA


def _masked(x: torch.Tensor, mask) -> torch.Tensor:
    return x if mask is None else torch.where(mask, x, torch.zeros_like(x))


def rotate_reward(q_prev, q_curr, axis, dt: float, lo: float, hi: float) -> torch.Tensor:
    """angdiff = axis-angle(q_curr q_prev^-1); object_angvel = angdiff / dt;
    rotate_reward = clip(angvel . k, lo, hi)."""
    angvel = quat_to_rotvec(quat_mul(q_curr, quat_conjugate(q_prev))) / float(dt)
    return (angvel * axis).sum(dim=-1).clamp(min=lo, max=hi)


def linvel_penalty(p_prev, p_curr, dt: float) -> torch.Tensor:
    """object_linvel = (pos - pos_prev) / dt; penalty = ||object_linvel||_1."""
    return ((p_curr - p_prev) / float(dt)).abs().sum(dim=-1)


def pose_diff_penalty(q, q_init, mask=None) -> torch.Tensor:
    return (_masked(q - q_init, mask) ** 2).sum(dim=-1)


def torque_penalty(tau, mask=None) -> torch.Tensor:
    return (_masked(tau, mask) ** 2).sum(dim=-1)


def work_penalty(tau, dq, mask=None) -> torch.Tensor:
    """((torques * dof_vel).sum(-1)) ** 2."""
    return _masked(tau * dq, mask).sum(dim=-1) ** 2


def combine_reward(terms: dict, scales: dict) -> torch.Tensor:
    """compute_hand_reward: rotate x 1 + each penalty x its (negative) scale."""
    return sum(scales[k] * terms[k] for k in ("rotate", "linvel", "pose", "torque", "work"))


def unscale(x, lower, upper) -> torch.Tensor:
    """(2x - upper - lower) / (upper - lower); a locked joint (range < 1e-6)
    maps to 0."""
    rng = upper - lower
    return torch.where(rng > 1e-6, (2.0 * x - upper - lower) / rng.clamp(min=1e-6), torch.zeros_like(x))


def push_history(hist: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Sliding window ``(n, L, w)``: drop the oldest frame, append ``frame``."""
    return torch.cat([hist[:, 1:], frame.unsqueeze(1)], dim=1)


def fill_history(hist: torch.Tensor, env_ids: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """At reset every slot of these envs' window holds the current frame."""
    out = hist.clone()
    out[env_ids] = frame.unsqueeze(1).expand(-1, hist.shape[1], -1)
    return out


def dropped(z, z0, drop_dz: float) -> torch.Tensor:
    """check_termination: object z below the threshold, here ``drop_dz``
    below the episode's start height."""
    return z < (z0 - float(drop_dz))


# --------------------------------------------------------------------------
# Shared controllers across hands (options; HORA trains one hand)
# --------------------------------------------------------------------------


class DesignRewardScale:
    """Per-design reward scaling: a running (EMA over steps) second moment of
    each design's per-step reward, and each env's reward divided by its
    design's RMS, floored at ``floor``. Scale only (no mean shift), so the
    sign of every term is kept."""

    def __init__(self, n_designs: int, decay: float = 0.999, floor: float = 0.05, device=None):
        self.decay, self.floor = float(decay), float(floor)
        self.m2 = torch.zeros(n_designs, device=device)
        self.seen = torch.zeros(n_designs, dtype=torch.bool, device=device)

    @property
    def rms(self) -> torch.Tensor:
        return self.m2.sqrt()

    def normalize(self, reward: torch.Tensor, design_idx: torch.Tensor) -> torch.Tensor:
        n = self.m2.shape[0]
        sq = torch.zeros(n, device=reward.device).index_add_(0, design_idx, reward.detach() ** 2)
        cnt = torch.zeros(n, device=reward.device).index_add_(0, design_idx, torch.ones_like(reward))
        has = cnt > 0
        batch = torch.where(has, sq / cnt.clamp(min=1.0), self.m2)
        first = has & ~self.seen
        self.m2 = torch.where(first, batch, torch.where(has, self.decay * self.m2 + (1 - self.decay) * batch, self.m2))
        self.seen = self.seen | has
        return reward / self.rms.clamp(min=self.floor)[design_idx]


MORPH_PER_SLOT = 10  # valid, axis (3), origin (3), length, lower, upper


def morphology_table(design) -> "np.ndarray":
    """``(32, 10)`` per envelope slot, root (palm) frame, q = 0: validity,
    joint axis, joint origin, link length, joint limits; ghost slots 0."""
    import numpy as np

    from .scene import grammar_envelope as ge

    T0 = ge.authored_fk(design, np.zeros(ge.N_SLOTS))
    out = np.zeros((ge.N_SLOTS, MORPH_PER_SLOT))
    for s in range(ge.N_SLOTS):
        if not design.slot_valid[s]:
            continue
        out[s, 0] = 1.0
        out[s, 1:4] = T0[s][:3, :3] @ np.asarray(design.slot_axis[s], dtype=float)
        out[s, 4:7] = T0[s][:3, 3]
        out[s, 7] = float(design.slot_length[s])
        out[s, 8:10] = design.slot_limits[s]
    return out


def disjoint_slot_map(valid) -> "np.ndarray":
    """``(D, J)`` int: the policy column of each design's articulation
    column, such that no two designs share a policy column for a real
    joint. Designs in order: each real joint keeps its own column when no
    earlier design's real joint holds it, else takes the lowest free one;
    the design's ghost joints fill its remaining columns in order. Raises
    when the designs have more real joints than columns."""
    import numpy as np

    valid = np.asarray(valid, dtype=bool)
    n_designs, n_cols = valid.shape
    if int(valid.sum()) > n_cols:
        raise ValueError(f"{int(valid.sum())} real joints over {n_designs} designs do not fit {n_cols} columns")
    taken = np.zeros(n_cols, dtype=bool)
    out = np.zeros((n_designs, n_cols), dtype=np.int64)
    for d in range(n_designs):
        used = np.zeros(n_cols, dtype=bool)
        for p in np.flatnonzero(valid[d]):
            c = p if not taken[p] else int(np.flatnonzero(~taken & ~used)[0])
            out[d, p] = c
            used[c] = taken[c] = True
        free = iter(np.flatnonzero(~used))
        for p in np.flatnonzero(~valid[d]):
            out[d, p] = next(free)
    return out


def to_policy_columns(x: torch.Tensor, phys_of_pol: torch.Tensor) -> torch.Tensor:
    """Per-joint values ``(n, J)`` in articulation order -> policy order."""
    return x.gather(1, phys_of_pol)


def to_phys_columns(x: torch.Tensor, pol_of_phys: torch.Tensor) -> torch.Tensor:
    """Per-joint values ``(n, J)`` in policy order -> articulation order."""
    return x.gather(1, pol_of_phys)
