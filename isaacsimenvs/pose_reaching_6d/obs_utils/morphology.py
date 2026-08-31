"""Morphology descriptor: what the policy needs to know about the hand it has.

A cross-embodied policy commands per-joint position targets on a hand it has
never seen. Proprioception alone cannot tell it what those commands do. Two
designs can present identical joint angles and identical NORMALIZED limits while
their fingers point in completely different directions, because all of that
lives in the mount transform -- which never entered the observation.

Worse, normalization actively hides the one limit that does vary: a joint_pos of
0.5 means 1 degree of travel on a +/-2 deg abduction joint and 12.5 on a
+/-25 deg one. The policy sees the same number for a twelvefold difference in
what its action achieves.

So this emits, per design, a fixed-width vector describing the mechanism. It is
CONSTANT for the life of an env -- computed once at scene build, then indexed
per env and concatenated to the observation every step.

WHAT IS IN IT, and why that and not something else
--------------------------------------------------
The sampler (hand_sampler.params.sample) draws 53 continuous and 11
discrete numbers for a five-finger hand. Those numbers are a complete sufficient
statistic -- the kinematics are a deterministic function of them -- but they are
in the wrong form. `face`, `u_frac`, `v_frac`, `roll`, `tilt` and `tilt_azimuth`
only mean anything after ``mount_on_face`` composes them into a transform, and
making the network relearn that composition wastes capacity on arithmetic we can
just do. So the mount is emitted as a POSE IN THE PALM FRAME, which is the frame
the rest of the observation already lives in.

Per finger slot, in slot order, ghosted slots included:

    mount position        3   palm frame
    mount orientation     6   first two columns of R (the 6D rotation
                              representation; continuous, unlike Euler or
                              quaternion, which matters for regression targets)
    link lengths          4   mc, pp, mp, dp
    link radii            4   mc, pp, mp, dp -- FOUR values, not one. radius_scale
                              is a single sampled knob but the tier nominals span
                              2.6x (19.3 / 10.2 / 8.9 / 7.3 mm), and dp is the
                              radius the fingertip actually grasps with.
    fingertip pad offset  3   distal cap, where contact happens
    enabled mask          6   per joint slot; a locked joint is still in the
                              action vector and does nothing
    AA half-range         1   the ONE limit the sampler varies per finger
    active flag           1   0 for a ghosted finger
                         --
                         28   x 5 slots = 140

plus palm extents (3), for 143 total.

Deliberately NOT included: per-joint axes and origins. Within this generator the
chain structure is a template constant (the FE/AA perpendicularity is carried
entirely by fixed ROLL_FE_TO_AA segments), so mount orientation already
determines every axis. They would be needed to describe a hand this generator
cannot make -- SHARPA or Allegro -- and that is the extension point if one
policy ever has to span the fixed and generated families.

Also not included: gains, armature and densities. They are fixed per tier across
the whole population, so they carry no information that distinguishes designs.
"""

from __future__ import annotations

import math

import numpy as np

from hand_sampler import params as P
from hand_sampler import sharpa_anchors as anchors
from hand_sampler.rotations import rpy_to_rot6d as _rpy_to_rot6d

# Tier order used by every per-link block below.
TIERS: tuple[str, ...] = ("mc", "pp", "mp", "dp")

PER_FINGER_DIM = 3 + 6 + 4 + 4 + 3 + P.N_JOINT_SLOTS + 1 + 1   # 28
DESCRIPTOR_DIM = P.N_FINGER_SLOTS * PER_FINGER_DIM + 3         # 143

FIELD_LAYOUT: tuple[tuple[str, int], ...] = (
    ("mount_pos", 3),
    ("mount_rot6d", 6),
    ("link_lengths", 4),
    ("link_radii", 4),
    ("tip_offset", 3),
    ("enabled_mask", P.N_JOINT_SLOTS),
    ("aa_half_range", 1),
    ("active", 1),
)



def finger_descriptor(f: P.FingerParams) -> list[float]:
    """The 28-number block for one finger slot, ghosted or not."""
    lengths = {
        "mc": f.mc.length,
        "pp": f.pp_length,
        "mp": f.mp_length,
        "dp": f.dp_length,
    }
    # Four distinct radii from one sampled knob: radius_scale multiplies each
    # tier's nominal, and the nominals are not equal.
    radii = {t: anchors.TIER_RADIUS_M[t] * f.radius_scale for t in TIERS}

    aa_lo, aa_hi = dict(zip(P.JOINT_SLOTS, f.limits))["MCP_AA"]
    aa_half = 0.5 * (float(aa_hi) - float(aa_lo))

    out: list[float] = []
    out += [float(v) for v in f.mount.xyz]
    out += _rpy_to_rot6d(f.mount.rpy)
    out += [float(lengths[t]) for t in TIERS]
    out += [float(radii[t]) for t in TIERS]
    # The distal cap: the capsule runs along +x from the link origin, so the
    # part that touches the object is one segment length out.
    out += [float(f.dp_length), 0.0, 0.0]
    out += [1.0 if bool(e) else 0.0 for e in f.enabled]
    out += [aa_half]
    out += [1.0 if f.active else 0.0]

    if len(out) != PER_FINGER_DIM:
        raise RuntimeError(
            f"finger descriptor is {len(out)} wide, expected {PER_FINGER_DIM}")
    return out


def hand_descriptor(hand: P.HandParams) -> np.ndarray:
    """Fixed-width descriptor for one design. Shape ``(DESCRIPTOR_DIM,)``.

    Ghosted fingers still occupy their slot -- the width cannot depend on how
    many fingers a design happens to use, or designs could not share a policy.
    """
    vec: list[float] = []
    for f in hand.fingers:
        vec += finger_descriptor(f)
    vec += [float(v) for v in hand.palm_extents]
    arr = np.asarray(vec, dtype=np.float32)
    if arr.shape != (DESCRIPTOR_DIM,):
        raise RuntimeError(
            f"hand descriptor is {arr.shape}, expected ({DESCRIPTOR_DIM},)")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{hand.name}: non-finite morphology descriptor")
    return arr


def population_descriptors(hands) -> np.ndarray:
    """Stack one descriptor per design. Shape ``(k, DESCRIPTOR_DIM)``."""
    return np.stack([hand_descriptor(h) for h in hands]).astype(np.float32)


def describe_layout() -> str:
    """Human-readable field map, for logs and for debugging an obs vector."""
    lines, off = [], 0
    for slot in range(P.N_FINGER_SLOTS):
        for name, width in FIELD_LAYOUT:
            lines.append(f"  [{off:>3}:{off + width:>3}) finger{slot}.{name}")
            off += width
    lines.append(f"  [{off:>3}:{off + 3:>3}) palm_extents")
    return "\n".join(lines)


__all__ = [
    "DESCRIPTOR_DIM",
    "PER_FINGER_DIM",
    "FIELD_LAYOUT",
    "hand_descriptor",
    "population_descriptors",
    "finger_descriptor",
    "describe_layout",
]
