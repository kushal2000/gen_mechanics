"""Parameter table for the hand-kinematics grammar (iteration 3).

This module is the *only* place sampling ranges/choices live. ``rules.py``
defines the typed productions; ``derive.py`` walks them. Every function here
takes an explicit ``numpy.random.Generator`` (never seeds a generator
itself), so callers control reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

DEG = math.pi / 180.0

# Angle grid used everywhere a rotation/orientation is sampled (mount pose,
# palm-segment direction, joint axis elevation/azimuth): 15 degrees, as the
# existing (design-space) sampler already uses.
ANGLE_STEP_DEG = 15.0
N_ANGLE_STEPS = 360 // int(ANGLE_STEP_DEG)  # 24 choices spanning [-180, 180)
N_ELEVATION_STEPS = 180 // int(ANGLE_STEP_DEG) + 1  # 13 choices spanning [0, 180]


@dataclass(frozen=True)
class Distribution:
    digit_count_range: Tuple[int, int] = (1, 6)
    # Number of palm bodies *in addition to* the root -- the root is always
    # a palm body with its own real segment (see rules.py's RootProduction
    # docstring), so 0 here still yields a hand with a palm, never a
    # palm-less one.
    palm_body_count_range: Tuple[int, int] = (0, 3)
    phalanx_count_range: Tuple[int, int] = (1, 6)

    branch_probability: float = 0.12
    max_branch_digits: int = 2
    # Depth 0 = a top-level digit's own phalanges; a branch digit spawned
    # from one of those is depth 1; a branch of a branch is depth 2; etc.
    # Branching is only offered while the *current* digit's depth is below
    # this bound, so the tree stays finite.
    max_branch_depth: int = 2

    link_length_grid_m: float = 0.005
    link_length_range_m: Tuple[float, float] = (0.015, 0.080)
    palm_length_range_m: Tuple[float, float] = (0.020, 0.080)

    revolute_limit_choices_deg: Tuple[Tuple[float, float], ...] = (
        (-10.0, 90.0),
        (-20.0, 100.0),
        (0.0, 110.0),
        (-30.0, 60.0),
        (-15.0, 75.0),
        (-45.0, 45.0),
    )
    palm_joint_limit_choices_deg: Tuple[Tuple[float, float], ...] = (
        (-20.0, 20.0),
        (-30.0, 30.0),
        (-20.0, 30.0),
        (-30.0, 20.0),
    )
    prismatic_limit_choices_m: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.020),
        (-0.010, 0.010),
        (0.0, 0.030),
    )

    # (kind, weight) — weights need not sum to 1; sampling renormalizes.
    # "Coupled" is dropped and renormalized whenever no earlier movable
    # joint exists yet in the current digit (i.e. at a digit's first
    # phalanx).
    module_probabilities: Tuple[Tuple[str, float], ...] = (
        ("R", 0.60),
        ("C", 0.09),
        ("P", 0.09),
        ("Coupled", 0.22),
    )
    palm_joint_probability: float = 0.55

    coupling_multiplier_choices: Tuple[float, ...] = (1.0, 0.5, 0.75, -0.5, -1.0, 1.5)
    coupling_offset_choices_rad: Tuple[float, ...] = (0.0, 0.1, -0.1, 0.2, -0.2)

    mount_frac_choices: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


DEFAULT_DISTRIBUTION = Distribution()


# --- Low-level grid samplers (each takes an explicit Generator) -------------


def sample_grid_angle_rad(rng: np.random.Generator) -> float:
    k = int(rng.integers(0, N_ANGLE_STEPS))
    return (k * ANGLE_STEP_DEG - 180.0) * DEG


def sample_grid_length_m(rng: np.random.Generator, length_range_m: Tuple[float, float], grid_m: float) -> float:
    lo, hi = length_range_m
    n = int(round((hi - lo) / grid_m))
    k = int(rng.integers(0, n + 1))
    return round(lo + k * grid_m, 10)


def sample_axis(rng: np.random.Generator) -> Tuple[float, float, float]:
    """Unit joint axis on a 15-degree spherical grid, including oblique
    (non-axis-aligned) directions -- not restricted to X/Y/Z."""
    el_k = int(rng.integers(0, N_ELEVATION_STEPS))
    az_k = int(rng.integers(0, N_ANGLE_STEPS))
    el = el_k * ANGLE_STEP_DEG * DEG
    az = (az_k * ANGLE_STEP_DEG - 180.0) * DEG
    x = math.sin(el) * math.cos(az)
    y = math.sin(el) * math.sin(az)
    z = math.cos(el)
    return (float(x), float(y), float(z))


def sample_revolute_limits_rad(rng: np.random.Generator, dist: Distribution) -> Tuple[float, float]:
    lo_deg, hi_deg = dist.revolute_limit_choices_deg[int(rng.integers(0, len(dist.revolute_limit_choices_deg)))]
    return (lo_deg * DEG, hi_deg * DEG)


def sample_palm_joint_limits_rad(rng: np.random.Generator, dist: Distribution) -> Tuple[float, float]:
    lo_deg, hi_deg = dist.palm_joint_limit_choices_deg[int(rng.integers(0, len(dist.palm_joint_limit_choices_deg)))]
    return (lo_deg * DEG, hi_deg * DEG)


def sample_prismatic_limits_m(rng: np.random.Generator, dist: Distribution) -> Tuple[float, float]:
    lo, hi = dist.prismatic_limit_choices_m[int(rng.integers(0, len(dist.prismatic_limit_choices_m)))]
    return (float(lo), float(hi))


def sample_module_kind(rng: np.random.Generator, dist: Distribution, allow_coupled: bool) -> str:
    options = [(k, w) for k, w in dist.module_probabilities if allow_coupled or k != "Coupled"]
    total = sum(w for _, w in options)
    r = float(rng.random()) * total
    acc = 0.0
    for k, w in options:
        acc += w
        if r <= acc:
            return k
    return options[-1][0]


def sample_module(rng: np.random.Generator, dist: Distribution, phalanx_index: int) -> dict:
    """Sample a full module spec (dict, JSON-safe) for a phalanx at index
    ``phalanx_index`` (0-based) within its digit. ``phalanx_index == 0`` can
    never yield ``Coupled`` (no earlier joint in the digit to couple to)."""
    kind = sample_module_kind(rng, dist, allow_coupled=phalanx_index > 0)
    axis = sample_axis(rng)
    if kind == "R":
        return {"kind": "R", "axis": axis, "limits": sample_revolute_limits_rad(rng, dist)}
    if kind == "C":
        return {"kind": "C", "axis": axis}
    if kind == "P":
        return {"kind": "P", "axis": axis, "limits": sample_prismatic_limits_m(rng, dist)}
    # Coupled: source is an earlier phalanx (by relative index) in the same digit.
    source_p = int(rng.integers(0, phalanx_index))
    multiplier = float(dist.coupling_multiplier_choices[int(rng.integers(0, len(dist.coupling_multiplier_choices)))])
    offset = float(dist.coupling_offset_choices_rad[int(rng.integers(0, len(dist.coupling_offset_choices_rad)))])
    return {"kind": "Coupled", "axis": axis, "source_p": source_p, "multiplier": multiplier, "offset": offset}
