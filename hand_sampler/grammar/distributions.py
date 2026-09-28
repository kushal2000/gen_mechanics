"""Parameter table for the hand-kinematics grammar (iteration 3).

This module is the *only* place sampling ranges/choices live. ``rules.py``
defines the typed productions; ``derive.py`` walks them. Every function here
takes an explicit ``numpy.random.Generator`` (never seeds a generator
itself), so callers control reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

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

    # Iteration 7 (M2/geometry): one capsule radius, sampled once per hand and
    # stamped onto every ``Body.radius`` -- see ``rules.py``'s module-level
    # note and ``derive.py``'s ``sample_derivation``/``derive``.
    capsule_radius_choices_m: Tuple[float, ...] = (0.008, 0.010, 0.012)

    # Grammar 0.5 (I16 support-widening, priority 1): the rest-bend
    # primitive. A phalanx's own joint origin (its mount joint, for a
    # digit's first phalanx, or its mid-digit continuation joint otherwise
    # -- see rules.py's convention note / derive.py's ``_process_digit``)
    # may deviate from the existing convention by a small additional
    # rotation (``bend_rpy``, one whole (roll, pitch, yaw) triple drawn from
    # this choice set) and a small lateral (x, y) offset on the host
    # segment (``bend_offset``). The default choice sets contain only the
    # "no bend" value and ``bend_probability`` defaults to 0.0 (never drawn
    # at all -- see ``sample_bend`` below), so default sampling/replay is
    # byte-identical to before this field existed. ``variants.G_BEND`` sets
    # a real 15-degree/5-mm grid and a nonzero probability.
    bend_rpy_choices_rad: Tuple[Tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    bend_offset_choices_m: Tuple[Tuple[float, float], ...] = ((0.0, 0.0),)
    bend_probability: float = 0.0

    # Grammar 0.5 (I16 support-widening, priority 2): continuous,
    # sign-normalised revolute limits. ``revolute_limit_range_deg`` is the
    # range a continuous draw is bounded by; its default (-45, 110) is the
    # union of every ``revolute_limit_choices_deg``/``palm_joint_limit_choices_deg``
    # entry (see ``coverage.global_revolute_limit_range_deg``, which computes
    # this dynamically for other callers -- this field is the static,
    # per-``Distribution`` counterpart ``coverage.py``'s own limit check now
    # reads). ``limits_continuous`` (default False) keeps existing
    # choice-set sampling; when True, ``sample_revolute_limits_rad`` draws a
    # continuous ``(lo, hi)`` pair (``lo < hi``) inside this range instead
    # (see ``sample_revolute_limits_continuous_rad``), and the small-step
    # ``step_limits`` operator (``derive.py``) moves one bound by
    # ``limit_step_deg`` degrees instead of jumping between choice-tuple
    # entries.
    revolute_limit_range_deg: Tuple[float, float] = (-45.0, 110.0)
    limits_continuous: bool = False
    limit_step_deg: float = 15.0

    # I14 fix 5: an optional separate ``Distribution`` used by the GROWTH
    # operators (``add_digit``, ``add_palm_body``, ``regrow_subtree``,
    # ``add_minimal_digit`` -- see ``derive.py``'s ``_growth_dist``) to
    # sample brand-NEW material, instead of the (outer) ``Distribution`` a
    # ``vary`` call is otherwise parameterized by. ``None`` (the default)
    # means "no separate insertion distribution": every growth operator
    # samples new material from the same outer ``Distribution``, exactly as
    # before this fix (this is what every existing named ``Distribution`` --
    # ``G_FULL``, ``G_SERIAL``, etc. -- still does, so default sampling and
    # every existing replay/test is byte-identical). A caller-level cap
    # (``digit_count_range``, ``palm_body_count_range``) is always read off
    # the OUTER distribution, never the insertion one, since those bound the
    # whole hand, not one newly-inserted piece. See ``variants.py``'s
    # ``G_FULL_INS``/``G_NOBRANCH_INS`` for a concrete insertion
    # distribution (phalanx_count_range=(1, 3), branch_probability=0.0).
    insertion: Optional["Distribution"] = None

    # G0 CPU grammar screen (plan revision 2026-09-27, step 2; I29/I30):
    # three optional, off-by-default generative rules the screen's V2/V3
    # variants turn on. Every one of them defaults to "off" in a way that
    # leaves ``sample_derivation``'s RNG-stream and output byte-identical to
    # before these fields existed for every EXISTING named ``Distribution``
    # (``G_FULL``, ``G_SERIAL``, etc.) -- see each field's own docstring and
    # ``derive.py``'s corresponding hook for the exact "off means identical"
    # argument.

    # I29 mount-spacing rule (V2): when set, top-level digit mounts (host
    # body + ``mount_frac``) are PLANNED (``derive._plan_top_level_mounts``)
    # to spread digits across ``mount_bodies`` and, within one host, across
    # ``mount_frac_choices``, instead of drawn i.i.d. per digit -- a
    # generative placement rule, not a rejection filter: it always returns
    # exactly as many (host, frac) pairs as there are digits, even when the
    # target below cannot be met by the discrete grid. The value is the
    # TARGET minimum physical separation (m) between two mounts sharing one
    # host, e.g. ``2 * capsule_radius_m + margin_m``. ``None`` (default)
    # disables the planner entirely: ``sample_derivation`` draws each
    # digit's mount exactly as it always has.
    mount_min_separation_m: Optional[float] = None

    # I30 curl-axis prior (V3): restricts a DIGIT phalanx's revolute-module
    # axis (``distributions.sample_module``, never a palm joint's own axis)
    # to this ``(lo_deg, hi_deg)`` elevation band from +z (see
    # ``sample_axis``'s ``elevation_band_deg`` argument) -- a band centered
    # on 90 deg keeps the axis roughly IN the plane transverse to the
    # segment's own forward (+z) direction, i.e. hinge-like, so successive
    # phalanges bend within a shared plane instead of a fully isotropic 3D
    # axis (which is what makes a random hand's rest curl point in an
    # incoherent direction per finger -- I30). ``None`` (default) disables
    # the band: ``sample_axis`` draws the elevation uniformly over the full
    # grid exactly as before this field existed.
    digit_axis_elevation_band_deg: Optional[Tuple[float, float]] = None

    # I30 opposition prior (V3): when ``True`` and a hand has >= 2 top-level
    # digits, the LAST-sampled digit's mount orientation is chosen (not
    # drawn) to oppose the mean forward direction of the earlier digits'
    # mounts (``derive._best_opposing_rpy``), a thumb-like opposition rather
    # than an independently random one. ``False`` (default) disables this:
    # every digit's ``mount_rpy`` is drawn i.i.d. exactly as before this
    # field existed.
    opposition_prior: bool = False

    # Opus review of G0 (opus-review-g0.md, item 2 / I39), V3s fix -- three
    # more optional, off-by-default fields. Every one leaves V1/V2/V3 (and
    # every other existing named ``Distribution``) byte-identical: each is
    # read only at a site gated on it being non-default, so an untouched
    # field never perturbs the RNG stream or the derived geometry.

    # Surface mounting (review item 2): when ``True``, a digit's mount
    # origin (top-level OR branch) is offset OFF the host's own centre axis
    # by ``host_radius_m`` (the hand's one ``capsule_radius_m``, passed down
    # by ``derive.py`` -- the host's own capsule radius) at an azimuth angle
    # ("the rule": either drawn i.i.d. from the same 15-degree grid
    # ``sample_grid_angle_rad`` uses, when no placement plan supplies one, or
    # the placement plan's own chosen azimuth -- see
    # ``derive._plan_top_level_mounts_surface``), instead of sitting ON the
    # host's centre axis (``(0, 0, mount_frac * host_length)``). ``False``
    # (default) keeps every digit mount exactly on-axis, byte-identical to
    # before this field existed -- see ``derive._emit_digit``.
    mount_on_host_surface: bool = False

    # Cross-host spacing (review item 2), dispatched together with
    # ``mount_on_host_surface``: when BOTH it and ``mount_min_separation_m``
    # (declared above, reused as the SAME target -- V2's own dispatch path
    # is unaffected: it is reached only when ``mount_on_host_surface`` is
    # ``False``) are set, top-level digit mounts are planned by
    # ``derive._plan_top_level_mounts_surface`` -- a greedy furthest-point
    # placement over every (host, mount_frac, azimuth) grid point's ACTUAL
    # 3-D position (via a scratch forward-kinematics pass over the
    # already-sampled root/palm bodies, ``derive._host_transforms_from_steps``),
    # so mounts on DIFFERENT hosts are spaced apart too, not just same-host
    # mounts (the existing ``_plan_top_level_mounts``/V2 planner only
    # reasons about same-host axial spacing). Using the surface azimuth (not
    # just the axial ``mount_frac``) is what lets a short host (root length
    # as low as 20 mm) still separate 4-5 digits: two mounts diametrically
    # opposite in azimuth are already ``2 * host_radius_m`` apart before any
    # axial offset at all.

    # Curl-axis prior (review item 5 / V3s fix): when ``True``, a digit's
    # FIRST phalanx (index 0 -- the one whose joint origin composes with the
    # digit's own ``mount_rpy``, see ``derive._compose_bend_rpy``) never
    # receives a rest-bend, regardless of ``bend_probability``/
    # ``bend_rpy_choices_rad`` -- every phalanx AFTER the first still bends
    # normally. V3's bug (opus-review-g0.md item 5): bending phalanx 0 tilts
    # the digit's own MOUNT frame (not just its own shape), which silently
    # invalidates the opposition prior's premise that ``mount_rpy`` is the
    # digit's true rest-pose forward direction. ``False`` (default) leaves
    # ``G_BEND`` (which deliberately bends phalanx 0 too, per I16/I11 -- a
    # real mid-chain frame rotation) and V3 exactly as before this field
    # existed.
    curl_skip_first_phalanx: bool = False

    # Opposition prior host frame fix (review item 5 / V3s fix): when
    # ``True`` (and ``opposition_prior`` is also ``True``), the mean-forward
    # direction of the earlier digits, and the last digit's own opposing
    # target, are computed in the ROOT frame -- each digit's own HOST body's
    # accumulated rotation (``derive._host_transforms_from_steps``) is
    # applied to its local ``mount_rpy`` forward direction before averaging
    # or opposing -- instead of treating every digit's ``mount_rpy`` as if
    # it were already expressed in a shared frame. V3's bug: two digits
    # mounted on DIFFERENT hosts (root vs. a rotated palm body) have
    # ``mount_rpy`` in DIFFERENT local frames, so naively averaging them is
    # only correct for root-only hands -- exactly the "105 deg in mixed-host
    # hands" case the review measured. ``False`` (default) leaves V3 exactly
    # as before this field existed.
    opposition_use_host_frame: bool = False


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


def sample_axis(
    rng: np.random.Generator, elevation_band_deg: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float]:
    """Unit joint axis on a 15-degree spherical grid, including oblique
    (non-axis-aligned) directions -- not restricted to X/Y/Z.

    ``elevation_band_deg`` (I30 curl-axis prior, default ``None``): when
    given, the elevation grid index is drawn only from grid points whose
    angle (from +z) falls inside ``(lo_deg, hi_deg)`` -- e.g. ``(60, 120)``
    keeps the axis roughly transverse to +z (hinge-like). Falls back to the
    full grid if the band contains no grid point. When ``None`` (every
    existing call site), this draws EXACTLY the same two ``rng.integers``
    calls, in the same order, as before this argument existed -- the RNG
    stream is untouched."""
    if elevation_band_deg is None:
        el_k = int(rng.integers(0, N_ELEVATION_STEPS))
    else:
        lo_deg, hi_deg = elevation_band_deg
        candidates = [k for k in range(N_ELEVATION_STEPS) if lo_deg <= k * ANGLE_STEP_DEG <= hi_deg]
        if not candidates:
            candidates = list(range(N_ELEVATION_STEPS))
        el_k = candidates[int(rng.integers(0, len(candidates)))]
    az_k = int(rng.integers(0, N_ANGLE_STEPS))
    el = el_k * ANGLE_STEP_DEG * DEG
    az = (az_k * ANGLE_STEP_DEG - 180.0) * DEG
    x = math.sin(el) * math.cos(az)
    y = math.sin(el) * math.sin(az)
    z = math.cos(el)
    return (float(x), float(y), float(z))


def sample_revolute_limits_continuous_rad(rng: np.random.Generator, dist: Distribution) -> Tuple[float, float]:
    """A continuous ``(lo, hi)`` pair, ``lo < hi``, drawn uniformly within
    ``dist.revolute_limit_range_deg`` (in radians). Two independent uniform
    draws, sorted -- the degenerate ``a == b`` case (probability ~0 for a
    continuous draw) is nudged apart so ``lo < hi`` always holds."""
    lo_deg, hi_deg = dist.revolute_limit_range_deg
    a = float(rng.uniform(lo_deg, hi_deg))
    b = float(rng.uniform(lo_deg, hi_deg))
    if a > b:
        a, b = b, a
    if a == b:
        b = min(hi_deg, a + 1e-9)
        if a == b:
            a = max(lo_deg, a - 1e-9)
    return a * DEG, b * DEG


def sample_revolute_limits_rad(rng: np.random.Generator, dist: Distribution) -> Tuple[float, float]:
    if dist.limits_continuous:
        return sample_revolute_limits_continuous_rad(rng, dist)
    lo_deg, hi_deg = dist.revolute_limit_choices_deg[int(rng.integers(0, len(dist.revolute_limit_choices_deg)))]
    return (lo_deg * DEG, hi_deg * DEG)


def sample_bend(rng: np.random.Generator, dist: Distribution) -> Tuple[Tuple[float, float, float], Tuple[float, float]]:
    """``(bend_rpy, bend_offset)`` for one Phalanx step -- see
    ``Distribution.bend_probability``'s docstring. Never touches ``rng`` at
    all when ``dist.bend_probability <= 0.0`` (true of every existing named
    ``Distribution``), so every existing seed's default-sampling replay
    (RNG-stream position included) is exactly as before this field
    existed."""
    if dist.bend_probability <= 0.0:
        return (0.0, 0.0, 0.0), (0.0, 0.0)
    if float(rng.random()) < dist.bend_probability:
        bend_rpy = dist.bend_rpy_choices_rad[int(rng.integers(0, len(dist.bend_rpy_choices_rad)))]
        bend_offset = dist.bend_offset_choices_m[int(rng.integers(0, len(dist.bend_offset_choices_m)))]
        return tuple(float(v) for v in bend_rpy), tuple(float(v) for v in bend_offset)
    return (0.0, 0.0, 0.0), (0.0, 0.0)


def sample_palm_joint_limits_rad(rng: np.random.Generator, dist: Distribution) -> Tuple[float, float]:
    lo_deg, hi_deg = dist.palm_joint_limit_choices_deg[int(rng.integers(0, len(dist.palm_joint_limit_choices_deg)))]
    return (lo_deg * DEG, hi_deg * DEG)


def sample_prismatic_limits_m(rng: np.random.Generator, dist: Distribution) -> Tuple[float, float]:
    lo, hi = dist.prismatic_limit_choices_m[int(rng.integers(0, len(dist.prismatic_limit_choices_m)))]
    return (float(lo), float(hi))


def sample_capsule_radius_m(rng: np.random.Generator, dist: Distribution) -> float:
    return float(dist.capsule_radius_choices_m[int(rng.integers(0, len(dist.capsule_radius_choices_m)))])


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


def sample_module(rng: np.random.Generator, dist: Distribution, phalanx_index: int,
                   revolute_source_indices: Optional[Tuple[int, ...]] = None) -> dict:
    """Sample a full module spec (dict, JSON-safe) for a phalanx at index
    ``phalanx_index`` (0-based) within its digit.

    Coupling-source rule: a ``Coupled`` module may only take a *revolute*
    source, never a continuous or prismatic one (a continuous joint has no
    fixed extent to take an affine image of, and coupling to a prismatic
    joint's meters-valued range would silently reinterpret it as radians).
    ``revolute_source_indices`` is the (0-based, within this digit) set of
    earlier phalanx indices whose own module is already known to be ``"R"``
    -- the only valid ``source_p`` choices. ``Coupled`` is dropped and
    renormalized (like every other kind gated on ``phalanx_index == 0``)
    whenever this set is empty, e.g. at a digit's first phalanx, or when no
    earlier phalanx happens to be revolute. Callers that have not yet
    decided earlier phalanges' kinds (there are none, at ``phalanx_index ==
    0``) may omit it; it then defaults to ``range(phalanx_index)``, which is
    only correct when the caller already knows every earlier phalanx is
    revolute (true only for ``phalanx_index == 0``, where the range is
    empty either way) -- every other caller must pass the real set."""
    if revolute_source_indices is None:
        revolute_source_indices = tuple(range(phalanx_index))
    kind = sample_module_kind(rng, dist, allow_coupled=phalanx_index > 0 and len(revolute_source_indices) > 0)
    axis = sample_axis(rng, dist.digit_axis_elevation_band_deg)
    if kind == "R":
        return {"kind": "R", "axis": axis, "limits": sample_revolute_limits_rad(rng, dist)}
    if kind == "C":
        return {"kind": "C", "axis": axis}
    if kind == "P":
        return {"kind": "P", "axis": axis, "limits": sample_prismatic_limits_m(rng, dist)}
    # Coupled: source is an earlier, revolute-only phalanx in the same digit
    # (see the coupling-source rule above).
    source_p = revolute_source_indices[int(rng.integers(0, len(revolute_source_indices)))]
    multiplier = float(dist.coupling_multiplier_choices[int(rng.integers(0, len(dist.coupling_multiplier_choices)))])
    offset = float(dist.coupling_offset_choices_rad[int(rng.integers(0, len(dist.coupling_offset_choices_rad)))])
    return {"kind": "Coupled", "axis": axis, "source_p": source_p, "multiplier": multiplier, "offset": offset}
