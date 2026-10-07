"""Sampling and (pure, deterministic) derivation for the hand-kinematics
grammar (iteration 3).

``sample_derivation`` is the only place randomness happens; it produces a
``Derivation`` -- a flat, ordered trace of every production applied and
every parameter value sampled. ``derive`` is pure (no RNG) and rebuilds the
identical ``KinematicModel`` from that trace, so ``replay`` (calling
``derive`` twice on the same derivation, or deriving two derivations
sampled from the same seed) is exact.

No production here ever accepts a raw model or URDF: models are built only
through ``derive`` walking productions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .coords import CONTINUOUS_SAMPLE_RANGE
from .distributions import (
    ANGLE_STEP_DEG,
    DEFAULT_DISTRIBUTION,
    DEG,
    Distribution,
    N_ANGLE_STEPS,
    N_ELEVATION_STEPS,
    sample_axis,
    sample_bend,
    sample_capsule_radius_m,
    sample_grid_angle_rad,
    lateral_offset_choices_m,
    sample_grid_length_m,
    sample_lateral_offset,
    sample_module,
    sample_palm_joint_limits_rad,
    sample_revolute_limits_rad,
)
from .fk import forward_kinematics, matrix_to_rpy, rpy_to_matrix
from .kinematics import (
    AffineCoupling,
    Body,
    Frame,
    Joint,
    KinematicModel,
    ModelError,
    Pose,
    validate,
)
from .limits import GenerationLimits, LimitContext, Structure
from .limits import context as _limit_context
from .rules import GRAMMAR_VERSION

DERIVATION_SCHEMA = "hand_grammar_derivation/0.1"


class VariationImpossible(Exception):
    """Raised by ``vary`` when the requested operator has no valid
    application after 32 attempts (e.g. ``remove_digit`` at 1 digit)."""


class DerivationError(ModelError):
    """Raised by ``derive`` when ``validate_derivation`` finds the
    *derivation itself* malformed -- before any attempt is made to build a
    ``KinematicModel`` from it. A ``ModelError`` subclass (same ``issues``
    list interface), so every existing ``except ModelError`` catch site
    (e.g. ``vary``'s own retry loop) already handles it."""


@dataclass(frozen=True)
class DerivationStep:
    path: str
    production: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class Derivation:
    seed: int
    grammar_version: str
    steps: Tuple[DerivationStep, ...]
    # Provenance of a ``vary``-produced derivation: an ordered tuple of
    # (operator, parent_seed) entries, one per ``vary`` application, oldest
    # first. Empty for a freshly sampled derivation. ``seed`` itself is
    # always the *founder* seed (the seed originally passed to
    # ``sample_derivation``) and is never changed by ``vary`` -- a varied
    # derivation is replayed by re-running its own stored ``steps`` through
    # ``derive``, never by resampling from ``seed`` again, so ``lineage`` is
    # the only record of which operators were applied and in what order;
    # ``seed`` alone does not recover it.
    lineage: Tuple[Tuple[str, int], ...] = ()


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def _coerce_rng(rng_or_seed):
    if isinstance(rng_or_seed, np.random.Generator):
        return -1, rng_or_seed
    seed = int(rng_or_seed)
    return seed, np.random.default_rng(seed)


def _mount_bodies_from_steps(steps) -> List[str]:
    hand = next(s for s in steps if s.path == "hand")
    n = hand.params["palm_body_count"]
    return ["root"] + [f"palm{i}" for i in range(n)]


def _step_digit_id(st: "DerivationStep") -> Optional[str]:
    if st.production in ("Digit", "Phalanx"):
        return st.params["digit_id"]
    return None


def _is_descendant_digit(host_id: str, candidate_id: Optional[str]) -> bool:
    """True if ``candidate_id`` is ``host_id`` itself or a branch digit nested
    (at any depth) under one of ``host_id``'s phalanges. Branch ids are always
    ``f"{parent_id}p{phalanx}b{slot}"``, so a ``host_id + "p"`` prefix match is
    exact: top-level ids are plain integers and never contain ``"p"``, so they
    can never collide with this pattern."""
    if candidate_id is None:
        return False
    return candidate_id == host_id or candidate_id.startswith(host_id + "p")


def _revolute_source_indices(steps, digit_id: str, upto: int) -> Tuple[int, ...]:
    """0-based indices < ``upto`` of ``digit_id``'s own Phalanx steps (already
    present in ``steps``) whose module is revolute ("R") -- the only valid
    ``Coupled`` source per the coupling-source rule (see
    ``distributions.sample_module``)."""
    return tuple(
        sorted(
            s.params["p"] for s in steps
            if s.production == "Phalanx" and s.params["digit_id"] == digit_id
            and s.params["p"] < upto and s.params["module"]["kind"] == "R"
        )
    )


def _max_uid(steps: Sequence[DerivationStep]) -> int:
    """Highest ``uid`` (I15 fix 1) already stamped on any step's ``params``
    (-1 if none). ``uid`` lives INSIDE ``params`` (not as a separate
    ``DerivationStep`` field) so every existing in-place-edit operator
    (``p = dict(s.params); p[...] = ...; DerivationStep(..., params=p)``)
    already preserves it for free via the ``dict(s.params)`` copy -- only
    the handful of sites that create a genuinely NEW step need to consult
    this to allocate a fresh one."""
    best = -1
    for s in steps:
        u = s.params.get("uid")
        if isinstance(u, int) and u > best:
            best = u
    return best


def _plan_top_level_mounts(
    rng, dist: Distribution, mount_bodies: List[str], digit_count: int, host_length: Dict[str, float],
    lim: Optional[LimitContext] = None,
) -> List[Tuple[str, float]]:
    """G0 screen, I29 mount-spacing rule (V2): ``digit_count`` (host,
    mount_frac) pairs for the top-level digits, spread across
    ``mount_bodies`` and, within a shared host, across
    ``dist.mount_frac_choices``, so neighbouring same-host mounts are as
    physically separated (given that host's OWN sampled length) as the
    discrete frac grid allows -- targeting ``dist.mount_min_separation_m``.
    A placement rule, not a rejection: always returns exactly
    ``digit_count`` pairs, even when the target cannot be met (more digits
    than a short host's grid can space out), in which case digits are
    spread as far apart as the grid allows (best effort, never retried).
    Only called when ``dist.mount_min_separation_m is not None`` -- see
    ``sample_derivation``."""
    frac_choices = sorted(set(dist.mount_frac_choices))
    n_frac = len(frac_choices)
    min_sep = dist.mount_min_separation_m

    def capacity(host: str) -> int:
        length = host_length.get(host, 0.0)
        if length <= 0.0 or n_frac <= 1:
            return 1
        count = 1
        last = frac_choices[0]
        for f in frac_choices[1:]:
            if (f - last) * length >= min_sep:
                count += 1
                last = f
        return max(1, count)

    caps = {h: capacity(h) for h in mount_bodies}
    counts = {h: 0 for h in mount_bodies}
    for _ in range(digit_count):
        # Generation limits: only hosts with room for another top-level digit
        # (every host when ``lim`` is None or does not bind).
        pool = mount_bodies if lim is None else (lim.eligible_hosts(mount_bodies) or mount_bodies)
        host = max(pool, key=lambda h: (caps[h] - counts[h], -mount_bodies.index(h)))
        counts[host] += 1
        if lim is not None:
            lim.take_host(host)

    assignments: List[Tuple[str, float]] = []
    for host in mount_bodies:
        k = counts[host]
        if k == 0:
            continue
        if k == 1:
            idx = int(rng.integers(0, n_frac))
            fracs = [frac_choices[idx]]
        else:
            positions = np.linspace(0, n_frac - 1, k)
            fracs = [frac_choices[int(round(pos))] for pos in positions]
        assignments.extend((host, f) for f in fracs)

    order = rng.permutation(len(assignments))
    return [assignments[i] for i in order]


def _host_transforms_from_steps(steps: Sequence[DerivationStep], root_length: float) -> Dict[str, np.ndarray]:
    """``{host_body_name: 4x4 root-frame transform}`` for ``"root"`` plus
    every ``PalmBody`` step already present in ``steps`` -- the REST-POSE
    (q=0) transform each host body would have once ``derive()`` builds it,
    computed here from a scratch, throwaway ``KinematicModel`` containing
    only those bodies (no digits: they are not sampled yet at the point
    ``sample_derivation`` needs this, see ``_plan_top_level_mounts_surface``).
    Exact, not approximate: every palm joint contributes identity rotation
    at q=0 regardless of type (revolute or fixed), so this is precisely the
    same rest transform ``forward_kinematics`` would report for these bodies
    off the FULL derivation. Used only to PLAN digit mount positions (G0
    screen review item 2, V3s' cross-host spacing) -- never exposed outside
    ``derive.py``, never stored in the derivation itself."""
    bodies: List[Body] = [Body(name="root")]
    joints: List[Joint] = []
    body_length: Dict[str, float] = {"root": root_length}
    for s in steps:
        if s.production != "PalmBody":
            continue
        p = s.params
        name = p["name"]
        parent = p["parent"]
        mount_offset = p.get("mount_offset", (0.0, 0.0))
        base_xyz = (mount_offset[0], mount_offset[1], p["mount_frac"] * body_length[parent])
        jtype = "revolute" if p["has_joint"] else "fixed"
        axis = tuple(p["axis"]) if p["has_joint"] else (1.0, 0.0, 0.0)
        limits = tuple(p["limits"]) if p["has_joint"] else None
        joints.append(Joint(
            name=f"{name}_j", type=jtype, parent=parent, child=name,
            origin=Pose(xyz=base_xyz, rpy=tuple(p["direction_rpy"])), axis=axis, limits=limits,
        ))
        bodies.append(Body(name=name))
        body_length[name] = p["length"]
    model = KinematicModel(name="_host_probe", root="root", bodies=tuple(bodies), joints=tuple(joints),
                            frames=(), couplings=())
    transforms = forward_kinematics(model, {})
    return {name: transforms[name] for name in body_length}


def _plan_top_level_mounts_surface(
    rng, dist: Distribution, mount_bodies: List[str], digit_count: int, host_length: Dict[str, float],
    host_radius_m: float, host_transforms: Dict[str, np.ndarray], lim: Optional[LimitContext] = None,
) -> List[Tuple[str, float, float]]:
    """G0 screen (opus-review-g0.md item 2), V3s cross-host spacing:
    ``digit_count`` (host, mount_frac, azimuth_rad) triples for the
    top-level digits, chosen by a greedy furthest-point search over EVERY
    (host, frac, azimuth) grid point's actual root-frame 3-D position --
    azimuth on the same 15-degree grid ``sample_grid_angle_rad`` draws from,
    position ``host_transforms[host] @ (host_radius_m*cos(az),
    host_radius_m*sin(az), frac*host_length[host], 1)`` (the mount's surface
    point, see ``_emit_digit``) -- so mounts on DIFFERENT hosts (via
    ``host_transforms``, ``_host_transforms_from_steps``) are spaced apart
    just as much as same-host mounts, unlike ``_plan_top_level_mounts``
    (same-host axial spacing only). A placement rule, not a rejection:
    always returns exactly ``digit_count`` triples. The first point is
    picked uniformly at random (``rng``); each subsequent point is the
    remaining candidate maximizing its minimum distance to every
    already-chosen point (ties broken by ``rng``) -- a standard greedy
    farthest-point placement, never retried/rejected. Only called when
    ``dist.mount_on_host_surface and dist.mount_min_separation_m is not
    None`` -- see ``sample_derivation``."""
    frac_choices = sorted(set(dist.mount_frac_choices))
    az_choices = [k * ANGLE_STEP_DEG * DEG for k in range(N_ANGLE_STEPS)]

    candidates: List[Tuple[str, float, float]] = []
    positions: List[np.ndarray] = []
    for host in mount_bodies:
        T = host_transforms[host]
        R, t = T[:3, :3], T[:3, 3]
        length = host_length.get(host, 0.0)
        for f in frac_choices:
            for az in az_choices:
                local = np.array([host_radius_m * math.cos(az), host_radius_m * math.sin(az), f * length])
                candidates.append((host, f, az))
                positions.append(R @ local + t)
    pos = np.stack(positions)

    remaining = list(range(len(candidates)))
    chosen: List[int] = []

    def eligible() -> List[int]:
        # Generation limits: only grid points on hosts with room for another
        # top-level digit. Without limits (or when they do not bind) this is
        # ``remaining`` itself, so every draw below is unchanged.
        if lim is None:
            return remaining
        room = {h: lim.host_has_room(h) for h in mount_bodies}
        out = [r for r in remaining if room[candidates[r][0]]]
        return out if len(out) < len(remaining) else remaining

    def take(r: int) -> None:
        remaining.remove(r)
        chosen.append(r)
        if lim is not None:
            lim.take_host(candidates[r][0])

    pool = eligible()
    take(pool[int(rng.integers(0, len(pool)))])
    while len(chosen) < digit_count and remaining:
        pool = eligible()
        if not pool:
            break
        chosen_pos = pos[chosen]
        dists = np.array([np.min(np.linalg.norm(chosen_pos - pos[ridx], axis=1)) for ridx in pool])
        best = float(dists.max())
        best_local = [k for k, d in enumerate(dists) if d >= best - 1e-9]
        pick = best_local[int(rng.integers(0, len(best_local)))]
        take(pool[pick])
    # The grid always has >= n_frac * n_azimuth (>= 5*24=120) points per
    # host; digit_count is capped at 5 by the envelope, so this fallback
    # (repeat the last chosen point) is unreachable in practice but keeps
    # the "always exactly digit_count triples" contract total.
    while len(chosen) < digit_count:
        chosen.append(chosen[-1])

    order = rng.permutation(len(chosen))
    return [candidates[chosen[i]] for i in order]


def _snap_to_angle_grid_rad(angle_rad: float) -> float:
    """Nearest point (radians) on the same 24-point, 15-degree grid
    ``sample_grid_angle_rad`` draws from (``{k * 15 - 180 : k in 0..23}``,
    i.e. ``[-180, 165]`` degrees, wrapping)."""
    deg = math.degrees(angle_rad) % 360.0
    k = int(round((deg + 180.0) / ANGLE_STEP_DEG)) % N_ANGLE_STEPS
    return (k * ANGLE_STEP_DEG - 180.0) * DEG


def _best_opposing_rpy(oppose_forward: np.ndarray) -> Tuple[float, float, float]:
    """G0 screen, I30 opposition prior (V3): a ``(roll=0, pitch, yaw)``
    triple, snapped onto the same 15-degree grid ``sample_grid_angle_rad``
    draws from, whose local +z direction (``rpy_to_matrix(rpy) @ (0,0,1)``)
    most nearly opposes ``oppose_forward`` (a unit vector). Solved
    analytically, not searched: with ``roll = 0``, ``rpy_to_matrix((0,
    pitch, yaw)) @ (0,0,1) == (cos(yaw) sin(pitch), sin(yaw) sin(pitch),
    cos(pitch))`` -- the usual spherical-coordinates parametrisation of a
    unit vector by colatitude ``pitch`` and azimuth ``yaw`` -- so the exact
    (unsnapped) ``pitch``/``yaw`` recovering the target direction
    ``-oppose_forward`` are ``acos(target_z)`` and ``atan2(target_y,
    target_x)``. A pure, deterministic computation (no RNG, no grid
    search, no retry), so the result stays exactly reproducible from the
    derivation that already fixed every earlier digit's own mount
    orientation."""
    target = -np.asarray(oppose_forward, dtype=float)
    norm = float(np.linalg.norm(target))
    if norm < 1e-9:
        return (0.0, 0.0, 0.0)
    target = target / norm
    pitch = math.acos(float(np.clip(target[2], -1.0, 1.0)))
    sp = math.sin(pitch)
    yaw = math.atan2(float(target[1]), float(target[0])) if sp > 1e-9 else 0.0
    return (0.0, _snap_to_angle_grid_rad(pitch), _snap_to_angle_grid_rad(yaw))


def _sample_phalanx(rng, dist: Distribution, steps: List[DerivationStep], digit_id: str, p: int,
                     depth: int, is_last: bool, next_uid: List[int], host_radius_m: float = 0.0,
                     lim: Optional[LimitContext] = None) -> None:
    """``lim`` (generation limits, default ``None``): module kinds come from
    the allowed set, and branch digits are spawned only when branches are
    allowed and the digit's joint budget has room for them (each branch digit
    needs at least ``phalanx_count_range[0]`` joints, reserved up front so
    sibling branches always fit). ``None`` draws exactly as before."""
    module = sample_module(rng, dist, p, _revolute_source_indices(steps, digit_id, p),
                           allowed=None if lim is None else lim.allowed_modules)
    length = sample_grid_length_m(rng, dist.link_length_range_m, dist.link_length_grid_m)
    # Opus review of G0 (item 5 / V3s fix): ``curl_skip_first_phalanx`` never
    # touches ``rng`` for phalanx 0 -- it must not draw (and discard) a bend
    # that would otherwise have been sampled, since that would desync the
    # RNG stream from every phalanx AFTER it relative to the same field
    # being off. Phalanx 0's own origin composes with the digit's
    # ``mount_rpy`` (see ``_compose_bend_rpy``), so bending it would tilt
    # the mount frame the opposition prior already committed to -- every
    # later phalanx (a plain continuation joint) still bends normally.
    if dist.curl_skip_first_phalanx and p == 0:
        bend_rpy, bend_offset = (0.0, 0.0, 0.0), (0.0, 0.0)
    else:
        bend_rpy, bend_offset = sample_bend(rng, dist)
    branch_digit_count = 0
    if depth < dist.max_branch_depth and float(rng.random()) < dist.branch_probability:
        # A phalanx's body must end up with >= 2 child joints for this to be
        # real branching (measured structurally, see the support-audit /
        # structural-validity assertions). A non-last phalanx already gets a
        # "next phalanx" child, so 1 branch digit suffices; the *last*
        # phalanx in a digit has no next-phalanx child, so it needs >= 2
        # branch digits on its own (skipped entirely if the distribution's
        # ``max_branch_digits`` can't reach 2).
        min_branches = 2 if is_last else 1
        max_branches = dist.max_branch_digits
        if lim is not None:
            max_branches = lim.branch_cap(max_branches, _phalanx_unit(dist))
        if min_branches <= max_branches:
            branch_digit_count = int(rng.integers(min_branches, max_branches + 1))
    uid = next_uid[0]
    next_uid[0] += 1
    steps.append(DerivationStep(path=f"digit/{digit_id}/phalanx/{p}", production="Phalanx", params={
        "digit_id": digit_id, "p": p, "module": module, "length": length,
        "branch_digit_count": branch_digit_count, "uid": uid,
        "bend_rpy": bend_rpy, "bend_offset": bend_offset,
    }))
    if branch_digit_count:
        # The branch mounts on THIS phalanx's own body (distal to its
        # joint), never on a palm body, so that body genuinely gets >= 2
        # child joints (its own next phalanx plus one per branch digit).
        phalanx_body = f"d{digit_id}p{p + 1}"
        if lim is not None:
            lim.reserve(branch_digit_count * _phalanx_unit(dist))
        for b in range(branch_digit_count):
            sub_id = f"{digit_id}p{p + 1}b{b}"
            if lim is not None:
                lim.release(_phalanx_unit(dist))
            _emit_digit(rng, dist, steps, sub_id, [phalanx_body], top_level=False, depth=depth + 1,
                        next_uid=next_uid, host_radius_m=host_radius_m, lim=lim)


def _emit_digit(rng, dist: Distribution, steps: List[DerivationStep], digit_id: str,
                 mount_bodies: List[str], top_level: bool, depth: int, next_uid: List[int],
                 forced_mount: Optional[Tuple] = None,
                 oppose_forward: Optional[np.ndarray] = None,
                 host_radius_m: float = 0.0,
                 host_transforms: Optional[Dict[str, np.ndarray]] = None,
                 lim: Optional[LimitContext] = None) -> None:
    """``forced_mount``/``oppose_forward`` (G0 screen, I29/I30; both default
    ``None``) are used ONLY by ``sample_derivation``'s top-level digit loop
    when the corresponding prior is enabled -- every other caller (branch
    digits, growth operators) omits both, taking the exact same i.i.d.
    (host, frac) and random ``mount_rpy`` draws as before either argument
    existed. ``forced_mount`` is a ``(host, frac)`` pair (old V2 planner,
    ``_plan_top_level_mounts``) or a ``(host, frac, azimuth_rad)`` triple
    (V3s' surface planner, ``_plan_top_level_mounts_surface``) -- the third
    element, when present, is the PLANNED azimuth (skips the i.i.d. draw
    below).

    ``host_radius_m`` (opus review of G0 item 2 / V3s fix, default 0.0):
    the hand's own ``capsule_radius_m`` -- read only when
    ``dist.mount_on_host_surface`` is ``True``, in which case the mount
    origin gets a radial ``(host_radius_m*cos(az), host_radius_m*sin(az))``
    offset off the host's centre axis, azimuth ``az`` either the planned
    one (``forced_mount``'s third element) or drawn i.i.d. from the same
    15-degree grid ``mount_rpy`` itself uses. ``False`` (default) draws no
    extra ``rng`` value at all and leaves ``mount_offset`` exactly
    ``(0.0, 0.0)`` -- byte-identical to before this field existed.

    ``lim`` (generation limits, default ``None``): a top-level digit without a
    planned mount picks its host only among hosts with room
    (``LimitContext.host_has_room``); the phalanx count is capped by the
    digit's joint budget (``LimitContext.begin_digit``, set by the caller),
    and the budget is passed down to the phalanges and branches. Hosts of
    planned mounts were already booked by the planner."""
    forced_azimuth: Optional[float] = None
    if forced_mount is not None:
        mount, mount_frac = forced_mount[0], forced_mount[1]
        if len(forced_mount) > 2:
            forced_azimuth = forced_mount[2]
    else:
        hosts = mount_bodies
        if lim is not None and top_level:
            hosts = lim.eligible_hosts(mount_bodies) or mount_bodies
        mount = hosts[int(rng.integers(0, len(hosts)))]
        if lim is not None and top_level:
            lim.take_host(mount)
        mount_frac = float(dist.mount_frac_choices[int(rng.integers(0, len(dist.mount_frac_choices)))])
    if dist.mount_on_host_surface:
        azimuth = forced_azimuth if forced_azimuth is not None else sample_grid_angle_rad(rng)
        mount_offset = (host_radius_m * math.cos(azimuth), host_radius_m * math.sin(azimuth))
    elif top_level and dist.mount_lateral_grid_m is not None:
        # Lateral digit mounts (off by default; see Distribution).
        mount_offset = sample_lateral_offset(rng, dist)
    else:
        mount_offset = (0.0, 0.0)
    if oppose_forward is not None:
        # Opus review item 5 / V3s fix: ``oppose_forward`` is a ROOT-frame
        # direction; when the host-frame fix is on, rotate it into THIS
        # digit's own host's local frame (``_best_opposing_rpy`` solves for
        # a LOCAL +z direction) so the digit's actual root-frame forward
        # direction -- not its local ``mount_rpy`` in isolation -- is what
        # ends up opposing the others.
        if host_transforms is not None:
            R_host = host_transforms.get(mount, np.eye(4))[:3, :3]
            mount_rpy = _best_opposing_rpy(R_host.T @ oppose_forward)
        else:
            mount_rpy = _best_opposing_rpy(oppose_forward)
    else:
        mount_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
    lo, hi = dist.phalanx_count_range
    if lim is not None:
        cap = lim.phalanx_cap()
        if cap is not None and cap < hi:
            hi = max(1, cap)
            lo = min(lo, hi)
    phalanx_count = int(rng.integers(lo, hi + 1))
    if lim is not None:
        lim.spend(phalanx_count)
    uid = next_uid[0]
    next_uid[0] += 1
    steps.append(DerivationStep(path=f"digit/{digit_id}", production="Digit", params={
        "digit_id": digit_id, "mount": mount, "mount_frac": mount_frac, "mount_rpy": mount_rpy,
        "mount_offset": mount_offset,
        "phalanx_count": phalanx_count, "top_level": top_level, "depth": depth, "uid": uid,
    }))
    for p in range(phalanx_count):
        _sample_phalanx(rng, dist, steps, digit_id, p, depth, is_last=(p == phalanx_count - 1), next_uid=next_uid,
                         host_radius_m=host_radius_m, lim=lim)


def _sample_digit(rng, dist: Distribution, steps: List[DerivationStep], next_id: List[int],
                   mount_bodies: List[str], next_uid: List[int],
                   forced_mount: Optional[Tuple] = None,
                   oppose_forward: Optional[np.ndarray] = None,
                   host_radius_m: float = 0.0,
                   host_transforms: Optional[Dict[str, np.ndarray]] = None,
                   lim: Optional[LimitContext] = None) -> None:
    """Sample a fresh *top-level* digit (id is the next 1-based integer).
    With ``lim``, its joint budget starts at ``max_joints_per_digit``."""
    digit_id = str(next_id[0])
    next_id[0] += 1
    if lim is not None:
        lim.begin_digit(lim.limits.max_joints_per_digit)
    _emit_digit(rng, dist, steps, digit_id, mount_bodies, top_level=True, depth=0, next_uid=next_uid,
                forced_mount=forced_mount, oppose_forward=oppose_forward, host_radius_m=host_radius_m,
                host_transforms=host_transforms, lim=lim)


def _capped_range(lo: int, hi: int, cap) -> Tuple[int, int]:
    """``(lo, hi)`` with ``hi`` lowered to ``cap`` (``lo`` too if needed);
    unchanged when ``cap`` is ``None``/infinite or not below ``hi``."""
    if cap is None or cap >= hi:
        return lo, hi
    hi = int(cap)
    return min(lo, hi), hi


def _phalanx_unit(dist: Distribution) -> int:
    """Fewest joints a digit sampled from ``dist`` can have (joint budgets
    reserve this much per pending branch digit)."""
    return max(1, int(dist.phalanx_count_range[0]))


def sample_derivation(rng_or_seed, dist: Distribution = DEFAULT_DISTRIBUTION,
                      limits: Optional[GenerationLimits] = None) -> Derivation:
    """Sample one derivation from ``dist``.

    ``limits`` (``limits.GenerationLimits``, default ``None``): hard
    generation limits, obeyed constructively (see ``limits.py``): the digit
    and palm-body counts are drawn from ranges capped by the limits, a palm
    body gets a joint only where the limits allow one (no stacking, below the
    cap, and leaving room for every digit), each top-level digit mounts on a
    host with room, phalanx counts respect the per-digit joint budget, module
    kinds come from the allowed set, and branches only when allowed. Every
    draw is made in the same order as without limits, and a limit that does
    not bind leaves its draw unchanged, so ``limits=None`` and ``UNLIMITED``
    give byte-identical derivations. Where a limit binds, values are drawn
    from the restricted options, so the result is not distributed like
    rejection sampling would be."""
    seed, rng = _coerce_rng(rng_or_seed)
    steps: List[DerivationStep] = []
    lim = _limit_context(limits, fresh=True)

    lo, hi = dist.digit_count_range
    if lim is not None:
        lo, hi = _capped_range(lo, hi, lim.digit_cap())
    digit_count = int(rng.integers(lo, hi + 1))
    # Additional palm bodies beyond the root -- the root is always a palm
    # body with a real segment of its own (root_length below), so a hand
    # never lacks a palm even when palm_body_count == 0.
    lo, hi = dist.palm_body_count_range
    if lim is not None:
        lo, hi = _capped_range(lo, hi, lim.limits.max_palm_bodies)
    palm_body_count = int(rng.integers(lo, hi + 1))
    root_length = sample_grid_length_m(rng, dist.palm_length_range_m, dist.link_length_grid_m)
    # One capsule radius per hand (iteration 7 / M2, geometry overlay -- see
    # rules.py's module note): stamped onto every Body.radius by ``derive``.
    capsule_radius_m = sample_capsule_radius_m(rng, dist)
    steps.append(DerivationStep(path="hand", production="Hand", params={
        "digit_count": digit_count, "palm_body_count": palm_body_count, "root_length": root_length,
        "capsule_radius_m": capsule_radius_m,
    }))

    # I15 fix 1: a stable ``uid`` counter, shared across every PalmBody/
    # Digit/Phalanx step this derivation creates (never resets, never
    # reused) -- see ``_max_uid``'s docstring and ``joint_identity`` below.
    next_uid: List[int] = [0]

    palm_names: List[str] = []
    host_length: Dict[str, float] = {"root": root_length}
    for i in range(palm_body_count):
        name = f"palm{i}"
        parent_choices = ["root"] + palm_names
        parent = parent_choices[int(rng.integers(0, len(parent_choices)))]
        mount_frac = float(dist.mount_frac_choices[int(rng.integers(0, len(dist.mount_frac_choices)))])
        length = sample_grid_length_m(rng, dist.palm_length_range_m, dist.link_length_grid_m)
        direction_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
        has_joint = bool(float(rng.random()) < dist.palm_joint_probability)
        if has_joint and lim is not None and not lim.can_add_jointed_palm(parent, digit_count):
            has_joint = False
        if lim is not None:
            lim.add_palm(name, parent, has_joint)
        axis = sample_axis(rng)
        joint_limits: Optional[Tuple[float, float]] = None
        if has_joint:
            joint_limits = sample_palm_joint_limits_rad(rng, dist)
        uid = next_uid[0]
        next_uid[0] += 1
        palm_params = {
            "name": name, "parent": parent, "mount_frac": mount_frac, "length": length,
            "direction_rpy": direction_rpy, "has_joint": has_joint, "axis": axis, "limits": joint_limits,
            "uid": uid,
        }
        if dist.mount_lateral_grid_m is not None:
            # Lateral mounts (off by default; see Distribution).
            palm_params["mount_offset"] = sample_lateral_offset(rng, dist)
        steps.append(DerivationStep(path=f"palm/{i}", production="PalmBody", params=palm_params))
        palm_names.append(name)
        host_length[name] = length

    mount_bodies = ["root"] + palm_names
    next_id = [1]

    # Opus review of G0 (item 2 / V3s fix): a scratch root-frame transform
    # for "root" plus every palm body, computed once (only when a rule
    # below actually needs it -- never for an untouched ``Distribution``),
    # reused by BOTH the surface/cross-host mount planner and the
    # host-frame-aware opposition prior below.
    host_transforms: Optional[Dict[str, np.ndarray]] = None
    if (dist.mount_on_host_surface and dist.mount_min_separation_m is not None) or dist.opposition_use_host_frame:
        host_transforms = _host_transforms_from_steps(steps, root_length)

    # G0 screen (I29 / opus review item 2): plan every top-level digit's
    # mount up front when a spacing rule is on, instead of drawing each
    # digit's mount independently. ``_plan_top_level_mounts`` (V2, same-host
    # axial spacing only) is reached exactly as before this review
    # (``mount_on_host_surface`` false); ``_plan_top_level_mounts_surface``
    # (V3s, cross-host + azimuth) is reached only when BOTH
    # ``mount_on_host_surface`` and ``mount_min_separation_m`` are set.
    # ``None`` (either field left at its default) keeps the original
    # per-digit i.i.d. draw.
    planned_mounts: Optional[List[Tuple]] = None
    if dist.mount_min_separation_m is not None and digit_count > 0:
        if dist.mount_on_host_surface:
            planned_mounts = _plan_top_level_mounts_surface(
                rng, dist, mount_bodies, digit_count, host_length, capsule_radius_m, host_transforms, lim=lim,
            )
        else:
            planned_mounts = _plan_top_level_mounts(rng, dist, mount_bodies, digit_count, host_length, lim=lim)
    for i in range(digit_count):
        forced_mount = planned_mounts[i] if planned_mounts is not None else None
        # G0 screen (I30): the LAST top-level digit, when the opposition
        # prior is on and there are >= 2 digits, opposes the mean forward
        # direction of the EARLIER digits' own (already-sampled) mounts --
        # see ``_best_opposing_rpy``. Every earlier digit, and every digit
        # under the default (off) prior, draws ``mount_rpy`` exactly as
        # before this feature existed.
        oppose_forward = None
        if dist.opposition_prior and digit_count >= 2 and i == digit_count - 1:
            prior_digits = [
                s.params for s in steps
                if s.production == "Digit" and s.params.get("top_level")
            ]
            if prior_digits:
                if dist.opposition_use_host_frame and host_transforms is not None:
                    # Opus review item 5 / V3s fix: each earlier digit's own
                    # ``mount_rpy`` is expressed in ITS OWN host's local
                    # frame, not the root frame -- rotate each one into the
                    # root frame with its host's own accumulated rotation
                    # before averaging (mixed-host hands otherwise average
                    # vectors from incompatible frames, opus-review-g0.md's
                    # "105 deg in mixed-host hands").
                    fwds = [
                        host_transforms.get(p["mount"], np.eye(4))[:3, :3]
                        @ (rpy_to_matrix(tuple(p["mount_rpy"])) @ np.array([0.0, 0.0, 1.0]))
                        for p in prior_digits
                    ]
                else:
                    fwds = [
                        rpy_to_matrix(tuple(p["mount_rpy"])) @ np.array([0.0, 0.0, 1.0])
                        for p in prior_digits
                    ]
                mean_fwd = np.mean(fwds, axis=0)
                norm = float(np.linalg.norm(mean_fwd))
                if norm > 1e-9:
                    oppose_forward = mean_fwd / norm
        _sample_digit(rng, dist, steps, next_id, mount_bodies, next_uid,
                      forced_mount=forced_mount, oppose_forward=oppose_forward, host_radius_m=capsule_radius_m,
                      host_transforms=host_transforms if dist.opposition_use_host_frame else None, lim=lim)

    return Derivation(seed=seed, grammar_version=GRAMMAR_VERSION, steps=tuple(steps))


# --------------------------------------------------------------------------
# derive: pure, deterministic reconstruction
# --------------------------------------------------------------------------


def validate_derivation(derivation: Derivation) -> List[str]:
    """Structural validation of a ``Derivation`` *as a trace* -- independent
    of, and prior to, building any ``KinematicModel`` from it. Returns a list
    of issue strings (empty if clean). Checks:

    - ``grammar_version`` matches the current ``GRAMMAR_VERSION``.
    - every ``Phalanx`` step's ``digit_id`` names a ``Digit`` step present in
      the same derivation.
    - the ``hand`` step's ``digit_count`` equals the number of top-level
      ``Digit`` steps.
    - each ``Digit`` step's ``Phalanx`` steps have exactly the indices
      ``0..phalanx_count - 1`` (contiguous, no gaps or duplicates).
    - every ``Digit`` step's ``mount`` names a body that another step in the
      derivation actually creates (``"root"``, a ``PalmBody`` step's
      ``name``, or some digit's own phalanx body ``f"d{digit_id}p{p+1}"``).
    """
    issues: List[str] = []

    if derivation.grammar_version != GRAMMAR_VERSION:
        issues.append(
            f"grammar_version mismatch: derivation has {derivation.grammar_version!r}, "
            f"expected {GRAMMAR_VERSION!r}"
        )

    hand_steps = [s for s in derivation.steps if s.path == "hand"]
    hand = hand_steps[0].params if len(hand_steps) == 1 else None
    if len(hand_steps) != 1:
        issues.append(f"expected exactly one 'hand' step, found {len(hand_steps)}")

    digit_steps: Dict[str, DerivationStep] = {}
    for s in derivation.steps:
        if s.production == "Digit":
            digit_id = s.params["digit_id"]
            if digit_id in digit_steps:
                issues.append(f"duplicate Digit step for digit id {digit_id!r}")
            digit_steps[digit_id] = s

    phalanx_by_digit: Dict[str, Dict[int, DerivationStep]] = {}
    for s in derivation.steps:
        if s.production != "Phalanx":
            continue
        digit_id = s.params["digit_id"]
        if digit_id not in digit_steps:
            issues.append(f"Phalanx step {s.path!r} references unknown digit id {digit_id!r}")
            continue
        by_p = phalanx_by_digit.setdefault(digit_id, {})
        p = s.params["p"]
        if p in by_p:
            issues.append(f"duplicate Phalanx step for digit {digit_id!r} index {p}")
        by_p[p] = s

    if hand is not None:
        n_top_level = sum(1 for s in digit_steps.values() if s.params.get("top_level"))
        if n_top_level != hand["digit_count"]:
            issues.append(
                f"hand digit_count={hand['digit_count']} disagrees with "
                f"{n_top_level} top-level Digit step(s)"
            )

    for digit_id, dstep in digit_steps.items():
        expected = list(range(dstep.params["phalanx_count"]))
        got = sorted(phalanx_by_digit.get(digit_id, {}))
        if got != expected:
            issues.append(
                f"digit {digit_id!r} phalanx indices {got} are not contiguous 0..{dstep.params['phalanx_count'] - 1}"
            )

    known_bodies = {"root"}
    for s in derivation.steps:
        if s.production == "PalmBody":
            known_bodies.add(s.params["name"])
    for digit_id, by_p in phalanx_by_digit.items():
        for p in by_p:
            known_bodies.add(f"d{digit_id}p{p + 1}")

    for digit_id, dstep in digit_steps.items():
        mount = dstep.params["mount"]
        if mount not in known_bodies:
            issues.append(f"digit {digit_id!r} mount {mount!r} is not a body any step creates")

    return issues


def _compose_bend_rpy(existing_rpy: Tuple[float, float, float],
                       bend_rpy: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Grammar 0.5's rest-bend primitive (I16 priority 1; I22 fix 1
    corrects the composition): combine a phalanx joint's existing origin
    orientation (the digit's own sampled ``mount_rpy`` for its first
    phalanx, or the identity ``(0,0,0)`` for a mid-digit continuation joint
    -- see rules.py's convention note) with the small additional
    ``bend_rpy`` perturbation.

    Composition order/method: proper rotation-matrix composition
    ``R = R(existing_rpy) @ R(bend_rpy)`` -- ``bend_rpy`` is applied in the
    base/mount frame, i.e. it rotates about the MOUNT's own local axes, not
    about world/componentwise axes -- converted back to a single fixed-axis
    XYZ triple via ``fk.matrix_to_rpy`` (a deterministic gimbal-lock rule;
    any triple reproducing ``R`` is FK-equivalent, since ``rpy_to_matrix`` /
    ``to_urdf`` only ever consume the recomposed matrix). The old
    COMPONENTWISE Euler-angle addition (roll+roll, pitch+pitch, yaw+yaw) was
    exact only by accident, because every caller that samples a nonzero
    ``existing_rpy`` (a digit's ``mount_rpy``) never also samples a nonzero
    phalanx-0 ``bend_rpy`` at the same time (the grammar's own productions
    keep phalanx-0 bend at 0); componentwise addition is NOT a valid small-
    rotation composition in general (Opus review: median 4.7 deg axis error
    across real mounts, up to 90 deg, if a mutation ever put a nonzero bend
    on phalanx 0). When ``bend_rpy == (0.0, 0.0, 0.0)`` this returns
    ``existing_rpy`` UNCHANGED (the identical object, not a recomputed
    triple) so every existing derivation/replay hash stays byte-identical --
    this is also exactly the continuation-joint case (``existing_rpy ==
    (0,0,0)``), which then returns ``(0,0,0)`` unchanged, matching what
    ``coverage.py``'s bend-grid check judges against
    ``Distribution.bend_rpy_choices_rad``."""
    if bend_rpy == (0.0, 0.0, 0.0):
        return existing_rpy
    if existing_rpy == (0.0, 0.0, 0.0):
        # A mid-digit continuation joint's own base orientation is always
        # exactly identity (see this function's docstring): composing with
        # it is mathematically a no-op (R(0) @ R(bend) == R(bend)), so
        # return ``bend_rpy`` unchanged rather than round-tripping it
        # through rpy_to_matrix/matrix_to_rpy, which would perturb it by
        # floating-point noise for no reason. This keeps every continuation
        # phalanx's ``bend_rpy`` exactly on ``Distribution.bend_rpy_choices_rad``'s
        # own grid, which ``coverage.py``'s bend-grid check relies on.
        return bend_rpy
    R = rpy_to_matrix(existing_rpy) @ rpy_to_matrix(bend_rpy)
    return matrix_to_rpy(R)


def derive(derivation: Derivation) -> KinematicModel:
    issues = validate_derivation(derivation)
    if issues:
        raise DerivationError(issues)

    steps_by_path = {s.path: s for s in derivation.steps}
    hand = steps_by_path["hand"].params
    palm_body_count = hand["palm_body_count"]
    root_length = hand["root_length"]
    # One scalar capsule radius for the whole hand (see rules.py's module
    # note / distributions.py's capsule_radius_choices_m); stamped onto every
    # Body below -- geometry itself is never derived here (see geometry.py),
    # only this one per-hand parameter that geometry.py later reads off
    # Body.radius.
    capsule_radius_m = hand["capsule_radius_m"]

    # The root always owns a real segment (see rules.py's RootProduction /
    # convention-change note): a hand always has a palm, so body_length for
    # "root" is never 0, and the root gets its own "<body>_tip" frame just
    # like every other segment-owning body.
    bodies: List[Body] = [Body(name="root", palm=True, radius=capsule_radius_m)]
    joints: List[Joint] = []
    frames: List[Frame] = [Frame(name="root_tip", body="root", pose=Pose(xyz=(0.0, 0.0, root_length)))]
    couplings: List[AffineCoupling] = []
    body_length: Dict[str, float] = {"root": root_length}
    joints_by_name: Dict[str, Joint] = {}

    for i in range(palm_body_count):
        p = steps_by_path[f"palm/{i}"].params
        name = p["name"]
        parent = p["parent"]
        length = p["length"]
        rpy = tuple(p["direction_rpy"])
        # Mount point ON the parent's own segment: T = Trans(ox,oy,frac*L) *
        # Rot(rpy) -- xyz is the translation to the mount point in the
        # parent's own frame, rpy is the child frame's orientation relative
        # to the parent, exactly as a URDF joint origin means (see rules.py's
        # convention-change note / fk.py's docstring). Never place the
        # child a further ``length`` out from that mount point -- that was
        # the old bug that left the parent's segment with an unowned
        # stretch between the mount and the parent's own tip.
        #
        # ``mount_offset`` (representation-check item 1): a lateral (x, y)
        # displacement of the mount point off the parent's own z-axis line,
        # read via ``.get`` for backward compatibility with any
        # hand-authored/older PalmBody params dict predating this field
        # (mirrors ``bend_offset``'s ``.get`` default below). Default
        # ``(0.0, 0.0)`` keeps ``base_xyz`` exactly ``(0.0, 0.0, frac*L)`` --
        # byte-identical to every pre-existing derivation/replay hash.
        mount_offset = p.get("mount_offset", (0.0, 0.0))
        mount_len = body_length[parent]
        base_xyz = (mount_offset[0], mount_offset[1], p["mount_frac"] * mount_len)
        jtype = "revolute" if p["has_joint"] else "fixed"
        axis = tuple(p["axis"]) if p["has_joint"] else (1.0, 0.0, 0.0)
        limits = tuple(p["limits"]) if p["has_joint"] else None
        j = Joint(
            name=f"{name}_j", type=jtype, parent=parent, child=name,
            origin=Pose(xyz=base_xyz, rpy=rpy),
            axis=axis, limits=limits,
        )
        joints.append(j)
        joints_by_name[j.name] = j
        bodies.append(Body(name=name, palm=True, radius=capsule_radius_m))
        frames.append(Frame(name=f"{name}_tip", body=name, pose=Pose(xyz=(0.0, 0.0, length))))
        body_length[name] = length

    def _process_digit(p: Dict[str, Any]) -> None:
        digit_id = p["digit_id"]
        mount = p["mount"]
        # Mount point ON the host segment (see the palm-mount comment above
        # -- the same convention, applied to a digit/branch mounting on a
        # palm or phalanx body): xyz is the unrotated translation along the
        # host's own z-axis, mount_rpy is the child frame's orientation.
        # ``mount_offset`` (opus review of G0 item 2 / V3s surface mounting):
        # a lateral (x, y) displacement of the mount point off the host's
        # own z-axis line -- the SAME field/convention PalmBody's own
        # ``mount_offset`` already uses (see the palm-mount comment above),
        # read via ``.get`` for backward compatibility with any
        # hand-authored/older Digit params dict predating this field.
        # Default ``(0.0, 0.0)`` keeps ``base_xyz`` exactly
        # ``(0.0, 0.0, frac*L)`` -- byte-identical to every pre-existing
        # derivation/replay hash.
        mount_len = body_length.get(mount, 0.0)
        mount_offset = p.get("mount_offset", (0.0, 0.0))
        base_xyz = (mount_offset[0], mount_offset[1], p["mount_frac"] * mount_len)
        base_rpy = tuple(p["mount_rpy"])

        prev_body = mount
        prev_len = 0.0
        phalanx_joint_names: Dict[int, str] = {}
        for pi in range(p["phalanx_count"]):
            pp = steps_by_path[f"digit/{digit_id}/phalanx/{pi}"].params
            body_name = f"d{digit_id}p{pi + 1}"
            joint_name = f"{body_name}_j"
            # Grammar 0.5 rest-bend primitive (I16 priority 1): ``bend_rpy``/
            # ``bend_offset`` default to (0,0,0)/(0,0) via ``.get`` for
            # backward compatibility with any hand-authored Phalanx params
            # dict predating this field (see distributions.Distribution
            # .bend_probability's docstring). xyz becomes
            # (bend_offset_x, bend_offset_y, t) -- t as today (the mount
            # fraction along the host segment for pi==0, or the previous
            # phalanx's own length for a continuation) -- since the existing
            # convention's own x/y are always exactly 0 in both cases.
            # rpy is composed with the existing convention's own rpy (the
            # digit's sampled ``mount_rpy`` for pi==0, or identity for a
            # continuation) by plain componentwise Euler-angle addition, NOT
            # a rotation-matrix product -- see ``_compose_bend_rpy``'s own
            # docstring for why this is FK-exact regardless.
            bend_rpy = tuple(pp.get("bend_rpy", (0.0, 0.0, 0.0)))
            bend_offset = tuple(pp.get("bend_offset", (0.0, 0.0)))
            if pi == 0:
                # ``base_xyz``'s own (x, y) is the digit's ``mount_offset``
                # (opus review item 2 / V3s surface mounting; (0.0, 0.0)
                # unless ``mount_on_host_surface`` is on) -- additive with
                # ``bend_offset`` (Grammar 0.5's independent rest-bend
                # primitive), not overwritten by it, so the two features
                # compose rather than one silently discarding the other.
                # Both are (0.0, 0.0) for every distribution that sets
                # neither, so this is byte-identical to before either field
                # existed.
                origin_xyz = (base_xyz[0] + bend_offset[0], base_xyz[1] + bend_offset[1], base_xyz[2])
                origin_rpy = _compose_bend_rpy(base_rpy, bend_rpy)
            else:
                origin_xyz = (bend_offset[0], bend_offset[1], prev_len)
                origin_rpy = _compose_bend_rpy((0.0, 0.0, 0.0), bend_rpy)
            origin = Pose(xyz=origin_xyz, rpy=origin_rpy)

            mod = pp["module"]
            axis = tuple(mod["axis"])
            coupling = None
            if mod["kind"] == "R":
                jtype = "revolute"
                limits = tuple(mod["limits"])
            elif mod["kind"] == "C":
                jtype = "continuous"
                limits = None
            elif mod["kind"] == "P":
                jtype = "prismatic"
                limits = tuple(mod["limits"])
            elif mod["kind"] == "Coupled":
                jtype = "revolute"
                src_name = phalanx_joint_names[mod["source_p"]]
                src_joint = joints_by_name[src_name]
                if src_joint.type == "continuous":
                    slo, shi = CONTINUOUS_SAMPLE_RANGE
                else:
                    slo, shi = src_joint.limits
                mult = mod["multiplier"]
                off_c = mod["offset"]
                lo, hi = mult * slo + off_c, mult * shi + off_c
                if lo > hi:
                    lo, hi = hi, lo
                limits = (lo, hi)
                coupling = AffineCoupling(dependent=joint_name, source=src_name, multiplier=mult, offset=off_c)
            else:
                raise ModelError([f"unknown module kind {mod['kind']!r}"])

            j = Joint(name=joint_name, type=jtype, parent=prev_body, child=body_name,
                      origin=origin, axis=axis, limits=limits)
            joints.append(j)
            joints_by_name[joint_name] = j
            if coupling is not None:
                couplings.append(coupling)
            bodies.append(Body(name=body_name, radius=capsule_radius_m))
            length = pp["length"]
            frames.append(Frame(name=f"{body_name}_tip", body=body_name, pose=Pose(xyz=(0.0, 0.0, length))))
            body_length[body_name] = length
            phalanx_joint_names[pi] = joint_name
            prev_body = body_name
            prev_len = length

    # Digits form a DAG by mount (top-level digits mount on root/palm bodies,
    # always available; branch digits mount on a phalanx body created by
    # processing their host digit) -- process in that dependency order (a
    # simple fixed-point/BFS-by-depth over ``derivation.steps``' own order)
    # rather than assuming any particular position in the flat step list, so
    # ``vary``'s list-splicing operators can never silently reorder a branch
    # ahead of the phalanx body it mounts on.
    pending = [s.params for s in derivation.steps if s.production == "Digit"]
    while pending:
        ready = [p for p in pending if p["mount"] in body_length]
        if not ready:
            raise ModelError([
                f"digit {p['digit_id']!r} mount {p['mount']!r} is never created"
                for p in pending
            ])
        still_pending = [p for p in pending if p["mount"] not in body_length]
        for p in ready:
            _process_digit(p)
        pending = still_pending

    model = KinematicModel(
        name="grammar_hand", root="root", bodies=tuple(bodies), joints=tuple(joints),
        frames=tuple(frames), couplings=tuple(couplings),
    )
    validate(model)
    return model


def joint_identity(derivation: Derivation) -> Dict[str, int]:
    """I15 fix 1: ``{joint_name: uid}`` for every joint ``derive(derivation)``
    would produce, where ``uid`` is the stable identifier of the
    ``Phalanx``/``PalmBody`` step that CREATED that joint (see
    ``_max_uid``'s docstring). ``derive`` itself keeps joint NAMES
    positional/renumbering-sensitive (so replay hashes and URDF names are
    unchanged) -- this function is the alignment key callers (``phenodist.
    phenotype_distance``, ``e1_locality``) should use instead of a bare name
    match whenever they have a parent/child pair descended from a common
    ancestor derivation, since insert/delete_phalanx, remove_digit and
    remove_palm_body can renumber a joint's NAME (``d{digit_id}p{p+1}_j``)
    without that joint being a new piece of structure at all. A joint whose
    creating step predates this fix (no ``"uid"`` key in its params, e.g. a
    derivation produced by an older ``sample_derivation``) maps to ``-1``."""
    out: Dict[str, int] = {}
    for s in derivation.steps:
        if s.production == "PalmBody":
            out[f"{s.params['name']}_j"] = int(s.params.get("uid", -1))
        elif s.production == "Phalanx":
            out[f"d{s.params['digit_id']}p{s.params['p'] + 1}_j"] = int(s.params.get("uid", -1))
    return out


def segment(model: KinematicModel, body: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return ``(start, end)``: the endpoints, in the root frame at q=0, of
    ``body``'s own geometric segment. ``start`` is ``body``'s own origin;
    ``end`` is its ``"<body>_tip"`` frame (every body the grammar produces
    carries one -- see rules.py). This is a thin q=0 forward-kinematics
    query; it lives here (rather than in coverage.py) because it is about
    the derived model's geometry itself, not about judging that geometry
    against a ``Distribution``."""
    transforms = forward_kinematics(model, {})
    start = tuple(float(v) for v in transforms[body][:3, 3])
    end = tuple(float(v) for v in transforms[f"{body}_tip"][:3, 3])
    return start, end


def generate(seed, dist: Distribution = DEFAULT_DISTRIBUTION,
             limits: Optional[GenerationLimits] = None) -> Tuple[Derivation, KinematicModel]:
    derivation = sample_derivation(seed, dist, limits=limits)
    model = derive(derivation)
    return derivation, model


# --------------------------------------------------------------------------
# JSON round trip for Derivation
# --------------------------------------------------------------------------


def _encode(v: Any) -> Any:
    if isinstance(v, tuple):
        return {"__tuple__": [_encode(x) for x in v]}
    if isinstance(v, dict):
        return {k: _encode(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_encode(x) for x in v]
    return v


def _decode(v: Any) -> Any:
    if isinstance(v, dict):
        if set(v.keys()) == {"__tuple__"}:
            return tuple(_decode(x) for x in v["__tuple__"])
        return {k: _decode(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_decode(x) for x in v]
    return v


def _step_to_dict(s: DerivationStep) -> Dict[str, Any]:
    return {"path": s.path, "production": s.production, "params": _encode(s.params)}


def _step_from_dict(d: Dict[str, Any]) -> DerivationStep:
    return DerivationStep(path=d["path"], production=d["production"], params=_decode(d["params"]))


def derivation_to_dict(d: Derivation) -> Dict[str, Any]:
    return {
        "schema": DERIVATION_SCHEMA,
        "seed": d.seed,
        "grammar_version": d.grammar_version,
        "steps": [_step_to_dict(s) for s in d.steps],
        "lineage": _encode(d.lineage),
    }


def derivation_from_dict(d: Dict[str, Any]) -> Derivation:
    if d.get("schema") != DERIVATION_SCHEMA:
        raise ValueError(f"unsupported schema {d.get('schema')!r}, expected {DERIVATION_SCHEMA!r}")
    return Derivation(
        seed=d["seed"],
        grammar_version=d["grammar_version"],
        steps=tuple(_step_from_dict(s) for s in d["steps"]),
        lineage=_decode(d["lineage"]) if "lineage" in d else (),
    )


def derivation_to_json(d: Derivation) -> str:
    return json.dumps(derivation_to_dict(d), allow_nan=False)


def derivation_from_json(text: str) -> Derivation:
    return derivation_from_dict(json.loads(text))


# --------------------------------------------------------------------------
# vary: structural / parametric mutation operators
# --------------------------------------------------------------------------

OPERATORS: Tuple[str, ...] = (
    "resample_parameter",
    "perturb_parameter",
    "regrow_subtree",
    "insert_phalanx",
    "delete_phalanx",
    "add_digit",
    "remove_digit",
)


def _growth_dist(dist: Distribution) -> Distribution:
    """I14 fix 5: the ``Distribution`` a GROWTH operator (``add_digit``,
    ``add_palm_body``, ``regrow_subtree``, ``add_minimal_digit``) should
    draw its brand-new material from -- ``dist.insertion`` if set, else
    ``dist`` itself (current behaviour, unchanged). Caller-level caps
    (``digit_count_range``, ``palm_body_count_range``) are never read from
    this: they bound the whole hand, so callers must keep reading those off
    the outer ``dist``, not this function's return value."""
    return dist.insertion if dist.insertion is not None else dist


def _op_resample_parameter(rng, dist: Distribution, derivation: Derivation,
                           lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """``lim``: a palm body keeps its joint state when the drawn one is not
    allowed; a phalanx draws its module from the allowed kinds; a top-level
    digit re-mounts only on a host with room."""
    steps = list(derivation.steps)
    idx = int(rng.integers(0, len(steps)))
    s = steps[idx]
    if s.production == "PalmBody":
        p = dict(s.params)
        length = sample_grid_length_m(rng, dist.palm_length_range_m, dist.link_length_grid_m)
        direction_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
        has_joint = bool(float(rng.random()) < dist.palm_joint_probability)
        if (lim is not None and has_joint != bool(p["has_joint"])
                and not lim.allows(lim.struct.with_joint(p["name"], has_joint))):
            has_joint = bool(p["has_joint"])
        axis = sample_axis(rng)
        limits = sample_palm_joint_limits_rad(rng, dist) if has_joint else None
        p.update({"length": length, "direction_rpy": direction_rpy, "has_joint": has_joint,
                  "axis": axis, "limits": limits})
        steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
        return steps
    if s.production == "Phalanx":
        p = dict(s.params)
        module = sample_module(rng, dist, p["p"], _revolute_source_indices(steps, p["digit_id"], p["p"]),
                               allowed=None if lim is None else lim.allowed_modules)
        length = sample_grid_length_m(rng, dist.link_length_range_m, dist.link_length_grid_m)
        p.update({"module": module, "length": length})
        steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
        return steps
    if s.production == "Digit":
        p = dict(s.params)
        if p["top_level"]:
            mount_bodies = _mount_bodies_from_steps(steps)
            if lim is not None:
                mount_bodies = lim.eligible_hosts(mount_bodies, moving=p["digit_id"]) or [p["mount"]]
            p["mount"] = mount_bodies[int(rng.integers(0, len(mount_bodies)))]
        # A branch digit's mount is fixed to its host phalanx's body (it is
        # not a free choice); only its pose within that body is resampled.
        p["mount_frac"] = float(dist.mount_frac_choices[int(rng.integers(0, len(dist.mount_frac_choices)))])
        p["mount_rpy"] = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
        steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
        return steps
    return None


def _op_perturb_parameter(rng, dist: Distribution, derivation: Derivation,
                          lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    candidates = [i for i, s in enumerate(steps) if s.production in ("PalmBody", "Phalanx")]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    length_range = dist.palm_length_range_m if s.production == "PalmBody" else dist.link_length_range_m
    grid = dist.link_length_grid_m
    lo, hi = length_range
    direction = 1.0 if float(rng.random()) < 0.5 else -1.0
    new_len = p["length"] + direction * grid
    if new_len > hi:
        new_len = hi - (new_len - hi)
    if new_len < lo:
        new_len = lo + (lo - new_len)
    new_len = min(max(new_len, lo), hi)
    p["length"] = round(new_len, 10)
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


def _regrow_context(lim: LimitContext, steps: List[DerivationStep], digit_id: str,
                    gdist: Distribution) -> Optional[LimitContext]:
    """A ``LimitContext`` over ``steps`` without ``digit_id``'s subtree, with
    the joint budget the regrown digit may use; ``None`` if it has no room
    (fewer joints than a digit needs, or no host for a top-level digit)."""
    kept = [st for st in steps if not _is_descendant_digit(digit_id, _step_digit_id(st))]
    sub = LimitContext(lim.limits, kept, base=lim.base)
    digit = next(st for st in steps if st.production == "Digit" and st.params["digit_id"] == digit_id)
    if digit.params["top_level"]:
        if not sub.eligible_hosts(_mount_bodies_from_steps(steps)):
            return None
        room = lim.limits.max_joints_per_digit
    else:
        owner = lim.struct.top_of().get(digit_id, digit_id)
        room = None if lim.limits.max_joints_per_digit is None else (
            lim.limits.max_joints_per_digit - sub.struct.joints_per_digit().get(owner, 0))
    if room is not None and room < _phalanx_unit(gdist):
        return None
    sub.begin_digit(room)
    return sub


def _op_regrow_subtree(rng, dist: Distribution, derivation: Derivation,
                       lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """``lim``: only digits that can be regrown within the limits are
    candidates; the new subtree is sampled under them (joint budget, allowed
    modules, branches, a top-level digit's host)."""
    steps = list(derivation.steps)
    digit_idxs = [i for i, s in enumerate(steps) if s.production == "Digit"]
    sub_lim: Dict[int, LimitContext] = {}
    if lim is not None:
        for i in digit_idxs:
            ctx = _regrow_context(lim, steps, steps[i].params["digit_id"], _growth_dist(dist))
            if ctx is not None:
                sub_lim[i] = ctx
        if len(sub_lim) < len(digit_idxs):
            digit_idxs = [i for i in digit_idxs if i in sub_lim]
    if not digit_idxs:
        return None
    idx = digit_idxs[int(rng.integers(0, len(digit_idxs)))]
    s = steps[idx]
    digit_id = s.params["digit_id"]
    top_level = s.params["top_level"]
    depth = s.params["depth"]
    mount = s.params["mount"]
    # Top-level digits may re-pick which palm body they mount on; a branch
    # digit's mount is fixed to its host phalanx's body.
    mount_bodies = _mount_bodies_from_steps(steps) if top_level else [mount]

    kept = [st for st in steps if not _is_descendant_digit(digit_id, _step_digit_id(st))]

    new_steps: List[DerivationStep] = []
    next_uid = [_max_uid(steps) + 1]
    hand_params = next(st.params for st in steps if st.path == "hand")
    host_radius_m = hand_params.get("capsule_radius_m", 0.0)
    _emit_digit(rng, _growth_dist(dist), new_steps, digit_id, mount_bodies, top_level, depth, next_uid,
                host_radius_m=host_radius_m, lim=sub_lim.get(idx))
    return kept + new_steps


def _rename_branch_mounts(steps: List[DerivationStep], host_digit_id: str,
                           renumber: Dict[int, int]) -> List[DerivationStep]:
    """After insert/delete-phalanx renumbers ``host_digit_id``'s local
    phalanx indices, update the ``mount`` field of any branch ``Digit`` step
    hosted on one of those phalanges so it still names the (renumbered) body
    it is actually mounted on. The branch digit's own id/body names are
    never renamed (see ``rules.py``'s module docstring): they stay valid,
    globally unique identifiers even if they no longer literally encode the
    host's *current* phalanx index."""
    rename = {
        f"d{host_digit_id}p{old + 1}": f"d{host_digit_id}p{new + 1}"
        for old, new in renumber.items() if old != new
    }
    if not rename:
        return steps
    out = []
    for st in steps:
        if (st.production == "Digit" and not st.params.get("top_level", True)
                and st.params.get("mount") in rename):
            out.append(DerivationStep(path=st.path, production="Digit",
                                       params={**st.params, "mount": rename[st.params["mount"]]}))
        else:
            out.append(st)
    return out


def _op_insert_phalanx(rng, dist: Distribution, derivation: Derivation,
                       lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """``lim``: only digits whose top-level digit has a joint to spare; the
    new module from the allowed kinds."""
    steps = list(derivation.steps)
    digit_idxs = [i for i, s in enumerate(steps)
                  if s.production == "Digit" and s.params["phalanx_count"] < dist.phalanx_count_range[1]]
    if lim is not None and lim.limits.max_joints_per_digit is not None:
        digit_idxs = [i for i in digit_idxs if lim.joint_room(steps[i].params["digit_id"]) >= 1]
    if not digit_idxs:
        return None
    idx = digit_idxs[int(rng.integers(0, len(digit_idxs)))]
    dstep = steps[idx]
    digit_id = dstep.params["digit_id"]
    old_count = dstep.params["phalanx_count"]
    ins_p = int(rng.integers(0, old_count + 1))

    others = [st for st in steps if st is not dstep]
    phalanx_steps = {st.params["p"]: st for st in others
                      if st.production == "Phalanx" and st.params["digit_id"] == digit_id}
    non_digit_others = [st for st in others
                         if not (st.production == "Phalanx" and st.params["digit_id"] == digit_id)]

    renumber: Dict[int, int] = {}
    new_phalanx_list: List[DerivationStep] = []
    for old_p in range(old_count):
        new_p = old_p if old_p < ins_p else old_p + 1
        renumber[old_p] = new_p
        st = phalanx_steps[old_p]
        params = dict(st.params)
        mod = dict(params["module"])
        if mod.get("kind") == "Coupled":
            sp = mod["source_p"]
            mod["source_p"] = sp if sp < ins_p else sp + 1
        params["module"] = mod
        params["p"] = new_p
        new_phalanx_list.append(
            DerivationStep(path=f"digit/{digit_id}/phalanx/{new_p}", production="Phalanx", params=params))

    revolute_source_indices = tuple(
        old_p for old_p in range(ins_p) if phalanx_steps[old_p].params["module"]["kind"] == "R"
    )
    module = sample_module(rng, dist, ins_p, revolute_source_indices,
                           allowed=None if lim is None else lim.allowed_modules)
    length = sample_grid_length_m(rng, dist.link_length_range_m, dist.link_length_grid_m)
    bend_rpy, bend_offset = sample_bend(rng, dist)
    new_uid = _max_uid(steps) + 1
    new_phalanx_list.append(DerivationStep(path=f"digit/{digit_id}/phalanx/{ins_p}", production="Phalanx", params={
        "digit_id": digit_id, "p": ins_p, "module": module, "length": length, "branch_digit_count": 0,
        "uid": new_uid, "bend_rpy": bend_rpy, "bend_offset": bend_offset,
    }))

    new_dstep = DerivationStep(path=dstep.path, production="Digit",
                                params={**dstep.params, "phalanx_count": old_count + 1})
    result = non_digit_others + [new_dstep] + new_phalanx_list
    return _rename_branch_mounts(result, digit_id, renumber)


def _op_delete_phalanx(rng, dist: Distribution, derivation: Derivation,
                        target: Optional[int] = None,
                        lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """``target`` (grammar 0.5, reversibility): when given, the uid of the
    ``Phalanx`` step to delete -- restricts the candidate pool to exactly
    that phalanx (still subject to every safety check below) instead of
    drawing uniformly, so a caller (the reversibility test) can undo a
    specific ``insert_phalanx`` application precisely.

    I15 fix 4 (I18 fix 1 corrects the ``del_p == 0`` case below): a phalanx
    that hosts a branch MAY now be deleted -- its branch sub-digit(s) are
    re-attached (their ``Digit`` step's ``mount`` field updated) rather than
    orphaned: to the PROXIMAL neighbour phalanx's body (index ``del_p - 1``,
    whose body name is never itself renumbered by this deletion, since only
    phalanges with an index ABOVE ``del_p`` shift down) if ``del_p > 0``, or
    to the digit's NEW phalanx 0 (the phalanx that was index 1, before it is
    itself renumbered down to index 0) if ``del_p == 0`` (the first
    phalanx) -- NEVER to the digit's own ``mount`` (I18 fix 1: that body can
    be a palm body when this digit is top-level, and a branch digit must
    never mount on a palm body; the old code reattached there, which broke
    that invariant). A digit with only 1 phalanx is never a delete_phalanx
    candidate at all (see the ``phalanx_count <= 1: continue`` guard below),
    so this case never has to fall back to the digit's own mount: there is
    always a new phalanx 0 to reattach to once ``del_p == 0`` is reached.
    Deleting any phalanx except the digit's *current last one* never
    changes which phalanx is last, so it is always safe. Deleting the last
    one promotes the previous phalanx to "last" -- still refused when that
    PREVIOUS phalanx (not ``del_p`` itself) hosts exactly 1 branch digit of
    its OWN (valid only for a non-last phalanx, which already has a
    next-phalanx child; as the new last phalanx it would drop to a single
    child) -- this pre-existing safety check is unrelated to ``del_p``'s own
    branches, which are simply relocated, never left dangling."""
    steps = list(derivation.steps)
    candidates: List[Tuple[int, List[int]]] = []
    for i, s in enumerate(steps):
        if s.production != "Digit" or s.params["phalanx_count"] <= 1:
            continue
        digit_id = s.params["digit_id"]
        phalanx_count = s.params["phalanx_count"]
        last_idx = phalanx_count - 1
        phalanx_by_p = {
            pp.params["p"]: pp for pp in steps
            if pp.production == "Phalanx" and pp.params["digit_id"] == digit_id
        }
        second_last = phalanx_by_p.get(last_idx - 1)
        last_deletion_safe = not (second_last is not None and second_last.params["branch_digit_count"] == 1)
        deletable = [
            p_idx for p_idx in phalanx_by_p
            if (p_idx != last_idx or last_deletion_safe)
            and (target is None or phalanx_by_p[p_idx].params.get("uid") == target)
        ]
        if deletable:
            candidates.append((i, deletable))
    if not candidates:
        return None
    idx, deletable = candidates[int(rng.integers(0, len(candidates)))]
    dstep = steps[idx]
    digit_id = dstep.params["digit_id"]
    old_count = dstep.params["phalanx_count"]
    del_p = deletable[int(rng.integers(0, len(deletable)))]

    others = [st for st in steps if st is not dstep]
    phalanx_steps = {st.params["p"]: st for st in others
                      if st.production == "Phalanx" and st.params["digit_id"] == digit_id}
    non_digit_others = [st for st in others
                         if not (st.production == "Phalanx" and st.params["digit_id"] == digit_id)]

    # Re-attach ``del_p``'s own branch sub-digits (if any) BEFORE the
    # renumbering pass below, since the deleted phalanx's body never
    # survives renumbering at all (it is not merely renamed, it is gone).
    deleted_body = f"d{digit_id}p{del_p + 1}"
    if del_p > 0:
        # Proximal neighbour's body name is unaffected by this deletion
        # (only phalanges with an index ABOVE del_p shift down).
        reattach_mount = f"d{digit_id}p{del_p}"
    else:
        # I18 fix 1: reattach to the phalanx that BECOMES the new phalanx 0
        # (old index 1), using ITS pre-renumbering name -- the renumbering
        # pass + ``_rename_branch_mounts`` call below then retargets this
        # same mount (along with every other branch already hosted there)
        # from "d{digit_id}p2" to "d{digit_id}p1", so the reattached branch
        # ends up on the correct final phalanx-0 body. Never the digit's
        # own ``mount``: that can be a palm body when this digit is
        # top-level, and a branch digit must never mount on a palm body.
        reattach_mount = f"d{digit_id}p{del_p + 2}"
    non_digit_others = [
        DerivationStep(path=st.path, production="Digit", params={**st.params, "mount": reattach_mount})
        if (st.production == "Digit" and not st.params.get("top_level", True)
            and st.params.get("mount") == deleted_body)
        else st
        for st in non_digit_others
    ]

    renumber: Dict[int, int] = {}
    new_phalanx_list: List[DerivationStep] = []
    for old_p in range(old_count):
        if old_p == del_p:
            continue
        new_p = old_p if old_p < del_p else old_p - 1
        renumber[old_p] = new_p
        st = phalanx_steps[old_p]
        params = dict(st.params)
        mod = dict(params["module"])
        if mod.get("kind") == "Coupled":
            sp = mod["source_p"]
            if sp == del_p:
                # Source phalanx removed: fall back to a fresh, non-Coupled module.
                mod = sample_module(rng, dist, 0, allowed=None if lim is None else lim.allowed_modules)
            elif sp > del_p:
                mod["source_p"] = sp - 1
        params["module"] = mod
        params["p"] = new_p
        new_phalanx_list.append(
            DerivationStep(path=f"digit/{digit_id}/phalanx/{new_p}", production="Phalanx", params=params))

    new_dstep = DerivationStep(path=dstep.path, production="Digit",
                                params={**dstep.params, "phalanx_count": old_count - 1})
    result = non_digit_others + [new_dstep] + new_phalanx_list
    return _rename_branch_mounts(result, digit_id, renumber)


def _op_add_digit(rng, dist: Distribution, derivation: Derivation,
                  lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """``lim``: needs a host with room; the new digit is sampled under the
    limits."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    if hand_params["digit_count"] >= dist.digit_count_range[1]:
        return None
    mount_bodies = _mount_bodies_from_steps(steps)
    if lim is not None and not lim.eligible_hosts(mount_bodies):
        return None
    top_ids = [int(s.params["digit_id"]) for s in steps if s.production == "Digit" and s.params.get("top_level")]
    next_id = [max(top_ids, default=0) + 1]
    new_steps: List[DerivationStep] = []
    next_uid = [_max_uid(steps) + 1]
    _sample_digit(rng, _growth_dist(dist), new_steps, next_id, mount_bodies, next_uid, lim=lim)
    new_hand = DerivationStep(path="hand", production="Hand",
                               params={**hand_params, "digit_count": hand_params["digit_count"] + 1})
    rest = [s for i, s in enumerate(steps) if i != hand_idx]
    return rest + [new_hand] + new_steps


def _op_remove_digit(rng, dist: Distribution, derivation: Derivation,
                     lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    if hand_params["digit_count"] <= 1:
        return None
    top_ids = [s.params["digit_id"] for s in steps if s.production == "Digit" and s.params.get("top_level")]
    if not top_ids:
        return None
    digit_id = top_ids[int(rng.integers(0, len(top_ids)))]
    kept = [s for s in steps if not _is_descendant_digit(digit_id, _step_digit_id(s))]
    new_hand = DerivationStep(path="hand", production="Hand",
                               params={**hand_params, "digit_count": hand_params["digit_count"] - 1})
    return [s if s.path != "hand" else new_hand for s in kept]


# --------------------------------------------------------------------------
# Small-step operators (opt-in only -- see ``vary``'s ``operators`` argument
# and ``SMALL_STEP_OPERATORS`` below). Each edits EXACTLY one field of one
# existing step (never adds/removes a step), by moving that field to a
# grid-neighbouring choice rather than resampling it fresh -- unlike
# ``resample_parameter``/``perturb_parameter`` above, which redraw a field
# from its whole distribution. Adding these to ``_OPERATOR_FNS`` does not
# change ``OPERATORS`` or default ``vary`` behaviour: a caller only reaches
# them by naming them explicitly (``operator=...``) or opting into
# ``operators=SMALL_STEP_OPERATORS``.
# --------------------------------------------------------------------------

SMALL_STEP_OPERATORS: Tuple[str, ...] = (
    "step_axis",
    "step_limits",
    "step_mount",
    "step_coupling",
    # I14 fix 5: ``step_length`` was an exact alias of ``perturb_parameter``
    # (same function, see ``_OPERATOR_FNS`` below) -- included here it gave
    # any pool containing both double weight on the same effect. The
    # function itself (and the ``"step_length"`` key in ``_OPERATOR_FNS``,
    # reachable via ``operator="step_length"``) stays, for compatibility;
    # only the default small-step POOL no longer draws it.
    "step_root_length",
    "step_radius",
    # Grammar 0.5 (I16 priority 1): one grid step on one component of a
    # phalanx's own bend_rpy/bend_offset (see _op_step_bend_rpy/_offset
    # below).
    "step_bend_rpy",
    "step_bend_offset",
)


def _axis_grid_indices(axis: Tuple[float, float, float]) -> Tuple[int, int]:
    """Invert ``distributions.sample_axis``'s grid: find the (elevation,
    azimuth) step indices whose axis matches ``axis`` exactly (every axis
    stored in a derivation was produced by that same formula, so this is an
    exact float match, not a nearest-neighbour search)."""
    x, y, z = axis
    for el_k in range(N_ELEVATION_STEPS):
        el = el_k * ANGLE_STEP_DEG * DEG
        for az_k in range(N_ANGLE_STEPS):
            az = (az_k * ANGLE_STEP_DEG - 180.0) * DEG
            xx, yy, zz = math.sin(el) * math.cos(az), math.sin(el) * math.sin(az), math.cos(el)
            if abs(xx - x) < 1e-9 and abs(yy - y) < 1e-9 and abs(zz - z) < 1e-9:
                return el_k, az_k
    raise ValueError(f"axis {axis!r} is not on the sampling grid")


def _step_axis_value(rng, axis: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """One grid step in elevation (clamped to [0, N_ELEVATION_STEPS-1]) or
    azimuth (wrapped, since azimuth is circular), chosen at random."""
    el_k, az_k = _axis_grid_indices(axis)
    step_elevation = bool(rng.integers(0, 2))
    direction = 1 if bool(rng.integers(0, 2)) else -1
    if step_elevation:
        el_k = max(0, min(N_ELEVATION_STEPS - 1, el_k + direction))
    else:
        az_k = (az_k + direction) % N_ANGLE_STEPS
    el = el_k * ANGLE_STEP_DEG * DEG
    az = (az_k * ANGLE_STEP_DEG - 180.0) * DEG
    return (float(math.sin(el) * math.cos(az)), float(math.sin(el) * math.sin(az)), float(math.cos(el)))


def _op_step_axis(rng, dist: Distribution, derivation: Derivation,
                  lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    candidates = [
        i for i, s in enumerate(steps)
        if (s.production == "PalmBody" and s.params.get("has_joint"))
        or s.production == "Phalanx"
    ]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    try:
        if s.production == "PalmBody":
            p["axis"] = _step_axis_value(rng, p["axis"])
        else:
            mod = dict(p["module"])
            mod["axis"] = _step_axis_value(rng, mod["axis"])
            p["module"] = mod
    except ValueError:
        return None
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


def _find_choice_index(value: Tuple[float, float], choices: Sequence[Tuple[float, float]],
                        scale: float) -> Optional[int]:
    for i, (lo, hi) in enumerate(choices):
        if abs(lo * scale - value[0]) < 1e-9 and abs(hi * scale - value[1]) < 1e-9:
            return i
    return None


def _step_choice_index(rng, idx: int, n: int) -> int:
    if n <= 1:
        return idx
    direction = 1 if bool(rng.integers(0, 2)) else -1
    return max(0, min(n - 1, idx + direction))


def _op_step_limits(rng, dist: Distribution, derivation: Derivation,
                    lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """I14 fix: ``dist.*_limit_choices_*`` are sorted HERE (a local copy, on
    every call) before finding/stepping the current choice's index, so a
    "neighbouring" choice is numerically adjacent (e.g. never jumps a
    revolute joint's limits from ``(0, 110)`` straight to ``(-30, 60)``, nor
    flips a multiplier's sign) -- WITHOUT reordering the ``Distribution``
    field itself, since ``resample_parameter``/``sample_module`` pick a
    choice by RAW index into that same (unsorted) tuple, and reordering it
    would silently change every existing seed's default-sampling replay."""
    steps = list(derivation.steps)
    candidates = [
        i for i, s in enumerate(steps)
        if (s.production == "PalmBody" and s.params.get("has_joint"))
        or (s.production == "Phalanx" and s.params["module"]["kind"] in ("R", "P"))
    ]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    if s.production == "PalmBody":
        choices = sorted(dist.palm_joint_limit_choices_deg)
        ci = _find_choice_index(p["limits"], choices, scale=DEG)
        if ci is None:
            return None
        new_ci = _step_choice_index(rng, ci, len(choices))
        if new_ci == ci:
            return None
        lo_deg, hi_deg = choices[new_ci]
        p["limits"] = (lo_deg * DEG, hi_deg * DEG)
        steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
        return steps
    mod = dict(p["module"])
    if mod["kind"] == "R" and dist.limits_continuous:
        # Grammar 0.5 (I16 priority 2): move ONE bound by one
        # ``dist.limit_step_deg`` grid step, clamped within
        # ``dist.revolute_limit_range_deg`` and away from the other bound
        # (never crossing it, so ``lo < hi`` always still holds).
        lo, hi = mod["limits"]
        range_lo_deg, range_hi_deg = dist.revolute_limit_range_deg
        range_lo, range_hi = range_lo_deg * DEG, range_hi_deg * DEG
        step = dist.limit_step_deg * DEG
        direction = 1.0 if bool(rng.integers(0, 2)) else -1.0
        if bool(rng.integers(0, 2)):
            new_lo = max(range_lo, min(lo + direction * step, min(range_hi, hi - 1e-9)))
            if abs(new_lo - lo) < 1e-12:
                return None
            mod["limits"] = (new_lo, hi)
        else:
            new_hi = min(range_hi, max(hi + direction * step, max(range_lo, lo + 1e-9)))
            if abs(new_hi - hi) < 1e-12:
                return None
            mod["limits"] = (lo, new_hi)
        p["module"] = mod
        steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
        return steps
    if mod["kind"] == "R":
        choices, scale = sorted(dist.revolute_limit_choices_deg), DEG
    else:
        choices, scale = sorted(dist.prismatic_limit_choices_m), 1.0
    ci = _find_choice_index(mod["limits"], choices, scale=scale)
    if ci is None:
        return None
    new_ci = _step_choice_index(rng, ci, len(choices))
    if new_ci == ci:
        return None
    lo, hi = choices[new_ci]
    mod["limits"] = (lo * scale, hi * scale)
    p["module"] = mod
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


def _step_one_grid_angle(rng, current: float) -> float:
    k = int(round((current / DEG + 180.0) / ANGLE_STEP_DEG)) % N_ANGLE_STEPS
    direction = 1 if bool(rng.integers(0, 2)) else -1
    new_k = (k + direction) % N_ANGLE_STEPS
    return (new_k * ANGLE_STEP_DEG - 180.0) * DEG


def _op_step_mount(rng, dist: Distribution, derivation: Derivation,
                   lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    candidates = [i for i, s in enumerate(steps) if s.production in ("Digit", "PalmBody")]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    # A ``PalmBody`` step has no ``mount_rpy`` field of its own; its
    # ``direction_rpy`` (the segment's own orientation) plays the analogous
    # role and is stepped the same way.
    rpy_field = "mount_rpy" if s.production == "Digit" else "direction_rpy"
    if dist.mount_lateral_grid_m is not None and (
            s.production == "PalmBody"
            or (s.production == "Digit" and p.get("top_level") and not dist.mount_on_host_surface)):
        # Lateral digit mounts (off by default): a third choice, one grid
        # step of one offset component. Never reached for an existing
        # variant, so their draws are unchanged.
        which = int(rng.integers(0, 3))
        if which == 2:
            grid = list(lateral_offset_choices_m(dist))
            off = list(p.get("mount_offset", (0.0, 0.0)))
            comp = int(rng.integers(0, 2))
            if off[comp] not in grid:
                return None
            ci = grid.index(off[comp])
            new_ci = _step_choice_index(rng, ci, len(grid))
            if new_ci == ci:
                return None
            off[comp] = grid[new_ci]
            p["mount_offset"] = tuple(off)
            steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
            return steps
        step_frac = which == 1
    else:
        step_frac = bool(rng.integers(0, 2))
    if step_frac:
        choices = list(dist.mount_frac_choices)
        if p["mount_frac"] not in choices or len(choices) <= 1:
            return None
        ci = choices.index(p["mount_frac"])
        new_ci = _step_choice_index(rng, ci, len(choices))
        if new_ci == ci:
            return None
        p["mount_frac"] = choices[new_ci]
    else:
        rpy = list(p[rpy_field])
        axis_i = int(rng.integers(0, 3))
        rpy[axis_i] = _step_one_grid_angle(rng, rpy[axis_i])
        p[rpy_field] = tuple(rpy)
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


def _op_step_coupling(rng, dist: Distribution, derivation: Derivation,
                      lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    candidates = [
        i for i, s in enumerate(steps)
        if s.production == "Phalanx" and s.params["module"]["kind"] == "Coupled"
    ]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    mod = dict(p["module"])
    field = "multiplier" if bool(rng.integers(0, 2)) else "offset"
    # I14 fix: sorted locally (see ``_op_step_limits``'s docstring) so a
    # neighbour is numerically adjacent, e.g. never flips a multiplier's
    # sign in one step.
    choices = sorted(dist.coupling_multiplier_choices if field == "multiplier" else dist.coupling_offset_choices_rad)
    cur = mod[field]
    if cur not in choices or len(choices) <= 1:
        return None
    ci = choices.index(cur)
    new_ci = _step_choice_index(rng, ci, len(choices))
    if new_ci == ci:
        return None
    mod[field] = choices[new_ci]
    p["module"] = mod
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


def _op_step_root_length(rng, dist: Distribution, derivation: Derivation,
                         lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """I14 fix 5: the ``hand`` step's own ``root_length`` (the root/palm
    segment's length), previously never mutated by ANY operator, stepped by
    one ``dist.link_length_grid_m`` grid increment (up or down, clamped
    within ``dist.palm_length_range_m`` -- the same range/grid
    ``root_length`` is originally sampled from)."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand = steps[hand_idx]
    lo, hi = dist.palm_length_range_m
    grid = dist.link_length_grid_m
    n = int(round((hi - lo) / grid))
    cur = hand.params["root_length"]
    ci = int(round((cur - lo) / grid))
    ci = max(0, min(n, ci))
    new_ci = _step_choice_index(rng, ci, n + 1)
    if new_ci == ci:
        return None
    p = dict(hand.params)
    p["root_length"] = round(lo + new_ci * grid, 10)
    steps[hand_idx] = DerivationStep(path="hand", production="Hand", params=p)
    return steps


def _op_step_radius(rng, dist: Distribution, derivation: Derivation,
                    lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """I14 fix 5: the ``hand`` step's own ``capsule_radius_m`` (one scalar
    stamped onto every ``Body.radius`` -- previously never mutated by any
    operator), stepped to a numerically-neighbouring choice in
    ``dist.capsule_radius_choices_m`` (sorted locally, same rationale as
    ``_op_step_limits``)."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand = steps[hand_idx]
    choices = sorted(dist.capsule_radius_choices_m)
    cur = hand.params["capsule_radius_m"]
    if cur not in choices or len(choices) <= 1:
        return None
    ci = choices.index(cur)
    new_ci = _step_choice_index(rng, ci, len(choices))
    if new_ci == ci:
        return None
    p = dict(hand.params)
    p["capsule_radius_m"] = choices[new_ci]
    steps[hand_idx] = DerivationStep(path="hand", production="Hand", params=p)
    return steps


def _bend_component_grid(choices, axis_index: int) -> List[float]:
    """Sorted, de-duplicated set of the ``axis_index``-th component across
    every whole triple/pair in ``choices`` (``dist.bend_rpy_choices_rad`` --
    3 components -- or ``dist.bend_offset_choices_m`` -- 2 components).
    ``variants.G_BEND`` builds each of these choice sets as the full
    Cartesian product of one small per-component grid, so this recovers
    that per-component grid for the small-step operators below (which move
    ONE component by one grid step, unlike ``step_limits``/``step_coupling``,
    which step the whole tuple to its neighbour in a flat choice list)."""
    return sorted({c[axis_index] for c in choices})


def _op_step_bend_rpy(rng, dist: Distribution, derivation: Derivation,
                      lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """Grammar 0.5 (I16 priority 1): move ONE component (roll, pitch, or
    yaw) of one Phalanx step's own ``bend_rpy`` to its numerically
    neighbouring value on ``dist.bend_rpy_choices_rad``'s own per-component
    grid (see ``_bend_component_grid``). A no-op (``None``) whenever that
    component's grid has only one value (true of every default
    ``Distribution``, whose ``bend_rpy_choices_rad`` is the single value
    ``(0.0, 0.0, 0.0)``) or the current value is not itself on that grid."""
    steps = list(derivation.steps)
    candidates = [i for i, s in enumerate(steps) if s.production == "Phalanx"]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    bend_rpy = list(p.get("bend_rpy", (0.0, 0.0, 0.0)))
    comp = int(rng.integers(0, 3))
    grid = _bend_component_grid(dist.bend_rpy_choices_rad, comp)
    if len(grid) <= 1 or bend_rpy[comp] not in grid:
        return None
    ci = grid.index(bend_rpy[comp])
    new_ci = _step_choice_index(rng, ci, len(grid))
    if new_ci == ci:
        return None
    bend_rpy[comp] = grid[new_ci]
    p["bend_rpy"] = tuple(bend_rpy)
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


def _op_step_bend_offset(rng, dist: Distribution, derivation: Derivation,
                         lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """Grammar 0.5 (I16 priority 1): the ``bend_offset`` counterpart of
    ``_op_step_bend_rpy`` -- moves ONE component (x or y) of one Phalanx
    step's own ``bend_offset`` to its numerically neighbouring value on
    ``dist.bend_offset_choices_m``'s own per-component grid."""
    steps = list(derivation.steps)
    candidates = [i for i, s in enumerate(steps) if s.production == "Phalanx"]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    bend_offset = list(p.get("bend_offset", (0.0, 0.0)))
    comp = int(rng.integers(0, 2))
    grid = _bend_component_grid(dist.bend_offset_choices_m, comp)
    if len(grid) <= 1 or bend_offset[comp] not in grid:
        return None
    ci = grid.index(bend_offset[comp])
    new_ci = _step_choice_index(rng, ci, len(grid))
    if new_ci == ci:
        return None
    bend_offset[comp] = grid[new_ci]
    p["bend_offset"] = tuple(bend_offset)
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


# --------------------------------------------------------------------------
# Minimal structural operators (opt-in only -- see ``MINIMAL_STRUCTURAL_OPERATORS``
# below). Motivation (E1): ``add_digit``/``remove_digit`` change ~7 joints in
# one application because a new digit is sampled at full size, and no
# default operator changes palm structure at all. Each operator here edits
# structure by the smallest possible increment (one single-phalanx digit, one
# palm body, one palm joint's presence). Adding these to ``_OPERATOR_FNS``
# does not change ``OPERATORS`` or default ``vary`` behaviour: a caller only
# reaches them via ``operator=...`` or ``operators=MINIMAL_STRUCTURAL_OPERATORS``.
# --------------------------------------------------------------------------

MINIMAL_STRUCTURAL_OPERATORS: Tuple[str, ...] = (
    "add_minimal_digit",
    "add_palm_body",
    "remove_palm_body",
    "toggle_palm_joint",
    "remove_digit_minimal",
)


def _op_add_minimal_digit(rng, dist: Distribution, derivation: Derivation,
                          lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """Add a new top-level digit with exactly one phalanx (module kind
    always ``"R"`` -- a single-phalanx digit has no earlier phalanx to
    couple to, so ``"Coupled"`` is never valid there anyway; forcing ``"R"``
    keeps this operator's effect exactly one movable revolute joint, never a
    continuous/prismatic one). Mounts on a uniformly sampled existing body
    (root or a palm body), like a fresh top-level digit."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    if hand_params["digit_count"] >= dist.digit_count_range[1]:
        return None
    gdist = _growth_dist(dist)
    mount_bodies = _mount_bodies_from_steps(steps)
    if lim is not None:
        # Generation limits: the new digit is one hinge, so "R" must be
        # allowed, and it mounts only on a host with room.
        if not lim.module_allowed("R"):
            return None
        mount_bodies = lim.eligible_hosts(mount_bodies)
        if not mount_bodies:
            return None
    top_ids = [int(s.params["digit_id"]) for s in steps if s.production == "Digit" and s.params.get("top_level")]
    digit_id = str(max(top_ids, default=0) + 1)
    mount = mount_bodies[int(rng.integers(0, len(mount_bodies)))]
    mount_frac = float(gdist.mount_frac_choices[int(rng.integers(0, len(gdist.mount_frac_choices)))])
    mount_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
    axis = sample_axis(rng)
    limits = sample_revolute_limits_rad(rng, gdist)
    length = sample_grid_length_m(rng, gdist.link_length_range_m, gdist.link_length_grid_m)
    bend_rpy, bend_offset = sample_bend(rng, gdist)
    uid_base = _max_uid(steps) + 1
    digit_params = {
        "digit_id": digit_id, "mount": mount, "mount_frac": mount_frac, "mount_rpy": mount_rpy,
        "phalanx_count": 1, "top_level": True, "depth": 0, "uid": uid_base,
    }
    if gdist.mount_lateral_grid_m is not None and not gdist.mount_on_host_surface:
        digit_params["mount_offset"] = sample_lateral_offset(rng, gdist)
    digit_step = DerivationStep(path=f"digit/{digit_id}", production="Digit", params=digit_params)
    phalanx_step = DerivationStep(path=f"digit/{digit_id}/phalanx/0", production="Phalanx", params={
        "digit_id": digit_id, "p": 0, "module": {"kind": "R", "axis": axis, "limits": limits},
        "length": length, "branch_digit_count": 0, "uid": uid_base + 1,
        "bend_rpy": bend_rpy, "bend_offset": bend_offset,
    })
    new_hand = DerivationStep(path="hand", production="Hand",
                               params={**hand_params, "digit_count": hand_params["digit_count"] + 1})
    rest = [s for i, s in enumerate(steps) if i != hand_idx]
    return rest + [new_hand, digit_step, phalanx_step]


def _op_remove_digit_minimal(rng, dist: Distribution, derivation: Derivation,
                              target: Optional[int] = None,
                              lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """Reversible counterpart of ``add_minimal_digit``: remove a top-level
    digit with 1 or 2 phalanges (I15 fix 4, widened from exactly 1 --
    ``add_minimal_digit`` itself only ever ADDS a 1-phalanx digit, but
    ``insert_phalanx``/other operators applied afterward can grow that
    digit to 2 phalanges without this operator's own ratchet ever letting
    it back down again), none of which host a branch (removing the whole
    digit here -- via ``_is_descendant_digit`` -- also removes any branch
    nested under it, which would make this "minimal" operator silently
    remove a much larger subtree; ``delete_phalanx`` is the operator that
    handles branch re-attachment).

    ``target`` (grammar 0.5, reversibility): when given, the uid of the
    ``Digit`` step to remove -- restricts the candidate pool to exactly that
    digit (still subject to every safety check above) instead of drawing
    uniformly, so a caller (the reversibility test) can undo a specific
    ``add_minimal_digit`` application precisely."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    if hand_params["digit_count"] <= 1:
        return None
    phalanx_by_digit: Dict[str, List[DerivationStep]] = {}
    for s in steps:
        if s.production == "Phalanx":
            phalanx_by_digit.setdefault(s.params["digit_id"], []).append(s)
    candidates = [
        s.params["digit_id"] for s in steps
        if s.production == "Digit" and s.params.get("top_level") and s.params["phalanx_count"] in (1, 2)
        and len(phalanx_by_digit.get(s.params["digit_id"], [])) == s.params["phalanx_count"]
        and all(ph.params["branch_digit_count"] == 0 for ph in phalanx_by_digit.get(s.params["digit_id"], []))
        and (target is None or s.params.get("uid") == target)
    ]
    if not candidates:
        return None
    digit_id = candidates[int(rng.integers(0, len(candidates)))]
    kept = [s for s in steps if not _is_descendant_digit(digit_id, _step_digit_id(s))]
    new_hand = DerivationStep(path="hand", production="Hand",
                               params={**hand_params, "digit_count": hand_params["digit_count"] - 1})
    return [s if s.path != "hand" else new_hand for s in kept]


def _op_add_palm_body(rng, dist: Distribution, derivation: Derivation,
                      lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """Add one new palm body (index ``palm_body_count``), sampled exactly
    like ``sample_derivation``'s own palm-body loop: a sampled parent among
    root and every existing palm body, a sampled mount fraction/length/
    direction, and a palm joint present with probability
    ``dist.palm_joint_probability``. Always appended at the end, so no
    existing palm/digit step's name or path needs renumbering."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    palm_body_count = hand_params["palm_body_count"]
    if palm_body_count >= dist.palm_body_count_range[1]:
        return None
    if lim is not None and not lim.allows(lim.struct.with_palm("__new_palm", "root", False)):
        return None             # generation limits: no room for another palm body
    gdist = _growth_dist(dist)
    palm_names = [f"palm{i}" for i in range(palm_body_count)]
    parent_choices = ["root"] + palm_names
    parent = parent_choices[int(rng.integers(0, len(parent_choices)))]
    mount_frac = float(gdist.mount_frac_choices[int(rng.integers(0, len(gdist.mount_frac_choices)))])
    length = sample_grid_length_m(rng, gdist.palm_length_range_m, gdist.link_length_grid_m)
    direction_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
    has_joint = bool(float(rng.random()) < gdist.palm_joint_probability)
    if has_joint and lim is not None and not lim.allows(lim.struct.with_palm("__new_palm", parent, True)):
        has_joint = False       # generation limits: a joint here is not allowed, so the body is rigid
    axis = sample_axis(rng)
    limits = sample_palm_joint_limits_rad(rng, gdist) if has_joint else None
    name = f"palm{palm_body_count}"
    palm_params = {
        "name": name, "parent": parent, "mount_frac": mount_frac, "length": length,
        "direction_rpy": direction_rpy, "has_joint": has_joint, "axis": axis, "limits": limits,
        "uid": _max_uid(steps) + 1,
    }
    if gdist.mount_lateral_grid_m is not None:
        palm_params["mount_offset"] = sample_lateral_offset(rng, gdist)
    new_step = DerivationStep(path=f"palm/{palm_body_count}", production="PalmBody", params=palm_params)
    new_hand = DerivationStep(path="hand", production="Hand",
                               params={**hand_params, "palm_body_count": palm_body_count + 1})
    rest = [s for i, s in enumerate(steps) if i != hand_idx]
    return rest + [new_hand, new_step]


def _op_remove_palm_body(rng, dist: Distribution, derivation: Derivation,
                          target: Optional[int] = None,
                          lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """``target`` (grammar 0.5, reversibility): when given, the uid of the
    ``PalmBody`` step to remove -- restricts the candidate pool to exactly
    that body instead of drawing uniformly, so a caller (the reversibility
    test) can undo a specific ``add_palm_body`` application precisely.

    Remove ANY existing palm body (I14 fix 5 -- previously restricted to
    a leaf palm body with no digit/palm children, which made this operator
    far less often applicable than ``add_palm_body``, an asymmetry the
    review flagged). If the removed body ``X`` hosts palm children
    (``PalmBody`` steps whose ``parent == X``) or digit mounts (``Digit``
    steps whose ``mount == X``), those are RE-ATTACHED to ``X``'s own
    parent, at ``X``'s own ``mount_frac`` on that parent (i.e. wherever
    ``X`` itself used to attach) -- an approximation (the reattached
    child's position along ``X`` itself, and ``X``'s own segment length,
    are both dropped), not an exact geometric inverse of ``add_palm_body``,
    but one that makes add/remove close to symmetric: applying
    ``add_palm_body`` then ``remove_palm_body`` on the body it just added
    always succeeds and returns to a body count matching the start (the
    grammar's own `parent index < child index` invariant guarantees ``X``'s
    parent is never itself renumbered by this removal, since only bodies
    with an index ABOVE ``X`` shift down).

    Every palm body with a higher index than the removed one is renumbered
    down by one (name and ``path``)."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    palm_body_count = hand_params["palm_body_count"]
    if palm_body_count == 0:
        return None
    palm_steps = {s.params["name"]: s for s in steps if s.production == "PalmBody"}
    if target is None:
        candidates = list(palm_steps.keys())
    else:
        candidates = [name for name, s in palm_steps.items() if s.params.get("uid") == target]
    if lim is not None:
        # Generation limits: re-attaching the removed body's digits to its
        # parent must not overfill the parent's finger chain or carrier.
        candidates = [n for n in candidates if lim.allows(lim.struct.without_palm_reattach(n))]
    if not candidates:
        return None
    remove_name = candidates[int(rng.integers(0, len(candidates)))]
    remove_step = palm_steps[remove_name]
    remove_idx = int(remove_name[len("palm"):])
    remove_parent = remove_step.params["parent"]
    remove_mount_frac = remove_step.params["mount_frac"]

    rename: Dict[str, str] = {
        f"palm{i}": f"palm{i - 1}" for i in range(remove_idx + 1, palm_body_count)
    }

    new_steps: List[DerivationStep] = []
    for s in steps:
        if s.path == f"palm/{remove_idx}":
            continue
        if s.production == "PalmBody":
            old_i = int(s.params["name"][len("palm"):])
            new_i = old_i if old_i < remove_idx else old_i - 1
            p = dict(s.params)
            p["name"] = f"palm{new_i}"
            if p["parent"] == remove_name:
                p["parent"] = remove_parent
                p["mount_frac"] = remove_mount_frac
            elif p["parent"] in rename:
                p["parent"] = rename[p["parent"]]
            new_steps.append(DerivationStep(path=f"palm/{new_i}", production="PalmBody", params=p))
        elif s.production == "Digit" and s.params.get("mount") == remove_name:
            new_steps.append(DerivationStep(path=s.path, production="Digit",
                                             params={**s.params, "mount": remove_parent,
                                                     "mount_frac": remove_mount_frac}))
        elif s.production == "Digit" and s.params.get("mount") in rename:
            new_steps.append(DerivationStep(path=s.path, production="Digit",
                                             params={**s.params, "mount": rename[s.params["mount"]]}))
        elif s.path == "hand":
            new_steps.append(DerivationStep(path="hand", production="Hand",
                                             params={**hand_params, "palm_body_count": palm_body_count - 1}))
        else:
            new_steps.append(s)
    return new_steps


def _op_remove_palm_body_empty(rng, dist: Distribution, derivation: Derivation,
                                target: Optional[int] = None,
                                lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """Exact-inverse counterpart of ``add_palm_body`` (grammar 0.5, I18 fix
    2): remove a LEAF palm body -- one with no ``PalmBody`` child (no other
    palm body's ``parent`` names it) and no ``Digit`` mounted on it -- i.e.
    exactly the shape ``add_palm_body`` always produces (it is always
    appended at the end, with nothing yet attached to it). Unlike the
    general ``remove_palm_body`` (which accepts ANY existing body and
    approximately re-attaches its children to its parent), this never
    re-attaches anything, since a leaf by construction has nothing to
    re-attach; it is used as the shrink half of the ``add_palm_body`` pair
    in ``EVOLUTION_PAIRS``/``EVOLUTION_OPERATORS`` so that pair's neutral
    walk is a genuine exact inverse rather than the approximate,
    asymmetric general removal. ``remove_palm_body`` itself stays available
    outside the evolution pool as a larger, non-exact-inverse move.

    ``target``: uid of the ``PalmBody`` step to remove (restricts the
    candidate pool to exactly that body instead of drawing uniformly, so
    the reversibility test can undo a specific ``add_palm_body``
    application precisely)."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    palm_body_count = hand_params["palm_body_count"]
    if palm_body_count == 0:
        return None
    palm_steps = {s.params["name"]: s for s in steps if s.production == "PalmBody"}
    parent_names = {s.params["parent"] for s in steps if s.production == "PalmBody"}
    mount_names = {s.params["mount"] for s in steps if s.production == "Digit"}
    leaf_names = [name for name in palm_steps if name not in parent_names and name not in mount_names]
    if target is not None:
        leaf_names = [name for name in leaf_names if palm_steps[name].params.get("uid") == target]
    if not leaf_names:
        return None
    remove_name = leaf_names[int(rng.integers(0, len(leaf_names)))]
    remove_idx = int(remove_name[len("palm"):])

    rename: Dict[str, str] = {
        f"palm{i}": f"palm{i - 1}" for i in range(remove_idx + 1, palm_body_count)
    }
    new_steps: List[DerivationStep] = []
    for s in steps:
        if s.path == f"palm/{remove_idx}":
            continue
        if s.production == "PalmBody":
            old_i = int(s.params["name"][len("palm"):])
            new_i = old_i if old_i < remove_idx else old_i - 1
            p = dict(s.params)
            p["name"] = f"palm{new_i}"
            if p["parent"] in rename:
                p["parent"] = rename[p["parent"]]
            new_steps.append(DerivationStep(path=f"palm/{new_i}", production="PalmBody", params=p))
        elif s.production == "Digit" and s.params.get("mount") in rename:
            new_steps.append(DerivationStep(path=s.path, production="Digit",
                                             params={**s.params, "mount": rename[s.params["mount"]]}))
        elif s.path == "hand":
            new_steps.append(DerivationStep(path="hand", production="Hand",
                                             params={**hand_params, "palm_body_count": palm_body_count - 1}))
        else:
            new_steps.append(s)
    return new_steps


def _op_toggle_palm_joint(rng, dist: Distribution, derivation: Derivation,
                           target: Optional[int] = None,
                           lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """On one existing (necessarily non-root -- the root has no ``PalmBody``
    step of its own) palm body, add a palm joint (sampled axis/limits) if
    absent, or remove it (``has_joint=False``, ``limits=None``) if present.

    Self-inverse (grammar 0.5, reversibility): toggling twice on the SAME
    body restores it exactly (on/off keeps the stored axis only if the
    limits were also byte-identical -- toggling off then on again resamples
    axis/limits, so use ``target`` below rather than relying on a second
    random toggle happening to land on the same body/axis). ``target``, when
    given, is the uid of the ``PalmBody`` step to toggle -- restricts the
    candidate pool to exactly that body instead of drawing uniformly."""
    steps = list(derivation.steps)
    if target is None:
        candidates = [i for i, s in enumerate(steps) if s.production == "PalmBody"]
    else:
        candidates = [i for i, s in enumerate(steps)
                      if s.production == "PalmBody" and s.params.get("uid") == target]
    if lim is not None:
        # Generation limits: only bodies whose joint may be added (cap, no
        # stacking, carried digits, finger chains) or removed (its digits fit
        # where they fall back to).
        candidates = [i for i in candidates
                      if lim.allows(lim.struct.with_joint(steps[i].params["name"], not steps[i].params["has_joint"]))]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    if p["has_joint"]:
        p["has_joint"] = False
        p["limits"] = None
    else:
        p["has_joint"] = True
        p["axis"] = sample_axis(rng)
        p["limits"] = sample_palm_joint_limits_rad(rng, dist)
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


# --------------------------------------------------------------------------
# Branch-digit operators (grammar 0.5, iteration B): the exact-inverse pair
# for "branch" in ``EVOLUTION_OPERATORS``' operator table (add a
# single-phalanx branch digit on a sampled existing phalanx body / remove a
# branch digit with exactly 1 phalanx and no sub-branches of its own). Each
# mirrors ``add_minimal_digit``/``remove_digit_minimal`` but for a BRANCH
# digit (mounted on a phalanx body, ``top_level=False``) rather than a
# top-level one -- unlike a fresh top-level digit, a branch digit's HOST
# phalanx step tracks how many branches it hosts (``branch_digit_count``,
# see ``rules.PhalanxProduction``'s docstring), which both operators keep
# accurate (read by ``_op_delete_phalanx``'s own safety check and by
# ``_op_remove_digit_minimal``'s "no branch" gate).
# --------------------------------------------------------------------------


def _op_add_branch_digit(rng, dist: Distribution, derivation: Derivation,
                         lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """Add one new branch digit (1 phalanx, mirrors ``add_minimal_digit``)
    on a uniformly sampled existing ``Phalanx`` step's own body, subject to
    ``dist.max_branch_depth`` (the host digit's own depth must still be
    below it) and ``dist.max_branch_digits`` (the host phalanx must not
    already host the max). New material (module/length/bend/mount pose) is
    drawn from ``_growth_dist(dist)``, like every other growth operator."""
    steps = list(derivation.steps)
    if lim is not None and not (lim.limits.allow_branches and lim.module_allowed("R")):
        return None             # generation limits: no branches, or no hinge for the one-hinge branch digit
    digit_depth = {s.params["digit_id"]: s.params["depth"] for s in steps if s.production == "Digit"}
    candidates = [
        i for i, s in enumerate(steps)
        if s.production == "Phalanx"
        and digit_depth.get(s.params["digit_id"], 0) < dist.max_branch_depth
        and s.params["branch_digit_count"] < dist.max_branch_digits
    ]
    if lim is not None and lim.limits.max_joints_per_digit is not None:
        candidates = [i for i in candidates if lim.joint_room(steps[i].params["digit_id"]) >= 1]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    host = steps[idx]
    host_digit_id = host.params["digit_id"]
    host_p = host.params["p"]
    host_body = f"d{host_digit_id}p{host_p + 1}"
    slot = host.params["branch_digit_count"]
    sub_id = f"{host_digit_id}p{host_p + 1}b{slot}"
    depth = digit_depth[host_digit_id] + 1

    gdist = _growth_dist(dist)
    mount_frac = float(gdist.mount_frac_choices[int(rng.integers(0, len(gdist.mount_frac_choices)))])
    mount_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
    axis = sample_axis(rng)
    limits = sample_revolute_limits_rad(rng, gdist)
    length = sample_grid_length_m(rng, gdist.link_length_range_m, gdist.link_length_grid_m)
    bend_rpy, bend_offset = sample_bend(rng, gdist)
    uid_base = _max_uid(steps) + 1

    digit_step = DerivationStep(path=f"digit/{sub_id}", production="Digit", params={
        "digit_id": sub_id, "mount": host_body, "mount_frac": mount_frac, "mount_rpy": mount_rpy,
        "phalanx_count": 1, "top_level": False, "depth": depth, "uid": uid_base,
    })
    phalanx_step = DerivationStep(path=f"digit/{sub_id}/phalanx/0", production="Phalanx", params={
        "digit_id": sub_id, "p": 0, "module": {"kind": "R", "axis": axis, "limits": limits},
        "length": length, "branch_digit_count": 0, "uid": uid_base + 1,
        "bend_rpy": bend_rpy, "bend_offset": bend_offset,
    })
    new_host = DerivationStep(path=host.path, production="Phalanx",
                               params={**host.params, "branch_digit_count": slot + 1})
    rest = [s if i != idx else new_host for i, s in enumerate(steps)]
    return rest + [digit_step, phalanx_step]


def _op_remove_branch_digit(rng, dist: Distribution, derivation: Derivation,
                             target: Optional[int] = None,
                             lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """Exact-inverse counterpart of ``add_branch_digit``: remove a branch
    digit (``top_level=False``) with exactly 1 phalanx and no sub-branches
    of its own, decrementing its HOST phalanx's ``branch_digit_count``.
    ``target``: uid of the branch ``Digit`` step to remove (restricts the
    candidate pool instead of drawing uniformly, so a caller -- the
    reversibility test -- can undo a specific ``add_branch_digit``
    application precisely)."""
    steps = list(derivation.steps)
    phalanx_by_digit: Dict[str, DerivationStep] = {
        s.params["digit_id"]: s for s in steps if s.production == "Phalanx"
    }
    candidates = [
        s for s in steps
        if s.production == "Digit" and not s.params.get("top_level", True)
        and s.params["phalanx_count"] == 1
        and phalanx_by_digit.get(s.params["digit_id"]) is not None
        and phalanx_by_digit[s.params["digit_id"]].params["branch_digit_count"] == 0
        and (target is None or s.params.get("uid") == target)
    ]
    if not candidates:
        return None
    branch_step = candidates[int(rng.integers(0, len(candidates)))]
    branch_digit_id = branch_step.params["digit_id"]
    host_body = branch_step.params["mount"]

    kept = [s for s in steps if not _is_descendant_digit(branch_digit_id, _step_digit_id(s))]

    def _decrement_host(s: DerivationStep) -> DerivationStep:
        if s.production != "Phalanx":
            return s
        body_name = f"d{s.params['digit_id']}p{s.params['p'] + 1}"
        if body_name != host_body:
            return s
        return DerivationStep(path=s.path, production="Phalanx",
                               params={**s.params, "branch_digit_count": s.params["branch_digit_count"] - 1})

    return [_decrement_host(s) for s in kept]


# --------------------------------------------------------------------------
# Segment-length step (2026-10-06, Martin: "we should be able to lengthen or
# shorten, and do so by a fixed amount like 5mm for now"). One existing
# segment -- a phalanx's link or an extra palm body (the root palm has its own
# ``step_root_length``) -- moves by exactly one length-grid step
# (``dist.link_length_grid_m``, 5 mm in every variant), never past the
# variant's range (``link_length_range_m`` for a phalanx,
# ``palm_length_range_m`` for a palm body). Unlike ``perturb_parameter``
# (whose step reflects off a bound, so it is not always exactly one step, and
# which is not in ``EVOLUTION_OPERATORS``), a move that would leave the range
# is simply not offered: the operator draws uniformly among the (segment,
# direction) moves that stay inside it, and is inapplicable when there are
# none. ``lengthen_segment``/``shorten_segment`` are the two directions on
# their own: an exact inverse pair (both take ``target``, a segment's uid).
# No generation limit concerns lengths, so limits never forbid it today; it
# still runs under ``vary``'s limit check like every other operator.
# --------------------------------------------------------------------------


def _segment_length_moves(dist: Distribution, steps: Sequence[DerivationStep], signs: Sequence[int],
                          target: Optional[int]) -> List[Tuple[int, float]]:
    grid = dist.link_length_grid_m
    moves: List[Tuple[int, float]] = []
    for sign in signs:
        for i, s in enumerate(steps):
            if s.production == "Phalanx":
                lo, hi = dist.link_length_range_m
            elif s.production == "PalmBody":
                lo, hi = dist.palm_length_range_m
            else:
                continue
            if target is not None and s.params.get("uid") != target:
                continue
            new = round(float(s.params["length"]) + sign * grid, 10)
            if lo - 1e-9 <= new <= hi + 1e-9:
                moves.append((i, new))
    return moves


def _apply_length_move(rng, dist: Distribution, derivation: Derivation, signs: Sequence[int],
                       target: Optional[int]) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    moves = _segment_length_moves(dist, steps, signs, target)
    if not moves:
        return None
    idx, new = moves[int(rng.integers(0, len(moves)))]
    s = steps[idx]
    steps[idx] = DerivationStep(path=s.path, production=s.production, params={**s.params, "length": new})
    return steps


def _op_step_segment_length(rng, dist: Distribution, derivation: Derivation, target: Optional[int] = None,
                            lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """Lengthen or shorten one segment by exactly one grid step (5 mm), within
    the variant's range; ``target``: restrict to the segment with that uid."""
    return _apply_length_move(rng, dist, derivation, (1, -1), target)


def _op_lengthen_segment(rng, dist: Distribution, derivation: Derivation, target: Optional[int] = None,
                         lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """``step_segment_length``, lengthening only (inverse: ``shorten_segment``)."""
    return _apply_length_move(rng, dist, derivation, (1,), target)


def _op_shorten_segment(rng, dist: Distribution, derivation: Derivation, target: Optional[int] = None,
                        lim: Optional[LimitContext] = None) -> Optional[List[DerivationStep]]:
    """``step_segment_length``, shortening only (inverse: ``lengthen_segment``)."""
    return _apply_length_move(rng, dist, derivation, (-1,), target)


_OPERATOR_FNS = {
    "resample_parameter": _op_resample_parameter,
    "perturb_parameter": _op_perturb_parameter,
    "regrow_subtree": _op_regrow_subtree,
    "insert_phalanx": _op_insert_phalanx,
    "delete_phalanx": _op_delete_phalanx,
    "add_digit": _op_add_digit,
    "remove_digit": _op_remove_digit,
    "step_axis": _op_step_axis,
    "step_limits": _op_step_limits,
    "step_mount": _op_step_mount,
    "step_length": _op_perturb_parameter,
    "step_coupling": _op_step_coupling,
    "step_root_length": _op_step_root_length,
    "step_radius": _op_step_radius,
    "step_bend_rpy": _op_step_bend_rpy,
    "step_bend_offset": _op_step_bend_offset,
    "add_minimal_digit": _op_add_minimal_digit,
    "remove_digit_minimal": _op_remove_digit_minimal,
    "add_palm_body": _op_add_palm_body,
    "remove_palm_body": _op_remove_palm_body,
    "remove_palm_body_empty": _op_remove_palm_body_empty,
    "toggle_palm_joint": _op_toggle_palm_joint,
    "add_branch_digit": _op_add_branch_digit,
    "remove_branch_digit": _op_remove_branch_digit,
    "step_segment_length": _op_step_segment_length,
    "lengthen_segment": _op_lengthen_segment,
    "shorten_segment": _op_shorten_segment,
}

# --------------------------------------------------------------------------
# EVOLUTION_OPERATORS (grammar 0.5, iteration B / balanced-grammar-synthesis
# .md section 3's operator table): the exact-inverse operator pool -- five
# growth/shrink pairs (digit, phalanx, palm body, palm joint -- self-inverse
# -- branch) plus every small-step operator. Each pair's growth move draws
# new material from ``dist.insertion`` when set (``_growth_dist``, see
# above); each pair's shrink move accepts an optional ``target`` uid so a
# caller can undo a specific growth application precisely (see
# ``apply_operator`` below and the reversibility test in
# ``experiments/e12_balance.py`` / ``grammar_bench/tests/test_grammar05_b.py``).
# Deliberately excludes ``regrow_subtree``/full-size ``add_digit``/
# ``remove_digit``/``resample_parameter`` (no exact inverse, large jumps --
# see balanced-grammar-synthesis.md section 3's "dropped from the default
# pool"). I18 fix 2: the palm-body pair uses ``remove_palm_body_empty`` (an
# EXACT inverse of ``add_palm_body`` -- it only ever removes a leaf with no
# children, exactly what ``add_palm_body`` produces), not the general
# ``remove_palm_body`` (which accepts any body and approximately
# re-attaches its children) -- that general operator stays defined and in
# ``_OPERATOR_FNS``/``TARGETABLE_OPERATORS`` as a larger move available
# OUTSIDE this pool.
# --------------------------------------------------------------------------

EVOLUTION_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("add_minimal_digit", "remove_digit_minimal"),
    ("insert_phalanx", "delete_phalanx"),
    ("add_palm_body", "remove_palm_body_empty"),
    ("toggle_palm_joint", "toggle_palm_joint"),
    ("add_branch_digit", "remove_branch_digit"),
)

# The segment-length step's two directions: an exact inverse pair (kept out
# of ``EVOLUTION_PAIRS``, whose five structural grow/shrink pairs E2/E12
# report on; the pool itself carries the combined ``step_segment_length``).
LENGTH_STEP_PAIR: Tuple[str, str] = ("lengthen_segment", "shorten_segment")

INVERSE_OF: Dict[str, str] = {}
for _growth, _shrink in EVOLUTION_PAIRS + (LENGTH_STEP_PAIR,):
    INVERSE_OF[_growth] = _shrink
    INVERSE_OF[_shrink] = _growth
del _growth, _shrink

# The pool as it was until 2026-10-06 (17 operators), frozen so runs made with
# it (the evolution driver, E1/E2/E3/E12, the G0 screen) can be reproduced:
# pass ``operators=EVOLUTION_OPERATORS_V1``.
EVOLUTION_OPERATORS_V1: Tuple[str, ...] = (
    "add_minimal_digit", "remove_digit_minimal",
    "insert_phalanx", "delete_phalanx",
    "add_palm_body", "remove_palm_body_empty",
    "toggle_palm_joint",
    "add_branch_digit", "remove_branch_digit",
) + tuple(SMALL_STEP_OPERATORS)

# The current pool: V1 plus ``step_segment_length`` (one segment +/- 5 mm),
# so evolution can change an existing segment's length (before, only new
# segments got a length, and only the root palm's length could step).
EVOLUTION_OPERATORS: Tuple[str, ...] = EVOLUTION_OPERATORS_V1 + ("step_segment_length",)
assert len(EVOLUTION_OPERATORS) == len(set(EVOLUTION_OPERATORS)), "EVOLUTION_OPERATORS has duplicates"

# Shrink operators (and self-inverse ``toggle_palm_joint``) that accept an
# optional ``target`` uid, per ``_OPERATOR_FNS`` above.
TARGETABLE_OPERATORS = frozenset({
    "remove_digit_minimal", "delete_phalanx", "remove_palm_body", "remove_palm_body_empty",
    "toggle_palm_joint", "remove_branch_digit",
    "step_segment_length", "lengthen_segment", "shorten_segment",
})


def _limit_base(derivation: Derivation, limits: Optional[GenerationLimits]) -> Optional[Dict[str, int]]:
    """The parent's per-limit excess (``None`` without limits)."""
    if limits is None:
        return None
    return Structure.from_steps(derivation.steps).excess(limits)


def _call_operator(fn, operator: str, rng, dist: Distribution, derivation: Derivation,
                   target: Optional[int], limits: Optional[GenerationLimits],
                   base: Optional[Dict[str, int]]) -> Optional[List[DerivationStep]]:
    """One raw application of an operator function, with a fresh
    ``LimitContext`` when ``limits`` is given (``None``: called exactly as
    before limits existed). Returns ``None`` when the operator has nothing to
    act on, or (a safety net the constructive rules make unreachable for a
    parent within the limits, see ``test_generation_limits.py``) when its
    result would make a limit worse than the parent's."""
    kwargs: Dict[str, Any] = {}
    if operator in TARGETABLE_OPERATORS:
        kwargs["target"] = target
    ctx = None
    if limits is not None:
        ctx = LimitContext(limits, derivation.steps, base=base)
        kwargs["lim"] = ctx
    out = fn(rng, dist, derivation, **kwargs)
    if out is not None and ctx is not None and not ctx.allows_steps(out):
        return None
    return out


def apply_operator(derivation: Derivation, rng: np.random.Generator, dist: Distribution,
                    operator: str, target: Optional[int] = None,
                    limits: Optional[GenerationLimits] = None) -> Optional[Derivation]:
    """Apply ``operator`` exactly ONCE (unlike ``vary``: no 32-attempt retry
    loop, no re-drawing a different operator on failure) to ``derivation``,
    optionally passing ``target`` through to the underlying ``_op_*``
    function when ``operator in TARGETABLE_OPERATORS`` (a uid selecting
    which digit/phalanx/palm-body to act on, instead of drawing uniformly
    -- see each targetable ``_op_*``'s own docstring). Returns ``None`` when
    the operator has no valid application (the underlying function returned
    ``None``, or the candidate failed to derive/validate) -- the caller
    decides what "not applicable" means for its own accounting (E1's
    ``VariationImpossible`` convention does not apply here, since this
    function never retries).

    ``limits`` (``limits.GenerationLimits``, default ``None``): the operator
    acts only in ways the limits allow (see ``limits.py``); with no such
    application it returns ``None``. ``None`` behaves exactly as before."""
    if operator not in _OPERATOR_FNS:
        raise ValueError(f"unknown vary operator {operator!r}")
    fn = _OPERATOR_FNS[operator]
    candidate_steps = _call_operator(fn, operator, rng, dist, derivation, target, limits,
                                     _limit_base(derivation, limits))
    if candidate_steps is None:
        return None
    if tuple(candidate_steps) == derivation.steps:
        return None
    candidate = Derivation(
        seed=derivation.seed, grammar_version=derivation.grammar_version,
        steps=tuple(candidate_steps), lineage=derivation.lineage + ((operator, derivation.seed),),
    )
    try:
        derive(candidate)
    except ModelError:
        return None
    return candidate


def vary(derivation: Derivation, rng: np.random.Generator, dist: Distribution = DEFAULT_DISTRIBUTION,
         operator: Optional[str] = None, operators: Optional[Sequence[str]] = None,
         limits: Optional[GenerationLimits] = None) -> Derivation:
    """Apply ``operator`` (random from ``operators`` if given, else random
    from ``OPERATORS`` -- unchanged default behaviour -- if both are
    omitted) to ``derivation``, retrying up to 32 times with fresh
    randomness until the result derives to a valid model and differs from
    the parent. Raises ``VariationImpossible`` if no valid application is
    found. Passing ``operators=SMALL_STEP_OPERATORS`` (or any other
    explicit operator/pool) is the only way to reach an operator outside
    ``OPERATORS``; nothing here changes what a bare ``vary(derivation,
    rng, dist)`` call does.

    ``limits`` (``limits.GenerationLimits``, default ``None``): the operator
    acts only in ways the limits allow, constructively (see ``limits.py``);
    an operator with no such application raises ``VariationImpossible`` like
    one with nothing to act on, so a caller drawing operators at random just
    draws again. A parent already outside the limits may still be mutated,
    as long as no limit gets worse. ``None`` behaves exactly as before."""
    if operator is not None:
        op = operator
    else:
        pool = operators if operators is not None else OPERATORS
        op = pool[int(rng.integers(0, len(pool)))]
    if op not in _OPERATOR_FNS:
        raise ValueError(f"unknown vary operator {op!r}")
    fn = _OPERATOR_FNS[op]
    base = _limit_base(derivation, limits)
    for _ in range(32):
        candidate_steps = _call_operator(fn, op, rng, dist, derivation, None, limits, base)
        if candidate_steps is None:
            continue
        if tuple(candidate_steps) == derivation.steps:
            continue
        candidate = Derivation(
            seed=derivation.seed, grammar_version=derivation.grammar_version,
            steps=tuple(candidate_steps),
            lineage=derivation.lineage + ((op, derivation.seed),),
        )
        try:
            derive(candidate)
        except ModelError:
            continue
        return candidate
    raise VariationImpossible(f"could not apply operator {op!r} to derivation after 32 attempts")


def vary_tracked(derivation: Derivation, rng: np.random.Generator, dist: Distribution,
                  operators: Sequence[str],
                  limits: Optional[GenerationLimits] = None) -> Tuple[Optional[Derivation], str]:
    """Like ``vary(derivation, rng, dist, operators=operators)`` -- one
    operator drawn uniformly from ``operators``, retried up to 32 times via
    ``apply_operator`` (exactly ``vary``'s own per-attempt logic, since
    ``apply_operator`` runs the same ``fn`` / no-op / ``ModelError`` checks)
    -- but returns ``(candidate_or_None, op)`` instead of raising
    ``VariationImpossible`` on total failure, so a caller (E12's per-pair
    growth/shrink applicability-rate reporting, grammar 0.5 I18 fix 4) can
    tally which operator was drawn and whether IT was applicable, without
    duplicating ``vary``'s algorithm."""
    op = operators[int(rng.integers(0, len(operators)))]
    for _ in range(32):
        candidate = apply_operator(derivation, rng, dist, op, limits=limits)
        if candidate is not None:
            return candidate, op
    return None, op
