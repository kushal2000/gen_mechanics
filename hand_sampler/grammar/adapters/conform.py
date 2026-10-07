"""Conform a projected derivation to the grammar's own discretisation.

``adapters/projection.py`` expresses a real hand EXACTLY: continuous link
lengths, free joint axes and mount poses, the URDF's own joint limits, a
lateral offset for every finger's mount, a rest bend at every joint and a
fixed 10 mm capsule radius. ``derive`` accepts all of that (it is the
representation compiler), but the grammar never generates such values, so
the projection is not a member of the grammar's search space, and the
small-step operators (which step a value to its neighbour on the grid) do
not apply to it.

``conform_to_grammar(derivation, dist, limits)`` snaps every parameter onto
what ``dist`` (a grammar variant) can generate, keeping the structure (palm
bodies, digits, phalanges, module kinds) unchanged. Snapping is closed-loop:
bodies are placed root first and each value is chosen to put its own body
where the real hand's body is, given where the already-snapped parents ended
up, so grid errors do not add up along a finger:

- lengths onto the 5 mm grid within the variant's range: a link length reaches
  for the hand's next joint (or fingertip); a host's length is chosen jointly
  with the mount fractions of what hangs on it; the root frame slides down its
  own axis when a mount lies below it or the palm is longer than the grammar's
  longest palm segment (an exact re-expression: the root is fixed);
- joint axes onto the 15 degree spherical grid (inside the variant's
  elevation band for digit joints, if it has one), matching the hand's axis
  direction in space;
- mount and palm-body orientations onto the nearest rotation of the 15 degree
  roll/pitch/yaw grid, mount positions onto the allowed fractions;
- lateral mount offsets onto what the variant can generate: none, the host's
  surface at a 15 degree azimuth (surface-mounting variants), or the 5 mm
  lateral grid (``Distribution.mount_lateral_grid_m``, e.g. ``G_WIDE``);
- rest bends onto the variant's bend menu (zero for most variants);
- joint limits onto the variant's menu, or (a continuous-limit variant,
  which samples any limits inside its range) clamped into that range;
- the capsule radius onto the allowed choice closest to half the median
  spacing between neighbouring finger bases (fingers are about as thick as
  their spacing).

A hand the grammar sampled conforms to itself exactly (for variants without
a bend on a digit's first link; with one, the mount and that bend are chosen
by a bounded joint search).

The report lists how far each kind of parameter moved and every RULE
CONFLICT: a feature the real hand needs that the variant's rules forbid
(a zero-length link between co-located joints, a module kind the variant
never samples, a lateral finger offset, limits far from the menu, ...).
``fidelity`` measures the result against the original model with the same
metric as E13 (zero plus 64 random configurations; joint positions and axes,
fingertips), and ``operator_applicability`` says which mutation operators
can act on it.

Stdlib + numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..derive import Derivation, DerivationStep, VariationImpossible, derive, vary
from ..distributions import (ANGLE_STEP_DEG, DEG, N_ANGLE_STEPS, N_ELEVATION_STEPS, Distribution,
                             lateral_offset_choices_m)
from ..fk import forward_kinematics, matrix_to_rpy, pose_to_matrix, rpy_to_matrix
from ..kinematics import MOVABLE_TYPES, KinematicModel, ModelError
from ..limits import GenerationLimits, LimitReport, check as check_limits

# Fidelity target (E13): 5 mm joint position, 10 deg joint axis.
POS_TOL_MM = 5.0
AXIS_TOL_DEG = 10.0
FIDELITY_SEED = 20260926
FIDELITY_CONFIGS = 64

# A snapped value further than this from the hand's own is reported as a
# rule conflict (the grid itself moves values by up to half a step: 2.5 mm,
# 7.5 deg).
CONFLICT_MM = 5.0
CONFLICT_DEG = 10.0


# --------------------------------------------------------------------------
# Grids (the exact floats the grammar's samplers produce)
# --------------------------------------------------------------------------


def _grid_angle(k: int) -> float:
    return (k * ANGLE_STEP_DEG - 180.0) * DEG


_ANGLES = [_grid_angle(k) for k in range(N_ANGLE_STEPS)]


def _grid_axis(el_k: int, az_k: int) -> Tuple[float, float, float]:
    el = el_k * ANGLE_STEP_DEG * DEG
    az = (az_k * ANGLE_STEP_DEG - 180.0) * DEG
    return (float(math.sin(el) * math.cos(az)), float(math.sin(el) * math.sin(az)), float(math.cos(el)))


def _axis_candidates(band: Optional[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
    els = list(range(N_ELEVATION_STEPS))
    if band is not None:
        inside = [k for k in els if band[0] <= k * ANGLE_STEP_DEG <= band[1]]
        els = inside or els
    return [_grid_axis(e, a) for e in els for a in range(N_ANGLE_STEPS)]


_RPY_GRID: Optional[Tuple[np.ndarray, List[Tuple[float, float, float]]]] = None


def _rpy_grid():
    global _RPY_GRID
    if _RPY_GRID is None:
        triples = [(r, p, y) for r in _ANGLES for p in _ANGLES for y in _ANGLES]
        mats = np.stack([rpy_to_matrix(t) for t in triples])
        _RPY_GRID = (mats, triples)
    return _RPY_GRID


def _rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (float(np.trace(Ra.T @ Rb)) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def snap_rpy(rpy: Sequence[float]) -> Tuple[Tuple[float, float, float], float]:
    """Nearest rotation on the 15 degree roll/pitch/yaw grid; (triple, error rad)."""
    mats, triples = _rpy_grid()
    R = rpy_to_matrix(tuple(rpy))
    tr = np.einsum("kij,ij->k", mats, R)          # trace(M^T R) for every grid M
    k = int(np.argmax(tr))
    return triples[k], _rot_angle(mats[k], R)


def snap_axis(axis: Sequence[float], band: Optional[Tuple[float, float]] = None
              ) -> Tuple[Tuple[float, float, float], float]:
    a = np.asarray(axis, dtype=float)
    a = a / max(np.linalg.norm(a), 1e-12)
    cands = _axis_candidates(band)
    dots = [float(np.dot(a, c)) for c in cands]
    k = int(np.argmax(dots))
    return cands[k], math.acos(max(-1.0, min(1.0, dots[k])))


def _snap_length(v: float, rng: Tuple[float, float], grid: float) -> float:
    lo, hi = rng
    n = int(round((hi - lo) / grid))
    k = int(round((v - lo) / grid))
    k = max(0, min(n, k))
    return round(lo + k * grid, 10)


def _length_grid(rng: Tuple[float, float], grid: float) -> List[float]:
    lo, hi = rng
    n = int(round((hi - lo) / grid))
    return [round(lo + k * grid, 10) for k in range(n + 1)]


def _nearest_limits(lim: Sequence[float], choices_deg, scale: float) -> Tuple[Tuple[float, float], float]:
    best, err = None, float("inf")
    for lo, hi in choices_deg:
        e = max(abs(lo * scale - lim[0]), abs(hi * scale - lim[1]))
        if e < err:
            best, err = (lo * scale, hi * scale), e
    return best, err


def _snap_revolute_limits(lim: Sequence[float], dist: Distribution) -> Tuple[Tuple[float, float], float]:
    if dist.limits_continuous:
        # A continuous-limit variant draws any (lo, hi) inside its range (and
        # ``step_limits`` moves one bound by ``limit_step_deg``), so the
        # hand's own limits are already in its support once inside the range.
        lo_r, hi_r = (v * DEG for v in dist.revolute_limit_range_deg)
        lo, hi = min(hi_r, max(lo_r, float(lim[0]))), min(hi_r, max(lo_r, float(lim[1])))
        if hi <= lo:
            step = dist.limit_step_deg * DEG
            hi = min(hi_r, lo + step)
            lo = hi - step
        return (lo, hi), max(abs(lo - lim[0]), abs(hi - lim[1]))
    return _nearest_limits(lim, dist.revolute_limit_choices_deg, DEG)


def _bend_support(dist: Distribution, first_phalanx: bool):
    """Every (bend_rpy, bend_offset) pair ``sample_bend`` can produce."""
    zero = ((0.0, 0.0, 0.0), (0.0, 0.0))
    if first_phalanx and dist.curl_skip_first_phalanx:
        return [zero]
    out = []
    if dist.bend_probability < 1.0:
        out.append(zero)
    if dist.bend_probability > 0.0:
        out += [(tuple(r), tuple(o)) for r in dist.bend_rpy_choices_rad for o in dist.bend_offset_choices_m]
    return out


_BEND_CACHE: Dict[Tuple[int, bool], Any] = {}


def _bend_arrays(dist: Distribution, first: bool):
    key = (id(dist), first)
    hit = _BEND_CACHE.get(key)
    if hit is None or hit[0] is not dist:
        pairs = _bend_support(dist, first)
        rpys = sorted({r for r, _ in pairs})
        offs = sorted({o for _, o in pairs})
        index = {(r, o) for r, o in pairs}
        mats = np.stack([rpy_to_matrix(r) for r in rpys])
        hit = (dist, rpys, offs, index, mats, np.asarray(offs, dtype=float))
        _BEND_CACHE[key] = hit
    return hit[1:]


def _snap_bend_matrix(R: np.ndarray, off: Sequence[float], dist: Distribution, first: bool):
    """The (bend_rpy, bend_offset) pair of the variant's support nearest to a
    relative rotation ``R`` and lateral offset ``off``, by rotation angle (in
    15 degree steps) plus offset (in 5 mm steps); returns (rpy, offset,
    (angle error rad, offset error m))."""
    rpys, offs, index, mats, off_arr = _bend_arrays(dist, first)
    tr = np.einsum("kij,ij->k", mats, R)
    ea = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    eo = np.hypot(off_arr[:, 0] - off[0], off_arr[:, 1] - off[1])
    best, best_cost, best_errs = None, float("inf"), (0.0, 0.0)
    order_r = np.argsort(ea)
    order_o = np.argsort(eo)
    # The support is a product (or contains the zero pair): try the nearest
    # few of each and keep the cheapest pair that is in the support.
    for ri in order_r[:8]:
        for oi in order_o[:8]:
            pair = (rpys[ri], offs[oi])
            if pair not in index:
                continue
            cost = math.degrees(float(ea[ri])) / ANGLE_STEP_DEG + float(eo[oi]) / 0.005
            if cost < best_cost:
                best, best_cost, best_errs = pair, cost, (float(ea[ri]), float(eo[oi]))
    if best is None:                      # fall back to an exhaustive search
        for r, o in index:
            e1 = _rot_angle(rpy_to_matrix(r), R)
            e2 = math.hypot(o[0] - off[0], o[1] - off[1])
            cost = math.degrees(e1) / ANGLE_STEP_DEG + e2 / 0.005
            if cost < best_cost:
                best, best_cost, best_errs = (r, o), cost, (e1, e2)
    return best[0], best[1], best_errs


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass
class RuleConflict:
    feature: str          # short key, e.g. "colocated_joints"
    where: str            # body/joint names
    detail: str           # one line with the numbers

    def line(self) -> str:
        return f"{self.feature}: {self.where} ({self.detail})"


@dataclass
class ConformReport:
    max_change: Dict[str, float] = field(default_factory=dict)   # "length_mm", "angle_deg", ...
    conflicts: List[RuleConflict] = field(default_factory=list)
    radius_hint_m: Optional[float] = None
    radius_m: Optional[float] = None
    # Shift (m) of the root frame along its own z so that no root mount lies
    # below the root segment's start; compose with the projection's
    # ``root_transform`` via ``root_transform(...)`` below.
    root_shift_m: float = 0.0
    valid: bool = False
    error: Optional[str] = None
    limits: Optional[LimitReport] = None

    def bump(self, key: str, value: float) -> None:
        self.max_change[key] = max(self.max_change.get(key, 0.0), float(value))

    def conflict_features(self) -> List[str]:
        return sorted({c.feature for c in self.conflicts})

    def root_transform(self, projection_root_transform: np.ndarray) -> np.ndarray:
        """The projection's root transform composed with this report's root
        shift: what ``fidelity`` (and any overlay of the original hand) must
        use for the conformed derivation."""
        T = np.eye(4)
        T[2, 3] = self.root_shift_m
        return np.asarray(projection_root_transform) @ T


CONFLICT_TEXT = {
    "colocated_joints": "co-located joints (a zero-length link), grammar links are 15-80 mm",
    "zero_length_palm": "zero-length palm body, grammar palm bodies are 20-80 mm",
    "lateral_mount_offset": "finger mounted off the host's axis (palm width), the grammar mounts on the axis "
                            "or on the capsule surface",
    "palm_mount_offset": "palm body mounted off its parent's axis",
    "mount_off_segment": "mount beyond the host segment's ends",
    "link_length_range": "link length outside the grammar's range",
    "rest_bend": "rest bend between links that the variant's bend menu cannot express",
    "forced_curl": "the variant forces a rest curl the straight finger does not have",
    "limits_menu": "joint limits far from the variant's menu",
    "module_kind": "module kind the variant never samples",
    "digit_count": "digit count outside the variant's range",
    "phalanx_count": "joints per digit outside the variant's range",
    "palm_body_count": "palm body count outside the variant's range",
    "axis_band": "digit joint axis outside the variant's elevation band",
}


# --------------------------------------------------------------------------
# conform_to_grammar
# --------------------------------------------------------------------------


def _choose_host_length(L0: float, axial: Sequence[float], rng: Tuple[float, float], grid: float,
                        fracs: Sequence[float]) -> float:
    """The grid length for a host whose children sit at ``axial`` positions
    along it, minimising their squared distance to the nearest allowed mount
    fraction (ties: closest to the hand's own length ``L0``)."""
    best, best_cost = None, float("inf")
    for L in _length_grid(rng, grid):
        cost = sum(min((f * L - a) ** 2 for f in fracs) for a in axial) + 1e-3 * (L - L0) ** 2
        if cost < best_cost - 1e-15:
            best, best_cost = L, cost
    return best


def _radius_hint(T0: Dict[str, np.ndarray], steps) -> Optional[float]:
    bases = [T0[f"d{s.params['digit_id']}p1"][:3, 3] for s in steps
             if s.production == "Digit" and s.params.get("top_level") and f"d{s.params['digit_id']}p1" in T0]
    if len(bases) < 2:
        return None
    P = np.stack(bases)
    nn = []
    for i in range(len(P)):
        d = np.linalg.norm(P - P[i], axis=1)
        d[i] = np.inf
        nn.append(float(d.min()))
    return 0.5 * float(np.median(nn))


def _trans(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def _rot4(R: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    return T


PARAMETER_CLASSES: Tuple[str, ...] = ("lengths", "mounts", "lateral", "rotations", "axes", "bends", "limits",
                                      "radius")


def conform_to_grammar(derivation: Derivation, dist: Distribution,
                       limits: Optional[GenerationLimits] = None,
                       exact: Sequence[str] = ()) -> Tuple[Derivation, ConformReport]:
    """``derivation`` (typically ``project_to_derivation(...).derivation``)
    with every parameter snapped onto ``dist``'s discretisation, plus a
    report (see the module docstring). The structure is unchanged, so the
    projection's ``name_map`` still applies; its ``root_transform`` must be
    composed with ``report.root_transform(...)`` (the root frame may move
    down its own axis).

    Snapping is closed-loop: bodies are placed root first, and each
    parameter is chosen to put ITS body where the hand's own body is, given
    where the already-snapped parents ended up (a link length reaches for
    the hand's next joint, a rest bend for the hand's next link direction, an
    axis for the hand's own joint axis in space), so grid errors do not
    accumulate along a finger.

    ``limits``: also check the result against these generation limits
    (``report.limits``). ``exact`` (diagnosis only): parameter classes
    (``PARAMETER_CLASSES``) to set to the hand's exact value instead of a grid
    value, to attribute the fidelity loss to each class. Sampling priors that
    only choose among grid values (V2/V2s mount spacing, the opposition
    prior) are not imposed; the axis elevation band and the bend menu are."""
    exact = set(exact)
    unknown = exact - set(PARAMETER_CLASSES)
    if unknown:
        raise ValueError(f"unknown parameter class(es) {sorted(unknown)}")
    rep = ConformReport()
    steps = list(derivation.steps)
    grid = dist.link_length_grid_m
    fracs = sorted(set(dist.mount_frac_choices))
    lrange, prange = dist.link_length_range_m, dist.palm_length_range_m
    band = dist.digit_axis_elevation_band_deg
    hand_i = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand = dict(steps[hand_i].params)

    # ---- structure counts against the variant's ranges ----------------------
    top = [s for s in steps if s.production == "Digit" and s.params.get("top_level")]
    lo, hi = dist.digit_count_range
    if not lo <= len(top) <= hi:
        rep.conflicts.append(RuleConflict("digit_count", "hand", f"{len(top)} digits, variant {lo}-{hi}"))
    lo, hi = dist.palm_body_count_range
    n_palm = sum(1 for s in steps if s.production == "PalmBody")
    if not lo <= n_palm <= hi:
        rep.conflicts.append(RuleConflict("palm_body_count", "hand", f"{n_palm} palm bodies, variant {lo}-{hi}"))
    lo, hi = dist.phalanx_count_range
    for s in steps:
        if s.production == "Digit" and not lo <= s.params["phalanx_count"] <= hi:
            rep.conflicts.append(RuleConflict("phalanx_count", f"digit {s.params['digit_id']}",
                                              f"{s.params['phalanx_count']} joints, variant {lo}-{hi}"))
    weights = dict(dist.module_probabilities)
    for s in steps:
        if s.production == "Phalanx" and weights.get(s.params["module"]["kind"], 0.0) <= 0.0:
            rep.conflicts.append(RuleConflict("module_kind", f"d{s.params['digit_id']}p{s.params['p'] + 1}_j",
                                              f"kind {s.params['module']['kind']}"))

    # ---- the hand's own frames (q = 0), and the root shift ---------------------
    original = derive(derivation)
    T0 = forward_kinematics(original, {})
    tip0 = {f.body: T0[f.body] @ pose_to_matrix(f.pose) for f in original.frames if f.name == f"{f.body}_tip"}
    root_kids = [s for s in steps if (s.production == "PalmBody" and s.params["parent"] == "root")
                 or (s.production == "Digit" and s.params["mount"] == "root")]
    child_body = {id(s): (s.params["name"] if s.production == "PalmBody" else f"d{s.params['digit_id']}p1")
                  for s in steps if s.production in ("PalmBody", "Digit")}
    axial = [float(T0[child_body[id(s)]][2, 3]) for s in root_kids]
    a_min, a_max = min(axial, default=0.0), max(axial, default=0.0)
    if (a_min < -1e-9 or a_max > prange[1] + 1e-9) and "mounts" not in exact:
        # Start the root segment at its lowest mount. A real hand often mounts
        # its thumb below the root frame's origin (where the grammar cannot
        # mount anything), or has a wrist stretch before the first mount that
        # makes the palm longer than the grammar's longest palm segment.
        # Moving the root frame along its own axis is an exact re-expression
        # of the same hand (the root is fixed in space); the root capsule
        # then spans the mounts. A grammar hand never needs it.
        rep.root_shift_m = a_min
        shift = _trans(0.0, 0.0, -a_min)
        T0 = {k: shift @ v for k, v in T0.items()}
        tip0 = {k: shift @ v for k, v in tip0.items()}

    # ---- capsule radius ------------------------------------------------------------------
    rep.radius_hint_m = _radius_hint(T0, steps)
    target_r = rep.radius_hint_m if rep.radius_hint_m is not None else hand["capsule_radius_m"]
    radius = float(min(dist.capsule_radius_choices_m, key=lambda r: (abs(r - target_r), r)))
    if "radius" in exact:
        radius = float(hand["capsule_radius_m"])
    hand["capsule_radius_m"] = radius
    rep.radius_m = radius

    # ---- helpers -----------------------------------------------------------------------------
    children: Dict[str, List[int]] = {}
    for i, s in enumerate(steps):
        if s.production == "PalmBody":
            children.setdefault(s.params["parent"], []).append(i)
        elif s.production == "Digit":
            children.setdefault(s.params["mount"], []).append(i)
    T: Dict[str, np.ndarray] = {"root": np.eye(4)}
    host_len: Dict[str, float] = {}

    def choose_host_length(host: str, L0: float, rng_m: Tuple[float, float]) -> float:
        axial = [float((np.linalg.inv(T[host]) @ T0[child_body[id(steps[i])]])[2, 3]) for i in children.get(host, [])]
        if "lengths" in exact:
            return L0
        L = _choose_host_length(L0, axial, rng_m, grid, fracs)
        rep.bump("length_mm", abs(L - L0) * 1000)
        if L0 < 1e-3:
            rep.conflicts.append(RuleConflict("zero_length_palm", host, f"{L0 * 1000:.1f} mm -> {L * 1000:.0f} mm"))
        return L

    def mount_point(host: str, body: str, path: str) -> Tuple[float, np.ndarray]:
        """(mount fraction, target origin in the host's frame)."""
        local = (np.linalg.inv(T[host]) @ T0[body])[:3, 3]
        L = host_len[host]
        if "mounts" in exact:
            return (float(local[2] / L) if L > 1e-9 else 0.0), local
        f = min(fracs, key=lambda f: (abs(f * L - local[2]), f))
        rep.bump("mount_axial_mm", abs(f * L - local[2]) * 1000)
        if local[2] < -CONFLICT_MM / 1000 or local[2] > L + CONFLICT_MM / 1000:
            rep.conflicts.append(RuleConflict("mount_off_segment", path,
                                              f"{local[2] * 1000:.1f} mm along a {L * 1000:.0f} mm host"))
        return f, local

    def rotation(R_parent: np.ndarray, R_target: np.ndarray) -> Tuple[Tuple[float, float, float], np.ndarray]:
        rel = R_parent.T @ R_target
        if "rotations" in exact:
            rpy = matrix_to_rpy(rel)
        else:
            rpy, _ = snap_rpy(matrix_to_rpy(rel))
        return tuple(float(v) for v in rpy), R_parent @ rpy_to_matrix(rpy)

    def axis_for(R_new: np.ndarray, R_orig: np.ndarray, a_orig, use_band: bool, name: str):
        a_local = R_new.T @ (R_orig @ np.asarray(a_orig, dtype=float))
        if "axes" in exact:
            return tuple(float(v) for v in a_local / np.linalg.norm(a_local))
        ax, e = snap_axis(a_local, band if use_band else None)
        if use_band and band is not None:
            _, free_e = snap_axis(a_local)
            if math.degrees(e) > math.degrees(free_e) + CONFLICT_DEG:
                rep.conflicts.append(RuleConflict("axis_band", name, f"axis moved {math.degrees(e):.0f} deg"))
        rep.bump("axis_deg", math.degrees(e))
        return ax

    # ---- root ------------------------------------------------------------------------------------
    host_len["root"] = choose_host_length("root", float(hand["root_length"]) - rep.root_shift_m, prange)
    hand["root_length"] = host_len["root"]
    steps[hand_i] = DerivationStep(path="hand", production="Hand", params=hand)

    # ---- palm bodies (parents first) -------------------------------------------------------------
    palm_idx = [i for i, s in enumerate(steps) if s.production == "PalmBody"]
    placed = {"root"}
    while palm_idx:
        ready = [i for i in palm_idx if steps[i].params["parent"] in placed]
        if not ready:
            raise ValueError("palm bodies form a cycle")
        for i in ready:
            p = dict(steps[i].params)
            name, parent = p["name"], p["parent"]
            f, local = mount_point(parent, name, steps[i].path)
            p["mount_frac"] = f
            lat_mm = math.hypot(local[0], local[1]) * 1000
            if "lateral" in exact:
                p["mount_offset"] = (float(local[0]), float(local[1]))
            else:
                if dist.mount_lateral_grid_m is not None:
                    grid_xy = lateral_offset_choices_m(dist)
                    p["mount_offset"] = tuple(min(grid_xy, key=lambda g: abs(g - float(v))) for v in local[:2])
                else:
                    p.pop("mount_offset", None)
                got = np.asarray(p.get("mount_offset", (0.0, 0.0)))
                lost_mm = float(np.hypot(*(np.asarray(local[:2]) - got))) * 1000
                rep.bump("lateral_offset_mm", lost_mm)
                if lost_mm > CONFLICT_MM:
                    rep.conflicts.append(RuleConflict("palm_mount_offset", name,
                                                      f"{lat_mm:.1f} mm off-axis, {lost_mm:.1f} mm lost"))
            off = p.get("mount_offset", (0.0, 0.0))
            origin = T[parent] @ _trans(off[0], off[1], f * host_len[parent])
            rpy, R_new = rotation(origin[:3, :3], T0[name][:3, :3])
            p["direction_rpy"] = rpy
            T[name] = _trans(*origin[:3, 3]) @ _rot4(R_new)
            rep.bump("angle_deg", math.degrees(_rot_angle(R_new, T0[name][:3, :3])))
            p["axis"] = axis_for(R_new, T0[name][:3, :3], steps[i].params["axis"], False, f"{name}_j")
            if p["has_joint"] and "limits" not in exact:
                new_lim, e = _nearest_limits(p["limits"], dist.palm_joint_limit_choices_deg, DEG)
                if math.degrees(e) > CONFLICT_DEG:
                    rep.conflicts.append(RuleConflict(
                        "limits_menu", f"{name}_j", "({:.0f}, {:.0f}) deg -> ({:.0f}, {:.0f})".format(
                            *(math.degrees(v) for v in tuple(p["limits"]) + new_lim))))
                p["limits"] = new_lim
                rep.bump("limits_deg", math.degrees(e))
            host_len[name] = choose_host_length(name, float(steps[i].params["length"]), prange)
            p["length"] = host_len[name]
            steps[i] = DerivationStep(path=steps[i].path, production="PalmBody", params=p)
            placed.add(name)
        palm_idx = [i for i in palm_idx if i not in ready]

    # ---- digits (top-level first, then branches once their host exists) ----------------------
    phalanges: Dict[str, Dict[int, int]] = {}
    for i, s in enumerate(steps):
        if s.production == "Phalanx":
            phalanges.setdefault(s.params["digit_id"], {})[s.params["p"]] = i
    digit_idx = [i for i, s in enumerate(steps) if s.production == "Digit"]
    while digit_idx:
        ready = [i for i in digit_idx if steps[i].params["mount"] in T]
        if not ready:
            raise ValueError("a digit's mount is never created")
        for i in ready:
            _conform_digit(i, steps, phalanges, dist, rep, exact, T, T0, tip0, host_len, mount_point, rotation,
                           axis_for, radius, lrange, grid)
        digit_idx = [i for i in digit_idx if i not in ready]

    conformed = Derivation(seed=derivation.seed, grammar_version=derivation.grammar_version, steps=tuple(steps),
                           lineage=derivation.lineage)
    try:
        derive(conformed)
        rep.valid = True
    except ModelError as exc:
        rep.error = f"{type(exc).__name__}: {exc}"
    if limits is not None:
        rep.limits = check_limits(conformed, limits)
    return conformed, rep


def _conform_digit(i, steps, phalanges, dist, rep, exact, T, T0, tip0, host_len, mount_point, rotation, axis_for,
                   radius, lrange, grid) -> None:
    p = dict(steps[i].params)
    did, host = p["digit_id"], p["mount"]
    b0 = f"d{did}p1"
    f, local = mount_point(host, b0, steps[i].path)
    p["mount_frac"] = f
    lateral = np.asarray(local[:2], dtype=float)
    lat_mm = float(np.hypot(*lateral)) * 1000
    if "lateral" in exact:
        p["mount_offset"] = (float(lateral[0]), float(lateral[1]))
    elif dist.mount_on_host_surface:
        az = math.atan2(float(lateral[1]), float(lateral[0])) if lat_mm > 1e-6 else -math.pi
        az = min(_ANGLES, key=lambda a: abs(math.remainder(a - az, 2 * math.pi)))
        p["mount_offset"] = (radius * math.cos(az), radius * math.sin(az))
    elif dist.mount_lateral_grid_m is not None and p.get("top_level"):
        grid_xy = lateral_offset_choices_m(dist)
        p["mount_offset"] = tuple(min(grid_xy, key=lambda g: abs(g - float(v))) for v in lateral)
    else:
        p.pop("mount_offset", None)
    residual = lateral - np.asarray(p.get("mount_offset", (0.0, 0.0)))

    # phalanx 0: the mount rotation, then a bend from the variant's support
    # (zero for most variants) for whatever rotation and lateral offset is left
    q0 = dict(steps[phalanges[did][0]].params)
    R_target = T0[b0][:3, :3]
    mount_rpy, R_mount = rotation(T[host][:3, :3], R_target)
    want_off = np.zeros(2) if "lateral" in exact else residual
    br, br_off, (ea, eo) = _snap_bend_matrix(R_mount.T @ R_target, want_off, dist, first=True)
    if "rotations" not in exact and len(_bend_support(dist, True)) > 1 and math.degrees(ea) > 1e-6:
        # The digit's first link may also bend: its orientation is
        # R(mount) @ R(bend), so search the mounts near the target jointly
        # with the bend (a grammar hand is then reproduced exactly).
        mats, triples = _rpy_grid()
        R_rel = T[host][:3, :3].T @ R_target
        near = np.argsort(-np.einsum("kij,ij->k", mats, R_rel))[:512]
        best = (math.degrees(ea), mount_rpy, R_mount, br, br_off, ea, eo)
        for kk in near:
            Rm = T[host][:3, :3] @ mats[kk]
            b_rpy, b_off, (e_a, e_o) = _snap_bend_matrix(Rm.T @ R_target, want_off, dist, first=True)
            if math.degrees(e_a) < best[0] - 1e-9:
                best = (math.degrees(e_a), tuple(float(v) for v in triples[kk]), Rm, b_rpy, b_off, e_a, e_o)
                if e_a < 1e-9:
                    break
        _, mount_rpy, R_mount, br, br_off, ea, eo = best
    p["mount_rpy"] = mount_rpy
    if "bends" in exact:
        br = tuple(float(v) for v in matrix_to_rpy(R_mount.T @ R_target))
    if "lateral" in exact:
        br_off, eo = (0.0, 0.0), 0.0
    else:
        rep.bump("lateral_offset_mm", eo * 1000)
        if eo * 1000 > CONFLICT_MM:
            rep.conflicts.append(RuleConflict("lateral_mount_offset", f"digit {did}",
                                              f"{lat_mm:.1f} mm off the host axis, {eo * 1000:.1f} mm lost"))
    q0["bend_rpy"], q0["bend_offset"] = br, br_off
    off = p.get("mount_offset", (0.0, 0.0))
    frame0 = T[host] @ _trans(off[0] + br_off[0], off[1] + br_off[1], f * host_len[host])
    frame0[:3, :3] = R_mount @ rpy_to_matrix(br)
    frames = [frame0]
    steps[i] = DerivationStep(path=steps[i].path, production="Digit", params=p)

    n = p["phalanx_count"]
    for k in range(n):
        j = phalanges[did][k]
        q = q0 if k == 0 else dict(steps[j].params)
        body = f"d{did}p{k + 1}"
        if k > 0:
            prev = frames[k - 1]
            L_prev = steps[phalanges[did][k - 1]].params["length"]
            # bend: reach for the hand's own orientation of this link, and
            # (with a bend-offset menu) for its joint's lateral position
            Rrel = prev[:3, :3].T @ T0[body][:3, :3]
            want = (np.linalg.inv(prev) @ T0[body][:, 3])[:2]
            if "bends" in exact:
                br, br_off, ea = tuple(float(v) for v in matrix_to_rpy(Rrel)), (float(want[0]), float(want[1])), 0.0
            else:
                br, br_off, (ea, _) = _snap_bend_matrix(Rrel, want, dist, first=False)
                orig_bend = math.degrees(_rot_angle(np.eye(3), T0[f"d{did}p{k}"][:3, :3].T @ T0[body][:3, :3]))
                new_bend = math.degrees(_rot_angle(np.eye(3), rpy_to_matrix(br)))
                if math.degrees(ea) > CONFLICT_DEG:
                    feature = "forced_curl" if new_bend > orig_bend + CONFLICT_DEG else "rest_bend"
                    rep.conflicts.append(RuleConflict(feature, f"{body}_j",
                                                      f"bend {orig_bend:.0f} deg -> {new_bend:.0f} deg"))
            q["bend_rpy"], q["bend_offset"] = br, br_off
            frame = prev @ _trans(br_off[0], br_off[1], L_prev)
            frame[:3, :3] = prev[:3, :3] @ rpy_to_matrix(br)
            frames.append(frame)
        cur = frames[k]
        rep.bump("angle_deg", math.degrees(_rot_angle(cur[:3, :3], T0[body][:3, :3])))
        rep.bump("joint_pos_mm", float(np.linalg.norm(cur[:3, 3] - T0[body][:3, 3])) * 1000)

        mod = dict(q["module"])
        mod["axis"] = axis_for(cur[:3, :3], T0[body][:3, :3], mod["axis"], True, f"{body}_j")
        if "limits" not in exact:
            if mod["kind"] == "R":
                new_lim, e = _snap_revolute_limits(mod["limits"], dist)
                if math.degrees(e) > CONFLICT_DEG:
                    rep.conflicts.append(RuleConflict(
                        "limits_menu", f"{body}_j", "({:.0f}, {:.0f}) deg -> ({:.0f}, {:.0f})".format(
                            *(math.degrees(v) for v in tuple(mod["limits"]) + new_lim))))
                mod["limits"] = new_lim
                rep.bump("limits_deg", math.degrees(e))
            elif mod["kind"] == "P":
                mod["limits"], _ = _nearest_limits(mod["limits"], dist.prismatic_limit_choices_m, 1.0)
        q["module"] = mod

        # length: reach for the hand's next joint (or its fingertip)
        target = T0[f"d{did}p{k + 2}"][:3, 3] if k < n - 1 else tip0[body][:3, 3]
        L0 = float(q["length"])
        reach = float(np.dot(target - cur[:3, 3], cur[:3, 2]))
        if "lengths" in exact:
            L = L0
        else:
            L = _snap_length(reach, lrange, grid)
            rep.bump("length_mm", abs(L - L0) * 1000)
            if L0 < 1e-3:
                rep.conflicts.append(RuleConflict("colocated_joints", f"{body}_j and the next joint",
                                                  f"{L0 * 1000:.1f} mm link -> {L * 1000:.0f} mm"))
            elif not lrange[0] - CONFLICT_MM / 1000 <= L0 <= lrange[1] + CONFLICT_MM / 1000:
                rep.conflicts.append(RuleConflict("link_length_range", body, f"{L0 * 1000:.1f} mm"))
        q["length"] = L
        host_len[body] = L
        steps[j] = DerivationStep(path=steps[j].path, production="Phalanx", params=q)
        T[body] = cur


# --------------------------------------------------------------------------
# Fidelity (E13's metric) and operator applicability
# --------------------------------------------------------------------------


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / (na * nb))))))


def fidelity(model: KinematicModel, name_map: Mapping[str, str], root_transform: np.ndarray,
             derived: KinematicModel, seed: int = FIDELITY_SEED, n_configs: int = FIDELITY_CONFIGS) -> Dict[str, Any]:
    """E13's measurement (``e13_representation.check_hand`` without its
    Pinocchio cross-check): the original model and ``derived`` are driven
    with the same joint values (zero plus ``n_configs`` random
    configurations within the original limits); max joint position error,
    max joint axis error and max fingertip error, in the projection's root
    frame (``root_transform``). The tolerance is 5 mm / 10 deg."""
    movable = [j for j in model.joints if j.type in MOVABLE_TYPES]
    derived_joint = {j.name: j for j in derived.joints}
    derived_frames = {f.name: f for f in derived.frames}
    rng = np.random.default_rng(seed)
    max_pos = max_axis = max_tip = 0.0
    n_tip = 0
    worst_joint, worst_pos = None, -1.0
    for trial in range(n_configs + 1):
        q_orig = {}
        for j in movable:
            if trial == 0:
                q_orig[j.name] = 0.0
            elif j.type == "continuous":
                q_orig[j.name] = float(rng.uniform(-math.pi, math.pi))
            else:
                q_orig[j.name] = float(rng.uniform(*j.limits))
        T_o_all = forward_kinematics(model, q_orig)
        T_d_all = forward_kinematics(derived, {name_map[jn]: v for jn, v in q_orig.items() if jn in name_map})
        for j in movable:
            djn = name_map.get(j.name)
            if djn is None or djn[:-2] not in T_d_all:
                continue
            T_o, T_d = T_o_all[j.child], root_transform @ T_d_all[djn[:-2]]
            pe = float(np.linalg.norm(T_o[:3, 3] - T_d[:3, 3]))
            if pe > worst_pos:
                worst_joint, worst_pos = j.name, pe
            max_pos = max(max_pos, pe)
            ao = T_o[:3, :3] @ np.asarray(j.axis, dtype=float)
            ad = T_d[:3, :3] @ np.asarray(derived_joint[djn].axis, dtype=float)
            max_axis = max(max_axis, _angle_between(ao, ad))
        for orig_body, dname in name_map.items():
            if not dname.endswith("_tip") or orig_body not in T_o_all:
                continue
            fr = derived_frames.get(dname)
            if fr is None:
                continue
            T_tip = root_transform @ (T_d_all[fr.body] @ pose_to_matrix(fr.pose))
            max_tip = max(max_tip, float(np.linalg.norm(T_o_all[orig_body][:3, 3] - T_tip[:3, 3])))
            n_tip += 1
    pos_mm, axis_deg = max_pos * 1000.0, math.degrees(max_axis)
    tip_mm = max_tip * 1000.0 if n_tip else None
    return {
        "max_pos_mm": pos_mm, "max_axis_deg": axis_deg, "max_tip_mm": tip_mm, "worst_joint": worst_joint,
        "within_target": bool(pos_mm <= POS_TOL_MM and axis_deg <= AXIS_TOL_DEG
                              and (tip_mm is None or tip_mm <= POS_TOL_MM)),
    }


def operator_applicability(derivation: Derivation, dist: Distribution, operators: Sequence[str],
                           limits: Optional[GenerationLimits] = None, seed: int = 0) -> Dict[str, str]:
    """Per operator: ``"ok"``, ``"impossible"`` (``VariationImpossible``: no
    valid application, under ``limits`` if given) or ``"error: ..."`` (the
    operator raised on a value it cannot handle)."""
    out: Dict[str, str] = {}
    for op in operators:
        try:
            vary(derivation, np.random.default_rng(seed), dist, operator=op, limits=limits)
            out[op] = "ok"
        except VariationImpossible:
            out[op] = "impossible"
        except Exception as exc:  # noqa: BLE001
            out[op] = f"error: {type(exc).__name__}"
    return out


__all__ = [
    "AXIS_TOL_DEG", "CONFLICT_TEXT", "ConformReport", "POS_TOL_MM", "RuleConflict", "conform_to_grammar",
    "fidelity", "operator_applicability", "snap_axis", "snap_rpy",
]
