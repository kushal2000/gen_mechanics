"""Coverage / support audit for the hand-kinematics grammar (iteration 4).

Two entry points:

``inventory(model)``
    A purely structural, grammar-agnostic census of the constructs present in
    a ``KinematicModel``: joint-type counts, couplings, palm/branch structure,
    axis health, limit symmetry, closures. Works on any valid model, imported
    or grammar-derived.

``coverage(model, dist)``
    Judges an ``inventory`` against ``rules.py``/``distributions.py``:
    ``expressible`` -- could the grammar's productions (see ``rules.py``)
    ever represent this structure at all, independent of parameter ranges?
    ``in_support`` -- does it also fall inside ``dist``'s sampled ranges/grids?
    Both come with named reason lists (``missing_constructs``,
    ``out_of_support``) -- never a bare boolean.

    ``coverage``'s digit-count/phalanx-run/branch-depth checks read the
    grammar's own palm-flag bookkeeping when it is present
    (``CoverageResult.digit_count_source == "palm_flags"``). derive.py always
    flags exactly the palm bodies it builds, so a model with *no* body flagged
    ``palm=True`` at all is never grammar output -- it identifies an imported
    model (e.g. a URDF) that the grammar's palm bookkeeping simply does not
    apply to. Reporting a vacuous ``digit_count`` of 0 in that case would
    misrepresent the model (every import would trip
    ``digit_count_out_of_range:0``), so these three checks fall back instead
    to a purely structural count: the number of movable-joint chains leaving
    the declared root, treating any run of fixed joints as transparent (a
    fixed subtree hanging off the root that never reaches a movable joint is
    not a digit; see ``_movable_children``). The same walk is reused for the
    fallback's phalanx-run and branch-depth checks so all three stay mutually
    consistent. This path is recorded as ``digit_count_source ==
    "root_chains"`` and, since it changes how the result was computed, also
    surfaces as an informational ``"digit_count_source:root_chains"`` entry
    in ``CoverageResult.notes`` (never in ``out_of_support`` -- it is not a
    defect in the model).

This module reads ``KinematicModel``/``Body``/``Joint``/... only; it never
imports ``derive.py`` and never constructs a model itself. It is deliberately
conservative: every heuristic below is documented at its use site, and where
a signal cannot be recovered from a bare model (e.g. "was this URDF axis
attribute present in the source file"), the field says so rather than
guessing. No claim of universality is made anywhere in this module.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from .canonical import normalize_axis_sign
from .coords import CONTINUOUS_SAMPLE_RANGE
from .kinematics import ALL_TYPES, KinematicModel, MOVABLE_TYPES
from .distributions import (Distribution, DEFAULT_DISTRIBUTION, ANGLE_STEP_DEG, DEG, FINE_ANGLE_STEP_DEG,
                            FINE_LENGTH_STEP_M, RESOLUTIONS, link_length_support_m, palm_body_length_support_m,
                            root_length_support_m)

GRID_TOL = 1e-9          # length-grid / axis-grid tolerance, per spec ("... to 1e-9")
UNIT_TOL = 1e-9          # axis unit-norm tolerance
PERP_TOL = 1e-6          # dot-product tolerance for "orthogonal to link direction"
SET_TOL = 1e-9           # membership-in-choice-set tolerance (limits, couplings)

DEFAULT_AXIS = (1.0, 0.0, 0.0)

# rules.py/derive.py name every phalanx body "d{digit_id}p{p+1}" (p the
# 0-based phalanx index). A body ending in "...p1" is therefore always a
# digit or branch-digit's *first* phalanx -- a mount joint, not a fresh
# grid-sampled segment. This pattern only ever matches grammar-produced
# names; on an imported model it simply never matches, so it can only
# remove (never add) a length-grid check -- it exists purely to break the
# rare coincidence where a mount joint's randomly sampled mount_rpy/xyz
# happens to look like a same-digit continuation (see the link-length
# check below).
_FIRST_PHALANX_RE = re.compile(r"^d.*p1$")


def _is_first_phalanx_body(name: str) -> bool:
    return bool(_FIRST_PHALANX_RE.match(name))


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstructInventory:
    joint_type_counts: Dict[str, int]
    movable_joint_count: int

    n_couplings: int
    couplings_with_offset: int
    couplings_negative: int

    branching_bodies: int
    branching_body_names: Tuple[str, ...]

    palm_bodies: int
    palm_joints: int
    palm_tree_nodes: int

    non_perpendicular_axes: int
    undetermined_link_direction: int
    non_unit_axes: int
    default_axis_joints: int  # joints whose stored axis == (1,0,0); a URDF axis
                               # left unspecified defaults here, but an explicit
                               # (1,0,0) axis is indistinguishable post-parse --
                               # this field reports the structural fact only.

    asymmetric_limits: int
    has_continuous: bool
    has_prismatic: bool

    limit_conflicts: int

    n_closures: int
    n_groups: int

    max_digits: int          # number of leaf-terminated chains from the root
    max_chain_depth: int     # longest root-to-leaf joint count


def _norm(v) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def inventory(model: KinematicModel) -> ConstructInventory:
    bodies_by_name = {b.name: b for b in model.bodies}
    joints_by_name = {j.name: j for j in model.joints}
    frames_by_body_tip: Dict[str, Tuple[float, float, float]] = {
        f.body: tuple(f.pose.xyz) for f in model.frames if f.name == f"{f.body}_tip"
    }

    joint_type_counts: Dict[str, int] = {t: 0 for t in ALL_TYPES}
    for j in model.joints:
        joint_type_counts[j.type] = joint_type_counts.get(j.type, 0) + 1
    movable_joint_count = sum(joint_type_counts.get(t, 0) for t in MOVABLE_TYPES)

    # Couplings.
    n_couplings = len(model.couplings)
    couplings_with_offset = sum(1 for c in model.couplings if abs(c.offset) > SET_TOL)
    couplings_negative = sum(1 for c in model.couplings if c.multiplier < 0.0)

    # Children map (structural; validate() already guarantees exactly one
    # parent joint per non-root body, so this is a simple tree).
    children: Dict[str, List] = {}
    for j in model.joints:
        children.setdefault(j.parent, []).append(j)

    palm_body_names = {b.name for b in model.bodies if b.palm}
    palm_bodies = len(palm_body_names)
    palm_joints = sum(1 for j in model.joints if j.child in palm_body_names)
    palm_tree_nodes = palm_bodies

    branching_body_names = tuple(
        sorted(
            name for name, kids in children.items()
            if name not in palm_body_names and len(kids) >= 2
        )
    )
    branching_bodies = len(branching_body_names)

    # Axis health + perpendicularity.
    non_perpendicular_axes = 0
    undetermined_link_direction = 0
    non_unit_axes = 0
    default_axis_joints = 0
    for j in model.joints:
        axis = tuple(float(v) for v in j.axis)
        n = _norm(axis)
        if axis == DEFAULT_AXIS:
            default_axis_joints += 1
        if n != 0.0 and abs(n - 1.0) > UNIT_TOL:
            non_unit_axes += 1
        if j.type not in MOVABLE_TYPES:
            continue
        # link direction = the child's own outgoing joint origin xyz, or its
        # "<body>_tip" frame if one exists; undetermined if neither is present
        # or the resulting vector is (numerically) zero-length.
        direction = None
        if j.child in frames_by_body_tip:
            direction = frames_by_body_tip[j.child]
        else:
            kids = sorted(children.get(j.child, []), key=lambda kj: kj.name)
            if kids:
                direction = tuple(float(v) for v in kids[0].origin.xyz)
        if direction is None or _norm(direction) < 1e-12 or n < 1e-12:
            undetermined_link_direction += 1
            continue
        au = tuple(a / n for a in axis)
        dn = _norm(direction)
        du = tuple(d / dn for d in direction)
        dot = sum(a * d for a, d in zip(au, du))
        if abs(dot) > PERP_TOL:
            non_perpendicular_axes += 1

    # Limits.
    asymmetric_limits = 0
    for j in model.joints:
        if j.limits is not None:
            lo, hi = j.limits
            if abs(lo + hi) > SET_TOL:
                asymmetric_limits += 1

    limit_conflicts = 0
    try:
        from .coords import admissible_box
        _box, conflicts = admissible_box(model)
        limit_conflicts = len(conflicts)
    except Exception:
        limit_conflicts = 0

    # Leaf chains / depth from the declared root.
    leaves = [b.name for b in model.bodies if b.name not in children]
    max_digits = len(leaves)
    depth_of: Dict[str, int] = {model.root: 0}
    order = [model.root]
    idx = 0
    while idx < len(order):
        cur = order[idx]
        idx += 1
        for j in children.get(cur, []):
            depth_of[j.child] = depth_of[cur] + 1
            order.append(j.child)
    max_chain_depth = max(depth_of.get(leaf, 0) for leaf in leaves) if leaves else 0

    return ConstructInventory(
        joint_type_counts=joint_type_counts,
        movable_joint_count=movable_joint_count,
        n_couplings=n_couplings,
        couplings_with_offset=couplings_with_offset,
        couplings_negative=couplings_negative,
        branching_bodies=branching_bodies,
        branching_body_names=branching_body_names,
        palm_bodies=palm_bodies,
        palm_joints=palm_joints,
        palm_tree_nodes=palm_tree_nodes,
        non_perpendicular_axes=non_perpendicular_axes,
        undetermined_link_direction=undetermined_link_direction,
        non_unit_axes=non_unit_axes,
        default_axis_joints=default_axis_joints,
        asymmetric_limits=asymmetric_limits,
        has_continuous=joint_type_counts.get("continuous", 0) > 0,
        has_prismatic=joint_type_counts.get("prismatic", 0) > 0,
        limit_conflicts=limit_conflicts,
        n_closures=len(model.closures),
        n_groups=len(model.groups),
        max_digits=max_digits,
        max_chain_depth=max_chain_depth,
    )


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageResult:
    # Renamed from "expressible" (iteration 6): could the grammar's own
    # productions ever build this *topology* at all, independent of sampled
    # parameter values -- necessary, not sufficient, for the grammar to
    # actually represent the hand (a topology match says nothing about
    # whether the specific lengths/axes/limits are ones the grammar would
    # ever sample; that is what ``in_support`` is for). Renamed because the
    # old name read as a stronger claim than the checks behind it justified.
    topology_expressible: bool
    in_support: bool
    missing_constructs: List[str]
    out_of_support: List[str]
    inventory: ConstructInventory
    digit_count: int
    digit_count_source: str  # "palm_flags" or "root_chains" -- see module docstring
    notes: List[str] = field(default_factory=list)


def _on_angle_grid(rad: float) -> bool:
    """True if ``rad`` lands on the 15-degree grid used everywhere a joint
    axis elevation/azimuth or an orientation is sampled (``distributions.py``:
    ``ANGLE_STEP_DEG``/``N_ANGLE_STEPS``/``N_ELEVATION_STEPS``)."""
    step = ANGLE_STEP_DEG * DEG
    k = rad / step
    return abs(k - round(k)) * step < GRID_TOL


def _axis_on_grid(axis: Tuple[float, float, float]) -> bool:
    x, y, z = (float(v) for v in axis)
    n = _norm((x, y, z))
    if n < 1e-12:
        return False
    x, y, z = x / n, y / n, z / n
    z = max(-1.0, min(1.0, z))
    el = math.acos(z)
    if not _on_angle_grid(el):
        return False
    # At the poles (el == 0 or pi) azimuth is degenerate -- any az is "on
    # grid" since sample_axis's own az_k choice is unobservable there.
    if el < GRID_TOL or abs(el - math.pi) < GRID_TOL:
        return True
    az = math.atan2(y, x)
    return _on_angle_grid(az)


def _length_on_grid(length: float, lo: float, grid: float) -> bool:
    k = (length - lo) / grid
    return abs(k - round(k)) * grid < GRID_TOL


def _in_choice_set(value: float, choices: Tuple[float, ...]) -> bool:
    return any(abs(value - c) <= SET_TOL for c in choices)


def _limits_in_choice_set(limits: Tuple[float, float], choices_deg: Tuple[Tuple[float, float], ...]) -> bool:
    lo, hi = limits
    for lo_deg, hi_deg in choices_deg:
        if abs(lo - lo_deg * DEG) <= SET_TOL and abs(hi - hi_deg * DEG) <= SET_TOL:
            return True
    return False


def _limits_in_choice_set_m(limits: Tuple[float, float], choices_m: Tuple[Tuple[float, float], ...]) -> bool:
    lo, hi = limits
    for lo_m, hi_m in choices_m:
        if abs(lo - lo_m) <= SET_TOL and abs(hi - hi_m) <= SET_TOL:
            return True
    return False


# ---------------------------------------------------------------------------
# relax (E11 support-widening audit): each named string below independently
# and cumulatively loosens one ``coverage`` check. ``coverage``'s own
# behaviour with ``relax=frozenset()`` (the default) is byte-identical to
# every version of this module before E11 -- every relax-aware branch below
# falls through to the pre-existing check when its own name is absent.
# ---------------------------------------------------------------------------

RELAX_NAMES: Tuple[str, ...] = (
    "limits_continuous",
    "limits_range_x1.5",
    "length_grid_1mm",
    "length_continuous",
    "length_range_x1.5",
    "axis_grid_5deg",
    "axis_continuous",
    "coupling_continuous",
    "rest_bend",
    "fixed_in_digit_ok",
    "children_unbounded",
)

LENGTH_GRID_1MM_M = 0.001
COUPLING_CONTINUOUS_MULT_RANGE = (-2.0, 2.0)
COUPLING_CONTINUOUS_OFFSET_RANGE_RAD = (-1.0, 1.0)


def _widen_range(lo: float, hi: float, factor: float) -> Tuple[float, float]:
    """``(lo, hi)`` widened ``factor``x about its own centre."""
    centre = (lo + hi) / 2.0
    half = (hi - lo) / 2.0 * factor
    return centre - half, centre + half


def global_revolute_limit_range_deg(dist: Distribution) -> Tuple[float, float]:
    """The (min lo, max hi), in degrees, spanning every choice in both
    ``dist.revolute_limit_choices_deg`` and ``dist.palm_joint_limit_choices_deg``
    -- the range ``limits_continuous``/``limits_range_x1.5`` judge a revolute
    joint's limits against, in place of exact choice-set membership."""
    los = [lo for lo, _hi in dist.revolute_limit_choices_deg] + [lo for lo, _hi in dist.palm_joint_limit_choices_deg]
    his = [hi for _lo, hi in dist.revolute_limit_choices_deg] + [hi for _lo, hi in dist.palm_joint_limit_choices_deg]
    return min(los), max(his)


def global_prismatic_limit_range_m(dist: Distribution) -> Tuple[float, float]:
    """The (min lo, max hi), in metres, spanning ``dist.prismatic_limit_choices_m``."""
    los = [lo for lo, _hi in dist.prismatic_limit_choices_m]
    his = [hi for _lo, hi in dist.prismatic_limit_choices_m]
    return min(los), max(his)


def _on_angle_grid_step(rad: float, step_deg: float) -> bool:
    step = step_deg * DEG
    k = rad / step
    return abs(k - round(k)) * step < GRID_TOL


def _axis_on_grid_step(axis: Tuple[float, float, float], step_deg: float) -> bool:
    x, y, z = (float(v) for v in axis)
    n = _norm((x, y, z))
    if n < 1e-12:
        return False
    x, y, z = x / n, y / n, z / n
    z = max(-1.0, min(1.0, z))
    el = math.acos(z)
    if not _on_angle_grid_step(el, step_deg):
        return False
    if el < GRID_TOL or abs(el - math.pi) < GRID_TOL:
        return True
    az = math.atan2(y, x)
    return _on_angle_grid_step(az, step_deg)


def _axis_ok(axis: Tuple[float, float, float], relax: FrozenSet[str]) -> bool:
    if "axis_continuous" in relax:
        return True
    step_deg = 5.0 if "axis_grid_5deg" in relax else ANGLE_STEP_DEG
    return _axis_on_grid_step(axis, step_deg)


def _revolute_limits_ok(axis: Tuple[float, float, float], limits: Tuple[float, float], palm: bool,
                         dist: Distribution, relax: FrozenSet[str]) -> bool:
    if "limits_continuous" in relax or "limits_range_x1.5" in relax:
        # Grammar 0.5 (I16 priority 2): read the RANGE off ``dist`` itself
        # (``revolute_limit_range_deg``) rather than deriving it from the
        # (fixed, choice-set) ``revolute_limit_choices_deg``/
        # ``palm_joint_limit_choices_deg`` -- so a ``Distribution`` that
        # widens its own range (e.g. ``variants.G_CONT``) is judged against
        # THAT range, not the unrelated default choice sets.
        # ``global_revolute_limit_range_deg`` (below) is kept as-is for
        # other callers (e11_support_widening's bits-per-joint table).
        #
        # Sign normalisation (I16 priority 2) is applied ONLY here, for the
        # RANGE check -- never for the plain choice-set membership check
        # below, whose choice set the grammar samples independently of the
        # joint's own axis sign (so flipping would judge a grammar-sampled
        # exact-choice-set value against choices it was never drawn against,
        # breaking self-consistency on the grammar's own generated output).
        #
        # ``normalize_axis_sign`` picks ONE fixed representative of the
        # equivalence class {(axis, limits), (-axis, (-hi, -lo))} -- correct
        # for canonical_form's hashing (which only needs equality between
        # the two), but NOT sufficient here on its own: a RANGE check is not
        # invariant under the swap (the range (-45, 110) is not symmetric),
        # so checking only the chosen representative can reject a limits
        # pair that already fit before normalising (regressing the
        # grammar's own generated-output check) while still not being what
        # actually answers "is this joint's motion, under EITHER physically
        # equivalent encoding, in support". The correct existential check is
        # to accept if EITHER the joint's own declared ``limits`` OR its
        # ``normalize_axis_sign`` representative fits the range -- which is
        # exactly what makes Barrett's (-180, 0) deg + a flipped axis judge
        # the same as (0, 180) deg un-flipped (one of the two fits, once the
        # range is wide enough), while never being stricter than checking
        # the declared value alone (monotonic: widening never removes a
        # previously-in-range value from support).
        lo_deg, hi_deg = dist.revolute_limit_range_deg
        if "limits_range_x1.5" in relax:
            lo_deg, hi_deg = _widen_range(lo_deg, hi_deg, 1.5)
        lo_rad, hi_rad = lo_deg * DEG, hi_deg * DEG

        def _fits(pair: Tuple[float, float]) -> bool:
            lo, hi = pair
            return (lo_rad - SET_TOL) <= lo and hi <= (hi_rad + SET_TOL)

        _, norm_limits = normalize_axis_sign(axis, limits)
        return _fits(limits) or _fits(norm_limits)
    choices = dist.palm_joint_limit_choices_deg if palm else dist.revolute_limit_choices_deg
    if getattr(dist, "limits_support_continuous", False) and not _limits_in_choice_set(limits, choices):
        # Support-continuous ranges (distributions.py): any range inside
        # ``revolute_limit_range_deg`` is in the support.
        lo_rad, hi_rad = (v * DEG for v in dist.revolute_limit_range_deg)
        return (lo_rad - SET_TOL) <= limits[0] < limits[1] <= (hi_rad + SET_TOL)
    return _limits_in_choice_set(limits, choices)


def _prismatic_limits_ok(axis: Tuple[float, float, float], limits: Tuple[float, float], dist: Distribution,
                          relax: FrozenSet[str]) -> bool:
    if "limits_continuous" in relax or "limits_range_x1.5" in relax:
        lo_m, hi_m = global_prismatic_limit_range_m(dist)
        if "limits_range_x1.5" in relax:
            lo_m, hi_m = _widen_range(lo_m, hi_m, 1.5)

        def _fits(pair: Tuple[float, float]) -> bool:
            lo, hi = pair
            return (lo_m - SET_TOL) <= lo and hi <= (hi_m + SET_TOL)

        _, norm_limits = normalize_axis_sign(axis, limits)  # see _revolute_limits_ok's comment
        return _fits(limits) or _fits(norm_limits)
    return _limits_in_choice_set_m(limits, dist.prismatic_limit_choices_m)


def _length_ok(length: float, base_range: Tuple[float, float], grid_m: float, relax: FrozenSet[str]) -> Tuple[bool, bool]:
    """Returns ``(in_range, on_grid)``. ``on_grid`` is only meaningful (and
    only checked) when ``in_range`` and ``length_continuous`` is absent from
    ``relax``.

    Grid alignment always uses the ORIGINAL (unwidened) ``base_range[0]`` as
    its reference point, never the widened lo: ``length_range_x1.5``
    extends how far an existing grid reaches, it must never re-anchor the
    grid itself, or an unrelated single relaxation could silently knock an
    already-on-grid length off it (breaking monotonicity -- adding a
    relaxation must never remove a model from support)."""
    grid_lo, orig_hi = base_range
    lo, hi = base_range
    if "length_range_x1.5" in relax:
        lo, hi = _widen_range(lo, hi, 1.5)
    if not (lo - GRID_TOL <= length <= hi + GRID_TOL):
        return False, False
    if "length_continuous" in relax:
        return True, True
    grid = LENGTH_GRID_1MM_M if "length_grid_1mm" in relax else grid_m
    return True, _length_on_grid(length, grid_lo, grid)


def _coupling_mult_ok(multiplier: float, dist: Distribution, relax: FrozenSet[str]) -> bool:
    if "coupling_continuous" in relax:
        lo, hi = COUPLING_CONTINUOUS_MULT_RANGE
        return (lo - SET_TOL) <= multiplier <= (hi + SET_TOL)
    return _in_choice_set(multiplier, dist.coupling_multiplier_choices)


def _coupling_offset_ok(offset: float, dist: Distribution, relax: FrozenSet[str]) -> bool:
    if "coupling_continuous" in relax:
        lo, hi = COUPLING_CONTINUOUS_OFFSET_RANGE_RAD
        return (lo - SET_TOL) <= offset <= (hi + SET_TOL)
    return _in_choice_set(offset, dist.coupling_offset_choices_rad)


def _bend_rpy_ok(rpy: Tuple[float, float, float], dist: Distribution, relax: FrozenSet[str]) -> bool:
    """Grammar 0.5 (I16 priority 1): True if ``rpy`` matches one whole
    triple in ``dist.bend_rpy_choices_rad`` exactly (the rest-bend grid), or
    ``rest_bend`` is in ``relax`` ("any bend" -- the flag's pre-existing
    meaning, unchanged)."""
    if "rest_bend" in relax:
        return True
    support = tuple(dist.bend_rpy_choices_rad) + tuple(getattr(dist, "bend_support_rpy_choices_rad", ()))
    return any(
        abs(rpy[0] - c[0]) <= SET_TOL and abs(rpy[1] - c[1]) <= SET_TOL and abs(rpy[2] - c[2]) <= SET_TOL
        for c in support
    )


def _bend_offset_ok(offset_xy: Tuple[float, float], dist: Distribution, relax: FrozenSet[str]) -> bool:
    """``bend_offset`` counterpart of ``_bend_rpy_ok``."""
    if "rest_bend" in relax:
        return True
    return any(
        abs(offset_xy[0] - c[0]) <= SET_TOL and abs(offset_xy[1] - c[1]) <= SET_TOL
        for c in dist.bend_offset_choices_m
    )


def _movable_children(body_name: str, children: Dict[str, List]) -> List:
    """``children[body_name]``, but with any run of fixed joints made
    transparent: a fixed child is skipped over (never returned itself) and
    replaced by its own children, recursively, until a movable joint (or a
    leaf) is reached. Used only by ``coverage``'s structural
    (``digit_count_source == "root_chains"``) fallback, where "child" must
    mean "next movable joint down this branch", not "next joint" -- a
    fixed-joint mounting frame spliced into a URDF must not itself be
    mistaken for a phalanx or a branch. A fixed subtree with no movable
    joint anywhere below it contributes nothing.
    """
    out: List = []
    stack = list(children.get(body_name, []))
    while stack:
        j = stack.pop()
        if j.type in MOVABLE_TYPES:
            out.append(j)
        else:
            stack.extend(children.get(j.child, []))
    return out


def _fine_bend_ok(rpy: Tuple[float, float, float], dist: Distribution) -> bool:
    """A rest bend in the fine support: each component a multiple of 5
    degrees within the span of that component in the coarse bend support."""
    from .derive import fine_bend_component_grid
    for comp in range(3):
        grid = fine_bend_component_grid(dist, comp)
        if not any(abs(rpy[comp] - g) <= SET_TOL for g in grid):
            return False
    return True


def _fine_limits_ok(limits: Tuple[float, float], dist: Distribution) -> bool:
    """A revolute range in the fine support: both ends on the 5 degree grid
    inside ``revolute_limit_range_deg``."""
    lo_rad, hi_rad = (v * DEG for v in dist.revolute_limit_range_deg)
    return ((lo_rad - SET_TOL) <= limits[0] < limits[1] <= (hi_rad + SET_TOL)
            and all(_on_angle_grid_step(v, FINE_ANGLE_STEP_DEG) for v in limits))


def coverage(
    model: KinematicModel,
    dist: Distribution = DEFAULT_DISTRIBUTION,
    relax: FrozenSet[str] = frozenset(),
    resolution: str = "coarse",
) -> CoverageResult:
    """As before, plus an optional ``relax`` (E11): a set of named support
    widenings (``RELAX_NAMES``), each applied independently and cumulatively.
    ``relax=frozenset()`` (the default) reproduces every pre-E11 caller's
    behaviour exactly -- every relax-aware check below falls through to its
    original, non-relaxed form when its own name is absent. Unknown names in
    ``relax`` are a caller error, not silently ignored.

    Lengths are judged against the distribution's support ranges
    (``link_length_support_m`` and friends; the sampling ranges for every
    variant that does not set one). ``resolution="fine"`` judges the fine
    grid instead of the coarse one (``distributions.FINE_*``): lengths on
    1 mm, axes on 5 degrees, rest bends with every component a multiple of 5
    degrees within the bend support's span, revolute ranges with both ends
    on 5 degrees inside the limit range (or anywhere in it, for a
    distribution whose ranges are continuous). The fine grid contains the
    coarse one, so a model in the coarse support is in the fine one."""
    unknown = set(relax) - set(RELAX_NAMES)
    if unknown:
        raise ValueError(f"coverage: unknown relax name(s): {sorted(unknown)}")
    if resolution not in RESOLUTIONS:
        raise ValueError(f"coverage: resolution must be one of {RESOLUTIONS}, got {resolution!r}")
    fine = resolution == "fine"
    inv = inventory(model)
    missing: List[str] = []
    out: List[str] = []

    joints_by_name = {j.name: j for j in model.joints}
    bodies_by_name = {b.name: b for b in model.bodies}
    palm_body_names = {b.name for b in model.bodies if b.palm}

    children: Dict[str, List] = {}
    for j in model.joints:
        children.setdefault(j.parent, []).append(j)

    non_root_palm = sum(1 for b in model.bodies if b.palm) - (1 if model.root in palm_body_names else 0)
    lo_p, hi_p = dist.palm_body_count_range
    if not (lo_p <= non_root_palm <= hi_p):
        out.append(f"palm_body_count_out_of_range:{non_root_palm}")

    # digit_starts / digit_count_source is needed by several of the new
    # topology checks below (continuation_pose, coupling_scope, fixed_in_digit)
    # as well as by the (pre-existing) in_support checks further down, so it
    # is computed once, up front, and reused everywhere.
    notes: List[str] = []
    if palm_body_names:
        # Top-level digit count, read structurally: a top-level Digit always
        # mounts on a palm body (rules.py), so its first joint has a palm
        # parent and a non-palm child. Exact for grammar output, whose
        # palm=True bookkeeping is set by derive.py for exactly the palm
        # bodies it builds.
        digit_starts = [
            j for j in model.joints
            if j.parent in palm_body_names and j.child not in palm_body_names
        ]
        digit_count_source = "palm_flags"
    else:
        # No body is flagged palm=True at all: never true of grammar output,
        # so this identifies an imported model (e.g. a URDF) that the
        # grammar's palm bookkeeping does not apply to. Fall back to a
        # purely structural count instead of a vacuous 0: the number of
        # movable-joint chains leaving the declared root, treating any run
        # of fixed joints as transparent (see _movable_children). A fixed
        # subtree hanging off the root that never reaches a movable joint is
        # not a digit.
        digit_starts = _movable_children(model.root, children)
        digit_count_source = "root_chains"
        notes.append("digit_count_source:root_chains")

    digit_count = len(digit_starts)
    lo_d, hi_d = dist.digit_count_range
    if not (lo_d <= digit_count <= hi_d):
        out.append(f"digit_count_out_of_range:{digit_count}")

    # ---- topology_expressible: could the grammar's own productions ever
    # build this shape at all, regardless of sampled parameter values? -----

    if inv.n_closures > 0:
        missing.append("loop_closure")

    for t in inv.joint_type_counts:
        if inv.joint_type_counts[t] > 0 and t not in ALL_TYPES:
            missing.append(f"unsupported_joint_type:{t}")

    body_parent_count: Dict[str, int] = {}
    for j in model.joints:
        body_parent_count[j.child] = body_parent_count.get(j.child, 0) + 1
    multi_parent = sorted(b for b, n in body_parent_count.items() if n > 1)
    if multi_parent:
        missing.append("multiple_parent_joints:" + ",".join(multi_parent))

    # Coupling depth: rules.py's ModuleCoupled sources an *earlier phalanx
    # within the same digit* (see rules.py docstring), so a coupling chain
    # can never be longer than one digit's own phalanx run, whose sampled
    # length is bounded above by dist.phalanx_count_range[1] - 1 hops.
    dependent_source = {c.dependent: c.source for c in model.couplings}
    max_chain_hops = 0
    for start in dependent_source:
        hops, cur, seen = 0, start, set()
        while cur in dependent_source and cur not in seen:
            seen.add(cur)
            cur = dependent_source[cur]
            hops += 1
        max_chain_hops = max(max_chain_hops, hops)
    max_allowed_hops = max(0, dist.phalanx_count_range[1] - 1)
    if max_chain_hops > max_allowed_hops:
        missing.append("coupling_chain_too_deep")

    if inv.n_groups > 0:
        # rules.py/derive.py never emit JointGroup productions.
        missing.append("joint_group")

    # (c)/(d)/(f) below need to know which digit's own subtree a body falls
    # in; (a) reuses the same partition (see its comment). Assign every body
    # to the digit whose own subtree (everything reachable, through any
    # joint type, from that digit's start joint) contains it -- a body never
    # in any digit_of entry is on the palm side (the root itself, or
    # whatever sits between it and the first digit-start joint), never
    # constrained by a digit's own branching cap. rules.py's ModuleCoupled
    # always sources an earlier phalanx of the SAME digit -- never another
    # digit, never a palm joint -- and derive.py always builds a Coupled
    # dependent as type "revolute" with limits that are exactly the affine
    # image of the (also always "revolute", after the coupling-source rule)
    # source's own limits.
    digit_of: Dict[str, int] = {}
    for i, start in enumerate(digit_starts):
        stack = [start.child]
        while stack:
            body = stack.pop()
            if body in digit_of:
                continue
            digit_of[body] = i
            stack.extend(kj.child for kj in children.get(body, []))

    # (a) a body inside some digit's own subtree can never end up with more
    # than 1 + max_branch_digits child joints: a phalanx body's own children
    # are, at most, its own next-phalanx joint (0 or 1) plus one joint per
    # branch digit mounted on it (<= max_branch_digits). A body on the palm
    # side (not in digit_of -- root, or any body strictly between root and a
    # digit's own start joint) is excluded: the grammar's own palm fans out
    # into an arbitrary tree (see palm_body_count_range/digit_count_range),
    # so an ordinary multi-finger hand's palm/root is never penalized here,
    # while a body already inside a digit that itself fans out further than
    # a phalanx ever could (e.g. a wrist body reached through a single
    # movable joint off the root, as in ORCA) is correctly flagged.
    excess_children = sorted(
        name for name, kids in children.items()
        if name in digit_of and len(kids) > 1 + dist.max_branch_digits
    )
    if excess_children and "children_unbounded" not in relax:
        missing.append("excess_children:" + ",".join(excess_children))

    coupling_scope_bad = set()
    coupling_type_bad = set()
    # Grammar 0.5 (I16 priority 3): renamed from ``coupling_limits_bad`` /
    # reported as ``coupling_limits_outside_image`` (was
    # ``coupling_limits_not_image``) -- it now fires only when the
    # dependent's declared limits are NOT CONTAINED in the affine image of
    # the source's limits (i.e. the dependent could move somewhere the
    # source's own image never reaches). A dependent whose own limits are
    # strictly tighter than (but still contained in) the image is accepted
    # -- real URDFs commonly declare a tighter dependent limit than the
    # source's own range would image (Ability, Inspire) -- and reported
    # instead as an informational ``dependent_limits_tighter:<joint>`` note
    # (never in ``out_of_support``: it is not a defect).
    coupling_limits_outside_image = set()
    coupling_limits_tighter = set()
    for c in model.couplings:
        dep_joint = joints_by_name.get(c.dependent)
        src_joint = joints_by_name.get(c.source)
        if dep_joint is None or src_joint is None:
            continue
        if src_joint.child in palm_body_names:
            coupling_scope_bad.add(c.dependent)
        elif digit_of.get(dep_joint.child, "?dep") != digit_of.get(src_joint.child, "?src"):
            coupling_scope_bad.add(c.dependent)
        if dep_joint.type != src_joint.type:
            coupling_type_bad.add(c.dependent)
        if dep_joint.limits is not None:
            slo = shi = None
            if src_joint.type == "continuous":
                slo, shi = CONTINUOUS_SAMPLE_RANGE
            elif src_joint.limits is not None:
                slo, shi = src_joint.limits
            if slo is not None:
                lo_img, hi_img = c.multiplier * slo + c.offset, c.multiplier * shi + c.offset
                if lo_img > hi_img:
                    lo_img, hi_img = hi_img, lo_img
                dlo, dhi = dep_joint.limits
                if dlo < lo_img - SET_TOL or dhi > hi_img + SET_TOL:
                    coupling_limits_outside_image.add(c.dependent)
                elif abs(dlo - lo_img) > SET_TOL or abs(dhi - hi_img) > SET_TOL:
                    coupling_limits_tighter.add(c.dependent)
    if coupling_scope_bad:
        missing.append("coupling_scope:" + ",".join(sorted(coupling_scope_bad)))
    if coupling_type_bad:
        missing.append("coupling_type_mismatch:" + ",".join(sorted(coupling_type_bad)))
    if coupling_limits_outside_image:
        missing.append("coupling_limits_outside_image:" + ",".join(sorted(coupling_limits_outside_image)))
    for name in sorted(coupling_limits_tighter):
        notes.append(f"dependent_limits_tighter:{name}")

    # (e) a fixed joint spliced between two movable joints, inside a digit
    # (as opposed to on the palm-to-digit mount edge, or a palm body's own
    # optional joint): the grammar's own Phalanx production always emits a
    # movable joint (R/C/P/Coupled), so a fixed joint can never legitimately
    # sit inside a digit's own chain.
    digit_start_names = {s.name for s in digit_starts}
    fixed_in_digit = sorted(
        j.name for j in model.joints
        if j.type == "fixed"
        and j.child in digit_of
        and j.name not in digit_start_names
        and _movable_children(j.child, children)
    )
    if fixed_in_digit and "fixed_in_digit_ok" not in relax:
        missing.append("fixed_in_digit:" + ",".join(fixed_in_digit))

    # Per-digit phalanx "run" length: walk forward from each digit start
    # while the current body has exactly one child (a lone continuation);
    # stop at a leaf or a branching body. This under-counts the grammar's
    # own phalanx_count whenever a branch is spliced mid-digit (branching
    # bodies end the run early), but it can never over-count -- so it is
    # safe against false "out of range" reports on genuine grammar output
    # (the true value is always >= this run length, and both are always
    # within [1, phalanx_count_range[1]] for a valid grammar model).
    #
    # In the structural (root_chains) fallback "child" means "next movable
    # joint down this branch" (_movable_children), i.e. the longest
    # movable-joint chain from a root child to a leaf/branch, so a fixed
    # mounting joint spliced into a URDF never breaks or inflates a run.
    #
    # (b) every joint visited here beyond a digit's own start is, by
    # construction, a mid-digit continuation joint (its parent has exactly
    # one child, so it is not a branch/mount point). Grammar 0.5 (I16
    # priority 1) gave every phalanx a rest-bend primitive -- a small
    # additional rotation/lateral offset on top of the old pure-z,
    # identity-orientation convention (see derive.py's ``_compose_bend_rpy``)
    # -- so a continuation joint's pose is now topology-expressible (the
    # grammar CAN build it) for ANY rpy/offset value; whether a SPECIFIC
    # value is in the grammar's own SUPPORT is a grid-membership question,
    # judged the same way axis/limit/length grids are (against
    # ``dist.bend_rpy_choices_rad``/``bend_offset_choices_m``, via
    # ``_bend_rpy_ok``/``_bend_offset_ok``) -- so "continuation_pose" now
    # lives in ``out_of_support``, not ``missing_constructs``. ``rest_bend``
    # in ``relax`` keeps its pre-existing meaning ("any bend", i.e. skip the
    # grid check entirely).
    lo_ph, hi_ph = dist.phalanx_count_range
    run_length_issues = []
    continuation_pose_issues = []
    for start in digit_starts:
        run, cur = 1, start.child
        while True:
            kids = _movable_children(cur, children) if digit_count_source == "root_chains" else children.get(cur, [])
            if len(kids) != 1:
                break
            kid = kids[0]
            kid_rpy = tuple(float(v) for v in kid.origin.rpy)
            kid_offset = (float(kid.origin.xyz[0]), float(kid.origin.xyz[1]))
            has_bend = (
                abs(kid_rpy[0]) > SET_TOL or abs(kid_rpy[1]) > SET_TOL or abs(kid_rpy[2]) > SET_TOL
                or abs(kid_offset[0]) > SET_TOL or abs(kid_offset[1]) > SET_TOL
            )
            bend_ok = _bend_rpy_ok(kid_rpy, dist, relax) or (fine and _fine_bend_ok(kid_rpy, dist))
            if has_bend and not (bend_ok and _bend_offset_ok(kid_offset, dist, relax)):
                continuation_pose_issues.append(kid.name)
            run += 1
            cur = kid.child
        if not (lo_ph <= run <= hi_ph):
            run_length_issues.append(start.name)
    if run_length_issues:
        out.append("phalanx_run_out_of_range:" + ",".join(sorted(run_length_issues)))
    if continuation_pose_issues:
        out.append("continuation_pose:" + ",".join(sorted(set(continuation_pose_issues))))

    topology_expressible = len(missing) == 0

    # Branch depth: BFS from every digit start. At a branching body, at most
    # one child continues the *same* digit (identified the same way as the
    # link-length continuation check below: origin.rpy == (0,0,0) and a
    # pure-z origin.xyz) and stays at the current depth; every other child
    # is a branch-digit start and sits one level deeper. A non-branching
    # body's single child (if any) always continues at the same depth.
    #
    # In the structural fallback, "child" again means _movable_children --
    # a fixed pass-through joint is never itself a branch.
    def _is_continuation(kj) -> bool:
        return (
            tuple(kj.origin.rpy) == (0.0, 0.0, 0.0)
            and kj.origin.xyz[0] == 0.0 and kj.origin.xyz[1] == 0.0
            and not _is_first_phalanx_body(kj.child)
        )

    branch_depth_issue = False
    for start in digit_starts:
        stack = [(start.child, 0)]
        seen = set()
        while stack:
            body, depth = stack.pop()
            if body in seen:
                continue
            seen.add(body)
            kids = _movable_children(body, children) if digit_count_source == "root_chains" else children.get(body, [])
            if len(kids) >= 2:
                cont = next((kj for kj in kids if _is_continuation(kj)), None)
                for kj in kids:
                    d2 = depth if kj is cont else depth + 1
                    if d2 > dist.max_branch_depth:
                        branch_depth_issue = True
                    stack.append((kj.child, d2))
            else:
                for kj in kids:
                    stack.append((kj.child, depth))
    if branch_depth_issue:
        out.append("branch_depth_out_of_range")

    # Link lengths: since iteration 5's convention fix, a joint's origin.xyz
    # encodes only *where on the parent's own segment* the child mounts
    # (never the child's own length -- see rules.py's convention-change
    # note), so a body's own segment length must be read from its own
    # "<body>_tip" frame instead, uniformly for every body (root, palm,
    # phalanx/branch alike; no more special-casing a digit/branch's first
    # phalanx, whose origin used to conflate a mount fraction with a grid
    # sample under the old, buggy convention). Root/palm bodies draw their
    # length fresh from palm_length_range_m/link_length_grid_m; every other
    # (phalanx) body draws its fresh from link_length_range_m/link_length_grid_m.
    body_tip_length: Dict[str, float] = {
        f.body: float(f.pose.xyz[2]) for f in model.frames if f.name == f"{f.body}_tip"
    }
    off_grid, out_of_range = [], []
    length_grid = FINE_LENGTH_STEP_M if fine else dist.link_length_grid_m
    for name, length in body_tip_length.items():
        if name == model.root:
            base_range = root_length_support_m(dist)
        elif name in palm_body_names:
            base_range = palm_body_length_support_m(dist)
        else:
            base_range = link_length_support_m(dist)
        in_range, on_grid = _length_ok(length, base_range, length_grid, relax)
        if not in_range:
            out_of_range.append(name)
        elif not on_grid:
            off_grid.append(name)
    if off_grid:
        out.append("link_length_off_grid:" + ",".join(sorted(off_grid)))
    if out_of_range:
        out.append("link_length_out_of_range:" + ",".join(sorted(out_of_range)))

    # Axis grid: every movable joint's axis should land on the 15-degree
    # spherical grid (distributions.sample_axis) in the joint's own frame --
    # unless ``axis_grid_5deg``/``axis_continuous`` relaxes the check. No
    # sign normalisation is needed here: the elevation grid spans [0, 180]
    # in steps that evenly divide 180, so ``-axis`` (elevation ``pi - el``,
    # azimuth ``az + pi``) is on-grid whenever ``axis`` is -- the check is
    # already sign-invariant by construction.
    axis_off_grid = [j.name for j in model.joints if j.type in MOVABLE_TYPES and not _axis_ok(j.axis, relax)
                     and not (fine and _axis_on_grid_step(j.axis, FINE_ANGLE_STEP_DEG))]
    if axis_off_grid:
        out.append("axis_off_grid:" + ",".join(sorted(axis_off_grid)))

    # Limits: revolute/prismatic joints not produced by a coupling must
    # match one of the distribution's own choice tuples -- palm joints
    # (child body palm=True) draw from palm_joint_limit_choices_deg,
    # everything else from revolute_limit_choices_deg / prismatic_limit_choices_m.
    # A coupled joint's own limits are *derived* (image of the source's
    # limits through the affine map), never sampled from a choice set, so
    # they are excluded here; its multiplier/offset are checked instead.
    # ``limits_continuous``/``limits_range_x1.5`` relax exact choice-set
    # membership to a (global, or widened-global) range check instead --
    # sign-normalised (I16 priority 2) INSIDE ``_revolute_limits_ok``/
    # ``_prismatic_limits_ok``'s own continuous-range branch only (never for
    # the plain choice-set membership check: the grammar samples a choice
    # independently of the joint's own axis sign, so normalising there would
    # judge a grammar-sampled exact value against choices it was never drawn
    # against -- see those functions' own comments).
    dependents = {c.dependent for c in model.couplings}
    limits_not_in_set = []
    for j in model.joints:
        if j.name in dependents or j.limits is None:
            continue
        if j.type == "revolute":
            if not (_revolute_limits_ok(j.axis, j.limits, j.child in palm_body_names, dist, relax)
                    or (fine and _fine_limits_ok(j.limits, dist))):
                limits_not_in_set.append(j.name)
        elif j.type == "prismatic":
            if not _prismatic_limits_ok(j.axis, j.limits, dist, relax):
                limits_not_in_set.append(j.name)
    if limits_not_in_set:
        out.append("limits_not_in_set:" + ",".join(sorted(limits_not_in_set)))

    coupling_bad = []
    for c in model.couplings:
        if not _coupling_mult_ok(c.multiplier, dist, relax):
            coupling_bad.append(f"{c.dependent}:multiplier")
        if not _coupling_offset_ok(c.offset, dist, relax):
            coupling_bad.append(f"{c.dependent}:offset")
    if coupling_bad:
        out.append("coupling_params_not_in_set:" + ",".join(sorted(coupling_bad)))

    in_support = topology_expressible and len(out) == 0

    return CoverageResult(
        topology_expressible=topology_expressible,
        in_support=in_support,
        missing_constructs=missing,
        out_of_support=out,
        inventory=inv,
        digit_count=digit_count,
        digit_count_source=digit_count_source,
        notes=notes,
    )
