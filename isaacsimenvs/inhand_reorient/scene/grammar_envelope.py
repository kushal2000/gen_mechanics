"""Padded fixed-topology envelope adapter: grammar `KinematicModel` ->
simulator-ready per-design tables.

See `project-notes/grammar/phase2-adapter-design.md` section 3-4 for the
design this implements. Numpy + `hand_sampler` only: no `isaaclab`/`pxr`
imports, so this module is importable and unit-testable without booting
Kit (`isaacsimenvs/inhand_reorient/scene/author_grammar.py` does the lazy
pxr USD authoring on top of it).

Envelope layout (fixed across every design so a whole population shares one
articulation topology):

    - 5 "finger" slots (0..4), 6 joint slots each (`f{f}_j{d}`), plus 2
      "palm-carrier" slots (`pc0_j`, `pc1_j`) -- 32 joint slots total.
    - finger slots 0, 1, 2 hang directly off root; finger slot 3 hangs off
      `pc0_j`; finger slot 4 hangs off `pc1_j`. `pc0_j`/`pc1_j` hang off
      root. This parent/child structure never changes between designs --
      only each slot's origin/axis/limits/validity does.
    - A design's finger may use fewer than 6 joint slots (its own digit has
      fewer phalanges) or none at all (padding); a design may have 0, 1 or 2
      real (jointed) palm carriers. Unused slots are "ghost": authored with
      the old sampler's convention (tiny link, no collider, locked near
      zero) so they never move and never collide, and the envelope's fixed
      topology is preserved regardless.

Admission (`admit`) restricts a derived `KinematicModel` to designs this
envelope can represent losslessly: revolute-only, no couplings, no in-digit
branching, <= 5 digits, <= 6 joints per digit, <= 2 jointed palm bodies each
carrying <= 1 digit with no jointed palm descendant, and few enough
root-mounted digits to fit the 3 root-only slots plus whichever of the two
carrier slots is not already spoken for by a real carrier.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hand_sampler.design_space import _ordered_box
from hand_sampler.grammar import geometry as grammar_geometry
from hand_sampler.grammar.envelope import fits_envelope
from hand_sampler.grammar.fk import forward_kinematics, pose_to_matrix, rodrigues
from hand_sampler.grammar.kinematics import Joint, KinematicModel

from ..palm_calibration import MIN_SPAWN_HEIGHT_ABOVE_PALM_M

# --------------------------------------------------------------------------
# Envelope layout constants
# --------------------------------------------------------------------------

N_FINGERS = 5
N_JOINTS_PER_FINGER = 6
N_SLOTS = N_FINGERS * N_JOINTS_PER_FINGER + 2  # 32
PC0_SLOT = N_FINGERS * N_JOINTS_PER_FINGER      # 30
PC1_SLOT = PC0_SLOT + 1                          # 31
ROOT_SENTINEL = -1

MAX_DIGITS = N_FINGERS
MAX_JOINTS_PER_DIGIT = N_JOINTS_PER_FINGER
MAX_JOINTED_PALM_BODIES = 2

# Old-sampler ghost-slot convention (see hand_sampler/build.py): a tiny,
# collider-less link that is locked near zero so it never moves and never
# collides, used to pad every design to the fixed 32-slot topology.
GHOST_LENGTH_M = 1e-4
GHOST_LIMITS = (0.0, 1e-8)
GHOST_AXIS = (0.0, 0.0, 1.0)

MAX_REST_PENETRATION_M = 0.003
"""Default `admit(check_overlap=True)` rejection threshold -- moved here
from `make_grammar_population.py` (which now imports it FROM here) so the
gate lives with `admit` itself; see that module's own historical docstring
for the Kit-traced blowup this threshold guards against."""

SLOT_NAMES: Tuple[str, ...] = tuple(
    [f"f{f}_j{d}" for f in range(N_FINGERS) for d in range(N_JOINTS_PER_FINGER)] + ["pc0_j", "pc1_j"]
)


def _slot_parent(idx: int) -> int:
    """Envelope-fixed parent slot index, or `ROOT_SENTINEL` for root."""
    if idx in (PC0_SLOT, PC1_SLOT):
        return ROOT_SENTINEL
    f, d = divmod(idx, N_JOINTS_PER_FINGER)
    if d > 0:
        return idx - 1
    if f in (0, 1, 2):
        return ROOT_SENTINEL
    return PC0_SLOT if f == 3 else PC1_SLOT


SLOT_PARENT: Tuple[int, ...] = tuple(_slot_parent(i) for i in range(N_SLOTS))
# Topological order: pc0, pc1 first (both root children), then every slot in
# increasing index order (every finger slot's parent is either root, an
# earlier same-finger slot, or pc0/pc1 -- all already emitted by then).
TOPOLOGICAL_ORDER: Tuple[int, ...] = (PC0_SLOT, PC1_SLOT) + tuple(range(PC0_SLOT))


class AdmissionError(Exception):
    """Raised by `canonicalize` when the model was not `admit`-ted."""

    def __init__(self, reasons: Sequence[str]):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons) if self.reasons else "design not admitted")


@dataclass(frozen=True)
class AdmissionResult:
    ok: bool
    reasons: Tuple[str, ...]


# --------------------------------------------------------------------------
# Small linear algebra helpers
# --------------------------------------------------------------------------


def _rotz(theta: float) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(theta), math.sin(theta)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    return T


def _shortest_rotation(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """3x3 rotation `R` with `R @ a_hat == b_hat` (minimal-angle solution;
    any twist about `b` is a valid extension since `Rot(axis,q)` is
    conjugation-invariant under an arbitrary choice of that twist -- see
    module docstring's `joint_local_frames` derivation)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        # 180 degrees: any axis perpendicular to `a` works.
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = perp - a * float(np.dot(perp, a))
        perp = perp / np.linalg.norm(perp)
        return rodrigues(tuple(perp), math.pi)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _mat3_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Standard trace-based rotation-matrix -> quaternion (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def _quat_apply_wxyz(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    """Rotate the 3-vector `v` by the unit quaternion `q` (w, x, y, z) --
    same convention as `palm_calibration.quat_apply`, reimplemented here so
    `spawn_height_above_palm_m` stays numpy-only (no `scipy` import)."""
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    w, axis = q[0], q[1:]
    t = 2.0 * np.cross(axis, v)
    return v + w * t + np.cross(axis, t)


# --------------------------------------------------------------------------
# Admission
# --------------------------------------------------------------------------


def _palm_body_names(model: KinematicModel) -> set:
    return {b.name for b in model.bodies if b.palm}


def _jointed_palm_joints(model: KinematicModel) -> List[Joint]:
    palm = _palm_body_names(model)
    return [j for j in model.joints if j.parent in palm and j.child in palm and j.type == "revolute"]


def _palm_joint_by_child(model: KinematicModel) -> Dict[str, Joint]:
    palm = _palm_body_names(model)
    return {j.child: j for j in model.joints if j.parent in palm and j.child in palm}


def _carrier_of_mount(mount: str, jointed_children: set, palm_joint_by_child: Dict[str, Joint], root: str) -> Optional[str]:
    cur = mount
    while True:
        if cur in jointed_children:
            return cur
        if cur == root:
            return None
        j = palm_joint_by_child.get(cur)
        if j is None:
            return None
        cur = j.parent


def _digit_id_of_mount_body(body: str) -> Optional[str]:
    """`"d{digit_id}p1"` -> `digit_id`, for a top-level digit's root body."""
    if not body.startswith("d"):
        return None
    rest = body[1:]
    if "p1" not in rest or not rest.endswith("p1"):
        return None
    digit_id = rest[: -len("p1")]
    return digit_id if digit_id.isdigit() else None


def _palm_index(body: str) -> int:
    """`"palm{i}"` -> `i`, used only to canonically order carriers."""
    return int(body[len("palm"):])


def _admit_structural(model: KinematicModel) -> AdmissionResult:
    """`(ok, reasons)` -- whether `model` fits the padded envelope's SHAPE
    losslessly (revolute-only, digit/joint counts, palm-carrier topology --
    see the module docstring). Independent of rest-pose geometry (no rest-
    overlap or spawn-height check -- those need a canonicalized design and a
    per-caller policy on whether to enforce or merely report them, see the
    public `admit` below, which wraps this). `canonicalize` calls THIS, not
    `admit`, so canonicalizing a structurally-fine-but-overlapping design
    (e.g. to inspect/report it, or before deciding whether to exempt it)
    never recurses through the public gate."""
    reasons: List[str] = []

    env_ok, env_reasons = fits_envelope(
        model, MAX_DIGITS, MAX_JOINTS_PER_DIGIT, allow_palm_joints=True, allow_branches=False
    )
    reasons.extend(env_reasons)

    non_fixed_types = {j.type for j in model.joints if j.type != "fixed"}
    bad_types = sorted(non_fixed_types - {"revolute"})
    if bad_types:
        reasons.append(f"non-revolute movable joint type(s) present: {bad_types}")

    if model.couplings:
        reasons.append(f"{len(model.couplings)} coupling(s) present; envelope requires revolute-only, no couplings")

    palm = _palm_body_names(model)
    jointed = _jointed_palm_joints(model)
    if len(jointed) > MAX_JOINTED_PALM_BODIES:
        reasons.append(f"{len(jointed)} jointed palm bodies exceed max {MAX_JOINTED_PALM_BODIES}")

    palm_joint_by_child = _palm_joint_by_child(model)
    jointed_children = {j.child for j in jointed}
    for j in jointed:
        cur = palm_joint_by_child.get(j.child)
        cur = cur.parent if cur is not None else None
        seen = set()
        while cur is not None:
            if cur in seen:
                break
            seen.add(cur)
            if cur in jointed_children:
                reasons.append(f"jointed palm body {j.child!r} has jointed palm ancestor {cur!r}")
                break
            pj = palm_joint_by_child.get(cur)
            cur = pj.parent if pj is not None else None

    digit_root_joints = [j for j in model.joints if j.parent in palm and j.child not in palm]
    carrier_of = {
        j.name: _carrier_of_mount(j.parent, jointed_children, palm_joint_by_child, model.root)
        for j in digit_root_joints
    }
    counts = Counter(c for c in carrier_of.values() if c is not None)
    for carrier, cnt in counts.items():
        if cnt > 1:
            reasons.append(f"jointed palm body {carrier!r} carries {cnt} digits, exceeds max 1")

    n_carriers = len(jointed)
    n_root_digits = sum(1 for c in carrier_of.values() if c is None)
    if n_root_digits + n_carriers > N_FINGERS:
        reasons.append(
            f"{n_root_digits} root-mounted digit(s) + {n_carriers} jointed palm carrier(s) "
            f"exceed the {N_FINGERS}-finger envelope"
        )

    return AdmissionResult(ok=len(reasons) == 0, reasons=tuple(reasons))


# --------------------------------------------------------------------------
# canonicalize: EnvelopeDesign
# --------------------------------------------------------------------------


@dataclass
class EnvelopeDesign:
    source: str
    model: KinematicModel
    slot_valid: np.ndarray           # (32,) bool
    slot_origin: np.ndarray          # (32,4,4) float64 -- joint origin, relative to the slot's ENVELOPE parent
    slot_axis: np.ndarray            # (32,3) float64 -- in the slot's own (child) local frame
    slot_limits: np.ndarray          # (32,2) float64
    slot_length: np.ndarray          # (32,) float64 -- segment length (tip frame z-offset)
    slot_joint_name: Tuple[Optional[str], ...]
    slot_body_name: Tuple[Optional[str], ...]
    capsule_radius_m: float
    root_length_m: float
    finger_digit_id: Tuple[Optional[str], ...]  # length 5
    grammar_version: str = ""
    reasons: Tuple[str, ...] = field(default_factory=tuple)  # empty iff admitted
    fingertip_marker_ok: np.ndarray = field(default_factory=lambda: np.zeros(N_FINGERS, dtype=bool))
    """(5,) bool -- see `canonicalize`'s `_fill_chain`: whether this finger's
    envelope fingertip-body slot is a ghost placed exactly at the real tip
    (review item 1). False (no ghost to place) for a finger whose real chain
    fills all 6 slots -- `palm_up` masks such a finger's `fingertip_valid`."""
    filtered_pairs: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    """Slot-index pairs (root capsule uses -1) that overlap at rest and are
    EXEMPTED from admission rejection (projected commercial hands only --
    `admit`'s `check_overlap=False` path, see its docstring) -- authoring
    must collision-filter exactly these pairs so PhysX's depenetration
    impulse doesn't blow up the ghost/carrier joints at step 0. Empty for
    every sampled design (those are REJECTED on overlap instead, never
    exempted)."""
    sha256: str = ""
    """This design's entry-level sha256 (over its raw derivation dict,
    `population_file._entry_sha256`) -- set by `population_file.
    load_population`, empty for a design built directly by `canonicalize`
    outside that path (e.g. most tests). Threaded through so per-design
    graded-score reports (Part C) can cite it without needing the
    population file's own JSON at report time."""


def _compose_palm_transform(mount_body: str, stop_body: str, palm_joint_by_child: Dict[str, Joint]) -> np.ndarray:
    """Root-to-`mount_body` (or carrier-to-`mount_body`) transform, composed
    from a chain of FIXED palm joints only (by construction the caller never
    asks this to cross a jointed palm joint -- `admit` already rejects any
    design where it would have to)."""
    if mount_body == stop_body:
        return np.eye(4)
    chain = [mount_body]
    cur = mount_body
    while cur != stop_body:
        j = palm_joint_by_child.get(cur)
        if j is None:
            raise ValueError(f"cannot reach {stop_body!r} from {mount_body!r} via palm joints")
        cur = j.parent
        chain.append(cur)
    chain.reverse()
    T = np.eye(4)
    for parent_b, child_b in zip(chain[:-1], chain[1:]):
        T = T @ pose_to_matrix(palm_joint_by_child[child_b].origin)
    return T


def canonicalize(model: KinematicModel, source: str = "") -> EnvelopeDesign:
    result = _admit_structural(model)
    if not result.ok:
        raise AdmissionError(result.reasons)

    palm = _palm_body_names(model)
    jointed = _jointed_palm_joints(model)
    jointed_sorted = sorted(jointed, key=lambda j: _palm_index(j.child))[:2]
    palm_joint_by_child = _palm_joint_by_child(model)
    jointed_children = {j.child for j in jointed}
    carrier_body_to_pc = {j.child: i for i, j in enumerate(jointed_sorted)}

    digit_root_joints = [j for j in model.joints if j.parent in palm and j.child not in palm]
    carrier_of = {
        j.name: _carrier_of_mount(j.parent, jointed_children, palm_joint_by_child, model.root)
        for j in digit_root_joints
    }

    root_digits = sorted(
        (j for j in digit_root_joints if carrier_of[j.name] is None),
        key=lambda j: int(_digit_id_of_mount_body(j.child)),
    )
    carrier_digit: Dict[int, Optional[Joint]] = {i: None for i in range(len(jointed_sorted))}
    for j in digit_root_joints:
        c = carrier_of[j.name]
        if c is not None and c in carrier_body_to_pc:
            carrier_digit[carrier_body_to_pc[c]] = j

    finger_root_joint: List[Optional[Joint]] = [None] * N_FINGERS
    finger_root_mount_transform: List[np.ndarray] = [np.eye(4)] * N_FINGERS
    reserved_fingers = {3 + pc for pc in carrier_digit}
    for pc, j in carrier_digit.items():
        if j is not None:
            carrier_body = jointed_sorted[pc].child
            finger_root_joint[3 + pc] = j
            finger_root_mount_transform[3 + pc] = _compose_palm_transform(j.parent, carrier_body, palm_joint_by_child)

    free_slots = [f for f in range(N_FINGERS) if f not in reserved_fingers]
    assert len(root_digits) <= len(free_slots), "admit() should have rejected this"
    for slot, j in zip(free_slots, root_digits):
        finger_root_joint[slot] = j
        finger_root_mount_transform[slot] = _compose_palm_transform(j.parent, model.root, palm_joint_by_child)

    slot_valid = np.zeros(N_SLOTS, dtype=bool)
    slot_origin = np.tile(np.eye(4), (N_SLOTS, 1, 1))
    slot_axis = np.tile(np.array(GHOST_AXIS), (N_SLOTS, 1))
    slot_limits = np.tile(np.array(GHOST_LIMITS), (N_SLOTS, 1))
    slot_length = np.full(N_SLOTS, GHOST_LENGTH_M)
    slot_joint_name: List[Optional[str]] = [None] * N_SLOTS
    slot_body_name: List[Optional[str]] = [None] * N_SLOTS
    finger_digit_id: List[Optional[str]] = [None] * N_FINGERS

    children_by_parent: Dict[str, List[Joint]] = {}
    for j in model.joints:
        children_by_parent.setdefault(j.parent, []).append(j)
    frames_by_name = {fr.name: fr for fr in model.frames}

    fingertip_marker_ok = np.zeros(N_FINGERS, dtype=bool)

    def _fill_chain(start_joint: Optional[Joint], base_slot: int, root_transform: np.ndarray) -> None:
        if start_joint is None:
            return
        finger_digit_id[base_slot // N_JOINTS_PER_FINGER] = _digit_id_of_mount_body(start_joint.child)
        chain: List[Joint] = []
        cur: Optional[Joint] = start_joint
        while cur is not None and len(chain) < N_JOINTS_PER_FINGER:
            chain.append(cur)
            nxt = children_by_parent.get(cur.child, [])
            cur = nxt[0] if len(nxt) == 1 else None
        cumulative = root_transform
        for d, j in enumerate(chain):
            idx = base_slot + d
            slot_valid[idx] = True
            slot_origin[idx] = cumulative @ pose_to_matrix(j.origin) if d == 0 else pose_to_matrix(j.origin)
            slot_axis[idx] = np.asarray(j.axis, dtype=float)
            slot_limits[idx] = np.asarray(j.limits, dtype=float)
            frame = frames_by_name.get(f"{j.child}_tip")
            slot_length[idx] = float(frame.pose.xyz[2]) if frame is not None else GHOST_LENGTH_M
            slot_joint_name[idx] = j.name
            slot_body_name[idx] = j.child

        # Review item 1 (fingertip one link short): a ghost slot's own
        # `slot_origin` defaults to identity (see the `np.tile(np.eye(4), ...)`
        # above), so with NO fix the first ghost slot after this finger's real
        # chain sits at the BASE of the last real link (its parent's own
        # origin), not that link's TIP -- and since every later ghost in the
        # same finger is itself identity-offset from the one before it, the
        # whole ghost tail (including `f{f}_link5`, whichever slot in this
        # finger every population/observation caller treats as "the
        # fingertip body") inherits that same wrong position. Translating
        # ONLY this first ghost by the last real link's own length puts it
        # (and everything chained after it) exactly at the real tip -- exact,
        # not approximate, because a ghost's own length is negligible
        # (`GHOST_LENGTH_M`) and its axis is the z-identity `GHOST_AXIS`, so
        # `authored_fk`'s local transform for it is a pure translation.
        if chain and len(chain) < N_JOINTS_PER_FINGER:
            last_idx = base_slot + len(chain) - 1
            first_ghost_idx = base_slot + len(chain)
            tip_translation = np.eye(4)
            tip_translation[2, 3] = float(slot_length[last_idx])
            slot_origin[first_ghost_idx] = tip_translation
        # Whether THIS finger's envelope-fixed "fingertip body" slot
        # (index `base_slot + N_JOINTS_PER_FINGER - 1`, e.g. `f{f}_link5`)
        # is a ghost that the translation above places exactly at the real
        # tip. False when the finger's real chain fills all
        # `N_JOINTS_PER_FINGER` slots (no ghost left to translate) -- see
        # `palm_up`'s `fingertip_valid`, which masks the runtime fingertip
        # OBSERVATION (not this finger's existence) in that case.
        fingertip_marker_ok[base_slot // N_JOINTS_PER_FINGER] = bool(chain) and len(chain) < N_JOINTS_PER_FINGER

    for f in range(N_FINGERS):
        _fill_chain(finger_root_joint[f], f * N_JOINTS_PER_FINGER, finger_root_mount_transform[f])

    for pc, j in enumerate(jointed_sorted):
        slot = PC0_SLOT + pc
        pre = _compose_palm_transform(j.parent, model.root, palm_joint_by_child)
        slot_valid[slot] = True
        slot_origin[slot] = pre @ pose_to_matrix(j.origin)
        slot_axis[slot] = np.asarray(j.axis, dtype=float)
        slot_limits[slot] = np.asarray(j.limits, dtype=float)
        frame = frames_by_name.get(f"{j.child}_tip")
        slot_length[slot] = float(frame.pose.xyz[2]) if frame is not None else GHOST_LENGTH_M
        slot_joint_name[slot] = j.name
        slot_body_name[slot] = j.child

    capsule_radius = next((b.radius for b in model.bodies if b.radius is not None), 0.01)
    root_frame = frames_by_name.get("root_tip")
    root_length = float(root_frame.pose.xyz[2]) if root_frame is not None else 0.0

    return EnvelopeDesign(
        source=source, model=model, slot_valid=slot_valid, slot_origin=slot_origin, slot_axis=slot_axis,
        slot_limits=slot_limits, slot_length=slot_length, slot_joint_name=tuple(slot_joint_name),
        slot_body_name=tuple(slot_body_name), capsule_radius_m=float(capsule_radius), root_length_m=root_length,
        finger_digit_id=tuple(finger_digit_id), grammar_version=getattr(model, "grammar_version", ""),
        reasons=(), fingertip_marker_ok=fingertip_marker_ok, filtered_pairs=(),
    )


def admit(
    model: KinematicModel,
    *,
    check_overlap: bool = True,
    check_spawn_height: bool = True,
    max_rest_penetration_m: float = MAX_REST_PENETRATION_M,
    min_spawn_height_m: float = MIN_SPAWN_HEIGHT_ABOVE_PALM_M,
) -> AdmissionResult:
    """The public admission gate for a SAMPLED design (review items 2 and 3):
    `_admit_structural(model)` (envelope shape) AND, by default, the design
    note's risk-4 rest-overlap filter AND the spawn-height requirement
    (review item 2 for populations -- `drop_detection` already uses world z
    for the RUNTIME check; this is the admission-time equivalent). Both
    physical checks canonicalize `model` once (structural admission has
    already passed, so this cannot raise) and are independent -- a design
    can fail either, both, or neither, and every violated reason is
    reported, not just the first.

    `check_overlap=False` is for PROJECTED commercial hands ONLY (review
    item 2's exemption: their overlaps are artifacts of the one-radius-per-
    hand capsule projection, not a real self-intersecting design) --
    `population_file.projected_entry` passes it; every SAMPLED-design path
    (`sampled_entries`, `load_population`, and hence `--variant
    sampled_only`) uses the default, closing review item 4's "not enforced,
    and sampled_only skips it" gap. `check_spawn_height=False` exists for
    symmetry/testing only; no caller in this package passes it."""
    structural = _admit_structural(model)
    if not structural.ok:
        return structural

    reasons: List[str] = []
    design = canonicalize(model)
    if check_overlap:
        pairs = rest_overlap_pairs(design)
        bad = [(i, j, pen) for i, j, pen in pairs if pen > max_rest_penetration_m]
        if bad:
            worst = max(pen for _, _, pen in bad)
            reasons.append(
                f"{len(bad)} rest-overlap pair(s) exceed {max_rest_penetration_m * 1000.0:.1f} mm "
                f"(worst {worst * 1000.0:.2f} mm)"
            )
    if check_spawn_height:
        height = spawn_height_above_palm_m(design, palm_up(design, n_sweep=0))
        if height < min_spawn_height_m:
            reasons.append(
                f"spawn height {height * 1000.0:.1f} mm above the palm along world z after base_rot "
                f"< required {min_spawn_height_m * 1000.0:.1f} mm"
            )
    return AdmissionResult(ok=len(reasons) == 0, reasons=tuple(reasons))


def viability_report(model: KinematicModel) -> dict:
    """Numpy + `hand_sampler` only (no isaaclab), for the CPU grammar screen
    (plan revision, step 2) -- everything a screen worker needs about ONE
    derived model, without touching Kit:

        {"admitted": bool, "reasons": [str, ...], "max_rest_overlap_mm": float,
         "fingertips_reachable": int, "spawn_height_mm": float,
         "digit_count": int, "joint_count": int}

    `reasons` is `admit(model)`'s own (structural + overlap + spawn-height,
    in that order -- structural failure short-circuits the rest, matching
    `admit`). The geometry fields (`max_rest_overlap_mm`, `spawn_height_mm`,
    `digit_count`, `joint_count`) are still reported even when `admitted` is
    `False` and even when the failure is a physical (not structural) one --
    the whole point of a screen is comparing near-miss designs, not only
    admitted ones -- but are `None` when the model fails the STRUCTURAL
    check (no envelope slots to measure at all)."""
    structural = _admit_structural(model)
    if not structural.ok:
        return {
            "admitted": False, "reasons": list(structural.reasons), "max_rest_overlap_mm": None,
            "fingertips_reachable": None, "spawn_height_mm": None, "digit_count": None, "joint_count": None,
        }

    design = canonicalize(model)
    pairs = rest_overlap_pairs(design)
    max_overlap_mm = max((pen for _, _, pen in pairs), default=0.0) * 1000.0
    pu = palm_up(design)
    spawn_height_mm = spawn_height_above_palm_m(design, pu) * 1000.0
    digit_count = int(sum(1 for d in design.finger_digit_id if d is not None))
    joint_count = int(design.slot_valid.sum())

    result = admit(model)
    return {
        "admitted": bool(result.ok), "reasons": list(result.reasons), "max_rest_overlap_mm": float(max_overlap_mm),
        "fingertips_reachable": int(pu.reachable_fingertips), "spawn_height_mm": float(spawn_height_mm),
        "digit_count": digit_count, "joint_count": joint_count,
    }


# --------------------------------------------------------------------------
# joint_local_frames / authored_fk
# --------------------------------------------------------------------------


def joint_local_frames(design: EnvelopeDesign) -> np.ndarray:
    """`(32, 2, 4, 4)`: per slot, `(frame0, frame1)` such that the slot's
    child-body world transform is `parent_world @ frame0 @ Rz(q) @
    inv(frame1)` for any `q` -- i.e. a dual-frame authoring convention (as
    PhysX/USD physics joints use) that always rotates about the LOCAL Z axis,
    with the real per-joint axis baked into `frame0`/`frame1` via a shared
    alignment rotation `R` (`R @ ez == axis`):

        frame0 = slot_origin @ [R, 0; 0, 1]
        frame1 = [R, 0; 0, 1]

    because `Rot(axis, q) == R @ Rz(q) @ inv(R)` for ANY rotation `R` with
    `R @ ez == axis` (conjugation of `SO(3)` by `R` rotates the axis by `R`
    and preserves the angle -- true for every such `R`, not just the
    minimal-angle one `_shortest_rotation` happens to return)."""
    out = np.zeros((N_SLOTS, 2, 4, 4))
    for idx in range(N_SLOTS):
        R = _shortest_rotation((0.0, 0.0, 1.0), design.slot_axis[idx])
        R4 = np.eye(4)
        R4[:3, :3] = R
        out[idx, 0] = design.slot_origin[idx] @ R4
        out[idx, 1] = R4
    return out


def authored_fk(design: EnvelopeDesign, q: np.ndarray) -> np.ndarray:
    """`(32,4,4)` world (root-frame) transforms of every slot's child body,
    at joint vector `q` (length 32, envelope slot order). Exact (not
    approximate) reconstruction of `hand_sampler.grammar.fk.forward_kinematics`
    restricted to this design's real joints -- see this module's tests."""
    frames = joint_local_frames(design)
    q = np.asarray(q, dtype=float).reshape(N_SLOTS)
    T = [np.eye(4) for _ in range(N_SLOTS)]
    for idx in TOPOLOGICAL_ORDER:
        parent = SLOT_PARENT[idx]
        parent_T = np.eye(4) if parent == ROOT_SENTINEL else T[parent]
        frame0, frame1 = frames[idx]
        local = frame0 @ _rotz(float(q[idx])) @ np.linalg.inv(frame1)
        T[idx] = parent_T @ local
    return np.stack(T)


def grammar_fk_reference(design: EnvelopeDesign, q: np.ndarray) -> np.ndarray:
    """`(32,4,4)`: the SAME quantity as `authored_fk`, computed instead by
    calling `hand_sampler.grammar.fk.forward_kinematics` directly on
    `design.model` (ghost/invalid slots get identity). Used by tests as the
    independent oracle `authored_fk` must match to 1e-9."""
    q = np.asarray(q, dtype=float).reshape(N_SLOTS)
    q_map = {}
    for idx in range(N_SLOTS):
        name = design.slot_joint_name[idx]
        if name is not None:
            q_map[name] = float(q[idx])
    world = forward_kinematics(design.model, q_map)
    out = np.tile(np.eye(4), (N_SLOTS, 1, 1))
    for idx in range(N_SLOTS):
        body = design.slot_body_name[idx]
        if body is not None:
            out[idx] = world[body]
    return out


# --------------------------------------------------------------------------
# token_boxes
# --------------------------------------------------------------------------


def _capsule_box_vertices(length: float, radius: float) -> np.ndarray:
    """8 corners of a capsule's bounding box, in the child body's own local
    frame: the body's own segment runs `(0,0,0)` -> `(0,0,length)` (grammar's
    "segments along +z" convention), cross-section `radius`. Mirrors what
    `design_space.joint_link_boxes` would compute from real collision mesh
    vertices -- grammar hands have no mesh, so this box IS the collision
    geometry, not an approximation of it."""
    zs = (0.0, max(length, 1e-9))
    xs = (-radius, radius)
    ys = (-radius, radius)
    return np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=np.float64)


def token_boxes(design: EnvelopeDesign) -> np.ndarray:
    """`(32, 4, 3)` float32 ordered boxes, via `design_space._ordered_box`
    on each slot's own capsule bounding box (ghost slots use the ghost
    link's tiny dimensions)."""
    out = np.zeros((N_SLOTS, 4, 3), dtype=np.float32)
    for idx in range(N_SLOTS):
        valid = bool(design.slot_valid[idx])
        length = float(design.slot_length[idx])
        radius = design.capsule_radius_m if valid else (GHOST_LENGTH_M / 2.0)
        verts = _capsule_box_vertices(length, radius)
        box, _ok = _ordered_box(verts, design.slot_axis[idx])
        out[idx] = box
    return out


# --------------------------------------------------------------------------
# mass_props
# --------------------------------------------------------------------------


DEFAULT_DENSITY_KG_M3 = 1200.0


def mass_props(design: EnvelopeDesign, density: float = DEFAULT_DENSITY_KG_M3) -> Dict[str, Tuple[float, np.ndarray, np.ndarray]]:
    """Per-body `{name: (mass, com, inertia_about_com)}`, from
    `hand_sampler.grammar.geometry.build_geometry`/`mass_properties` (which
    report inertia about each body's own ORIGIN, per that module's
    docstring) shifted to the body's own COM by the parallel-axis theorem."""
    spec = grammar_geometry.build_geometry(design.model)
    raw = grammar_geometry.mass_properties(spec, density)
    out: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}
    for name, props in raw.items():
        M = float(props.mass)
        com = np.asarray(props.com, dtype=float)
        I_origin = np.asarray(props.inertia, dtype=float)
        I_com = I_origin - M * (float(np.dot(com, com)) * np.eye(3) - np.outer(com, com))
        out[name] = (M, com, I_com)
    return out


# --------------------------------------------------------------------------
# rest_overlap_pairs
# --------------------------------------------------------------------------


ROOT_NODE = -1
"""Pseudo slot-index for the root/palm capsule in `rest_overlap_pairs`'s
output and `EnvelopeDesign.filtered_pairs` -- the root body is authored
(`author_grammar.author_design`'s `root_body_path`) but is not one of the 32
envelope joint SLOTS, so it has no slot index of its own."""


def _effective_parent(design: EnvelopeDesign, idx: int) -> int:
    """The slot (or `ROOT_NODE`) this VALID slot `idx` is actually mounted
    on, for adjacency purposes -- i.e. what it is EXPECTED to touch and
    should be excluded from the overlap check against. Usually just
    `SLOT_PARENT[idx]` (finger continuation joints, and PC0_SLOT/PC1_SLOT,
    whose own parent is always `ROOT_SENTINEL` -> `ROOT_NODE`). The one case
    `SLOT_PARENT` gets structurally wrong for THIS purpose: a finger 3/4 base
    slot (`d==0`) whose structural parent (PC0_SLOT/PC1_SLOT) is a GHOST
    (padding-only carrier, no real jointed palm body there) -- `canonicalize`
    already collapses that ghost's zero transform when composing this slot's
    own `slot_origin` (`_compose_palm_transform` walks straight to
    `model.root` in that case), so the slot is PHYSICALLY mounted directly on
    root, not on the (non-existent, uncollidable) ghost carrier -- review
    item 4's "pairs that meet across ghosts". A same-finger continuation
    joint's parent is always the previous slot in the SAME finger, which
    `canonicalize`'s contiguous-prefix fill guarantees is valid whenever
    `idx` itself is valid, so this can only ever fire for a base (`d==0`)
    slot."""
    parent = SLOT_PARENT[idx]
    if parent == ROOT_SENTINEL:
        return ROOT_NODE
    if design.slot_valid[parent]:
        return parent
    return ROOT_NODE


def rest_overlap_pairs(design: EnvelopeDesign) -> List[Tuple[int, int, float]]:
    """Capsule-capsule rest-pose (q=0) self-penetration filter over the
    AUTHORED geometry: the root/palm capsule (`ROOT_NODE`) plus every VALID
    joint slot, every pair whose capsules overlap by more than their own
    radii allow, EXCLUDING pairs that are expected to touch (envelope
    parent/child, walked through any ghost carrier via `_effective_parent`
    -- review item 4's two gaps: "it ignores the root capsule and pairs that
    meet across ghosts"). Returns `(slot_i, slot_j, penetration_depth_m)`
    for each violating pair, `slot_i`/`slot_j` possibly `ROOT_NODE`. Ghost
    slots themselves (other than the root pseudo-node) are never checked --
    they have no collider (see `author_grammar._author_body_and_collider`'s
    `real=False` path) and are physically incapable of a rest collision.
    This is a cheap capsule proxy for true self-penetration (real palm-cell
    geometry can be tighter or looser); see the design note's risk 4."""
    from hand_sampler.design_space import segment_distance

    T = authored_fk(design, np.zeros(N_SLOTS))
    valid_slots = [i for i in range(N_SLOTS) if design.slot_valid[i]]
    adjacent = set()
    for idx in valid_slots:
        parent = _effective_parent(design, idx)
        adjacent.add((idx, parent))
        adjacent.add((parent, idx))

    endpoints: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {
        ROOT_NODE: (np.zeros(3), np.array([0.0, 0.0, design.root_length_m]), design.capsule_radius_m),
    }
    for idx in valid_slots:
        p0 = T[idx][:3, 3]
        p1 = (T[idx] @ np.array([0.0, 0.0, float(design.slot_length[idx]), 1.0]))[:3]
        endpoints[idx] = (p0, p1, design.capsule_radius_m)

    nodes = [ROOT_NODE] + valid_slots
    out: List[Tuple[int, int, float]] = []
    for a_i, i in enumerate(nodes):
        for j in nodes[a_i + 1:]:
            if (i, j) in adjacent:
                continue
            p0, p1, ri = endpoints[i]
            q0, q1, rj = endpoints[j]
            d = segment_distance(p0, p1, q0, q1)
            pen = (ri + rj) - d
            if pen > 1e-6:
                out.append((i, j, float(pen)))
    return out


def mark_filtered_pairs(design: EnvelopeDesign, max_penetration_m: float = 0.0) -> EnvelopeDesign:
    """Return a copy of `design` with `filtered_pairs` set to every
    `rest_overlap_pairs` pair deeper than `max_penetration_m` -- for designs
    EXEMPTED from overlap rejection (projected commercial hands; see
    `admit`'s `check_overlap=False` and this module's docstring on
    `EnvelopeDesign.filtered_pairs`). `author_grammar.author_design` reads
    this to collision-filter exactly these pairs at authoring time, instead
    of relying on PhysX to resolve a real interpenetration itself (which, for
    a design like this one, blows up the ghost/carrier joints on step 0 --
    see this module's `MAX_REST_PENETRATION_M`)."""
    import dataclasses

    pairs = tuple((i, j) for i, j, pen in rest_overlap_pairs(design) if pen > max_penetration_m)
    return dataclasses.replace(design, filtered_pairs=pairs)


# --------------------------------------------------------------------------
# palm_up
# --------------------------------------------------------------------------


@dataclass
class PalmUpResult:
    normal: np.ndarray          # (3,) unit vector, local (root) frame
    base_rot_wxyz: np.ndarray   # (4,) quaternion mapping local -> world so `normal` -> world +Z
    default_q: np.ndarray       # (32,) mild-curl default pose
    spawn_offset: np.ndarray    # (3,) local-frame spawn point for the object
    fingertip_valid: np.ndarray  # (5,) bool
    fingertip_offsets: np.ndarray  # (5,3) local-frame fingertip positions at default_q
    reachable_fingertips: int   # CPU reachability metric (I24 lesson)
    n_swept: int


def palm_up(
    design: EnvelopeDesign,
    object_half_size: float = 0.03,
    n_sweep: int = 200,
    seed: int = 0,
    reach_tol_m: float = 0.05,
    curl_frac: float = 0.35,
) -> PalmUpResult:
    """Analytic palm-up calibration (design note section 4, "Analytic palm
    calibration") plus the I24 lesson applied to grammar hands: spawn the
    object over the fingertip workspace with a mild default curl, not over
    the palm/root origin, and report a CPU reachability metric (number of
    fingertips that reach within `reach_tol_m` of the spawn point over a
    random joint sweep) instead of assuming reachability."""
    valid_slots = [i for i in range(N_SLOTS) if design.slot_valid[i]]
    finger_valid = np.zeros(N_FINGERS, dtype=bool)
    finger_last_slot: List[Optional[int]] = [None] * N_FINGERS
    for f in range(N_FINGERS):
        base = f * N_JOINTS_PER_FINGER
        used = [base + d for d in range(N_JOINTS_PER_FINGER) if design.slot_valid[base + d]]
        if used:
            finger_valid[f] = True
            finger_last_slot[f] = max(used)

    mid_q = np.zeros(N_SLOTS)
    for idx in valid_slots:
        lo, hi = design.slot_limits[idx]
        mid_q[idx] = lo + curl_frac * (hi - lo)

    T_mid = authored_fk(design, mid_q)
    T0 = authored_fk(design, np.zeros(N_SLOTS))

    mount_pts = []
    tip_pts_mid = []
    for f in range(N_FINGERS):
        base = f * N_JOINTS_PER_FINGER
        if not finger_valid[f]:
            continue
        mount_pts.append(T0[base][:3, 3])
        last = finger_last_slot[f]
        tip_pts_mid.append((T_mid[last] @ np.array([0.0, 0.0, float(design.slot_length[last]), 1.0]))[:3])

    if mount_pts:
        mount_centroid = np.mean(np.stack(mount_pts), axis=0)
        tip_centroid = np.mean(np.stack(tip_pts_mid), axis=0)
    else:
        mount_centroid = np.zeros(3)
        tip_centroid = np.array([0.0, 0.0, 1.0])

    delta = tip_centroid - mount_centroid
    if np.linalg.norm(delta) < 1e-9:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = delta / np.linalg.norm(delta)

    if len(mount_pts) >= 2:
        x_raw = mount_pts[0] - mount_centroid
    else:
        x_raw = np.array([1.0, 0.0, 0.0])
    x_raw = x_raw - normal * float(np.dot(x_raw, normal))
    if np.linalg.norm(x_raw) < 1e-9:
        x_raw = np.array([1.0, 0.0, 0.0]) - normal * normal[0]
        if np.linalg.norm(x_raw) < 1e-9:
            x_raw = np.array([0.0, 1.0, 0.0]) - normal * normal[1]
    x_axis = x_raw / np.linalg.norm(x_raw)
    y_axis = np.cross(normal, x_axis)
    R_local_cols = np.column_stack([x_axis, y_axis, normal])  # local -> (x,y,n) frame
    base_rot = _mat3_to_quat_wxyz(R_local_cols.T)  # (x,y,n) frame -> world (ex,ey,ez)

    # Spawn point: over the fingertip workspace (I24 lesson), at the hull
    # extent along `normal` beyond the tip centroid, plus the object half
    # size and 5 mm clearance.
    if tip_pts_mid:
        extents = [float(np.dot(p - tip_centroid, normal)) for p in tip_pts_mid]
        hull_extent = max(0.0, max(extents, default=0.0))
    else:
        hull_extent = 0.0
    spawn_offset = tip_centroid + normal * (hull_extent + object_half_size + 0.005)

    rng = np.random.default_rng(seed)
    reach_count = np.zeros(N_FINGERS, dtype=int)
    for _ in range(n_sweep):
        q = np.array(mid_q)
        for idx in valid_slots:
            lo, hi = design.slot_limits[idx]
            q[idx] = rng.uniform(lo, hi)
        T = authored_fk(design, q)
        for f in range(N_FINGERS):
            if not finger_valid[f]:
                continue
            last = finger_last_slot[f]
            tip = (T[last] @ np.array([0.0, 0.0, float(design.slot_length[last]), 1.0]))[:3]
            if np.linalg.norm(tip - spawn_offset) <= reach_tol_m:
                reach_count[f] += 1

    fingertip_offsets = np.zeros((N_FINGERS, 3))
    for f in range(N_FINGERS):
        if finger_valid[f]:
            last = finger_last_slot[f]
            fingertip_offsets[f] = (T_mid[last] @ np.array([0.0, 0.0, float(design.slot_length[last]), 1.0]))[:3]

    # Review item 1: mask the exposed `fingertip_valid` (the runtime
    # OBSERVATION flag -- see `obs_utils`) for a finger whose real chain
    # fills every one of its 6 envelope slots, since then there is no ghost
    # slot left for `canonicalize` to translate to the true tip (see its
    # `fingertip_marker_ok`) -- the envelope-fixed "fingertip body"
    # (`f{f}_link5`) would sit at that link's BASE, not its tip. `finger_valid`
    # itself (this finger EXISTS) still gates every geometry computation
    # above (mount/tip centroids, spawn point, reach sweep) -- unaffected,
    # since those are pure FK and never depend on the marker-body trick.
    fingertip_valid = finger_valid & design.fingertip_marker_ok

    return PalmUpResult(
        normal=normal, base_rot_wxyz=base_rot, default_q=mid_q, spawn_offset=spawn_offset,
        fingertip_valid=fingertip_valid, fingertip_offsets=fingertip_offsets,
        reachable_fingertips=int(np.count_nonzero(reach_count > 0)), n_swept=n_sweep,
    )


def spawn_height_above_palm_m(design: EnvelopeDesign, pu: Optional[PalmUpResult] = None) -> float:
    """The population spawn point's height above the palm along WORLD z,
    AFTER `base_rot` -- `quat_apply(base_rot, spawn_offset_local)[2]`, the
    ONE convention `palm_calibration.MIN_SPAWN_HEIGHT_ABOVE_PALM_M`'s
    docstring pins down and `drop_detection.object_below_palm` (world z)
    checks every reset. `HAND_BASE_POS_M`/`env_origins` (author_grammar.py)
    are the SAME constant additive offset for every design's palm origin and
    every env's spawn point respectively, so they cancel out of this
    difference -- computing it here, from `spawn_offset`/`base_rot` alone,
    with no scene/env in scope, is exact, not an approximation of the
    runtime quantity. `pu` lets a caller that already ran `palm_up` (e.g.
    `admit`) reuse it instead of paying for another (possibly swept) call."""
    pu = pu if pu is not None else palm_up(design, n_sweep=0)
    return float(_quat_apply_wxyz(pu.base_rot_wxyz, pu.spawn_offset)[2])


# --------------------------------------------------------------------------
# GrammarPopulation
# --------------------------------------------------------------------------


@dataclass
class GrammarPopulation:
    n_designs: int
    sources: Tuple[str, ...]
    joint_link_boxes: np.ndarray    # (n,32,4,3) float32
    joint_valid: np.ndarray         # (n,32) bool
    joint_limits: np.ndarray        # (n,32,2) float64
    default_joint_pos: np.ndarray   # (n,32) float64
    hand_scale: np.ndarray          # (n,) float64
    fingertip_valid: np.ndarray     # (n,5) bool
    fingertip_offsets: np.ndarray   # (n,5,3) float64
    palm_center: np.ndarray         # (n,3) float64
    palm_keypoints: np.ndarray      # (n,4,3) float64
    palm_frame: np.ndarray          # (n,7) float64 (xyz + wxyz quat)
    base_rot: np.ndarray            # (n,4) float64 (wxyz)
    spawn_offset: np.ndarray        # (n,3) float64
    designs: Tuple[EnvelopeDesign, ...]
    palm_up_results: Tuple[PalmUpResult, ...]
    # Review item 11 (risk): a `drive` table (stiffness/damping/max_force
    # placeholders, `(n,32,5)`) used to live here. Removed: it was never
    # read anywhere (the AUTHORED USD joint drive uses hardcoded
    # `robot_param_constants` values, and the RUNTIME actuator gains come
    # from `hand_only.DEFAULT_HAND_STIFFNESS`/`DEFAULT_HAND_DAMPING`, scene-
    # wide, plus this env's own per-env carrier-gain override
    # (`scene_utils._apply_per_env_carrier_gains`) -- none of it consulted
    # `population.drive`), and its own placeholder values (3.0/0.1/0.5)
    # actively disagreed with the values that DO govern behavior (SHARPA-
    # mean stiffness/damping, 1.0 Nm effort). Wiring it up as the real
    # source of truth would mean per-design (not just per-carrier) runtime
    # gain overrides across all 30 finger joints -- a larger, Kit-risky
    # change deferred past this pass; see the worker report.


def build_population(designs: Sequence[EnvelopeDesign], **palm_up_kwargs) -> GrammarPopulation:
    n = len(designs)
    joint_link_boxes = np.zeros((n, N_SLOTS, 4, 3), dtype=np.float32)
    joint_valid = np.zeros((n, N_SLOTS), dtype=bool)
    joint_limits = np.zeros((n, N_SLOTS, 2))
    default_joint_pos = np.zeros((n, N_SLOTS))
    hand_scale = np.zeros(n)
    fingertip_valid = np.zeros((n, N_FINGERS), dtype=bool)
    fingertip_offsets = np.zeros((n, N_FINGERS, 3))
    palm_center = np.zeros((n, 3))
    palm_keypoints = np.zeros((n, 4, 3))
    palm_frame = np.zeros((n, 7))
    base_rot = np.zeros((n, 4))
    spawn_offset = np.zeros((n, 3))
    palm_up_results: List[PalmUpResult] = []

    for i, design in enumerate(designs):
        joint_link_boxes[i] = token_boxes(design)
        joint_valid[i] = design.slot_valid
        joint_limits[i] = design.slot_limits
        lengths = design.slot_length[design.slot_valid]
        hand_scale[i] = float(lengths.max()) if lengths.size else GHOST_LENGTH_M

        pu = palm_up(design, **palm_up_kwargs)
        palm_up_results.append(pu)
        default_joint_pos[i] = pu.default_q
        fingertip_valid[i] = pu.fingertip_valid
        fingertip_offsets[i] = pu.fingertip_offsets
        base_rot[i] = pu.base_rot_wxyz
        spawn_offset[i] = pu.spawn_offset

        T0 = authored_fk(design, np.zeros(N_SLOTS))
        finger_mounts = [T0[f * N_JOINTS_PER_FINGER][:3, 3] for f in range(N_FINGERS) if design.slot_valid[f * N_JOINTS_PER_FINGER]]
        palm_center[i] = np.mean(np.stack(finger_mounts), axis=0) if finger_mounts else np.zeros(3)
        kp = [np.zeros(3), np.array([0.0, 0.0, design.root_length_m])]
        kp += finger_mounts[:2] if len(finger_mounts) >= 2 else [np.zeros(3)] * max(0, 2 - len(finger_mounts))
        palm_keypoints[i] = np.stack(kp[:4] + [np.zeros(3)] * max(0, 4 - len(kp)))
        palm_frame[i, :3] = 0.0
        palm_frame[i, 3:] = _mat3_to_quat_wxyz(np.eye(3))

    return GrammarPopulation(
        n_designs=n, sources=tuple(d.source for d in designs), joint_link_boxes=joint_link_boxes,
        joint_valid=joint_valid, joint_limits=joint_limits, default_joint_pos=default_joint_pos,
        hand_scale=hand_scale, fingertip_valid=fingertip_valid, fingertip_offsets=fingertip_offsets,
        palm_center=palm_center, palm_keypoints=palm_keypoints, palm_frame=palm_frame, base_rot=base_rot,
        spawn_offset=spawn_offset, designs=tuple(designs), palm_up_results=tuple(palm_up_results),
    )
