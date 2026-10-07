"""Padded fixed-topology envelope adapter: grammar `KinematicModel` ->
simulator-ready per-design tables.

See `project-notes/grammar/phase2-adapter-design.md` section 3-4 for the
original design. Numpy + `hand_sampler` only: no `isaaclab`/`pxr` imports, so
this module is importable and unit-testable without booting Kit
(`isaacsimenvs/inhand_reorient/scene/author_grammar.py` does the lazy pxr USD
authoring on top of it).

Envelope layout (fixed across every design so a whole population shares one
articulation topology):

    - 6 finger slots. Finger slot `f` is one palm joint (its CARRIER, slot
      `6 f`, joint `f{f}_cj`, body `f{f}_c`, a child of the root) followed by
      up to 5 finger joints (slots `6 f + 1 + d`, joints `f{f}_j{d}`, bodies
      `f{f}_link{d}`, a chain on the carrier) -- 36 joint slots. Each finger
      slot also ends in a fingertip body `f{f}_tip`, fixed to `f{f}_link4`
      at the real fingertip (`tip_offsets`).
    - Each carrier has one of three roles in a design (`carrier_roles`):
      LOCKED (a finger on the rigid palm, or an empty finger slot: ghost
      limits, no collider, the finger's mount transform folded into the
      finger's base origin), LEADER (the first finger slot of a jointed palm
      part: the carrier IS that palm joint and carries the palm part's
      collider; a jointed palm part without a finger is a leader with an
      empty finger chain) or FOLLOWER (another finger on the same palm part:
      same origin, axis and limits as its leader, tied to it so the part
      moves as one piece -- `slot_tie` -- and no collider).
    - Fingers are assigned to finger slots in a canonical order
      (`canonicalize`): the fingers without a jointed palm part first (by
      digit id), then each jointed palm part in palm-index order, its
      fingers by digit id with the leader first. Any split of up to 6
      fingers fits: 5 on a rigid palm, 3 + 2 + 1, 2 + 2, ...
    - `slot_valid` marks the joints the policy controls: real finger joints
      and leader carriers. Follower and locked carriers are not valid (their
      tokens are disabled and their actions ignored); ghost slots (unused
      finger joints) are authored with the old sampler's convention (tiny
      link, no collider, locked near zero).

Admission (`_admit_structural`) restricts a derived `KinematicModel` to the
designs this envelope represents losslessly: revolute only, no couplings, no
in-digit branching, at most 5 joints per finger, no jointed palm part below
another jointed palm part, and at most 6 finger slots in use (one per finger,
plus one per jointed palm part that carries no finger). There is no cap on the
fingers one palm joint carries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hand_sampler import robot_param_constants as rpc
from hand_sampler.design_space import _ordered_box
from hand_sampler.grammar import geometry as grammar_geometry
from hand_sampler.grammar.envelope import fits_envelope
from hand_sampler.grammar.fk import forward_kinematics, pose_to_matrix, rodrigues
from hand_sampler.grammar.kinematics import Joint, KinematicModel

from ..palm_calibration import MIN_SPAWN_HEIGHT_ABOVE_PALM_M

# --------------------------------------------------------------------------
# Envelope layout constants
# --------------------------------------------------------------------------

N_FINGERS = 6                                  # finger slots
N_JOINTS_PER_FINGER = 5                        # finger joints per finger slot
SLOTS_PER_FINGER = 1 + N_JOINTS_PER_FINGER     # its carrier + its finger joints
N_SLOTS = N_FINGERS * SLOTS_PER_FINGER         # 36
ROOT_SENTINEL = -1

MAX_DIGITS = N_FINGERS
MAX_JOINTS_PER_DIGIT = N_JOINTS_PER_FINGER

LOCKED, LEADER, FOLLOWER = "locked", "leader", "follower"

# Old-sampler ghost-slot convention (see hand_sampler/build.py): a tiny,
# collider-less link that is locked near zero so it never moves and never
# collides, used to pad every design to the fixed 36-slot topology.
GHOST_LENGTH_M = 1e-4
GHOST_LIMITS = (0.0, 1e-8)
GHOST_AXIS = (0.0, 0.0, 1.0)

MAX_REST_PENETRATION_M = 0.003
"""Default `admit(check_overlap=True)` rejection threshold (deeper starting
overlaps made PhysX push links apart at over 100 rad in one or two steps)."""


def carrier_slot(f: int) -> int:
    """Slot index of finger slot `f`'s carrier (palm) joint."""
    return f * SLOTS_PER_FINGER


def finger_slot(f: int, d: int) -> int:
    """Slot index of finger joint `d` (0 = at the palm) of finger slot `f`."""
    return f * SLOTS_PER_FINGER + 1 + d


def slot_finger(idx: int) -> int:
    """Finger slot of slot `idx`."""
    return idx // SLOTS_PER_FINGER


def slot_joint_index(idx: int) -> int:
    """`-1` for a carrier slot, else the finger joint index `d` (0..4)."""
    return idx % SLOTS_PER_FINGER - 1


CARRIER_SLOTS: Tuple[int, ...] = tuple(carrier_slot(f) for f in range(N_FINGERS))
FINGER_BASE_SLOTS: Tuple[int, ...] = tuple(finger_slot(f, 0) for f in range(N_FINGERS))
LAST_FINGER_SLOTS: Tuple[int, ...] = tuple(finger_slot(f, N_JOINTS_PER_FINGER - 1) for f in range(N_FINGERS))


def slot_name(idx: int) -> str:
    """Joint name of slot `idx` (`author_grammar` authors it under this name)."""
    f, d = slot_finger(idx), slot_joint_index(idx)
    return f"f{f}_cj" if d < 0 else f"f{f}_j{d}"


def slot_body(idx: int) -> str:
    """Child body name of slot `idx`."""
    f, d = slot_finger(idx), slot_joint_index(idx)
    return f"f{f}_c" if d < 0 else f"f{f}_link{d}"


def tip_body(f: int) -> str:
    """Fingertip body of finger slot `f` (fixed to its last link)."""
    return f"f{f}_tip"


SLOT_NAMES: Tuple[str, ...] = tuple(slot_name(i) for i in range(N_SLOTS))
SLOT_BODY_NAMES: Tuple[str, ...] = tuple(slot_body(i) for i in range(N_SLOTS))
TIP_BODY_NAMES: Tuple[str, ...] = tuple(tip_body(f) for f in range(N_FINGERS))


def _slot_parent(idx: int) -> int:
    """Envelope-fixed parent slot index, or `ROOT_SENTINEL` for root."""
    d = slot_joint_index(idx)
    return ROOT_SENTINEL if d < 0 else idx - 1


SLOT_PARENT: Tuple[int, ...] = tuple(_slot_parent(i) for i in range(N_SLOTS))
# Every slot's parent is the root or an earlier slot of the same finger slot.
TOPOLOGICAL_ORDER: Tuple[int, ...] = tuple(range(N_SLOTS))


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
    """The nearest jointed palm body on the path from palm body `mount` to
    the root (`mount` itself included), or None."""
    cur = mount
    seen = set()
    while cur not in seen:
        seen.add(cur)
        if cur in jointed_children:
            return cur
        if cur == root:
            return None
        j = palm_joint_by_child.get(cur)
        if j is None:
            return None
        cur = j.parent
    return None


def _digit_id_of_mount_body(body: str) -> Optional[str]:
    """`"d{digit_id}p1"` -> `digit_id`, for a top-level digit's root body."""
    if not body.startswith("d"):
        return None
    rest = body[1:]
    if "p1" not in rest or not rest.endswith("p1"):
        return None
    digit_id = rest[: -len("p1")]
    return digit_id if digit_id.isdigit() else None


def _digit_key(j: Joint) -> Tuple[int, str]:
    did = _digit_id_of_mount_body(j.child)
    return (int(did), j.child) if did is not None else (1 << 30, j.child)


def _palm_key(body: str) -> Tuple[int, str]:
    """`"palm{i}"` -> `(i, name)`, used only to order palm parts canonically."""
    rest = body[len("palm"):] if body.startswith("palm") else ""
    return (int(rest), body) if rest.isdigit() else (1 << 30, body)


@dataclass(frozen=True)
class _Groups:
    """The finger-slot grouping of a model: fingers without a jointed palm
    part, and each jointed palm part with the fingers it carries."""

    root_digits: Tuple[Joint, ...]
    jointed: Tuple[Tuple[Joint, Tuple[Joint, ...]], ...]   # (palm joint, its digits' root joints)
    palm_joint_by_child: Dict[str, Joint]

    @property
    def n_slots(self) -> int:
        return len(self.root_digits) + sum(max(1, len(ds)) for _, ds in self.jointed)


def _groups(model: KinematicModel) -> _Groups:
    palm = _palm_body_names(model)
    jointed = sorted(_jointed_palm_joints(model), key=lambda j: _palm_key(j.child))
    palm_joint_by_child = _palm_joint_by_child(model)
    jointed_children = {j.child for j in jointed}
    digit_root_joints = [j for j in model.joints if j.parent in palm and j.child not in palm]
    carrier_of = {j.name: _carrier_of_mount(j.parent, jointed_children, palm_joint_by_child, model.root)
                  for j in digit_root_joints}
    root_digits = tuple(sorted((j for j in digit_root_joints if carrier_of[j.name] is None), key=_digit_key))
    groups = tuple(
        (pj, tuple(sorted((j for j in digit_root_joints if carrier_of[j.name] == pj.child), key=_digit_key)))
        for pj in jointed
    )
    return _Groups(root_digits=root_digits, jointed=groups, palm_joint_by_child=palm_joint_by_child)


def _admit_structural(model: KinematicModel) -> AdmissionResult:
    """`(ok, reasons)` -- whether `model` fits the envelope's SHAPE
    losslessly (see the module docstring). Independent of rest-pose geometry
    (no rest-overlap or spawn-height check -- the public `admit` below wraps
    this with those). `canonicalize` calls THIS, not `admit`, so
    canonicalizing a structurally fine but overlapping design never recurses
    through the public gate."""
    reasons: List[str] = []

    _env_ok, env_reasons = fits_envelope(
        model, MAX_DIGITS, MAX_JOINTS_PER_DIGIT, allow_palm_joints=True, allow_branches=False
    )
    reasons.extend(env_reasons)

    non_fixed_types = {j.type for j in model.joints if j.type != "fixed"}
    bad_types = sorted(non_fixed_types - {"revolute"})
    if bad_types:
        reasons.append(f"non-revolute movable joint type(s) present: {bad_types}")

    if model.couplings:
        reasons.append(f"{len(model.couplings)} coupling(s) present; envelope requires revolute-only, no couplings")

    jointed = _jointed_palm_joints(model)
    palm_joint_by_child = _palm_joint_by_child(model)
    jointed_children = {j.child for j in jointed}
    for j in jointed:
        cur = j.parent
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            if cur in jointed_children:
                reasons.append(f"jointed palm body {j.child!r} has jointed palm ancestor {cur!r}")
                break
            pj = palm_joint_by_child.get(cur)
            cur = pj.parent if pj is not None else None

    g = _groups(model)
    n_digits = len(g.root_digits) + sum(len(ds) for _, ds in g.jointed)
    n_empty = sum(1 for _, ds in g.jointed if not ds)
    if n_empty and g.n_slots > N_FINGERS:
        reasons.append(
            f"{n_digits} finger(s) and {n_empty} palm joint(s) without a finger need {g.n_slots} finger slots, "
            f"exceeding the {N_FINGERS}-slot envelope"
        )

    return AdmissionResult(ok=len(reasons) == 0, reasons=tuple(reasons))


# --------------------------------------------------------------------------
# canonicalize: EnvelopeDesign
# --------------------------------------------------------------------------


@dataclass
class EnvelopeDesign:
    source: str
    model: KinematicModel
    slot_valid: np.ndarray           # (36,) bool -- a joint the policy controls (finger joint or leader carrier)
    slot_origin: np.ndarray          # (36,4,4) float64 -- joint origin, relative to the slot's ENVELOPE parent
    slot_axis: np.ndarray            # (36,3) float64 -- in the slot's own (child) local frame
    slot_limits: np.ndarray          # (36,2) float64
    slot_length: np.ndarray          # (36,) float64 -- segment length (tip frame z-offset)
    slot_joint_name: Tuple[Optional[str], ...]
    slot_body_name: Tuple[Optional[str], ...]
    capsule_radius_m: float
    root_length_m: float
    finger_digit_id: Tuple[Optional[str], ...]  # length 6
    grammar_version: str = ""
    reasons: Tuple[str, ...] = field(default_factory=tuple)  # empty iff admitted
    slot_tie: np.ndarray = field(default_factory=lambda: np.full(N_SLOTS, -1, dtype=int))
    """(36,) int -- for a FOLLOWER carrier, the slot of the leader carrier it
    is tied to (same origin, axis and limits; authored as a PhysX mimic
    joint); -1 for every other slot."""
    filtered_pairs: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    """Node pairs (slot indices, the root capsule as `ROOT_NODE`) that
    authoring must collision-filter: (1) bodies joined through a chain of
    short bones (`short_bone_pairs`: a bone shorter than one capsule radius,
    e.g. a 0 mm bone between two joints at one point, so the bodies on either
    side of it meet at the joint like a parent and its child); (2) a finger's
    first link and the body it really sits on, when its PhysX joint parent
    (its own carrier) is a different body (`mount_pairs`: the palm for a
    finger on a locked carrier, the leader's palm part for a follower's
    finger -- PhysX only excludes a joint's own two bodies); both set by
    `canonicalize` for every design; (3) for EXEMPTED designs only (projected
    commercial hands, `admit(check_overlap=False)`), the pairs that overlap
    at rest (`mark_filtered_pairs`). Sampled designs are REJECTED on overlap
    instead, never exempted."""
    sha256: str = ""
    """This design's entry-level sha256 (over its raw derivation dict,
    `population_file._entry_sha256`) -- set by `population_file.
    load_population`, empty for a design built directly by `canonicalize`."""


def carrier_roles(design: EnvelopeDesign) -> Tuple[str, ...]:
    """Per finger slot, its carrier's role: `LEADER`, `FOLLOWER` or `LOCKED`."""
    out = []
    for c in CARRIER_SLOTS:
        if design.slot_valid[c]:
            out.append(LEADER)
        elif design.slot_tie[c] >= 0:
            out.append(FOLLOWER)
        else:
            out.append(LOCKED)
    return tuple(out)


def tied_q(design: EnvelopeDesign, q: np.ndarray) -> np.ndarray:
    """`q` (`(36,)` or `(N,36)`) with every follower carrier's value replaced
    by its leader's: the joint vector the tied articulation actually has."""
    q = np.array(q, dtype=float, copy=True)
    for s in np.nonzero(design.slot_tie >= 0)[0]:
        q[..., s] = q[..., int(design.slot_tie[s])]
    return q


def tie_columns(joint_tie: np.ndarray, slot_of_col: Sequence[int]) -> np.ndarray:
    """`(n, J)` int: for per-env tie rows in slot order (`(n, 36)`,
    `GrammarPopulation.joint_tie[design_idx]`) and an articulation whose
    column `c` holds slot `slot_of_col[c]`, the column whose value column `c`
    takes when joint positions are written: its own, or its leader's for a
    follower carrier (`obs_utils.tie_joints` gathers with it)."""
    slot_of_col = np.asarray(slot_of_col, dtype=int)
    col_of_slot = np.empty_like(slot_of_col)
    col_of_slot[slot_of_col] = np.arange(len(slot_of_col))
    tie = np.asarray(joint_tie, dtype=int)[:, slot_of_col]          # (n, J), in column order
    own = np.broadcast_to(np.arange(len(slot_of_col)), tie.shape)
    return np.where(tie >= 0, col_of_slot[np.clip(tie, 0, None)], own)


def tip_offsets(design: EnvelopeDesign) -> np.ndarray:
    """`(6,)`: per finger slot, the z offset of its fingertip body `f{f}_tip`
    in the frame of its last link `f{f}_link4`: that link's length when it is
    a real joint (a 5-joint finger), else 0 (`canonicalize` puts a shorter
    finger's ghost tail at the real tip)."""
    return np.array([float(design.slot_length[s]) if design.slot_valid[s] else 0.0 for s in LAST_FINGER_SLOTS])


def _compose_palm_transform(mount_body: str, stop_body: str, palm_joint_by_child: Dict[str, Joint]) -> np.ndarray:
    """`stop_body`-to-`mount_body` transform, composed from a chain of
    FIXED palm joints only (by construction the caller never asks this to
    cross a jointed palm joint)."""
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
    for _parent_b, child_b in zip(chain[:-1], chain[1:]):
        T = T @ pose_to_matrix(palm_joint_by_child[child_b].origin)
    return T


def canonicalize(model: KinematicModel, source: str = "") -> EnvelopeDesign:
    result = _admit_structural(model)
    if not result.ok:
        raise AdmissionError(result.reasons)

    g = _groups(model)
    pjbc = g.palm_joint_by_child

    # Canonical finger-slot assignment (see the module docstring).
    finger_root_joint: List[Optional[Joint]] = [None] * N_FINGERS
    finger_mount: List[np.ndarray] = [np.eye(4)] * N_FINGERS   # carrier frame -> the finger's mount body
    palm_joint_of: List[Optional[Joint]] = [None] * N_FINGERS
    leader_of: List[int] = [-1] * N_FINGERS
    f = 0
    for j in g.root_digits:
        finger_root_joint[f] = j
        finger_mount[f] = _compose_palm_transform(j.parent, model.root, pjbc)
        f += 1
    for pj, digits in g.jointed:
        lead = f
        for j in digits or (None,):
            palm_joint_of[f] = pj
            leader_of[f] = lead
            if j is not None:
                finger_root_joint[f] = j
                finger_mount[f] = _compose_palm_transform(j.parent, pj.child, pjbc)
            f += 1
    assert f <= N_FINGERS, "_admit_structural should have rejected this"

    slot_valid = np.zeros(N_SLOTS, dtype=bool)
    slot_origin = np.tile(np.eye(4), (N_SLOTS, 1, 1))
    slot_axis = np.tile(np.array(GHOST_AXIS), (N_SLOTS, 1))
    slot_limits = np.tile(np.array(GHOST_LIMITS), (N_SLOTS, 1))
    slot_length = np.full(N_SLOTS, GHOST_LENGTH_M)
    slot_tie = np.full(N_SLOTS, -1, dtype=int)
    slot_joint_name: List[Optional[str]] = [None] * N_SLOTS
    slot_body_name: List[Optional[str]] = [None] * N_SLOTS
    finger_digit_id: List[Optional[str]] = [None] * N_FINGERS

    children_by_parent: Dict[str, List[Joint]] = {}
    for j in model.joints:
        children_by_parent.setdefault(j.parent, []).append(j)
    frames_by_name = {fr.name: fr for fr in model.frames}

    def _length(body: str) -> float:
        frame = frames_by_name.get(f"{body}_tip")
        return float(frame.pose.xyz[2]) if frame is not None else GHOST_LENGTH_M

    for f in range(N_FINGERS):
        pj = palm_joint_of[f]
        c = carrier_slot(f)
        if pj is not None:
            pre = _compose_palm_transform(pj.parent, model.root, pjbc)
            slot_origin[c] = pre @ pose_to_matrix(pj.origin)
            slot_axis[c] = np.asarray(pj.axis, dtype=float)
            slot_limits[c] = np.asarray(pj.limits, dtype=float)
            slot_body_name[c] = pj.child
            if leader_of[f] == f:
                slot_valid[c] = True
                slot_length[c] = _length(pj.child)
                slot_joint_name[c] = pj.name
            else:
                slot_tie[c] = carrier_slot(leader_of[f])

        start = finger_root_joint[f]
        if start is None:
            continue
        finger_digit_id[f] = _digit_id_of_mount_body(start.child)
        chain: List[Joint] = []
        cur: Optional[Joint] = start
        while cur is not None and len(chain) < N_JOINTS_PER_FINGER:
            chain.append(cur)
            nxt = children_by_parent.get(cur.child, [])
            cur = nxt[0] if len(nxt) == 1 else None
        for d, j in enumerate(chain):
            idx = finger_slot(f, d)
            slot_valid[idx] = True
            slot_origin[idx] = finger_mount[f] @ pose_to_matrix(j.origin) if d == 0 else pose_to_matrix(j.origin)
            slot_axis[idx] = np.asarray(j.axis, dtype=float)
            slot_limits[idx] = np.asarray(j.limits, dtype=float)
            slot_length[idx] = _length(j.child)
            slot_joint_name[idx] = j.name
            slot_body_name[idx] = j.child
        # The first ghost slot after a finger's real chain is translated by
        # the last real link's length, so it and the identity-chained ghost
        # tail after it (up to `f{f}_link4`, which carries the fingertip
        # body) sit exactly at the real tip: a ghost's own length is
        # negligible and its axis is z, so its local transform at q = 0 is
        # this pure translation.
        if len(chain) < N_JOINTS_PER_FINGER:
            last_idx = finger_slot(f, len(chain) - 1)
            tip_translation = np.eye(4)
            tip_translation[2, 3] = float(slot_length[last_idx])
            slot_origin[finger_slot(f, len(chain))] = tip_translation

    capsule_radius = next((b.radius for b in model.bodies if b.radius is not None), 0.01)
    root_frame = frames_by_name.get("root_tip")
    root_length = float(root_frame.pose.xyz[2]) if root_frame is not None else 0.0

    design = EnvelopeDesign(
        source=source, model=model, slot_valid=slot_valid, slot_origin=slot_origin, slot_axis=slot_axis,
        slot_limits=slot_limits, slot_length=slot_length, slot_joint_name=tuple(slot_joint_name),
        slot_body_name=tuple(slot_body_name), capsule_radius_m=float(capsule_radius), root_length_m=root_length,
        finger_digit_id=tuple(finger_digit_id), grammar_version=getattr(model, "grammar_version", ""),
        reasons=(), slot_tie=slot_tie, filtered_pairs=(),
    )
    design.filtered_pairs = tuple(sorted(set(short_bone_pairs(design)) | set(mount_pairs(design))))
    return design


def _physical_reasons(
    design: EnvelopeDesign,
    *,
    check_overlap: bool,
    check_spawn_height: bool,
    max_rest_penetration_m: float,
    min_spawn_height_m: float,
    pu: PalmUpResult,
) -> Tuple[List[str], List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """Shared by `admit` and `viability_report`: `(reasons, pairs_q0,
    pairs_default_q)` for an already-`canonicalize`-d `design`, given a
    `palm_up` result `pu` the caller already computed (so this never pays
    for a second `palm_up` call). Opus review G0 item 3: overlap is checked
    at BOTH `q=0` (the design's authored rest pose) and `pu.default_q` (the
    reset pose the env actually uses every episode) -- a design must clear
    both; the worse of the two drives the reason string. Item 4's spawn-
    height check is unchanged, just reusing `pu` instead of recomputing it."""
    reasons: List[str] = []
    pairs_q0: List[Tuple[int, int, float]] = []
    pairs_default_q: List[Tuple[int, int, float]] = []
    if check_overlap:
        pairs_q0 = rest_overlap_pairs(design)
        pairs_default_q = rest_overlap_pairs(design, q=pu.default_q)
        bad_q0 = [(i, j, pen) for i, j, pen in pairs_q0 if pen > max_rest_penetration_m]
        bad_default_q = [(i, j, pen) for i, j, pen in pairs_default_q if pen > max_rest_penetration_m]
        if bad_q0 or bad_default_q:
            worst = max([pen for _, _, pen in bad_q0] + [pen for _, _, pen in bad_default_q])
            reasons.append(
                f"{len(bad_q0)} rest-overlap pair(s) at q=0 and {len(bad_default_q)} at default_q "
                f"exceed {max_rest_penetration_m * 1000.0:.1f} mm (worst {worst * 1000.0:.2f} mm)"
            )
    if check_spawn_height:
        height = spawn_height_above_palm_m(design, pu)
        if height < min_spawn_height_m:
            reasons.append(
                f"spawn height {height * 1000.0:.1f} mm above the palm along world z after base_rot "
                f"< required {min_spawn_height_m * 1000.0:.1f} mm"
            )
    return reasons, pairs_q0, pairs_default_q


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
    note's risk-4 rest-overlap filter (checked at BOTH `q=0` and the
    design's own `default_q` -- opus review G0 item 3, a design must clear
    both) AND the spawn-height requirement (review item 2 for populations --
    `drop_detection` already uses world z for the RUNTIME check; this is the
    admission-time equivalent). Both physical checks canonicalize `model`
    once (structural admission has already passed, so this cannot raise)
    and are independent -- a design can fail either, both, or neither, and
    every violated reason is reported, not just the first.

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

    design = canonicalize(model)
    # A single cheap (`n_sweep=0`, no reach sweep) `palm_up` call: both
    # physical checks only need `default_q`/`spawn_offset`/`base_rot_wxyz`,
    # never the reach sweep itself.
    pu = palm_up(design, n_sweep=0)
    reasons, _pairs_q0, _pairs_default_q = _physical_reasons(
        design, check_overlap=check_overlap, check_spawn_height=check_spawn_height,
        max_rest_penetration_m=max_rest_penetration_m, min_spawn_height_m=min_spawn_height_m, pu=pu,
    )
    return AdmissionResult(ok=len(reasons) == 0, reasons=tuple(reasons))


def viability_report(model: KinematicModel) -> dict:
    """Numpy + `hand_sampler` only (no isaaclab), for the CPU grammar screen
    (plan revision, step 2) -- everything a screen worker needs about ONE
    derived model, without touching Kit:

        {"admitted": bool, "reasons": [str, ...], "max_rest_overlap_mm": float,
         "fingertips_reachable": int, "spawn_height_mm": float,
         "digit_count": int, "joint_count": int}

    `reasons` mirrors `admit(model)`'s own (structural + overlap + spawn-
    height, in that order -- structural failure short-circuits the rest,
    matching `admit`); both share the `_physical_reasons` helper on the SAME
    canonicalized `design` and dense `palm_up` call, rather than this
    function calling `admit(model)` separately and re-doing that work (the
    per-design cost this function's own screening-throughput requirement,
    opus review G0 item 4, cares about). `max_rest_overlap_mm` is the worse
    of the q=0 and default_q poses (item 3). The geometry fields
    (`max_rest_overlap_mm`, `spawn_height_mm`, `digit_count`, `joint_count`)
    are still reported even when `admitted` is `False` and even when the
    failure is a physical (not structural) one -- the whole point of a
    screen is comparing near-miss designs, not only admitted ones -- but are
    `None` when the model fails the STRUCTURAL check (no envelope slots to
    measure at all)."""
    structural = _admit_structural(model)
    if not structural.ok:
        return {
            "admitted": False, "reasons": list(structural.reasons), "max_rest_overlap_mm": None,
            "fingertips_reachable": None, "spawn_height_mm": None, "digit_count": None, "joint_count": None,
        }

    design = canonicalize(model)
    pu = palm_up(design)  # dense reach sweep -- also supplies default_q for the overlap check below
    reasons, pairs_q0, pairs_default_q = _physical_reasons(
        design, check_overlap=True, check_spawn_height=True,
        max_rest_penetration_m=MAX_REST_PENETRATION_M, min_spawn_height_m=MIN_SPAWN_HEIGHT_ABOVE_PALM_M, pu=pu,
    )
    max_overlap_mm = max(
        [pen for _, _, pen in pairs_q0] + [pen for _, _, pen in pairs_default_q], default=0.0
    ) * 1000.0
    spawn_height_mm = spawn_height_above_palm_m(design, pu) * 1000.0
    digit_count = int(sum(1 for d in design.finger_digit_id if d is not None))
    joint_count = int(design.slot_valid.sum())

    return {
        "admitted": len(reasons) == 0, "reasons": reasons, "max_rest_overlap_mm": float(max_overlap_mm),
        "fingertips_reachable": int(pu.reachable_fingertips), "spawn_height_mm": float(spawn_height_mm),
        "digit_count": digit_count, "joint_count": joint_count,
    }


# --------------------------------------------------------------------------
# joint_local_frames / authored_fk
# --------------------------------------------------------------------------


def joint_local_frames(design: EnvelopeDesign) -> np.ndarray:
    """`(36, 2, 4, 4)`: per slot, `(frame0, frame1)` such that the slot's
    child-body world transform is `parent_world @ frame0 @ Rz(q) @
    inv(frame1)` for any `q` -- i.e. a dual-frame authoring convention (as
    PhysX/USD physics joints use) that always rotates about the LOCAL Z axis,
    with the real per-joint axis baked into `frame0`/`frame1` via a shared
    alignment rotation `R` (`R @ ez == axis`):

        frame0 = slot_origin @ [R, 0; 0, 1]
        frame1 = [R, 0; 0, 1]

    because `Rot(axis, q) == R @ Rz(q) @ inv(R)` for ANY rotation `R` with
    `R @ ez == axis`. A follower carrier has its leader's origin and axis, so
    its frames equal its leader's and the two bodies coincide at equal `q`."""
    out = np.zeros((N_SLOTS, 2, 4, 4))
    for idx in range(N_SLOTS):
        R = _shortest_rotation((0.0, 0.0, 1.0), design.slot_axis[idx])
        R4 = np.eye(4)
        R4[:3, :3] = R
        out[idx, 0] = design.slot_origin[idx] @ R4
        out[idx, 1] = R4
    return out


def authored_fk(design: EnvelopeDesign, q: np.ndarray) -> np.ndarray:
    """`(36,4,4)` world (root-frame) transforms of every slot's child body,
    at joint vector `q` (length 36, envelope slot order), with every follower
    carrier tied to its leader (`tied_q`). Exact (not approximate)
    reconstruction of `hand_sampler.grammar.fk.forward_kinematics`
    restricted to this design's real joints -- see this module's tests."""
    frames = joint_local_frames(design)
    q = tied_q(design, np.asarray(q, dtype=float).reshape(N_SLOTS))
    T = [np.eye(4) for _ in range(N_SLOTS)]
    for idx in TOPOLOGICAL_ORDER:
        parent = SLOT_PARENT[idx]
        parent_T = np.eye(4) if parent == ROOT_SENTINEL else T[parent]
        frame0, frame1 = frames[idx]
        local = frame0 @ _rotz(float(q[idx])) @ np.linalg.inv(frame1)
        T[idx] = parent_T @ local
    return np.stack(T)


def _rotz_batch(theta: np.ndarray) -> np.ndarray:
    """`(N,4,4)` batched version of `_rotz`."""
    theta = np.asarray(theta, dtype=float)
    n = theta.shape[0]
    c, s = np.cos(theta), np.sin(theta)
    T = np.zeros((n, 4, 4))
    T[:, 0, 0], T[:, 0, 1] = c, -s
    T[:, 1, 0], T[:, 1, 1] = s, c
    T[:, 2, 2] = 1.0
    T[:, 3, 3] = 1.0
    return T


def authored_fk_batch(design: EnvelopeDesign, q: np.ndarray) -> np.ndarray:
    """`(N,36,4,4)`: the SAME quantity as `authored_fk`, for `N` joint-vector
    samples at once (`q` shape `(N,36)`), via numpy-broadcast batched matmul
    (the reach sweep's >= 4000 samples per design)."""
    frames = joint_local_frames(design)  # (36,2,4,4), independent of q
    q = tied_q(design, np.asarray(q, dtype=float))
    n = q.shape[0]
    T: List[Optional[np.ndarray]] = [None] * N_SLOTS
    root_T = np.broadcast_to(np.eye(4), (n, 4, 4))
    for idx in TOPOLOGICAL_ORDER:
        parent = SLOT_PARENT[idx]
        parent_T = root_T if parent == ROOT_SENTINEL else T[parent]
        frame0, frame1 = frames[idx]
        Rz = _rotz_batch(q[:, idx])
        local = frame0[None, :, :] @ Rz @ np.linalg.inv(frame1)[None, :, :]
        T[idx] = parent_T @ local
    return np.stack(T, axis=1)  # (n, 36, 4, 4)


def tip_fk(design: EnvelopeDesign, T: np.ndarray) -> np.ndarray:
    """`(..., 6, 3)` fingertip-body positions (root frame) from slot
    transforms `T` (`(..., 36, 4, 4)`, `authored_fk`/`authored_fk_batch`)."""
    last = T[..., list(LAST_FINGER_SLOTS), :, :]
    off = tip_offsets(design)
    return last[..., :3, 3] + last[..., :3, 2] * off[:, None]


def grammar_fk_reference(design: EnvelopeDesign, q: np.ndarray) -> np.ndarray:
    """`(36,4,4)`: the SAME quantity as `authored_fk`, computed instead by
    calling `hand_sampler.grammar.fk.forward_kinematics` directly on
    `design.model` (ghost and locked slots get identity; a follower carrier
    gets its palm part's pose). Used by tests as the independent oracle
    `authored_fk` must match to 1e-9."""
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
    "segments along +z" convention), cross-section `radius`. Grammar hands
    have no mesh, so this box IS the collision geometry."""
    zs = (0.0, max(length, 1e-9))
    xs = (-radius, radius)
    ys = (-radius, radius)
    return np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=np.float64)


def token_boxes(design: EnvelopeDesign) -> np.ndarray:
    """`(36, 4, 3)` float32 ordered boxes, via `design_space._ordered_box`
    on each slot's own capsule bounding box (slots that are not valid use the
    ghost link's tiny dimensions)."""
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
(`author_grammar.author_design`'s `root_body_path`) but is not one of the 36
envelope joint SLOTS, so it has no slot index of its own."""


def _effective_parent(design: EnvelopeDesign, idx: int) -> int:
    """The node (a valid slot or `ROOT_NODE`) VALID slot `idx` really sits
    on, for adjacency: what it is EXPECTED to touch and is excluded from the
    overlap check against. Usually `SLOT_PARENT[idx]`; for a finger's first
    joint whose carrier is not valid it is the body the carrier stands in
    for: the leader's palm part for a follower, the palm for a locked
    carrier (`canonicalize` folds the finger's mount into its base origin).
    A finger joint's previous joint in the same finger is always valid when
    it is (contiguous chains), so only base slots ever take these branches."""
    parent = SLOT_PARENT[idx]
    if parent == ROOT_SENTINEL:
        return ROOT_NODE
    if design.slot_valid[parent]:
        return parent
    if design.slot_tie[parent] >= 0:
        return int(design.slot_tie[parent])
    return ROOT_NODE


SHORT_BONE_RADII = 1.0
"""A bone shorter than this many capsule radii joins its two neighbours
(`_short`). One radius: a 0 mm bone (two joints at one point, e.g. a
knuckle's abduction and flexion joints) and the few-millimetre bones of the
commercial hands make the capsules on either side meet at the joint, so they
overlap at any real bend; a longer bone keeps them apart at the rest and
reset poses, and when it does not (a finger folded back onto itself) that is
a real collision the overlap check must still see. No sampled bone is this
short (the grammar draws 15-80 mm with radii of at most 12 mm), so sampled
designs keep exactly their parent/child adjacency."""


def _short(design: EnvelopeDesign, node: int) -> bool:
    """A valid slot whose bone is shorter than `SHORT_BONE_RADII` capsule
    radii: the bodies on either side of it meet at its joint."""
    return node != ROOT_NODE and float(design.slot_length[node]) < SHORT_BONE_RADII * design.capsule_radius_m


def _chain_ancestors(design: EnvelopeDesign, idx: int) -> List[int]:
    """The nodes VALID slot `idx` is expected to touch towards the root: its
    effective parent, and beyond it every ancestor reached through a chain of
    short bones (`_short`), e.g. both neighbours of a 0 mm bone."""
    out = [_effective_parent(design, idx)]
    while _short(design, out[-1]):
        out.append(_effective_parent(design, out[-1]))
    return out


def adjacent_pairs(design: EnvelopeDesign) -> set:
    """Ordered node pairs (both orders) that `rest_overlap_pairs` excludes:
    every valid slot with its effective parent, and with every ancestor it
    meets through a chain of short bones (`_chain_ancestors`)."""
    out = set()
    for idx in range(N_SLOTS):
        if design.slot_valid[idx]:
            for a in _chain_ancestors(design, idx):
                out.add((idx, a))
                out.add((a, idx))
    return out


def short_bone_pairs(design: EnvelopeDesign) -> Tuple[Tuple[int, int], ...]:
    """Node pairs (sorted, root `ROOT_NODE`) that meet through a chain of
    short bones but are not parent and child: PhysX does not collide a
    joint's two bodies, but it would collide these, so authoring must
    collision-filter them (`canonicalize` puts them in `filtered_pairs`).
    Empty for a design without bones shorter than one capsule radius."""
    pairs = set()
    for idx in range(N_SLOTS):
        if design.slot_valid[idx]:
            for a in _chain_ancestors(design, idx)[1:]:
                pairs.add((min(idx, a), max(idx, a)))
    return tuple(sorted(pairs))


def mount_pairs(design: EnvelopeDesign) -> Tuple[Tuple[int, int], ...]:
    """`(node, base_slot)` for every finger whose PhysX joint parent (its own
    carrier) is not the body it really sits on: the palm (`ROOT_NODE`) for a
    finger on a locked carrier, the leader's carrier (the palm part) for a
    follower's finger. `rest_overlap_pairs` treats the two as adjacent
    (`_effective_parent`), but PhysX only excludes a joint's own two bodies,
    and the carrier between them has no collider, so without a filter they
    collide from step 0 wherever they overlap at rest. `canonicalize` puts
    these in `filtered_pairs`, which `author_grammar` collision-filters."""
    pairs = []
    for b in FINGER_BASE_SLOTS:
        c = SLOT_PARENT[b]
        if design.slot_valid[b] and not design.slot_valid[c]:
            node = int(design.slot_tie[c]) if design.slot_tie[c] >= 0 else ROOT_NODE
            pairs.append((min(node, b), max(node, b)))
    return tuple(sorted(pairs))


def capsule_core_endpoints_local(length: float, radius: float) -> Tuple[float, float]:
    """`(z0, z1)`: local-z interval of a capsule's CORE segment -- the
    segment between the two hemisphere-cap CENTERS, which is what a
    capsule-capsule distance query (PhysX's, and `segment_distance` below)
    actually operates on -- for a body whose nominal segment runs local
    `(0,0,0) -> (0,0,length)` with cross-section `radius`. Matches
    `author_grammar._author_body_and_collider` exactly: the collider mesh is
    centered at local z=`length/2`, with a cylindrical part of length
    `rpc.cylinder_part(length, radius) == max(length - 2r, 0)` and a
    hemispherical cap of `radius` on each end. When `length < 2r` both
    endpoints coincide at `length/2` (a sphere of radius `r`)."""
    half_cyl = rpc.cylinder_part(max(float(length), 1e-6), float(radius)) / 2.0
    center = float(length) / 2.0
    return center - half_cyl, center + half_cyl


def rest_overlap_pairs(design: EnvelopeDesign, q: Optional[np.ndarray] = None) -> List[Tuple[int, int, float]]:
    """Capsule-capsule self-penetration filter over the AUTHORED geometry,
    at joint vector `q` (default `None` -> `q=0`; pass `palm_up(design,
    n_sweep=0).default_q` for the env's per-episode reset pose --
    `admit`/`viability_report` check both): the root/palm capsule
    (`ROOT_NODE`) plus every VALID joint slot (followers' and locked
    carriers' bodies have no collider), every pair whose capsule CORE
    segments (`capsule_core_endpoints_local`) overlap by more than their
    radii allow, EXCLUDING pairs expected to touch (`adjacent_pairs`: parent
    and child, a finger and the body it really sits on, and bodies that meet
    through a chain of bones shorter than one radius). Returns `(slot_i,
    slot_j, penetration_depth_m)` per violating pair, `slot_i`/`slot_j`
    possibly `ROOT_NODE`. A cheap capsule proxy for true self-penetration."""
    from hand_sampler.design_space import segment_distance

    q_arr = np.zeros(N_SLOTS) if q is None else np.asarray(q, dtype=float).reshape(N_SLOTS)
    T = authored_fk(design, q_arr)
    valid_slots = [i for i in range(N_SLOTS) if design.slot_valid[i]]
    adjacent = adjacent_pairs(design)

    root_z0, root_z1 = capsule_core_endpoints_local(design.root_length_m, design.capsule_radius_m)
    endpoints: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {
        ROOT_NODE: (
            np.array([0.0, 0.0, root_z0]), np.array([0.0, 0.0, root_z1]), design.capsule_radius_m,
        ),
    }
    for idx in valid_slots:
        z0, z1 = capsule_core_endpoints_local(float(design.slot_length[idx]), design.capsule_radius_m)
        p0 = (T[idx] @ np.array([0.0, 0.0, z0, 1.0]))[:3]
        p1 = (T[idx] @ np.array([0.0, 0.0, z1, 1.0]))[:3]
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


def mark_filtered_pairs(
    design: EnvelopeDesign, max_penetration_m: float = 0.0, extra_qs: Sequence[np.ndarray] = (),
) -> EnvelopeDesign:
    """Return a copy of `design` with `filtered_pairs` set to its own pairs
    (`canonicalize`'s short-bone and mount pairs) plus the UNION, over `q=0`
    and every pose in `extra_qs` (pass `[palm_up(design, n_sweep=0).
    default_q]` so the reset pose is covered too), of every
    `rest_overlap_pairs` pair deeper than `max_penetration_m` -- for designs
    EXEMPTED from overlap rejection (projected commercial hands; see
    `admit`'s `check_overlap=False`). `author_grammar.author_design`
    collision-filters exactly these pairs, instead of relying on PhysX to
    resolve a real interpenetration itself (which blows up the joints on
    step 0, see `MAX_REST_PENETRATION_M`)."""
    import dataclasses

    pairs = set(design.filtered_pairs)          # keep canonicalize's own pairs
    for q in (None, *extra_qs):
        for i, j, pen in rest_overlap_pairs(design, q=q):
            if pen > max_penetration_m:
                pairs.add((i, j))
    return dataclasses.replace(design, filtered_pairs=tuple(sorted(pairs)))


# --------------------------------------------------------------------------
# palm_up
# --------------------------------------------------------------------------


@dataclass
class PalmUpResult:
    normal: np.ndarray          # (3,) unit vector, local (root) frame
    base_rot_wxyz: np.ndarray   # (4,) quaternion mapping local -> world so `normal` -> world +Z
    default_q: np.ndarray       # (36,) mild-curl default pose (followers equal their leader)
    spawn_offset: np.ndarray    # (3,) local-frame spawn point for the object
    fingertip_valid: np.ndarray  # (6,) bool: the finger slot holds a finger
    fingertip_offsets: np.ndarray  # (6,3) local-frame fingertip positions at default_q
    reachable_fingertips: int   # CPU reachability metric (I24 lesson)
    n_swept: int


def finger_valid(design: EnvelopeDesign) -> np.ndarray:
    """`(6,)` bool: finger slot `f` holds a finger (its first joint is real)."""
    return np.array([bool(design.slot_valid[b]) for b in FINGER_BASE_SLOTS])


def palm_up(
    design: EnvelopeDesign,
    object_half_size: float = 0.03,
    n_sweep: int = 4000,
    seed: int = 0,
    reach_tol_m: float = 0.05,
    curl_frac: float = 0.35,
) -> PalmUpResult:
    """Analytic palm-up calibration (design note section 4, "Analytic palm
    calibration") plus the I24 lesson applied to grammar hands: spawn the
    object over the fingertip workspace with a mild default curl, not over
    the palm/root origin, and report a CPU reachability metric (number of
    fingertips that reach within `reach_tol_m` of the spawn point over a
    dense (default 4000-sample) random joint sweep, evaluated as one batched
    FK call via `authored_fk_batch`) instead of assuming reachability.
    Fingertips are the fingertip bodies (`tip_fk`)."""
    valid_slots = [i for i in range(N_SLOTS) if design.slot_valid[i]]
    fvalid = finger_valid(design)

    mid_q = np.zeros(N_SLOTS)
    for idx in valid_slots:
        lo, hi = design.slot_limits[idx]
        mid_q[idx] = lo + curl_frac * (hi - lo)
    mid_q = tied_q(design, mid_q)

    T_mid = authored_fk(design, mid_q)
    T0 = authored_fk(design, np.zeros(N_SLOTS))
    tips_mid = tip_fk(design, T_mid)

    mount_pts = [T0[b][:3, 3] for f, b in enumerate(FINGER_BASE_SLOTS) if fvalid[f]]
    tip_pts_mid = [tips_mid[f] for f in range(N_FINGERS) if fvalid[f]]

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

    # Spawn point: the fingertip centroid at default_q, plus a clearance of
    # the object half size and 5 mm along the calibrated palm normal.
    # `spawn_height_above_palm_m`/`admit`'s spawn-height gate separately
    # guards against this landing at or below the palm.
    spawn_offset = tip_centroid + normal * (object_half_size + 0.005)

    # Reach sweep: a dense batched FK sweep. `n_sweep <= 0` (e.g. `admit`'s
    # cheap call, which only needs `default_q`/`spawn_offset`) skips it.
    reach_count = np.zeros(N_FINGERS, dtype=int)
    if n_sweep > 0:
        rng = np.random.default_rng(seed)
        q_batch = np.tile(mid_q, (n_sweep, 1))
        for idx in valid_slots:
            lo, hi = design.slot_limits[idx]
            q_batch[:, idx] = rng.uniform(lo, hi, size=n_sweep)
        tips = tip_fk(design, authored_fk_batch(design, q_batch))  # (n_sweep, 6, 3)
        dist = np.linalg.norm(tips - spawn_offset[None, None, :], axis=-1)
        reach_count = np.where(fvalid, np.count_nonzero(dist <= reach_tol_m, axis=0), 0)

    fingertip_offsets = np.where(fvalid[:, None], tips_mid, 0.0)

    return PalmUpResult(
        normal=normal, base_rot_wxyz=base_rot, default_q=mid_q, spawn_offset=spawn_offset,
        fingertip_valid=fvalid.copy(), fingertip_offsets=fingertip_offsets,
        reachable_fingertips=int(np.count_nonzero(reach_count > 0)), n_swept=n_sweep,
    )


def spawn_height_above_palm_m(design: EnvelopeDesign, pu: Optional[PalmUpResult] = None) -> float:
    """The population spawn point's height above the palm along WORLD z,
    AFTER `base_rot` -- `quat_apply(base_rot, spawn_offset_local)[2]`, the
    ONE convention `palm_calibration.MIN_SPAWN_HEIGHT_ABOVE_PALM_M`'s
    docstring pins down and `drop_detection.object_below_palm` (world z)
    checks every reset. `HAND_BASE_POS_M`/`env_origins` (author_grammar.py)
    are the SAME additive offset for the palm origin and the spawn point, so
    they cancel out of this difference. `pu` lets a caller that already ran
    `palm_up` reuse it."""
    pu = pu if pu is not None else palm_up(design, n_sweep=0)
    return float(_quat_apply_wxyz(pu.base_rot_wxyz, pu.spawn_offset)[2])


# --------------------------------------------------------------------------
# GrammarPopulation
# --------------------------------------------------------------------------


@dataclass
class GrammarPopulation:
    n_designs: int
    sources: Tuple[str, ...]
    joint_link_boxes: np.ndarray    # (n,36,4,3) float32
    joint_valid: np.ndarray         # (n,36) bool: joints the policy controls
    joint_tie: np.ndarray           # (n,36) int: a follower carrier's leader slot, -1 otherwise
    joint_limits: np.ndarray        # (n,36,2) float64
    default_joint_pos: np.ndarray   # (n,36) float64
    hand_scale: np.ndarray          # (n,) float64
    fingertip_valid: np.ndarray     # (n,6) bool
    fingertip_offsets: np.ndarray   # (n,6,3) float64
    tip_offsets: np.ndarray         # (n,6) float64: fingertip body z offset on its last link
    palm_center: np.ndarray         # (n,3) float64
    palm_keypoints: np.ndarray      # (n,4,3) float64
    palm_frame: np.ndarray          # (n,7) float64 (xyz + wxyz quat)
    base_rot: np.ndarray            # (n,4) float64 (wxyz)
    spawn_offset: np.ndarray        # (n,3) float64
    designs: Tuple[EnvelopeDesign, ...]
    palm_up_results: Tuple[PalmUpResult, ...]
    # Drive gains are not a population table: the authored USD drive uses
    # `robot_param_constants` values and the runtime actuator gains come
    # from `hand_only.DEFAULT_HAND_*`, plus `scene_utils`' per-env carrier
    # gains (locked, leader, follower).


def build_population(designs: Sequence[EnvelopeDesign], **palm_up_kwargs) -> GrammarPopulation:
    n = len(designs)
    joint_link_boxes = np.zeros((n, N_SLOTS, 4, 3), dtype=np.float32)
    joint_valid = np.zeros((n, N_SLOTS), dtype=bool)
    joint_tie = np.full((n, N_SLOTS), -1, dtype=int)
    joint_limits = np.zeros((n, N_SLOTS, 2))
    default_joint_pos = np.zeros((n, N_SLOTS))
    hand_scale = np.zeros(n)
    fingertip_valid = np.zeros((n, N_FINGERS), dtype=bool)
    fingertip_offsets = np.zeros((n, N_FINGERS, 3))
    tip_offs = np.zeros((n, N_FINGERS))
    palm_center = np.zeros((n, 3))
    palm_keypoints = np.zeros((n, 4, 3))
    palm_frame = np.zeros((n, 7))
    base_rot = np.zeros((n, 4))
    spawn_offset = np.zeros((n, 3))
    palm_up_results: List[PalmUpResult] = []

    for i, design in enumerate(designs):
        joint_link_boxes[i] = token_boxes(design)
        joint_valid[i] = design.slot_valid
        joint_tie[i] = design.slot_tie
        joint_limits[i] = design.slot_limits
        tip_offs[i] = tip_offsets(design)
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
        finger_mounts = [T0[b][:3, 3] for b in FINGER_BASE_SLOTS if design.slot_valid[b]]
        palm_center[i] = np.mean(np.stack(finger_mounts), axis=0) if finger_mounts else np.zeros(3)
        kp = [np.zeros(3), np.array([0.0, 0.0, design.root_length_m])]
        kp += finger_mounts[:2] if len(finger_mounts) >= 2 else [np.zeros(3)] * max(0, 2 - len(finger_mounts))
        palm_keypoints[i] = np.stack(kp[:4] + [np.zeros(3)] * max(0, 4 - len(kp)))
        palm_frame[i, :3] = 0.0
        palm_frame[i, 3:] = _mat3_to_quat_wxyz(np.eye(3))

    return GrammarPopulation(
        n_designs=n, sources=tuple(d.source for d in designs), joint_link_boxes=joint_link_boxes,
        joint_valid=joint_valid, joint_tie=joint_tie, joint_limits=joint_limits,
        default_joint_pos=default_joint_pos, hand_scale=hand_scale, fingertip_valid=fingertip_valid,
        fingertip_offsets=fingertip_offsets, tip_offsets=tip_offs, palm_center=palm_center,
        palm_keypoints=palm_keypoints, palm_frame=palm_frame, base_rot=base_rot, spawn_offset=spawn_offset,
        designs=tuple(designs), palm_up_results=tuple(palm_up_results),
    )
