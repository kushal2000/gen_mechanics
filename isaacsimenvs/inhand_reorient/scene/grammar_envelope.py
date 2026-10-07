"""Padded fixed-topology envelope: a grammar hand (`hand_sampler.grammar.hand.Hand`)
-> simulator-ready per-design tables.

Numpy + `hand_sampler` only: no `isaaclab`/`pxr` imports, so this module is
importable and unit-testable without Kit (`author_grammar.py` does the lazy
USD authoring on top of it).

Layout (fixed across every design, so a population shares one articulation
topology):

    - 6 finger slots. Finger slot `f` is one palm joint (its CARRIER, slot
      `6 f`, joint `f{f}_cj`, body `f{f}_c`, a child of the root) followed by
      up to 5 finger joints (slots `6 f + 1 + d`, joints `f{f}_j{d}`, bodies
      `f{f}_link{d}`) -- 36 joint slots -- and a fingertip body `f{f}_tip`
      fixed to `f{f}_link4` at the real fingertip (`tip_offsets`).
    - The root body is the palm frame (x = palm normal on the grasp side, y
      across the palm, z from the wrist toward the fingers; origin at the
      wrist centre) and carries the main palm plate.
    - Each carrier has a role (`carrier_roles`): LOCKED (a finger on the main
      palm, or an empty finger slot: ghost limits, no collider, the finger's
      mount folded into its base origin), LEADER (the first finger of a palm
      joint's section: the carrier IS that palm joint and carries the
      section's plate) or FOLLOWER (another finger on the same section: same
      origin, axis and limits as its leader, tied to it by a mimic joint,
      gearing -1, no collider).
    - A coupled joint is a mimic joint on the independent joint its coupling
      chain starts from (gearing -1.1 per link of the chain), so it is tied
      (`slot_tie`, `slot_gear`) and not policy-controlled.
    - Fingers take finger slots in a canonical order: the main palm's fingers
      in the hand's order, then each palm joint's fingers, leader first.
    - `slot_valid` marks the joints the policy controls (independent finger
      joints and leader carriers); `slot_real` every joint with a real link
      (coupled joints included) and leader carriers.
    - Links are rounded boxes (`hand_sampler.grammar.derive.link_core_box_m`:
      a core box grown by 6 mm), along each link body's +x. Ghost slots (no
      finger joint) are tiny, collider-less and locked near zero.

Every grammar hand fits (at most 6 fingers of at most 5 joints, at most one
palm joint per finger). `admit` adds the viability check C1 (self-overlap)
for sampled designs; commercial hands are exempt and collision-filter their
overlapping pairs instead (`mark_filtered_pairs`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace as dc_replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hand_sampler.design_space import _ordered_box
from hand_sampler.grammar import derive as gdv
from hand_sampler.grammar import viability as gvb
from hand_sampler.grammar.fk import forward_kinematics, pose_to_matrix, rodrigues
from hand_sampler.grammar.hand import (
    COUPLING_RATIO,
    LINK_DENSITY_KG_M3,
    LINK_RADIUS_MM,
    NO_RULES,
    PALM_THICKNESS_MM,
    Hand,
    check,
)
from hand_sampler.grammar.kinematics import KinematicModel

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

GHOST_LENGTH_M = 1e-4
GHOST_LIMITS = (0.0, 1e-8)
GHOST_AXIS = (0.0, 0.0, 1.0)

LINK_RADIUS_M = LINK_RADIUS_MM * 1e-3
MAX_REST_PENETRATION_M = gvb.C1_MAX_OVERLAP_MM * 1e-3


def carrier_slot(f: int) -> int:
    return f * SLOTS_PER_FINGER


def finger_slot(f: int, d: int) -> int:
    return f * SLOTS_PER_FINGER + 1 + d


def slot_finger(idx: int) -> int:
    return idx // SLOTS_PER_FINGER


def slot_joint_index(idx: int) -> int:
    """`-1` for a carrier slot, else the finger joint index `d` (0..4)."""
    return idx % SLOTS_PER_FINGER - 1


CARRIER_SLOTS: Tuple[int, ...] = tuple(carrier_slot(f) for f in range(N_FINGERS))
FINGER_BASE_SLOTS: Tuple[int, ...] = tuple(finger_slot(f, 0) for f in range(N_FINGERS))
LAST_FINGER_SLOTS: Tuple[int, ...] = tuple(finger_slot(f, N_JOINTS_PER_FINGER - 1) for f in range(N_FINGERS))


def slot_name(idx: int) -> str:
    f, d = slot_finger(idx), slot_joint_index(idx)
    return f"f{f}_cj" if d < 0 else f"f{f}_j{d}"


def slot_body(idx: int) -> str:
    f, d = slot_finger(idx), slot_joint_index(idx)
    return f"f{f}_c" if d < 0 else f"f{f}_link{d}"


def tip_body(f: int) -> str:
    return f"f{f}_tip"


SLOT_NAMES: Tuple[str, ...] = tuple(slot_name(i) for i in range(N_SLOTS))
SLOT_BODY_NAMES: Tuple[str, ...] = tuple(slot_body(i) for i in range(N_SLOTS))
TIP_BODY_NAMES: Tuple[str, ...] = tuple(tip_body(f) for f in range(N_FINGERS))


def _slot_parent(idx: int) -> int:
    d = slot_joint_index(idx)
    return ROOT_SENTINEL if d < 0 else idx - 1


SLOT_PARENT: Tuple[int, ...] = tuple(_slot_parent(i) for i in range(N_SLOTS))
TOPOLOGICAL_ORDER: Tuple[int, ...] = tuple(range(N_SLOTS))

ROOT_NODE = -1
"""The main palm plate (the root body) in node pairs (`filtered_pairs`)."""


class AdmissionError(Exception):
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


def _transz(d: float) -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = d
    return T


def _shortest_rotation(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """3x3 rotation `R` with `R @ a_hat == b_hat` (minimal angle)."""
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
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = perp - a * float(np.dot(perp, a))
        perp = perp / np.linalg.norm(perp)
        return rodrigues(tuple(perp), math.pi)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _mat3_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def _quat_apply_wxyz(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    w, axis = q[0], q[1:]
    t = 2.0 * np.cross(axis, v)
    return v + w * t + np.cross(axis, t)


def _translate(p: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = p
    return T


# --------------------------------------------------------------------------
# canonicalize: EnvelopeDesign
# --------------------------------------------------------------------------


@dataclass
class EnvelopeDesign:
    source: str
    hand: Hand
    model: KinematicModel            # `derive(hand)`, the FK oracle
    slot_valid: np.ndarray           # (36,) bool: a joint the policy controls
    slot_real: np.ndarray            # (36,) bool: a real joint with a real link (or a leader carrier)
    slot_prismatic: np.ndarray       # (36,) bool: a sliding joint
    slot_origin: np.ndarray          # (36,4,4): joint origin relative to the slot's envelope parent
    slot_axis: np.ndarray            # (36,3): in the slot's own (child) frame
    slot_limits: np.ndarray          # (36,2): rad (m for a sliding joint)
    slot_length: np.ndarray          # (36,): link length (m); 0 for carriers
    slot_joint_name: Tuple[Optional[str], ...]   # the grammar model's joint name
    slot_body_name: Tuple[Optional[str], ...]
    finger_index: Tuple[int, ...]    # (6,): the hand's finger index per finger slot, -1 if empty
    carrier_palm_joint: Tuple[int, ...]  # (6,): the palm joint per finger slot, -1 on the main palm
    slot_tie: np.ndarray = field(default_factory=lambda: np.full(N_SLOTS, -1, dtype=int))
    """(36,) int: for a follower carrier its leader's slot; for a coupled
    joint the independent slot its coupling chain starts from; else -1."""
    slot_gear: np.ndarray = field(default_factory=lambda: np.ones(N_SLOTS))
    """(36,): q[s] = slot_gear[s] * q[slot_tie[s]] for a tied slot."""
    filtered_pairs: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    """Node pairs (slots, `ROOT_NODE` for the main palm) authoring must
    collision-filter: bodies that touch by construction but are not a
    joint's own two bodies (a finger's first links and the plate it sits
    on, links joined through 0 mm links), plus, for exempted commercial
    hands only, pairs overlapping at the zero or start pose."""
    sha256: str = ""

    @property
    def finger_digit_id(self) -> Tuple[Optional[str], ...]:
        return tuple(None if i < 0 else str(i) for i in self.finger_index)

    @property
    def capsule_radius_m(self) -> float:   # the rounded links' corner radius
        return LINK_RADIUS_M


def carrier_roles(design: EnvelopeDesign) -> Tuple[str, ...]:
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
    """`q` (`(36,)` or `(N,36)`) with every tied slot (follower carrier,
    coupled joint) set to its gear times its source."""
    q = np.array(q, dtype=float, copy=True)
    for s in np.nonzero(design.slot_tie >= 0)[0]:
        q[..., s] = design.slot_gear[s] * q[..., int(design.slot_tie[s])]
    return q


def tie_columns(joint_tie: np.ndarray, slot_of_col: Sequence[int]) -> np.ndarray:
    """`(n, J)` int: for per-env tie rows in slot order and an articulation
    whose column `c` holds slot `slot_of_col[c]`, the column whose value
    column `c` takes when joint positions are written (its own, or its
    source's for a tied slot); `obs_utils.tie_joints` gathers with it and
    multiplies by `tie_gears`."""
    slot_of_col = np.asarray(slot_of_col, dtype=int)
    col_of_slot = np.empty_like(slot_of_col)
    col_of_slot[slot_of_col] = np.arange(len(slot_of_col))
    tie = np.asarray(joint_tie, dtype=int)[:, slot_of_col]
    own = np.broadcast_to(np.arange(len(slot_of_col)), tie.shape)
    return np.where(tie >= 0, col_of_slot[np.clip(tie, 0, None)], own)


def tie_gears(joint_tie: np.ndarray, joint_gear: np.ndarray, slot_of_col: Sequence[int]) -> np.ndarray:
    """`(n, J)`: the factor `tie_columns`' gathered value is multiplied by (1
    for an untied column)."""
    slot_of_col = np.asarray(slot_of_col, dtype=int)
    tie = np.asarray(joint_tie, dtype=int)[:, slot_of_col]
    gear = np.asarray(joint_gear, dtype=float)[:, slot_of_col]
    return np.where(tie >= 0, gear, 1.0)


def tip_offsets(design: EnvelopeDesign) -> np.ndarray:
    """`(6,)`: per finger slot, the x offset of its fingertip body on its last
    link `f{f}_link4`: that link's length for a 5-joint finger, else 0 (the
    ghost tail of a shorter finger already sits at its fingertip)."""
    return np.array([float(design.slot_length[s]) if design.slot_real[s] else 0.0 for s in LAST_FINGER_SLOTS])


def _assign_slots(hand: Hand) -> Tuple[List[int], List[int], List[int]]:
    """Per finger slot: (hand finger index or -1, palm joint or -1, leader
    finger slot or -1)."""
    order: List[Tuple[int, int]] = [(i, -1) for i, f in enumerate(hand.fingers) if f.palm_joint < 0]
    for k in range(len(hand.palm_joints)):
        order += [(i, k) for i, f in enumerate(hand.fingers) if f.palm_joint == k]
    fi = [-1] * N_FINGERS
    pk = [-1] * N_FINGERS
    lead = [-1] * N_FINGERS
    first_of: Dict[int, int] = {}
    for f, (i, k) in enumerate(order):
        fi[f], pk[f] = i, k
        if k >= 0:
            lead[f] = first_of.setdefault(k, f)
    return fi, pk, lead


def canonicalize(hand: Hand, source: str = "") -> EnvelopeDesign:
    problems = check(hand, NO_RULES)
    if problems:
        raise AdmissionError(problems)
    if len(hand.fingers) > N_FINGERS or any(len(f.joints) > N_JOINTS_PER_FINGER for f in hand.fingers):
        raise AdmissionError(["hand larger than the 36-slot envelope"])
    model = gdv.derive(hand, name=source or "grammar_hand")
    fi, pk, lead = _assign_slots(hand)

    slot_valid = np.zeros(N_SLOTS, dtype=bool)
    slot_real = np.zeros(N_SLOTS, dtype=bool)
    slot_prismatic = np.zeros(N_SLOTS, dtype=bool)
    slot_origin = np.tile(np.eye(4), (N_SLOTS, 1, 1))
    slot_axis = np.tile(np.array(GHOST_AXIS), (N_SLOTS, 1))
    slot_limits = np.tile(np.array(GHOST_LIMITS), (N_SLOTS, 1))
    slot_length = np.full(N_SLOTS, GHOST_LENGTH_M)
    slot_tie = np.full(N_SLOTS, -1, dtype=int)
    slot_gear = np.ones(N_SLOTS)
    jname: List[Optional[str]] = [None] * N_SLOTS
    bname: List[Optional[str]] = [None] * N_SLOTS

    for f in range(N_FINGERS):
        c = carrier_slot(f)
        k = pk[f]
        hinge = np.zeros(3)
        slot_length[c] = 0.0
        if k >= 0:
            hinge = gdv.palm_hinge_m(hand, k)
            slot_origin[c] = _translate(hinge)
            slot_axis[c] = gdv.palm_axis(hand, k)
            slot_limits[c] = gdv.palm_limits()
            bname[c] = f"palm{k}"
            if lead[f] == f:
                slot_valid[c] = slot_real[c] = True
                jname[c] = f"palm{k}_joint"
            else:
                slot_tie[c] = carrier_slot(lead[f])
        i = fi[f]
        if i < 0:
            continue
        fg = hand.fingers[i]
        base = np.eye(4)
        base[:3, :3] = gdv.finger_frame(fg)
        base[:3, 3] = gdv.mount_point_m(fg) - hinge
        n = len(fg.joints)
        for d, jt in enumerate(fg.joints):
            s = finger_slot(f, d)
            slot_real[s] = True
            slot_valid[s] = jt.type != "coupled"
            slot_prismatic[s] = jt.type == "sliding"
            slot_origin[s] = base if d == 0 else _translate((fg.joints[d - 1].length * 1e-3, 0.0, 0.0))
            slot_axis[s] = gdv.joint_axis(hand, i, d)
            slot_limits[s] = gdv.joint_limits(hand, i, d)
            slot_length[s] = jt.length * 1e-3
            jname[s] = f"f{i}_j{d}"
            bname[s] = f"f{i}_link{d}"
            if jt.type == "coupled":
                src, gear = d - 1, COUPLING_RATIO
                while fg.joints[src].type == "coupled":
                    src, gear = src - 1, gear * COUPLING_RATIO
                slot_tie[s] = finger_slot(f, src)
                slot_gear[s] = gear
        if n < N_JOINTS_PER_FINGER:
            # the first ghost slot sits at the real fingertip, so the ghost tail
            # (and the fingertip body on f{f}_link4) does too
            slot_origin[finger_slot(f, n)] = _translate((fg.joints[-1].length * 1e-3, 0.0, 0.0))

    design = EnvelopeDesign(
        source=source, hand=hand, model=model, slot_valid=slot_valid, slot_real=slot_real,
        slot_prismatic=slot_prismatic, slot_origin=slot_origin, slot_axis=slot_axis, slot_limits=slot_limits,
        slot_length=slot_length, slot_joint_name=tuple(jname), slot_body_name=tuple(bname),
        finger_index=tuple(fi), carrier_palm_joint=tuple(pk), slot_tie=slot_tie, slot_gear=slot_gear,
    )
    design.filtered_pairs = construction_pairs(design)
    return design


# --------------------------------------------------------------------------
# Nodes: grammar bodies <-> envelope slots
# --------------------------------------------------------------------------


def node_of_key(design: EnvelopeDesign, key: Tuple) -> int:
    """`hand_sampler.grammar.viability` body key -> envelope node."""
    if key[0] == "palm":
        if key[1] < 0:
            return ROOT_NODE
        f = next(f for f in range(N_FINGERS) if design.carrier_palm_joint[f] == key[1] and design.slot_valid[carrier_slot(f)])
        return carrier_slot(f)
    _, i, j = key
    f = design.finger_index.index(i)
    return finger_slot(f, j)


def _articulation_parent(design: EnvelopeDesign, node: int) -> int:
    if node == ROOT_NODE:
        return ROOT_NODE
    p = SLOT_PARENT[node]
    return ROOT_NODE if p == ROOT_SENTINEL else p


def construction_pairs(design: EnvelopeDesign) -> Tuple[Tuple[int, int], ...]:
    """Bodies that touch by construction (`viability.touching_pairs`) and
    are not a joint's own two bodies (PhysX already ignores those)."""
    pairs = set()
    for a, b in gvb.touching_pairs(design.hand):
        na, nb = node_of_key(design, a), node_of_key(design, b)
        if na == nb or _articulation_parent(design, na) == nb or _articulation_parent(design, nb) == na:
            continue
        pairs.add((min(na, nb), max(na, nb)))
    return tuple(sorted(pairs))


def overlap_pairs(design: EnvelopeDesign, q: Optional[np.ndarray] = None,
                  min_overlap_mm: float = 0.0) -> List[Tuple[int, int, float]]:
    """`(node_a, node_b, overlap_m)` for every pair of bodies that do not touch
    by construction and overlap (rounded-box distance) at slot joint vector
    `q` (default 0)."""
    qd = None if q is None else slot_q_to_dofs(design, q)
    out = []
    for a, b, ov in gvb.overlaps(design.hand, qd):
        if ov > min_overlap_mm:
            out.append((node_of_key(design, a), node_of_key(design, b), ov * 1e-3))
    return out


def mark_filtered_pairs(design: EnvelopeDesign, max_penetration_m: float = 0.0,
                        extra_qs: Sequence[np.ndarray] = ()) -> EnvelopeDesign:
    """A copy of `design` whose `filtered_pairs` also hold every pair that
    overlaps at q = 0 or at any of `extra_qs` (commercial hands only: their
    overlaps come from the shared cross-section, not from the design)."""
    pairs = set(design.filtered_pairs)
    for q in (None, *extra_qs):
        for a, b, _ in overlap_pairs(design, q, min_overlap_mm=max_penetration_m * 1e3):
            pairs.add((min(a, b), max(a, b)))
    return dc_replace(design, filtered_pairs=tuple(sorted(pairs)))


# --------------------------------------------------------------------------
# Joint vectors: slots <-> grammar dofs
# --------------------------------------------------------------------------


def slot_of_dof(design: EnvelopeDesign) -> List[int]:
    """For each grammar dof (`derive.dofs` order), its slot."""
    out = []
    for d in gdv.dofs(design.hand):
        if d.finger < 0:
            f = next(f for f in range(N_FINGERS)
                     if design.carrier_palm_joint[f] == d.index and design.slot_valid[carrier_slot(f)])
            out.append(carrier_slot(f))
        else:
            out.append(finger_slot(design.finger_index.index(d.finger), d.index))
    return out


def slot_q_to_dofs(design: EnvelopeDesign, q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q[..., slot_of_dof(design)]


def dofs_to_slot_q(design: EnvelopeDesign, qd: np.ndarray) -> np.ndarray:
    out = np.zeros(qd.shape[:-1] + (N_SLOTS,))
    out[..., slot_of_dof(design)] = qd
    return tied_q(design, out)


# --------------------------------------------------------------------------
# Admission and viability
# --------------------------------------------------------------------------


def admit(hand: Hand, *, check_overlap: bool = True,
          max_rest_penetration_m: float = MAX_REST_PENETRATION_M) -> AdmissionResult:
    """Whether the simulator builds `hand`: it is in the grammar (every
    grammar hand fits the envelope) and, for a sampled design
    (`check_overlap`), passes C1 (no two bodies overlap by more than 3 mm at
    the zero or the start pose). Commercial hands pass `check_overlap=False`
    and filter their overlapping pairs instead."""
    problems = check(hand, NO_RULES)
    if problems:
        return AdmissionResult(False, tuple(problems))
    if check_overlap:
        c1 = gvb.c1_self_overlap(hand, max_rest_penetration_m * 1e3)
        if not c1.ok:
            return AdmissionResult(False, (f"C1: {len(c1.pairs)} overlapping pair(s), worst {c1.worst_mm:.1f} mm "
                                           f"(allowed {max_rest_penetration_m * 1e3:.0f} mm)",))
    return AdmissionResult(True, ())


def viability_report(hand: Hand) -> dict:
    """Both checks for one hand (CPU only):
    `{"admitted", "reasons", "c1_ok", "c1_worst_mm", "c2_ok", "c2_best_mm",
    "c2_margin_mm", "digit_count", "joint_count"}`; `admitted` is C1 (what the
    simulator needs), `viable` is C1 and C2."""
    problems = check(hand, NO_RULES)
    if problems:
        return {"admitted": False, "viable": False, "reasons": problems, "c1_ok": None, "c1_worst_mm": None,
                "c2_ok": None, "c2_best_mm": None, "c2_margin_mm": None, "digit_count": None, "joint_count": None}
    v = gvb.viability(hand)
    reasons = []
    if not v.c1.ok:
        reasons.append(f"C1: worst overlap {v.c1.worst_mm:.1f} mm")
    if not v.c2.ok:
        reasons.append(f"C2: fingertip workspaces {v.c2.best_mm:.1f} mm apart above the palm")
    return {"admitted": bool(v.c1.ok), "viable": bool(v.ok), "reasons": reasons, "c1_ok": bool(v.c1.ok),
            "c1_worst_mm": float(v.c1.worst_mm), "c2_ok": bool(v.c2.ok), "c2_best_mm": float(v.c2.best_mm),
            "c2_margin_mm": float(v.c2.margin_mm), "digit_count": len(hand.fingers),
            "joint_count": sum(len(f.joints) for f in hand.fingers) + len(hand.palm_joints)}


# --------------------------------------------------------------------------
# joint_local_frames / authored_fk
# --------------------------------------------------------------------------


def joint_local_frames(design: EnvelopeDesign) -> np.ndarray:
    """`(36, 2, 4, 4)`: per slot `(frame0, frame1)` such that the child
    body's transform is `parent @ frame0 @ M(q) @ inv(frame1)`, `M` a
    rotation about (revolute) or translation along (prismatic) the local z:
    `frame0 = origin @ [R, 0]`, `frame1 = [R, 0]`, `R @ ez == axis`."""
    out = np.zeros((N_SLOTS, 2, 4, 4))
    for idx in range(N_SLOTS):
        R4 = np.eye(4)
        R4[:3, :3] = _shortest_rotation((0.0, 0.0, 1.0), design.slot_axis[idx])
        out[idx, 0] = design.slot_origin[idx] @ R4
        out[idx, 1] = R4
    return out


def _motion(design: EnvelopeDesign, idx: int, q: float) -> np.ndarray:
    return _transz(q) if design.slot_prismatic[idx] else _rotz(q)


def authored_fk(design: EnvelopeDesign, q: np.ndarray) -> np.ndarray:
    """`(36,4,4)` root-frame transforms of every slot's child body at slot
    vector `q` (tied slots follow their source)."""
    frames = joint_local_frames(design)
    q = tied_q(design, np.asarray(q, dtype=float).reshape(N_SLOTS))
    T = [np.eye(4) for _ in range(N_SLOTS)]
    for idx in TOPOLOGICAL_ORDER:
        parent = SLOT_PARENT[idx]
        parent_T = np.eye(4) if parent == ROOT_SENTINEL else T[parent]
        frame0, frame1 = frames[idx]
        T[idx] = parent_T @ frame0 @ _motion(design, idx, float(q[idx])) @ np.linalg.inv(frame1)
    return np.stack(T)


def authored_fk_batch(design: EnvelopeDesign, q: np.ndarray) -> np.ndarray:
    """`(N,36,4,4)`: `authored_fk` for `N` slot vectors at once."""
    frames = joint_local_frames(design)
    q = tied_q(design, np.asarray(q, dtype=float))
    n = q.shape[0]
    T: List[Optional[np.ndarray]] = [None] * N_SLOTS
    root_T = np.broadcast_to(np.eye(4), (n, 4, 4))
    for idx in TOPOLOGICAL_ORDER:
        parent = SLOT_PARENT[idx]
        parent_T = root_T if parent == ROOT_SENTINEL else T[parent]
        frame0, frame1 = frames[idx]
        M = np.tile(np.eye(4), (n, 1, 1))
        if design.slot_prismatic[idx]:
            M[:, 2, 3] = q[:, idx]
        else:
            c, s = np.cos(q[:, idx]), np.sin(q[:, idx])
            M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1] = c, -s, s, c
        T[idx] = parent_T @ frame0[None] @ M @ np.linalg.inv(frame1)[None]
    return np.stack(T, axis=1)


def tip_fk(design: EnvelopeDesign, T: np.ndarray) -> np.ndarray:
    """`(..., 6, 3)` fingertip-body positions from slot transforms `T`
    (links run along their body's +x)."""
    last = T[..., list(LAST_FINGER_SLOTS), :, :]
    off = tip_offsets(design)
    return last[..., :3, 3] + last[..., :3, 0] * off[:, None]


def grammar_fk_reference(design: EnvelopeDesign, q: np.ndarray) -> np.ndarray:
    """`(36,4,4)`: the same quantity as `authored_fk`, computed by
    `hand_sampler.grammar.fk.forward_kinematics` on `design.model` (ghost
    and locked slots identity; a follower carrier gets its section's pose)."""
    q = tied_q(design, np.asarray(q, dtype=float).reshape(N_SLOTS))
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
# Geometry: link boxes, palm plates, mass
# --------------------------------------------------------------------------


def link_core(design: EnvelopeDesign, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """(centre, half extents) of slot `idx`'s rounded link core box, in its
    body frame (metres)."""
    f, d = slot_finger(idx), slot_joint_index(idx)
    return gdv.link_core_box_m(design.hand, design.finger_index[f], d)


def link_core_points(design: EnvelopeDesign, idx: int) -> np.ndarray:
    """The 8 core-box corners: the collider's convex hull points."""
    return gdv.box_corners(*link_core(design, idx))


def plate_points(design: EnvelopeDesign, node: int) -> np.ndarray:
    """A palm plate's prism corners (<= 64) in its body frame: the main palm
    (`ROOT_NODE`, root frame) or a leader carrier's section (relative to its
    hinge)."""
    outlines = gdv.palm_outlines_mm(design.hand)
    if node == ROOT_NODE:
        return gdv.plate_points_m(outlines[-1])
    k = design.carrier_palm_joint[slot_finger(node)]
    p = design.hand.palm_joints[k]
    return gdv.plate_points_m(outlines[k], origin_mm=(p.y, p.z))


def plate_mass_props(points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """(mass, com, diagonal inertia about the com) of a plate prism (palm
    density), its inertia that of its bounding box at that mass."""
    top = points[: len(points) // 2]
    yz = top[:, 1:]
    x, y = yz[:, 0], yz[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))
    t = PALM_THICKNESS_MM * 1e-3
    m = LINK_DENSITY_KG_M3 * area * t
    c2 = gdv.polygon_centroid(yz)
    com = np.array([0.0, c2[0], c2[1]])
    ext = points.max(axis=0) - points.min(axis=0)
    a, b, cc = ext
    I = m / 12.0 * np.array([b * b + cc * cc, a * a + cc * cc, a * a + b * b])
    return float(m), com, I


def link_mass_props(design: EnvelopeDesign, idx: int) -> Tuple[float, np.ndarray, np.ndarray]:
    return gdv.rounded_box_mass_props(*link_core(design, idx))


def token_boxes(design: EnvelopeDesign) -> np.ndarray:
    """`(36, 4, 3)` float32 ordered boxes (`design_space._ordered_box`) of each
    slot's body in its own frame: a link's outer rounded box, a leader
    carrier's section plate, tiny for every other slot."""
    out = np.zeros((N_SLOTS, 4, 3), dtype=np.float32)
    for idx in range(N_SLOTS):
        if design.slot_real[idx] and slot_joint_index(idx) >= 0:
            c, h = link_core(design, idx)
            verts = gdv.box_corners(c, h + LINK_RADIUS_M)
        elif design.slot_real[idx]:
            pts = plate_points(design, idx)
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            verts = gdv.box_corners((lo + hi) / 2.0, (hi - lo) / 2.0)
        else:
            verts = gdv.box_corners(np.zeros(3), np.full(3, GHOST_LENGTH_M / 2.0))
        box, _ok = _ordered_box(verts, design.slot_axis[idx])
        out[idx] = box
    return out


# --------------------------------------------------------------------------
# palm_up: palm-up placement, start pose, object start
# --------------------------------------------------------------------------

PALM_UP_ROT = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
"""Palm frame -> world: the palm normal (+x) up (+z), the fingers (+z) along
world +x."""


@dataclass
class PalmUpResult:
    normal: np.ndarray            # (3,) the palm normal, root frame: +x
    base_rot_wxyz: np.ndarray     # (4,) root -> world, normal to world +z
    default_q: np.ndarray         # (36,) the episode start pose (tied slots follow)
    spawn_offset: np.ndarray      # (3,) root-frame object start point
    fingertip_valid: np.ndarray   # (6,) bool
    fingertip_offsets: np.ndarray  # (6,3) root-frame fingertips at default_q


def finger_valid(design: EnvelopeDesign) -> np.ndarray:
    return np.array([design.finger_index[f] >= 0 for f in range(N_FINGERS)])


def palm_up(design: EnvelopeDesign, object_half_size: float = 0.03) -> PalmUpResult:
    """The palm frame is known, so this is exact: base_rot turns the palm
    normal to world +z; the start pose is `derive.start_q` (every flexion
    joint at 0.35 of its upper limit); the object starts above the main
    palm's outline centroid, half its size plus 5 mm above the plate
    (`derive.object_start_m`)."""
    fvalid = finger_valid(design)
    default_q = dofs_to_slot_q(design, gdv.start_q(design.hand))
    tips = tip_fk(design, authored_fk(design, default_q))
    return PalmUpResult(
        normal=np.array([1.0, 0.0, 0.0]), base_rot_wxyz=_mat3_to_quat_wxyz(PALM_UP_ROT), default_q=default_q,
        spawn_offset=gdv.object_start_m(design.hand, object_half_size), fingertip_valid=fvalid,
        fingertip_offsets=np.where(fvalid[:, None], tips, 0.0),
    )


# --------------------------------------------------------------------------
# GrammarPopulation
# --------------------------------------------------------------------------


@dataclass
class GrammarPopulation:
    n_designs: int
    sources: Tuple[str, ...]
    joint_link_boxes: np.ndarray    # (n,36,4,3) float32
    joint_valid: np.ndarray         # (n,36) bool: joints the policy controls
    joint_real: np.ndarray          # (n,36) bool: real joints (coupled included) and leader carriers
    joint_tie: np.ndarray           # (n,36) int: a tied slot's source, -1 otherwise
    joint_gear: np.ndarray          # (n,36) float: a tied slot's gear
    joint_limits: np.ndarray        # (n,36,2)
    default_joint_pos: np.ndarray   # (n,36)
    hand_scale: np.ndarray          # (n,)
    fingertip_valid: np.ndarray     # (n,6) bool
    fingertip_offsets: np.ndarray   # (n,6,3)
    tip_offsets: np.ndarray         # (n,6): fingertip body x offset on its last link
    palm_center: np.ndarray         # (n,3)
    palm_keypoints: np.ndarray      # (n,4,3)
    palm_frame: np.ndarray          # (n,7)
    base_rot: np.ndarray            # (n,4) wxyz
    spawn_offset: np.ndarray        # (n,3)
    designs: Tuple[EnvelopeDesign, ...]
    palm_up_results: Tuple[PalmUpResult, ...]


def build_population(designs: Sequence[EnvelopeDesign], **palm_up_kwargs) -> GrammarPopulation:
    n = len(designs)
    out = dict(
        joint_link_boxes=np.zeros((n, N_SLOTS, 4, 3), dtype=np.float32),
        joint_valid=np.zeros((n, N_SLOTS), dtype=bool), joint_real=np.zeros((n, N_SLOTS), dtype=bool),
        joint_tie=np.full((n, N_SLOTS), -1, dtype=int), joint_gear=np.ones((n, N_SLOTS)),
        joint_limits=np.zeros((n, N_SLOTS, 2)), default_joint_pos=np.zeros((n, N_SLOTS)), hand_scale=np.zeros(n),
        fingertip_valid=np.zeros((n, N_FINGERS), dtype=bool), fingertip_offsets=np.zeros((n, N_FINGERS, 3)),
        tip_offsets=np.zeros((n, N_FINGERS)), palm_center=np.zeros((n, 3)), palm_keypoints=np.zeros((n, 4, 3)),
        palm_frame=np.zeros((n, 7)), base_rot=np.zeros((n, 4)), spawn_offset=np.zeros((n, 3)),
    )
    results: List[PalmUpResult] = []
    for i, d in enumerate(designs):
        out["joint_link_boxes"][i] = token_boxes(d)
        out["joint_valid"][i] = d.slot_valid
        out["joint_real"][i] = d.slot_real
        out["joint_tie"][i] = d.slot_tie
        out["joint_gear"][i] = d.slot_gear
        out["joint_limits"][i] = d.slot_limits
        out["tip_offsets"][i] = tip_offsets(d)
        lengths = d.slot_length[d.slot_real]
        out["hand_scale"][i] = float(lengths.max()) if lengths.size else GHOST_LENGTH_M
        pu = palm_up(d, **palm_up_kwargs)
        results.append(pu)
        out["default_joint_pos"][i] = pu.default_q
        out["fingertip_valid"][i] = pu.fingertip_valid
        out["fingertip_offsets"][i] = pu.fingertip_offsets
        out["base_rot"][i] = pu.base_rot_wxyz
        out["spawn_offset"][i] = pu.spawn_offset
        outline = gdv.palm_outlines_mm(d.hand)[-1]
        c = gdv.polygon_centroid(outline) * 1e-3
        top = PALM_THICKNESS_MM / 2.0 * 1e-3
        out["palm_center"][i] = np.array([top, c[0], c[1]])
        mounts = [gdv.mount_point_m(f) for f in d.hand.fingers][:2]
        kp = [np.zeros(3), out["palm_center"][i]] + mounts
        out["palm_keypoints"][i] = np.stack((kp + [np.zeros(3)] * 4)[:4])
        out["palm_frame"][i, 3:] = _mat3_to_quat_wxyz(np.eye(3))
    return GrammarPopulation(n_designs=n, sources=tuple(d.source for d in designs), designs=tuple(designs),
                             palm_up_results=tuple(results), **out)
