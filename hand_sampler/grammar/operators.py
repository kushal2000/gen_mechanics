"""Random hands and mutation operators for the hand grammar.

Every operator lists its applications (`candidates`) and offers only those
whose result keeps the rules: nothing is generated and then rejected. An
operator with no application cannot act (the viewer hides it). A hand that
already breaks the rules (a commercial hand with a sliding joint, under the
Evolution Rules) can still be mutated as long as it stays in the grammar and
no rule gets broken more often.

Stages (GRAMMAR-LOCK item 17): every parameter steps by 10 mm or 30 degrees
(coarse) or 1 mm or 5 degrees (fine). Structural operators belong to the
coarse stage; the fine stage only refines parameters. Every coarse value lies
on the fine grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hand import (
    KIND_AXIS,
    KIND_MIX,
    LINK_MAX_MM,
    LINK_MIN_MM,
    PALM_ANCHOR_MM,
    PALM_HINGE_DISTANCE_MM,
    REVOLUTE_TYPES,
    STEPS,
    TILT_RANGE_DEG,
    TIP_MIN_MM,
    Finger,
    Hand,
    Joint,
    PalmJoint,
    Rules,
    EVOLUTION_RULES,
    NO_RULES,
    canonical_axis,
    check,
    drop_unused_palm_joints,
    link_length_ok,
    with_finger,
    with_joint,
)

STAGES = ("coarse", "fine")

# Random-draw priors: uniform within the grammar's own limits, plus three
# shares measured on the commercial hands.
P_PALM_JOINT = 4 / 18
"""Share of the commercial hands with a palm joint (SHARPA, Shadow, SVH, ARMS)."""
P_ZERO_LINK = 27 / 220
"""Share of commercial bones that are 0 mm (two joints at one point)."""
P_COUPLED = 18 / 180
"""Within-finger couplings per joint after a finger's first joint."""
P_SLIDING = 2 / 265
"""Sliding joints per commercial joint (Dex1's two jaws), when allowed."""


def step_of(stage: str) -> Tuple[int, int]:
    return STEPS[stage]


def _grid(lo: int, hi: int, step: int) -> List[int]:
    start = int(math.ceil(lo / step) * step)
    return list(range(start, hi + 1, step))


def _pick(rng: np.random.Generator, values: Sequence):
    return values[int(rng.integers(len(values)))]


def _admissible(child: Hand, parent_violations: Optional[int], rules: Rules) -> bool:
    v = check(child, rules)
    if parent_violations is None or parent_violations == 0:
        return not v
    return len(v) <= parent_violations and not check(child, NO_RULES)


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Operator:
    name: str
    label: str            # the viewer's button, plain words
    hover: str            # one line of hover text
    structural: bool
    moves: Callable[[Hand, Rules, str], List[tuple]]   # raw applications (before the rules)
    apply: Callable[[Hand, tuple, str], Hand]


def _finger_param_moves(hand: Hand, rules: Rules, stage: str) -> List[tuple]:
    return [(i, s) for i in range(len(hand.fingers)) for s in (-1, 1)]


def _move_finger_moves(hand: Hand, rules: Rules, stage: str) -> List[tuple]:
    return [(i, c, s) for i in range(len(hand.fingers)) for c in ("y", "z") for s in (-1, 1)]


def _move_finger(hand: Hand, mv: tuple, stage: str) -> Hand:
    i, c, s = mv
    mm = step_of(stage)[0]
    f = hand.fingers[i]
    return with_finger(hand, i, replace(f, **{c: getattr(f, c) + s * mm}))


def _turn_finger(hand: Hand, mv: tuple, stage: str) -> Hand:
    i, s = mv
    f = hand.fingers[i]
    return with_finger(hand, i, replace(f, facing=(f.facing + s * step_of(stage)[1]) % 360))


def _tilt_finger(hand: Hand, mv: tuple, stage: str) -> Hand:
    i, s = mv
    f = hand.fingers[i]
    return with_finger(hand, i, replace(f, tilt=f.tilt + s * step_of(stage)[1]))


def _joint_moves(hand: Hand, rules: Rules, stage: str) -> List[tuple]:
    return [(i, j, s) for i, f in enumerate(hand.fingers) for j in range(len(f.joints)) for s in (-1, 1)]


def new_length(length: int, s: int, mm: int, is_tip: bool) -> Optional[int]:
    """One length step: +/- `mm`, skipping the 1-14 mm gap (shortening below
    15 mm gives 0, lengthening 0 gives 15); None outside the range."""
    if length == 0 and s > 0:
        return LINK_MIN_MM
    out = length + s * mm
    if not is_tip and 0 < out < LINK_MIN_MM:
        out = 0 if s < 0 else LINK_MIN_MM
    return out if link_length_ok(out, is_tip) else None


def _length(hand: Hand, mv: tuple, stage: str) -> Hand:
    i, j, s = mv
    f = hand.fingers[i]
    jt = f.joints[j]
    out = new_length(jt.length, s, step_of(stage)[0], j == len(f.joints) - 1)
    if out is None:
        return hand
    return with_joint(hand, i, j, replace(jt, length=out))


def step_axis(axis: Tuple[int, int], which: int, s: int, deg: int) -> Tuple[int, int]:
    az, el = axis
    if which == 0:
        return canonical_axis(az + s * deg, el)
    return canonical_axis(az, el + s * deg)


def _axis_az(hand: Hand, mv: tuple, stage: str) -> Hand:
    i, j, s = mv
    jt = hand.fingers[i].joints[j]
    return with_joint(hand, i, j, replace(jt, axis=step_axis(jt.axis, 0, s, step_of(stage)[1])))


def _axis_el(hand: Hand, mv: tuple, stage: str) -> Hand:
    i, j, s = mv
    jt = hand.fingers[i].joints[j]
    return with_joint(hand, i, j, replace(jt, axis=step_axis(jt.axis, 1, s, step_of(stage)[1])))


def _hinge_moves(hand: Hand, rules: Rules, stage: str) -> List[tuple]:
    return [(k, c, s) for k in range(len(hand.palm_joints)) for c in ("y", "z") for s in (-1, 1)]


def _hinge_axis_moves(hand: Hand, rules: Rules, stage: str) -> List[tuple]:
    return [(k, s) for k in range(len(hand.palm_joints)) for s in (-1, 1)]


def _with_palm_joint(hand: Hand, k: int, p: PalmJoint) -> Hand:
    ps = list(hand.palm_joints)
    ps[k] = p
    return replace(hand, palm_joints=tuple(ps))


def _move_hinge(hand: Hand, mv: tuple, stage: str) -> Hand:
    k, c, s = mv
    p = hand.palm_joints[k]
    return _with_palm_joint(hand, k, replace(p, **{c: getattr(p, c) + s * step_of(stage)[0]}))


def _hinge_az(hand: Hand, mv: tuple, stage: str) -> Hand:
    k, s = mv
    p = hand.palm_joints[k]
    return _with_palm_joint(hand, k, replace(p, axis=step_axis(p.axis, 0, s, step_of(stage)[1])))


def _hinge_el(hand: Hand, mv: tuple, stage: str) -> Hand:
    k, s = mv
    p = hand.palm_joints[k]
    return _with_palm_joint(hand, k, replace(p, axis=step_axis(p.axis, 1, s, step_of(stage)[1])))


# ---- structural ---------------------------------------------------------


def _free_spots(hand: Hand, rules: Rules, stage: str) -> List[Tuple[int, int]]:
    """Grid points (10 mm, or 1 mm thinned to 5 mm at the fine stage) inside
    the mount area and at least the spacing from every finger base."""
    lo, hi = rules.base_distance_mm
    step = 10
    out = []
    for y in range(-hi, hi + 1, step):
        for z in range(-hi, hi + 1, step):
            d = math.hypot(y - PALM_ANCHOR_MM[0], z - PALM_ANCHOR_MM[1])
            if not (lo <= d <= hi):
                continue
            if all(math.hypot(y - f.y, z - f.z) >= rules.min_spacing_mm for f in hand.fingers):
                out.append((y, z))
    return out


def _radial_facing(y: int, z: int, deg: int) -> int:
    a = math.degrees(math.atan2(y - PALM_ANCHOR_MM[0], z - PALM_ANCHOR_MM[1]))
    return int(round(a / deg) * deg) % 360


def _add_finger_moves(hand: Hand, rules: Rules, stage: str) -> List[tuple]:
    if stage != "coarse" or len(hand.fingers) >= rules.max_fingers:
        return []
    return [(y, z) for (y, z) in _free_spots(hand, rules, stage)]


def _add_finger(hand: Hand, mv: tuple, stage: str) -> Hand:
    y, z = mv
    f = Finger(y=y, z=z, facing=_radial_facing(y, z, step_of(stage)[1]), tilt=0,
               joints=(Joint("hinge", KIND_AXIS["flexion"], 40),))
    return replace(hand, fingers=hand.fingers + (f,))


def _remove_finger_moves(hand: Hand, rules: Rules, stage: str) -> List[tuple]:
    if stage != "coarse" or len(hand.fingers) <= rules.min_fingers:
        return []
    return [(i,) for i in range(len(hand.fingers))]


def _remove_finger(hand: Hand, mv: tuple, stage: str) -> Hand:
    (i,) = mv
    fs = hand.fingers[:i] + hand.fingers[i + 1:]
    return drop_unused_palm_joints(replace(hand, fingers=fs))


def _split_moves(hand: Hand, rules: Rules, stage: str) -> List[tuple]:
    if stage != "coarse":
        return []
    kinds = list(KIND_MIX) + (["sliding"] if "sliding" in rules.joint_types else [])
    out = []
    for i, f in enumerate(hand.fingers):
        if len(f.joints) >= rules.max_joints:
            continue
        for j, jt in enumerate(f.joints):
            is_tip = j == len(f.joints) - 1
            L = jt.length
            for a in sorted({0, L // 2, L}):
                if link_length_ok(a, False) and link_length_ok(L - a, is_tip):
                    out.extend((i, j, a, k) for k in kinds)
    return out


def _split(hand: Hand, mv: tuple, stage: str) -> Hand:
    """Split finger i's link j at `a` mm from joint j: a new joint of kind k
    starts the second part; the finger keeps its length."""
    i, j, a, k = mv
    f = hand.fingers[i]
    jt = f.joints[j]
    new = Joint("sliding" if k == "sliding" else "hinge", KIND_AXIS.get(k, (0, 0)), jt.length - a)
    js = list(f.joints)
    js[j] = replace(jt, length=a)
    js.insert(j + 1, new)
    return with_finger(hand, i, replace(f, joints=tuple(js)))


def _merge_moves(hand: Hand, rules: Rules, stage: str) -> List[tuple]:
    if stage != "coarse":
        return []
    return [(i, j) for i, f in enumerate(hand.fingers) for j in range(1, len(f.joints))]


def _merge(hand: Hand, mv: tuple, stage: str) -> Hand:
    """Remove finger i's joint j (j >= 1): links j-1 and j become one. A
    joint coupled to the removed one follows the joint before it instead
    (or becomes a hinge if that one is not revolute)."""
    i, j = mv
    f = hand.fingers[i]
    js = list(f.joints)
    js[j - 1] = replace(js[j - 1], length=js[j - 1].length + js[j].length)
    del js[j]
    if j < len(js) and js[j].type == "coupled" and js[j - 1].type not in REVOLUTE_TYPES:
        js[j] = replace(js[j], type="hinge")
    return with_finger(hand, i, replace(f, joints=tuple(js)))


def _couple_moves(hand: Hand, rules: Rules, stage: str) -> List[tuple]:
    if stage != "coarse":
        return []
    return [(i, j) for i, f in enumerate(hand.fingers) for j in range(1, len(f.joints))
            if f.joints[j].type in REVOLUTE_TYPES]


def _couple(hand: Hand, mv: tuple, stage: str) -> Hand:
    i, j = mv
    jt = hand.fingers[i].joints[j]
    return with_joint(hand, i, j, replace(jt, type="hinge" if jt.type == "coupled" else "coupled"))


def default_hinge(fingers: Sequence[Finger], deg: int, mm: int) -> PalmJoint:
    """A new palm joint for `fingers`: the hinge halfway between the wrist
    centre and their bases (kept inside the hinge range), along the palm
    (+z), so the section cups."""
    cy = float(np.mean([f.y for f in fingers])) - PALM_ANCHOR_MM[0]
    cz = float(np.mean([f.z for f in fingers])) - PALM_ANCHOR_MM[1]
    d = math.hypot(cy, cz)
    lo, hi = PALM_HINGE_DISTANCE_MM
    target = min(max(d / 2.0, lo), hi)
    s = target / d if d > 1e-9 else 0.0
    y = PALM_ANCHOR_MM[0] + int(math.ceil(abs(cy * s) / mm) * mm) * (1 if cy >= 0 else -1)
    z = PALM_ANCHOR_MM[1] + int(math.ceil(abs(cz * s) / mm) * mm) * (1 if cz >= 0 else -1)
    if math.hypot(y - PALM_ANCHOR_MM[0], z - PALM_ANCHOR_MM[1]) > hi:   # rounding outward crossed the top
        y = PALM_ANCHOR_MM[0] + int(cy * s / mm) * mm
        z = PALM_ANCHOR_MM[1] + int(cz * s / mm) * mm
    return PalmJoint(y=y, z=z, axis=(0, 0))


def _own_palm_joint_moves(hand: Hand, rules: Rules, stage: str) -> List[tuple]:
    if stage != "coarse" or len(hand.palm_joints) >= rules.max_palm_joints:
        return []
    count = [sum(1 for f in hand.fingers if f.palm_joint == k) for k in range(len(hand.palm_joints))]
    return [(i,) for i, f in enumerate(hand.fingers) if f.palm_joint < 0 or count[f.palm_joint] > 1]


def _own_palm_joint(hand: Hand, mv: tuple, stage: str) -> Hand:
    (i,) = mv
    f = hand.fingers[i]
    p = default_hinge([f], step_of(stage)[1], step_of(stage)[0])
    hand = replace(hand, palm_joints=hand.palm_joints + (p,))
    return with_finger(hand, i, replace(f, palm_joint=len(hand.palm_joints) - 1))


def _move_to_section_moves(hand: Hand, rules: Rules, stage: str) -> List[tuple]:
    if stage != "coarse":
        return []
    return [(i, k) for i, f in enumerate(hand.fingers) for k in range(-1, len(hand.palm_joints))
            if k != f.palm_joint]


def _move_to_section(hand: Hand, mv: tuple, stage: str) -> Hand:
    i, k = mv
    return drop_unused_palm_joints(with_finger(hand, i, replace(hand.fingers[i], palm_joint=k)))


def _remove_palm_joint_moves(hand: Hand, rules: Rules, stage: str) -> List[tuple]:
    if stage != "coarse":
        return []
    return [(k,) for k in range(len(hand.palm_joints))]


def _remove_palm_joint(hand: Hand, mv: tuple, stage: str) -> Hand:
    (k,) = mv
    fs = tuple(replace(f, palm_joint=-1) if f.palm_joint == k else f for f in hand.fingers)
    return drop_unused_palm_joints(replace(hand, fingers=fs))


OPERATORS: Tuple[Operator, ...] = (
    # parameters (both stages)
    Operator("move_finger", "move a finger", "Move one finger's base along the palm (y or z) by one step.",
             False, _move_finger_moves, _move_finger),
    Operator("turn_finger", "turn a finger", "Turn one finger's facing in the palm's plane by one step.",
             False, _finger_param_moves, _turn_finger),
    Operator("tilt_finger", "tilt a finger", "Tilt one finger out of the palm's plane by one step.",
             False, _finger_param_moves, _tilt_finger),
    Operator("length", "lengthen or shorten a link",
             "Change one link's length by one step (links are 0 mm or 15-90 mm; fingertips 10-90 mm).",
             False, _joint_moves, _length),
    Operator("axis_az", "turn a joint axis", "Turn one joint's axis around the link by one step.",
             False, _joint_moves, _axis_az),
    Operator("axis_el", "tip a joint axis", "Tip one joint's axis toward or away from the link by one step.",
             False, _joint_moves, _axis_el),
    Operator("move_hinge", "move a palm hinge", "Move one palm joint's hinge along the palm by one step.",
             False, _hinge_moves, _move_hinge),
    Operator("hinge_az", "turn a palm hinge", "Turn one palm joint's axis in the palm's plane by one step.",
             False, _hinge_axis_moves, _hinge_az),
    Operator("hinge_el", "tip a palm hinge", "Tip one palm joint's axis out of the palm's plane by one step.",
             False, _hinge_axis_moves, _hinge_el),
    # structure (coarse stage)
    Operator("add_finger", "add a finger",
             "Add a one-joint finger (a bending joint, 40 mm) at a free spot on the palm, pointing outward.",
             True, _add_finger_moves, _add_finger),
    Operator("remove_finger", "remove a finger",
             "Remove one whole finger; a palm joint left without a finger goes too.",
             True, _remove_finger_moves, _remove_finger),
    Operator("split", "add a joint (split a link)",
             "Split one link in two with a new joint; the finger keeps its length.", True, _split_moves, _split),
    Operator("merge", "remove a joint (merge two links)",
             "Remove one joint; the links on either side become one, so the finger keeps its length.",
             True, _merge_moves, _merge),
    Operator("couple", "couple or uncouple a joint",
             "Make a joint follow the joint before it (x1.1), or move on its own again.",
             True, _couple_moves, _couple),
    Operator("own_palm_joint", "give a finger its own palm joint",
             "Put one finger on a new hinged palm section (hinge halfway to the wrist, along the palm).",
             True, _own_palm_joint_moves, _own_palm_joint),
    Operator("move_to_section", "move a finger to another palm part",
             "Move one finger onto another palm joint's section, or back onto the main palm.",
             True, _move_to_section_moves, _move_to_section),
    Operator("remove_palm_joint", "remove a palm joint",
             "Remove one palm joint; its fingers return to the main palm.",
             True, _remove_palm_joint_moves, _remove_palm_joint),
)
OPERATOR_BY_NAME: Dict[str, Operator] = {op.name: op for op in OPERATORS}
INVERSES: Dict[str, str] = {
    "add_finger": "remove_finger", "split": "merge", "couple": "couple",
    "own_palm_joint": "remove_palm_joint", "move_to_section": "move_to_section",
}


def operators_for(stage: str) -> Tuple[Operator, ...]:
    return tuple(op for op in OPERATORS if stage == "coarse" or not op.structural)


def candidates(hand: Hand, op: Operator, rules: Rules = EVOLUTION_RULES, stage: str = "coarse") -> List[tuple]:
    """Every application of `op` to `hand` whose result keeps `rules`."""
    if op.structural and stage != "coarse":
        return []
    pv = len(check(hand, rules))
    out = []
    for mv in op.moves(hand, rules, stage):
        child = op.apply(hand, mv, stage)
        if child != hand and _admissible(child, pv, rules):
            out.append(mv)
    return out


def can_act(hand: Hand, op: Operator, rules: Rules = EVOLUTION_RULES, stage: str = "coarse") -> bool:
    if op.structural and stage != "coarse":
        return False
    pv = len(check(hand, rules))
    for mv in op.moves(hand, rules, stage):
        child = op.apply(hand, mv, stage)
        if child != hand and _admissible(child, pv, rules):
            return True
    return False


def apply_operator(hand: Hand, name: str, rng: np.random.Generator, rules: Rules = EVOLUTION_RULES,
                   stage: str = "coarse") -> Optional[Tuple[Hand, tuple]]:
    """One random application of operator `name`: `(child, move)`, or None
    if it cannot act. A split's new joint kind is drawn with the commercial
    mix; every other choice is uniform."""
    op = OPERATOR_BY_NAME[name]
    cands = candidates(hand, op, rules, stage)
    if not cands:
        return None
    if name == "split":
        kinds = [c[3] for c in cands]
        w = np.array([KIND_MIX.get(k, P_SLIDING) for k in kinds], dtype=float)
        mv = cands[int(rng.choice(len(cands), p=w / w.sum()))]
    else:
        mv = _pick(rng, cands)
    return op.apply(hand, mv, stage), mv


def mutate(hand: Hand, rng: np.random.Generator, rules: Rules = EVOLUTION_RULES,
           stage: str = "coarse") -> Tuple[Hand, str, tuple]:
    """One random mutation: an operator chosen uniformly among those that can
    act, then one of its applications. Returns `(child, operator, move)`."""
    ops = [op for op in operators_for(stage) if can_act(hand, op, rules, stage)]
    if not ops:
        raise ValueError("no operator can act on this hand under these rules")
    op = _pick(rng, ops)
    child, mv = apply_operator(hand, op.name, rng, rules, stage)
    return child, op.name, mv


# --------------------------------------------------------------------------
# Random hands
# --------------------------------------------------------------------------


def _draw_kind(rng: np.random.Generator, rules: Rules) -> str:
    kinds = list(KIND_MIX)
    w = [KIND_MIX[k] for k in kinds]
    if "sliding" in rules.joint_types:
        kinds.append("sliding")
        w.append(P_SLIDING)
    w = np.asarray(w) / np.sum(w)
    return kinds[int(rng.choice(len(kinds), p=w))]


def _draw_finger_joints(rng: np.random.Generator, rules: Rules, stage: str) -> Tuple[Joint, ...]:
    """1 to the rules' maximum joints, uniformly; each joint exactly on a kind
    drawn with the commercial mix; joints after the first coupled with
    probability `P_COUPLED`; links uniform over 15-90 mm on the stage grid
    (0 mm with probability `P_ZERO_LINK`), the fingertip link over 10-90 mm,
    each drawn among the lengths that leave the rest of the finger room under
    the finger-length cap."""
    mm = step_of(stage)[0]
    n = int(rng.integers(1, rules.max_joints + 1))
    budget = rules.max_finger_length_mm
    joints: List[Joint] = []
    for j in range(n):
        is_tip = j == n - 1
        cap = min(LINK_MAX_MM, budget - (0 if is_tip else TIP_MIN_MM))   # the fingertip still needs 10 mm
        if is_tip:
            values = _grid(TIP_MIN_MM, cap, mm) or [TIP_MIN_MM]
            length = _pick(rng, values)
        else:
            values = _grid(LINK_MIN_MM, cap, mm)
            length = 0 if (not values or rng.random() < P_ZERO_LINK) else _pick(rng, values)
        budget -= length
        kind = _draw_kind(rng, rules)
        jtype = "sliding" if kind == "sliding" else "hinge"
        if (jtype == "hinge" and j > 0 and "coupled" in rules.joint_types
                and joints[-1].type in REVOLUTE_TYPES and rng.random() < P_COUPLED):
            jtype = "coupled"
        joints.append(Joint(jtype, KIND_AXIS.get(kind, (0, 0)), length))
    return tuple(joints)


def _snap(v: float, step: int) -> int:
    return int(round(v / step) * step)


def random_hand(rng: np.random.Generator, rules: Rules = EVOLUTION_RULES, stage: str = "coarse") -> Hand:
    """A random hand within `rules`, on the stage's grid, uniform within the
    grammar's own limits (no hand-shaped template):

    - 2 to the rules' maximum fingers, uniformly;
    - each base uniform over the free grid points of the mount area (the
      ring 10-160 mm around the wrist centre, in the plate), so the spacing
      rule holds by construction;
    - facing uniform over the full circle, tilt uniform over -30 to +90 deg;
    - joints and links as `_draw_finger_joints` (kinds with the commercial
      mix);
    - with probability `P_PALM_JOINT`, one finger, chosen uniformly, on its
      own palm joint (`default_hinge`).
    """
    mm, deg = step_of(stage)
    n = int(rng.integers(rules.min_fingers, rules.max_fingers + 1))
    facings = _grid(0, 360 - deg, deg)
    tilts = _grid(TILT_RANGE_DEG[0], TILT_RANGE_DEG[1], deg)
    fingers: List[Finger] = []
    for _ in range(n):
        spots = _free_spots(Hand(fingers=tuple(fingers)) if fingers else Hand(fingers=()), rules, stage)
        y, z = _pick(rng, spots)
        fingers.append(Finger(y=y, z=z, facing=_pick(rng, facings), tilt=_pick(rng, tilts),
                              joints=_draw_finger_joints(rng, rules, stage)))
    hand = Hand(fingers=tuple(fingers))
    if rules.max_palm_joints > 0 and rng.random() < P_PALM_JOINT:
        k = int(rng.integers(len(fingers)))
        hand = Hand(fingers=tuple(replace(f, palm_joint=0) if i == k else f for i, f in enumerate(fingers)),
                    palm_joints=(default_hinge([fingers[k]], deg, mm),))
    problems = check(hand, rules)
    if problems:   # every draw keeps the limits by construction; this guards the construction itself
        raise AssertionError(f"random_hand broke its rules: {problems}")
    return hand
