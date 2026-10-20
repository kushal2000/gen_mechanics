"""Random hands and mutation operators for the hand grammar.

Every operator lists its applications, shuffles them and returns the first
whose child keeps the rules and C1 (Vatsal's method): nothing out of the
rules is ever produced. An operator with no such child cannot act (the
viewer hides it). A hand that already breaks the rules (a commercial hand
under the Feetech rules) or C1 can still be mutated as long as it stays in
the grammar and gets no worse.

One grid: every parameter steps by 10 mm (finger bases by the rule set's
pitch) or 30 degrees. A structural change keeps every link inside its
allowed lengths by moving it to the nearest allowed length
(`fit_lengths`).

Random hands are built finger by finger. Each finger's base (spot, facing,
tilt and first joint kind) is drawn from the placements whose base motor
clears the mount and the base motors already placed, its other joints and
links from their allowed values, and the finger is kept if the whole hand is
clear (C1); otherwise it is drawn again, up to 50 times, then the whole hand
again with the same number of fingers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import derive as dv
from . import viability as gvb
from .hand import (
    GRID_DEG,
    GRID_MM,
    KIND_AXIS,
    KIND_MIX,
    MIN_CUT_OFFSET_MM,
    REVOLUTE_TYPES,
    EVOLUTION_RULES,
    NO_RULES,
    Finger,
    Hand,
    Joint,
    PalmJoint,
    Rules,
    allowed_lengths,
    canonical_axis,
    check,
    cut_normal_yz,
    drop_empty_cut,
    faces_outward,
    finger_section,
    length_floor_mm,
    shifted,
    tilts_for,
    with_finger,
    with_joint,
)

P_PALM_JOINT = 0.5
"""Share of random hands drawn with a palm cut (when the rules allow one)."""
MAX_FINGER_ATTEMPTS = 50
MAX_HAND_ATTEMPTS = 20


def _pick(rng: np.random.Generator, values: Sequence):
    return values[int(rng.integers(len(values)))]


def _admissible(child: Hand, parent_violations: Optional[int], rules: Rules) -> bool:
    v = check(child, rules)
    if parent_violations is None or parent_violations == 0:
        return not v
    return len(v) <= parent_violations and not check(child, NO_RULES)


@lru_cache(maxsize=4096)
def _c1_worst(hand: Hand) -> float:
    return gvb.c1_self_overlap(hand).worst_mm


def _c1_keeps(child: Hand, parent_worst: float) -> bool:
    """C1 for a mutant: a parent that passes C1 needs a child that passes it;
    a parent that already fails it (a commercial or hand-made hand) needs a
    child no worse."""
    w = _c1_worst(child)
    if parent_worst <= gvb.C1_MAX_OVERLAP_MM + gvb.C1_TOLERANCE_MM:
        return w <= gvb.C1_MAX_OVERLAP_MM + gvb.C1_TOLERANCE_MM
    return w <= parent_worst + 1e-6


def clear(hand: Hand) -> bool:
    """Generation-time clearance: C1 (every two solids that do not touch by
    construction a wall apart at rest and at the start pose; no base motor
    in a cut's seam)."""
    return gvb.clear(hand)


# --------------------------------------------------------------------------
# Lengths
# --------------------------------------------------------------------------


def _nearest(values: Sequence[int], v: int) -> int:
    return min(values, key=lambda a: (abs(a - v), -a))


def fit_lengths(f: Finger, rules: Rules) -> Finger:
    """The finger with every link moved to its nearest allowed length (ties
    to the longer), from the palm out."""
    js = list(f.joints)
    for j in range(len(js)):
        allowed = allowed_lengths(replace(f, joints=tuple(js)), j, rules.buildable_links, rules.max_link_mm,
                                  rules.link_step_mm)
        if js[j].length not in allowed:
            js[j] = replace(js[j], length=_nearest(allowed, js[j].length))
    return replace(f, joints=tuple(js))


def step_length(f: Finger, j: int, s: int, rules: Rules) -> Optional[int]:
    """Link j's next allowed length up (s = +1) or down (s = -1): 10 mm
    steps, the floor down to 0 where a 0 mm link is allowed. None at the
    ends."""
    allowed = allowed_lengths(f, j, rules.buildable_links, rules.max_link_mm, rules.link_step_mm)
    L = f.joints[j].length
    if L in allowed:
        k = allowed.index(L) + s
        return allowed[k] if 0 <= k < len(allowed) else None
    beyond = [v for v in allowed if (v > L if s > 0 else v < L)]
    if not beyond:
        return None
    return min(beyond) if s > 0 else max(beyond)


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Operator:
    name: str
    label: str            # the viewer's button, plain words
    hover: str            # one line of hover text
    structural: bool
    moves: Callable[[Hand, Rules], List[tuple]]     # raw applications (before the rules)
    apply: Callable[[Hand, tuple, Rules], Hand]


def _finger_param_moves(hand: Hand, rules: Rules) -> List[tuple]:
    return [(i, s) for i in range(len(hand.fingers)) for s in (-1, 1)]


def _move_finger_moves(hand: Hand, rules: Rules) -> List[tuple]:
    return [(i, c, s) for i in range(len(hand.fingers)) for c in ("y", "z") for s in (-1, 1)]


def _move_finger(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    i, c, s = mv
    f = hand.fingers[i]
    return with_finger(hand, i, replace(f, **{c: getattr(f, c) + s * rules.base_pitch_mm}))


def _turn_finger(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    i, s = mv
    f = hand.fingers[i]
    return with_finger(hand, i, replace(f, facing=(f.facing + s * GRID_DEG) % 360))


def _tilt_finger(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    i, s = mv
    f = hand.fingers[i]
    return with_finger(hand, i, replace(f, tilt=f.tilt + s * GRID_DEG))


def _joint_moves(hand: Hand, rules: Rules) -> List[tuple]:
    return [(i, j, s) for i, f in enumerate(hand.fingers) for j in range(len(f.joints)) for s in (-1, 1)]


def _length(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    """Link j one allowed length up or down; a link after it that becomes
    too short (a shorter link can turn the next bend servo) grows to its
    shortest."""
    i, j, s = mv
    f = hand.fingers[i]
    out = step_length(f, j, s, rules)
    if out is None:
        return hand
    js = list(f.joints)
    js[j] = replace(js[j], length=out)
    return with_finger(hand, i, fit_lengths(replace(f, joints=tuple(js)), rules))


def step_axis(axis: Tuple[int, int], which: int, s: int, deg: int = GRID_DEG) -> Tuple[int, int]:
    az, el = axis
    if which == 0:
        return canonical_axis(az + s * deg, el)
    return canonical_axis(az, el + s * deg)


def _turn_axis(hand: Hand, mv: tuple, rules: Rules, which: int) -> Hand:
    i, j, s = mv
    f = hand.fingers[i]
    js = list(f.joints)
    js[j] = replace(js[j], axis=step_axis(js[j].axis, which, s, rules.axis_step_deg))
    return with_finger(hand, i, fit_lengths(replace(f, joints=tuple(js)), rules))


def _axis_az(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    return _turn_axis(hand, mv, rules, 0)


def _axis_el(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    return _turn_axis(hand, mv, rules, 1)


def _cut_moves(hand: Hand, rules: Rules) -> List[tuple]:
    return [(0, s) for s in (-1, 1)] if hand.palm_joints else []


def _move_cut(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    _, s = mv
    p = hand.palm_joints[0]
    return replace(hand, palm_joints=(replace(p, offset=p.offset + s * GRID_MM),))


def _turn_cut(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    _, s = mv
    p = hand.palm_joints[0]
    return replace(hand, palm_joints=(replace(p, facing=(p.facing + s * GRID_DEG) % 360),))


# ---- structural ---------------------------------------------------------


def base_spots(rules: Rules) -> List[Tuple[int, int]]:
    """Every grid point a base may take under `rules`."""
    lo, hi = rules.base_distance_mm
    step = rules.base_pitch_mm
    k = hi // step
    return [(y, z) for y in range(-k * step, k * step + 1, step) for z in range(-k * step, k * step + 1, step)
            if lo <= math.hypot(y, z) <= hi]


def facings_at(y: int, z: int, rules: Rules) -> List[int]:
    out = list(range(0, 360, GRID_DEG))
    if rules.outward_facing:
        out = [fa for fa in out if faces_outward(Finger(y=y, z=z, facing=fa))]
    return out


def _radial_facing(y: int, z: int) -> int:
    a = math.degrees(math.atan2(y, z))
    return int(round(a / GRID_DEG) * GRID_DEG) % 360


def _add_finger_moves(hand: Hand, rules: Rules) -> List[tuple]:
    """`(y, z, seed)` per free spot; the seed draws the new finger's joints
    (0 here; `iter_valid` gives each a random one when it has a generator)."""
    if len(hand.fingers) >= rules.max_fingers:
        return []
    taken = {(f.y, f.z) for f in hand.fingers}
    return [(y, z, 0) for (y, z) in base_spots(rules) if (y, z) not in taken]


def added_finger_joints(hand: Hand, rules: Rules) -> int:
    """How many joints an added finger has: the mean over the hand's
    fingers, rounded up (Martin, 2026-10-16), within the rules."""
    if not hand.fingers:
        return 1
    mean = sum(len(f.joints) for f in hand.fingers) / len(hand.fingers)
    return max(1, min(rules.max_joints, math.ceil(mean - 1e-9)))


def _add_finger(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    """A finger at spot (y, z), pointing straight out from the palm centre,
    tilt 0, with `added_finger_joints` joints drawn as a random hand's are
    (`_draw_joints`, from generator `seed`)."""
    y, z, seed = mv
    rng = np.random.default_rng(seed)
    base = Finger(y=y, z=z, facing=_radial_facing(y, z), tilt=0)
    joints = _draw_joints(rng, rules, _draw_kind(rng), base, n=added_finger_joints(hand, rules))
    if joints is None:
        return hand
    return replace(hand, fingers=hand.fingers + (replace(base, joints=joints),))


def _remove_finger_moves(hand: Hand, rules: Rules) -> List[tuple]:
    if len(hand.fingers) <= rules.min_fingers:
        return []
    return [(i,) for i in range(len(hand.fingers))]


def _remove_finger(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    (i,) = mv
    return drop_empty_cut(replace(hand, fingers=hand.fingers[:i] + hand.fingers[i + 1:]))


def _split_kinds(rules: Rules) -> List[str]:
    return list(KIND_MIX) + (["sliding"] if "sliding" in rules.joint_types else [])


def _split_moves(hand: Hand, rules: Rules) -> List[tuple]:
    out = []
    for i, f in enumerate(hand.fingers):
        if len(f.joints) >= rules.max_joints:
            continue
        for j, jt in enumerate(f.joints):
            for a in range(0, jt.length + 1, rules.link_step_mm):
                out.extend((i, j, a, k) for k in _split_kinds(rules))
    return out


def _split(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    """Split finger i's link j at `a` mm from joint j: a new joint of kind k
    starts the second part; then every link to its nearest allowed length
    (the finger keeps its length where the floors allow)."""
    i, j, a, k = mv
    f = hand.fingers[i]
    jt = f.joints[j]
    new = Joint("sliding" if k == "sliding" else "hinge", KIND_AXIS.get(k, (0, 0)), jt.length - a)
    js = list(f.joints)
    js[j] = replace(jt, length=a)
    js.insert(j + 1, new)
    return with_finger(hand, i, fit_lengths(replace(f, joints=tuple(js)), rules))


def _merge_moves(hand: Hand, rules: Rules) -> List[tuple]:
    return [(i, j) for i, f in enumerate(hand.fingers) for j in range(1, len(f.joints))]


def _merge(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    """Remove finger i's joint j (j >= 1): links j-1 and j become one (then
    every link to its nearest allowed length). A joint coupled to the removed
    one follows the joint before it instead (or becomes a hinge)."""
    i, j = mv
    f = hand.fingers[i]
    js = list(f.joints)
    js[j - 1] = replace(js[j - 1], length=js[j - 1].length + js[j].length)
    del js[j]
    if j < len(js) and js[j].type == "coupled" and js[j - 1].type not in REVOLUTE_TYPES:
        js[j] = replace(js[j], type="hinge")
    return with_finger(hand, i, fit_lengths(replace(f, joints=tuple(js)), rules))


def _couple_moves(hand: Hand, rules: Rules) -> List[tuple]:
    if "coupled" not in rules.joint_types:
        return []
    return [(i, j) for i, f in enumerate(hand.fingers) for j in range(1, len(f.joints))
            if f.joints[j].type in REVOLUTE_TYPES]


def _couple(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    i, j = mv
    jt = hand.fingers[i].joints[j]
    return with_joint(hand, i, j, replace(jt, type="hinge" if jt.type == "coupled" else "coupled"))


def cut_moves(hand: Hand) -> List[tuple]:
    """Every cut (facing, offset) with at least one finger on each side."""
    out = []
    pts = np.array([[f.y, f.z] for f in hand.fingers], dtype=float)
    for fa in range(0, 360, GRID_DEG):
        n = cut_normal_yz(PalmJoint(fa, 0))
        s = pts @ n
        top = int(math.ceil(float(s.max())))
        for off in range(MIN_CUT_OFFSET_MM, top + 1, GRID_MM):
            beyond = int((s > off).sum())
            if 0 < beyond < len(pts):
                out.append((fa, off))
    return out


def _add_cut_moves(hand: Hand, rules: Rules) -> List[tuple]:
    if hand.palm_joints or rules.max_palm_joints < 1:
        return []
    return cut_moves(hand)


def _add_cut(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    fa, off = mv
    return replace(hand, palm_joints=(PalmJoint(facing=fa, offset=off),))


def _remove_cut_moves(hand: Hand, rules: Rules) -> List[tuple]:
    return [(0,)] if hand.palm_joints else []


def _remove_cut(hand: Hand, mv: tuple, rules: Rules) -> Hand:
    return replace(hand, palm_joints=())


OPERATORS: Tuple[Operator, ...] = (
    # parameters
    Operator("move_finger", "move a finger", "Move one finger's base one grid step along the palm (y or z).",
             False, _move_finger_moves, _move_finger),
    Operator("turn_finger", "turn a finger", "Turn one finger 30 degrees in the palm's plane.",
             False, _finger_param_moves, _turn_finger),
    Operator("tilt_finger", "tilt a finger", "Tilt one finger 30 degrees out of the palm's plane.",
             False, _finger_param_moves, _tilt_finger),
    Operator("length", "lengthen or shorten a link",
             "Change one link to its next allowed length (10 mm steps; down from its shortest to 0 where allowed).",
             False, _joint_moves, _length),
    Operator("axis_az", "turn a joint axis", "Turn one joint's axis 30 degrees around the link.",
             False, _joint_moves, _axis_az),
    Operator("axis_el", "tip a joint axis", "Tip one joint's axis 30 degrees toward or away from the link.",
             False, _joint_moves, _axis_el),
    Operator("move_cut", "move the palm cut", "Move the palm's cut 10 mm.",
             False, _cut_moves, _move_cut),
    Operator("turn_cut", "turn the palm cut", "Turn the palm's cut 30 degrees.",
             False, _cut_moves, _turn_cut),
    # structure
    Operator("add_finger", "add a finger",
             "Add a finger at a free spot on the palm, pointing outward, with as many joints as the other fingers "
             "have on average (rounded up), its joints and links drawn as for a random hand.",
             True, _add_finger_moves, _add_finger),
    Operator("remove_finger", "remove a finger",
             "Remove one whole finger; a palm cut left with no finger on one side goes too.",
             True, _remove_finger_moves, _remove_finger),
    Operator("split", "add a joint (split a link)",
             "Split one link in two with a new joint; links grow to their shortest allowed length if needed.",
             True, _split_moves, _split),
    Operator("merge", "remove a joint (merge two links)",
             "Remove one joint; the links on either side become one (at most the longest link the rules allow).",
             True, _merge_moves, _merge),
    Operator("couple", "couple or uncouple a joint",
             "Make a joint follow the joint before it (x1.1), or move on its own again.",
             True, _couple_moves, _couple),
    Operator("add_cut", "cut the palm",
             "Cut the palm along a straight line; the part beyond it hinges on the rest.",
             True, _add_cut_moves, _add_cut),
    Operator("remove_cut", "remove the palm cut", "Join the palm into one piece again.",
             True, _remove_cut_moves, _remove_cut),
)
OPERATOR_BY_NAME: Dict[str, Operator] = {op.name: op for op in OPERATORS}
INVERSES: Dict[str, str] = {
    "add_finger": "remove_finger", "split": "merge", "couple": "couple", "add_cut": "remove_cut",
}


def recentre(hand: Hand, rules: Rules) -> Hand:
    """The hand moved on the base grid so that its palm's centre lies within
    half a base step of the mount (`Rules.mount_at_centre`): the same
    fingers, the mount at their centre. Unchanged under rules that leave the
    mount free."""
    if not hand.fingers or not rules.mount_at_centre:
        return hand
    p = rules.base_pitch_mm
    k = np.floor((dv.palm_centre_mm(hand) + p / 2.0) / p).astype(int)    # 0 when within half a step
    return shifted(hand, int(-k[0] * p), int(-k[1] * p)) if k.any() else hand


def _turned(hand: Hand, k: int, mirror: bool) -> Hand:
    """The hand mirrored (y -> -y) if `mirror`, then turned k x 90 degrees
    about the palm normal (a facing f goes to f + 90 k)."""
    def point(y: int, z: int) -> Tuple[int, int]:
        y = -y if mirror else y
        for _ in range(k % 4):
            y, z = z, -y
        return y, z

    def facing(f: int) -> int:
        return ((-f if mirror else f) + 90 * k) % 360

    fingers = []
    for f in hand.fingers:
        y, z = point(f.y, f.z)
        js = f.joints if not mirror else tuple(replace(j, axis=canonical_axis(180 - j.axis[0], j.axis[1]))
                                               for j in f.joints)
        fingers.append(replace(f, y=y, z=z, facing=facing(f.facing), joints=js))
    cuts = tuple(replace(p, facing=(((180 - p.facing) if mirror else p.facing) + 90 * k) % 360)
                 for p in hand.palm_joints)
    return replace(hand, fingers=tuple(fingers), palm_joints=cuts)


def _finger_key(f: Finger) -> tuple:
    return (f.y, f.z, f.facing, f.tilt, tuple((j.type, tuple(j.axis), j.length) for j in f.joints))


def _hand_key(hand: Hand) -> tuple:
    return (tuple(sorted(_finger_key(f) for f in hand.fingers)), tuple((p.facing, p.offset) for p in hand.palm_joints))


def canonical(hand: Hand, rules: Rules) -> Hand:
    """One hand for the eight that differ only by a turn of the palm by 90
    degrees or a mirror image, which the palm-up task cannot tell apart
    (Martin, 2026-10-20): the copy whose sorted fingers come first among
    those that keep the rules and C1 (a copy is checked again: a servo slid
    by a crossing keeps its side, so a mirror image is not always clear),
    its fingers in that order. A hand that breaks the rules is returned
    unchanged."""
    if check(hand, rules):
        return hand
    h = rules.base_pitch_mm / 2.0
    c = dv.palm_centre_mm(hand) if rules.mount_at_centre else None
    cands = []
    for mirror in (False, True):
        for k in range(4):
            if c is not None:            # the palm's centre turns with the hand
                y, z = (-c[0] if mirror else c[0]), c[1]
                for _ in range(k):
                    y, z = z, -y
                if not (-h <= y < h and -h <= z < h):
                    continue
            t = _turned(hand, k, mirror)
            cands.append((_hand_key(t), bool(k or mirror), t))
    for _key, moved, t in sorted(cands, key=lambda x: (x[0], x[1])):
        if not moved or (not check(t, rules) and gvb.clear(t)):
            return replace(t, fingers=tuple(sorted(t.fingers, key=_finger_key)))
    return replace(hand, fingers=tuple(sorted(hand.fingers, key=_finger_key)))


def _weighted_order(rng: np.random.Generator, weights: Sequence[float]) -> List[int]:
    """A random order in which item i comes early with weight w_i
    (Efraimidis-Spirakis keys u ** (1 / w))."""
    w = np.asarray(weights, dtype=float)
    keys = rng.random(len(w)) ** (1.0 / np.maximum(w, 1e-12))
    return list(np.argsort(-keys))


def iter_valid(hand: Hand, op: Operator, rules: Rules = EVOLUTION_RULES, rng: Optional[np.random.Generator] = None):
    """The applications of `op` whose child keeps `rules` and C1, as
    `(move, child)`, in shuffled order when `rng` is given (a split's new
    joint kind weighted by the commercial mix), else in list order. Lazy:
    a caller that needs one stops at the first (Vatsal's method)."""
    moves = op.moves(hand, rules)
    if not moves:
        return
    if rng is not None:
        if op.name == "split":
            order = _weighted_order(rng, [KIND_MIX.get(m[3], 0.01) for m in moves])
        else:
            order = list(rng.permutation(len(moves)))
        moves = [moves[int(k)] for k in order]
        if op.name == "add_finger":       # each new finger's joints from its own draw
            moves = [m[:-1] + (int(rng.integers(1 << 62)),) for m in moves]
    pv = len(check(hand, rules))
    parent_worst = None
    for mv in moves:
        child = op.apply(hand, mv, rules)
        centred = recentre(child, rules)          # the mount stays at the palm's centre
        if centred != child and (pv == 0 or _admissible(centred, pv, rules)):
            child = centred                       # (a hand already off-centre may also stay where it is)
        if child == hand or not _admissible(child, pv, rules):
            continue
        if parent_worst is None:
            parent_worst = _c1_worst(hand)
        if _c1_keeps(child, parent_worst):
            child = canonical(child, rules)       # turns and mirror images of a hand are one hand
            if child != hand:
                yield mv, child


def candidates(hand: Hand, op: Operator, rules: Rules = EVOLUTION_RULES) -> List[tuple]:
    """Every application of `op` to `hand` whose child keeps `rules` and C1
    (exhaustive: for tests and analysis; `apply_operator` stops at the first)."""
    return [mv for mv, _ in iter_valid(hand, op, rules)]


def can_act(hand: Hand, op: Operator, rules: Rules = EVOLUTION_RULES) -> bool:
    """Whether `op` has an application whose child keeps `rules` and C1
    (stops at the first)."""
    return next(iter_valid(hand, op, rules), None) is not None


def apply_operator(hand: Hand, name: str, rng: np.random.Generator, rules: Rules = EVOLUTION_RULES
                   ) -> Optional[Tuple[Hand, tuple]]:
    """Operator `name`'s candidates in random order; the first child that
    keeps `rules` and C1, as `(child, move)`, or None."""
    hit = next(iter_valid(hand, OPERATOR_BY_NAME[name], rules, rng), None)
    if hit is None:
        return None
    mv, child = hit
    return child, mv


def mutate(hand: Hand, rng: np.random.Generator, rules: Rules = EVOLUTION_RULES,
           tries: int = 1) -> Optional[Tuple[Hand, str, tuple]]:
    """One random mutation: an operator chosen uniformly among those with an
    application, its applications shuffled, the first child that keeps the
    rules and C1. Returns `(child, operator, move)`, or None when the operator
    has no such child; `tries` > 1 moves on to other operators (in random
    order) before giving up."""
    ops = [op for op in OPERATORS if op.moves(hand, rules)]
    for k in list(rng.permutation(len(ops)))[:max(1, tries)]:
        op = ops[int(k)]
        res = apply_operator(hand, op.name, rng, rules)
        if res is not None:
            return res[0], op.name, res[1]
    return None


# --------------------------------------------------------------------------
# Random hands
# --------------------------------------------------------------------------


class NoHand(RuntimeError):
    """No hand with the requested finger count fits the rules."""


def _base_placements(rules: Rules) -> Tuple[List[Tuple[int, int, int, int, str]], np.ndarray]:
    """Every (y, z, facing, tilt, first kind) a finger's base may take, and
    the draw's weight of each (the kind mix)."""
    return _base_placements_cached(rules)


@lru_cache(maxsize=16)
def _base_placements_cached(rules: Rules):
    out, w = [], []
    for (y, z) in base_spots(rules):
        for fa in facings_at(y, z, rules):
            for k, p in KIND_MIX.items():
                ts = tilts_for(k)            # a bend first: tilt 0 only
                for t in ts:
                    out.append((y, z, fa, t, k))
                    w.append(p / len(ts))    # the first joint's kinds keep the mix
    return out, np.array(w)


@lru_cache(maxsize=16)
def _base_cdf(rules: Rules) -> np.ndarray:
    _, w = _base_placements_cached(rules)
    c = np.cumsum(w)
    return c / c[-1]


@lru_cache(maxsize=200000)
def _base_box_corners(y: int, z: int, facing: int, tilt: int, kind: str) -> np.ndarray:
    """The base motor's corners (palm frame, metres) of a finger whose first
    joint has the kind's axis (no crossing)."""
    f = Finger(y=y, z=z, facing=facing, tilt=tilt, joints=(Joint("hinge", KIND_AXIS[kind], 90),))
    return dv.base_motor_boxes_mm(f)[0].corners() * dv.MM


@lru_cache(maxsize=200000)
def _clears_mount(y: int, z: int, facing: int, tilt: int, kind: str) -> bool:
    C = _base_box_corners(y, z, facing, tilt, kind)
    return dv.gjk_distance(C, dv.mount_points_m()) / dv.MM >= gvb.CLEARANCE_MM - gvb.C1_TOLERANCE_MM


def _base_clear(cand: Tuple[int, int, int, int, str], placed: List[np.ndarray]) -> bool:
    if not _clears_mount(*cand):
        return False
    C = _base_box_corners(*cand)
    c = C.mean(axis=0)
    r = float(np.max(np.linalg.norm(C - c, axis=1)))
    for P in placed:
        pc = P.mean(axis=0)
        if np.linalg.norm(pc - c) > r + float(np.max(np.linalg.norm(P - pc, axis=1))) + 0.01:
            continue
        if dv.gjk_distance(C, P) / dv.MM < gvb.CLEARANCE_MM - gvb.C1_TOLERANCE_MM:
            return False
    return True


def _draw_base(rng: np.random.Generator, rules: Rules, placed: List[np.ndarray],
               taken: set) -> Optional[Tuple[int, int, int, int, str]]:
    """A base drawn from the placements whose motor clears the mount and the
    base motors placed so far (each placement weighted by its kind's share):
    placements drawn by weight until one fits (the same distribution as
    drawing from the ones that fit), or, if few fit, the first that fits in a
    weighted random order."""
    cands, w = _base_placements(rules)
    cdf = _base_cdf(rules)
    for _ in range(200):
        c = cands[int(np.searchsorted(cdf, rng.random(), side="right"))]
        if (c[0], c[1]) not in taken and _base_clear(c, placed):
            return c
    for k in _weighted_order(rng, w):
        c = cands[int(k)]
        if (c[0], c[1]) in taken:
            continue
        if _base_clear(c, placed):
            return c
    return None


def _draw_kind(rng: np.random.Generator) -> str:
    kinds = list(KIND_MIX)
    w = np.array([KIND_MIX[k] for k in kinds])
    return kinds[int(rng.choice(len(kinds), p=w / w.sum()))]


def _shortest_rest_mm(base: Finger, js: List[Joint], j: int, v: int, rules: Rules) -> int:
    """The links after link j at their shortest, link j at `v` (j = -1: the
    whole finger), each from the palm out (a short link can turn the next
    bend servo, which changes the floors after it)."""
    js = list(js)
    if j >= 0:
        js[j] = replace(js[j], length=v)
    total = 0
    for k in range(j + 1, len(js)):
        L = length_floor_mm(replace(base, joints=tuple(js)), k, rules.buildable_links)
        js[k] = replace(js[k], length=L)
        total += L
    return total


def _draw_joints(rng: np.random.Generator, rules: Rules, first: str, base: Finger,
                 n: Optional[int] = None) -> Optional[Tuple[Joint, ...]]:
    """`n` joints (default 1 to the rules' maximum, uniformly); the first of
    the base's kind, the others drawn with the commercial mix, every axis
    exactly on its kind; each link uniform over its allowed lengths, among
    those that leave the rest of the finger room under the length cap."""
    fixed = n
    for _ in range(20):
        n = fixed if fixed is not None else int(rng.integers(1, rules.max_joints + 1))
        kinds = [first] + [_draw_kind(rng) for _ in range(n - 1)]
        js = [Joint("hinge", KIND_AXIS[k], rules.max_link_mm) for k in kinds]
        if _shortest_rest_mm(base, js, -1, 0, rules) > rules.max_finger_length_mm:
            continue
        budget = rules.max_finger_length_mm
        ok = True
        for j in range(n):
            f = replace(base, joints=tuple(js))
            values = [v for v in allowed_lengths(f, j, rules.buildable_links, rules.max_link_mm, rules.link_step_mm)
                      if v + _shortest_rest_mm(base, js, j, v, rules) <= budget]
            if not values:
                ok = False
                break
            L = _pick(rng, values)
            js[j] = replace(js[j], length=L)
            budget -= L
        if ok:
            return tuple(js)
    return None


def random_hand(rng: np.random.Generator, rules: Rules = EVOLUTION_RULES) -> Hand:
    """A random hand within `rules`, uniform within its limits (no
    hand-shaped template), built finger by finger:

    - 2 to the rules' maximum fingers, uniformly; the count is kept;
    - each finger's base (grid spot, facing, tilt and first joint kind) is
      drawn from the placements whose base motor clears the mount and the
      base motors already placed, weighted by the kind mix (69/27/4); its
      other joints and links as `_draw_joints`;
    - a finger is kept if the hand stays clear (C1); otherwise its base and
      joints are drawn again, up to 50 times, then the whole hand again with
      the same finger count (up to 20 times; `NoHand` after that);
    - the hand is moved on the base grid so the mount sits at the palm's
      centre (`recentre`); if it then breaks a rule or C1, it is drawn again;
    - with probability `P_PALM_JOINT` (when the rules allow a palm joint) a
      cut drawn uniformly among those that leave a finger on each side and
      keep the hand clear (none: the hand is drawn again).
    """
    n = int(rng.integers(rules.min_fingers, rules.max_fingers + 1))
    want_cut = rules.max_palm_joints > 0 and rng.random() < P_PALM_JOINT
    for _ in range(MAX_HAND_ATTEMPTS):
        hand = _draw_fingers(rng, rules, n)
        if hand is None:
            continue
        hand = recentre(hand, rules)          # the mount at the palm's centre
        if check(hand, rules) or not clear(hand):
            continue
        if want_cut:
            cut = None
            moves = cut_moves(hand)
            for k in rng.permutation(len(moves)):
                cand = _add_cut(hand, moves[int(k)], rules)
                if not check(cand, rules) and clear(cand):
                    cut = cand
                    break
            if cut is None:
                continue
            hand = cut
        problems = check(hand, rules)
        if problems:   # every draw keeps the limits by construction; this guards the construction itself
            raise AssertionError(f"random_hand broke its rules: {problems}")
        return canonical(hand, rules)     # turns and mirror images of a hand are one hand
    raise NoHand(f"no {n}-finger hand under these rules after {MAX_HAND_ATTEMPTS} attempts")


def _draw_fingers(rng: np.random.Generator, rules: Rules, n: int) -> Optional[Hand]:
    fingers: List[Finger] = []
    placed: List[np.ndarray] = []
    for _ in range(n):
        for _ in range(MAX_FINGER_ATTEMPTS):
            base = _draw_base(rng, rules, placed, {(f.y, f.z) for f in fingers})
            if base is None:
                return None
            y, z, fa, t, kind = base
            stub = Finger(y=y, z=z, facing=fa, tilt=t)
            joints = _draw_joints(rng, rules, kind, stub)
            if joints is None:
                continue
            f = replace(stub, joints=joints)
            hand = Hand(fingers=tuple(fingers) + (f,))
            if gvb.clear(hand, focus_finger=len(fingers)):
                fingers.append(f)
                placed.extend(b.corners() * dv.MM for b in dv.base_motor_boxes_mm(f))
                break
        else:
            return None
    return Hand(fingers=tuple(fingers))
