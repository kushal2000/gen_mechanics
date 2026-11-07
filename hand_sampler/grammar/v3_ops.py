"""v3 mutation operators (7 Nov 2026), on hand_sampler/grammar/v3.py's link-only hands. Every child keeps the v3 rules
(mutant priors relaxed: v3.valid(mutant=True)) and the clearance, and is re-centred on its palm's centroid.

  move_finger    a finger's base one 20 mm grid step (y or z)
  turn_finger    a finger's facing +-30 deg
  tilt_finger    +-30 deg, only on a 1-DOF base with an abduction first, 0-60 deg
  length         one link one step within 30 / 40 / 50 mm (never a knuckle's 0 mm link)
  kind           one joint flexion <-> abduction (never a knuckle joint; tilt reset when a base turns flexion)
  add_finger     copy a finger (its joints, links, tilt) to a free grid spot at the same angle to the outward direction
  remove_finger  remove a finger (>= 2 left)
  add_joint      insert a joint (kind 50 / 50) carrying a new 30-50 mm link (<= 4 joints; never inside a knuckle)
  remove_joint   remove a joint and its link (>= 2 joints; never a knuckle joint)
  roll_finger    a finger's roll about its own length +-30 deg (the full circle), every joint axis turning with it
  base_type      every finger's base switched together: 1-DOF <-> FE->AbAd <-> AbAd->FE. To a knuckle: the first joint
                 becomes the co-located pair (+1 joint; a finger at 4 joints drops its last); FA <-> AF: the pair's
                 axes swap; to 1-DOF: the 0 mm joint goes (a finger left with 1 joint gets a 30 mm link appended)

`mutate(hand, rng)` picks an operator uniformly among those with moves, tries its moves in random order and returns the
first valid child that differs from the parent: (child, operator, move), or None.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable, List, NamedTuple, Optional, Tuple

import numpy as np

from . import hand as gh
from . import v3

GRID = v3.BASE_PITCH_MM
LINKS = v3.LINKS_MM


def _set_finger(hand: gh.Hand, i: int, f: gh.Finger) -> gh.Hand:
    fs = list(hand.fingers)
    fs[i] = f
    return gh.Hand(fingers=tuple(fs))


def _knuckle_len(f: gh.Finger) -> int:
    """How many joints at the base belong to a knuckle (2) or none (0)."""
    return 2 if v3.base_type(f) in ("knFA", "knAF") else 0


# ---- the operators: moves(hand) -> list, apply(hand, move) -> hand
def _move_moves(h):
    return [(i, dy, dz) for i in range(len(h.fingers)) for dy, dz in ((GRID, 0), (-GRID, 0), (0, GRID), (0, -GRID))]


def _move(h, mv):
    i, dy, dz = mv
    f = h.fingers[i]
    return _set_finger(h, i, replace(f, y=f.y + dy, z=f.z + dz))


def _turn_moves(h):
    return [(i, s) for i in range(len(h.fingers)) for s in (30, -30)]


def _turn(h, mv):
    i, s = mv
    f = h.fingers[i]
    return _set_finger(h, i, replace(f, facing=(f.facing + s) % 360))


def _tilt_moves(h):
    out = []
    for i, f in enumerate(h.fingers):
        if v3.base_type(f) == "kn1dof" and v3.kind_of(f, f.joints[0]) == "abd":
            out += [(i, s) for s in (30, -30) if 0 <= f.tilt + s <= v3.MAX_TILT_DEG]
    return out


def _tilt(h, mv):
    i, s = mv
    f = h.fingers[i]
    return _set_finger(h, i, replace(f, tilt=f.tilt + s))


def _length_moves(h):
    out = []
    for i, f in enumerate(h.fingers):
        for j, jt in enumerate(f.joints):
            if jt.length == 0:
                continue
            k = LINKS.index(jt.length)
            out += [(i, j, LINKS[k + s]) for s in (1, -1) if 0 <= k + s < len(LINKS)]
    return out


def _length(h, mv):
    i, j, L = mv
    f = h.fingers[i]
    js = list(f.joints)
    js[j] = replace(js[j], length=L)
    return _set_finger(h, i, replace(f, joints=tuple(js)))


def _kind_moves(h):
    return [(i, j) for i, f in enumerate(h.fingers) for j in range(_knuckle_len(f), len(f.joints))]


def _kind(h, mv):
    i, j = mv
    f = h.fingers[i]
    js = list(f.joints)
    new = "flex" if v3.kind_of(f, js[j]) == "abd" else "abd"
    js[j] = replace(js[j], axis=v3.axis_for(new, v3.roll_of(f)))
    g = replace(f, joints=tuple(js))
    tilt = f.tilt if v3.kind_of(g, js[0]) == "abd" else 0
    return _set_finger(h, i, replace(g, tilt=tilt))


def _add_finger_moves(h):
    if len(h.fingers) >= 6:
        return []
    taken = {(f.y, f.z) for f in h.fingers}
    spots = [(y, z) for y in range(-100, 101, GRID) for z in range(-100, 101, GRID)
             if (y, z) not in taken and 20 <= math.hypot(y, z) <= 100]
    return [(i, y, z) for i in range(len(h.fingers)) for (y, z) in spots]


def _add_finger(h, mv):
    i, y, z = mv
    f = h.fingers[i]
    rel = f.facing - math.degrees(math.atan2(f.y, f.z)) if (f.y or f.z) else 0.0   # angle to the outward direction
    fa = int(round((math.degrees(math.atan2(y, z)) + rel) / gh.GRID_DEG) * gh.GRID_DEG) % 360
    return gh.Hand(fingers=h.fingers + (replace(f, y=y, z=z, facing=fa),))


def _remove_finger_moves(h):
    return list(range(len(h.fingers))) if len(h.fingers) > 2 else []


def _remove_finger(h, i):
    return gh.Hand(fingers=tuple(f for k, f in enumerate(h.fingers) if k != i))


def _add_joint_moves(h):
    out = []
    for i, f in enumerate(h.fingers):
        if len(f.joints) >= v3.MAX_JOINTS:
            continue
        for pos in range(max(_knuckle_len(f), 1), len(f.joints) + 1):    # after the base (and the knuckle)
            out += [(i, pos, k, L) for k in ("flex", "abd") for L in LINKS]
    return out


def _add_joint(h, mv):
    i, pos, kind, L = mv
    f = h.fingers[i]
    js = list(f.joints)
    js.insert(pos, gh.Joint("hinge", v3.axis_for(kind, v3.roll_of(f)), L))
    return _set_finger(h, i, replace(f, joints=tuple(js)))


def _remove_joint_moves(h):
    out = []
    for i, f in enumerate(h.fingers):
        if len(f.joints) <= v3.MIN_JOINTS:
            continue
        lo = _knuckle_len(f)
        out += [(i, j) for j in range(lo, len(f.joints)) if not (lo == 0 and j == 0 and len(f.joints) < 2)]
    return out


def _remove_joint(h, mv):
    i, j = mv
    f = h.fingers[i]
    js = [jt for k, jt in enumerate(f.joints) if k != j]
    g = replace(f, joints=tuple(js))
    tilt = f.tilt if (js and v3.kind_of(g, js[0]) == "abd" and v3.base_type(g) == "kn1dof") else 0
    return _set_finger(h, i, replace(g, tilt=tilt))


def _base_type_moves(h):
    cur = v3.base_type(h.fingers[0])
    return [t for t in v3.BASE_TYPES if t != cur]


def _to_base(f: gh.Finger, target: str) -> gh.Finger:
    cur = v3.base_type(f)
    js = list(f.joints)
    r = v3.roll_of(f)
    pair = {"knFA": (v3.axis_for("flex", r), v3.axis_for("abd", r)), "knAF": (v3.axis_for("abd", r), v3.axis_for("flex", r))}
    if cur == target:
        return f
    if cur == "kn1dof":                                   # first joint -> the co-located pair (+1 joint)
        k1, k2 = pair[target]
        js = [gh.Joint("hinge", k1, 0), replace(js[0], axis=k2)] + js[1:]
        if len(js) > v3.MAX_JOINTS:
            js = js[: v3.MAX_JOINTS]
        return replace(f, joints=tuple(js), tilt=0)
    if target in pair:                                    # FA <-> AF: swap the pair's axes
        k1, k2 = pair[target]
        js[0], js[1] = replace(js[0], axis=k1), replace(js[1], axis=k2)
        return replace(f, joints=tuple(js), tilt=0)
    js = js[1:]                                           # knuckle -> 1-DOF: the 0 mm joint goes (-1 joint)
    if len(js) < v3.MIN_JOINTS:
        js.append(gh.Joint("hinge", v3.axis_for("flex", r), min(LINKS)))
    return replace(f, joints=tuple(js), tilt=0)


def _base_type(h, target):
    return gh.Hand(fingers=tuple(_to_base(f, target) for f in h.fingers))


def _roll_moves(h):
    return [(i, s) for i, f in enumerate(h.fingers) for s in (30, -30)]           # the full circle (wrapped)


def _roll(h, mv):
    i, s = mv
    f = h.fingers[i]
    js = tuple(replace(jt, axis=(int(jt.axis[0]) + s, int(jt.axis[1]))) for jt in f.joints)
    return _set_finger(h, i, v3.as_v3(replace(f, joints=js), v3.wrap_roll(v3.roll_of(f) + s)))


class Op(NamedTuple):
    name: str
    moves: Callable
    apply: Callable


OPERATORS: Tuple[Op, ...] = (
    Op("move_finger", _move_moves, _move), Op("turn_finger", _turn_moves, _turn), Op("tilt_finger", _tilt_moves, _tilt),
    Op("length", _length_moves, _length), Op("kind", _kind_moves, _kind),
    Op("add_finger", _add_finger_moves, _add_finger), Op("remove_finger", _remove_finger_moves, _remove_finger),
    Op("add_joint", _add_joint_moves, _add_joint), Op("remove_joint", _remove_joint_moves, _remove_joint),
    Op("roll_finger", _roll_moves, _roll), Op("base_type", _base_type_moves, _base_type),
)
OPERATOR_BY_NAME = {o.name: o for o in OPERATORS}


def apply_operator(hand: gh.Hand, name: str, rng: np.random.Generator, max_moves: int = 64
                   ) -> Optional[Tuple[gh.Hand, tuple]]:
    """Operator `name`'s moves in random order (at most `max_moves` tried); the first valid child, re-centred."""
    op = OPERATOR_BY_NAME[name]
    moves = op.moves(hand)
    for k in list(rng.permutation(len(moves)))[:max_moves]:
        mv = moves[int(k)]
        child = v3.centred(op.apply(hand, mv))
        if child != hand and v3.valid(child, mutant=True):
            return child, mv
    return None


def mutate(hand: gh.Hand, rng: np.random.Generator, tries: int = 3) -> Optional[Tuple[gh.Hand, str, tuple]]:
    """One mutation: an operator uniformly among those with moves, its moves shuffled, the first valid child;
    `tries` > 1 moves on to other operators before giving up."""
    ops = [op for op in OPERATORS if op.moves(hand)]
    for k in list(rng.permutation(len(ops)))[:max(1, tries)]:
        res = apply_operator(hand, ops[int(k)].name, rng)
        if res is not None:
            return res[0], ops[int(k)].name, res[1]
    return None


__all__ = ["OPERATORS", "OPERATOR_BY_NAME", "apply_operator", "mutate"]
