"""How many hands a rule set allows, counted exactly (no geometry): the size
of the space random hands and mutation search before the clearance check.

A finger is a base spot, a facing (outward if the rules ask), a tilt and a
chain of joints, each axis from an axis set and each link from its allowed
lengths (`hand.allowed_lengths`), the finger within the length cap; the
joint types the rules allow multiply each joint (a coupled joint only after
a hinge or coupled one). A finger whose first joint is a bend hinge has tilt
0 (`hand.bend_first`). A hand is a set of fingers on distinct spots, so
finger order does not count. A palm cut multiplies a hand by at most one
plus the number of cut lines (not included).

The count leaves out the centre rule (the mount at the palm's centre, which
depends on the servos' footprints); `experiments/E16_grammar_rebuild/ours/
search_space.py` estimates how much of the space keeps it and passes C1 and
C2.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

from . import motor as mt
from .hand import (GRID_DEG, KIND_AXIS, TILT_RANGE_DEG, Rules, axis_unit, canonical_axis)

BEND = KIND_AXIS["flexion"]

AXES_KINDS: Tuple[Tuple[int, int], ...] = (KIND_AXIS["flexion"], KIND_AXIS["abduction"], KIND_AXIS["roll"])
AXES_ALL: Tuple[Tuple[int, int], ...] = tuple(sorted({canonical_axis(az, el) for az in range(0, 180, GRID_DEG)
                                                      for el in range(-90, 91, GRID_DEG)}))


def _elementary(values: Sequence[int], k: int) -> int:
    e = [1] + [0] * k
    for v in values:
        for m in range(k, 0, -1):
            e[m] += e[m - 1] * v
    return e[k]


@lru_cache(maxsize=64)
def chains_by_first(rules: Rules, axes: Tuple[Tuple[int, int], ...]) -> Tuple[Tuple[int, int], ...]:
    """Per joint count (1 to the rules' maximum): the number of joint chains
    whose first joint is a bend hinge, and of the others."""
    U = [axis_unit(a) for a in axes]
    buildable = rules.buildable_links
    extra_types = [t for t in ("sliding",) if t in rules.joint_types]
    coupled = "coupled" in rules.joint_types
    step = 10
    top = rules.max_link_mm

    def links(i: int, j: int, prev_zero: bool, turned_i: bool, bend_j: bool) -> List[Tuple[int, bool]]:
        """(length, joint j's servo turned) for the links after joint i before
        joint j: a bend servo turns when its link is too short for it to lie
        along the link (`hand.turned_servo`)."""
        if not buildable:
            return [(L, False) for L in [0] + list(range(10, top + 1, step))]
        std = mt.link_floor_mm(U[i], U[j], turned_i, False)
        lo = mt.link_floor_mm(U[i], U[j], turned_i, True) if bend_j else std
        out = [(L, L < std) for L in range(lo, top + 1, step)]
        return ([(0, False)] + out) if (not prev_zero and mt.crossing_ok(U[i], U[j])) else out

    def tips(i: int, turned_i: bool) -> List[int]:
        lo = mt.tip_floor_mm(U[i], turned_i) if buildable else 10
        return list(range(lo, top + 1, step))

    def types(revolute_before: bool) -> List[Tuple[str, bool]]:
        """(type, revolute) choices for a joint whose predecessor is
        revolute or not (the first joint: no predecessor)."""
        out = [("hinge", True)] + [(t, False) for t in extra_types]
        if coupled and revolute_before:
            out.append(("coupled", True))
        return out

    @lru_cache(maxsize=None)
    def rec(left: int, i: int, prev_zero: bool, budget: int, revolute: bool, turned_i: bool) -> int:
        """Ways to finish a finger from joint i (`left` joints to the end;
        `turned_i`: its servo is turned)."""
        if left == 1:
            return sum(1 for t in tips(i, turned_i) if t <= budget)
        total = 0
        for j in range(len(axes)):
            for _, rev in types(revolute):
                for L, turned_j in links(i, j, prev_zero, turned_i, rev and mt.is_bend(U[j])):
                    if L <= budget:
                        total += rec(left - 1, j, L == 0, budget - L, rev, turned_j)
        return total

    out = []
    for n in range(1, rules.max_joints + 1):
        bend = other = 0
        for i in range(len(axes)):
            for t, rev in types(False):
                c = rec(n, i, False, rules.max_finger_length_mm, rev, False)
                if t == "hinge" and axes[i] == BEND:
                    bend += c
                else:
                    other += c
        out.append((bend, other))
    return tuple(out)


def chains(rules: Rules, axes: Tuple[Tuple[int, int], ...]) -> Tuple[int, ...]:
    """The number of joint chains per joint count (1 to the rules' maximum)."""
    return tuple(b + o for b, o in chains_by_first(rules, axes))


def fingers_per_spot(rules: Rules, axes: Tuple[Tuple[int, int], ...] = AXES_KINDS) -> List[int]:
    """Per base spot: its facings times (the chains with a bend first, at
    tilt 0, plus the others at every tilt)."""
    from .operators import base_spots, facings_at

    tilts = len(range(TILT_RANGE_DEG[0], TILT_RANGE_DEG[1] + 1, GRID_DEG))
    rows = chains_by_first(rules, axes)
    per = sum(b for b, _ in rows) + tilts * sum(o for _, o in rows)
    return [len(facings_at(y, z, rules)) * per for (y, z) in base_spots(rules)]


def hands_log10(rules: Rules, axes: Tuple[Tuple[int, int], ...] = AXES_KINDS) -> Dict[int, float]:
    """log10 of the number of hands per finger count (no palm cut): sets of
    fingers on distinct spots."""
    o = fingers_per_spot(rules, axes)
    return {n: math.log10(_elementary(o, n))
            for n in range(rules.min_fingers, rules.max_fingers + 1) if _elementary(o, n) > 0}


def summary(rules: Rules) -> str:
    """One line for the viewer: hands with joints on the three kinds (four
    fingers, and any count), and with all 31 axis directions."""
    k = hands_log10(rules, AXES_KINDS)
    a = hands_log10(rules, AXES_ALL)
    total_k = math.log10(sum(10 ** v for v in k.values()))
    total_a = math.log10(sum(10 ** v for v in a.values()))
    four = f"10^{k[4]:.0f} four-finger hands, " if 4 in k else ""
    return (f"search space: {four}10^{total_k:.0f} in all with joints on the three kinds; "
            f"10^{total_a:.0f} with all 31 axis directions (before the centre rule and the clearance check)")
