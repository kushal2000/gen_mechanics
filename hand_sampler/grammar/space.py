"""How many hands a rule set allows, counted exactly (no geometry): the size
of the space random hands and mutation search before the clearance check.

A finger is a base spot, a facing (outward if the rules ask), a tilt and a
chain of joints, each axis from an axis set and each link from its allowed
lengths (`hand.allowed_lengths`), the finger within the length cap; the
joint types the rules allow multiply each joint (a coupled joint only after
a hinge or coupled one). A hand is a set of fingers on distinct spots, so
finger order does not count. A palm cut multiplies a hand by at most one
plus the number of cut lines (not included).

`experiments/E16_grammar_rebuild/ours/search_space.py` also estimates how
much of the space passes C1 and C2 (about 3% of four-finger hands under the
Feetech rules with joints on the three kinds).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

from . import motor as mt
from .hand import (GRID_DEG, KIND_AXIS, LINK_MAX_MM, TILT_RANGE_DEG, Rules, axis_unit, canonical_axis)

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
def chains(rules: Rules, axes: Tuple[Tuple[int, int], ...]) -> Tuple[int, ...]:
    """The number of joint chains per joint count (1 to the rules' maximum)."""
    U = [axis_unit(a) for a in axes]
    buildable = rules.buildable_links
    extra_types = [t for t in ("sliding",) if t in rules.joint_types]
    coupled = "coupled" in rules.joint_types
    step = 10

    def links(i: int, j: int, prev_zero: bool) -> List[int]:
        if not buildable:
            return [0] + list(range(10, LINK_MAX_MM + 1, step))
        out = list(range(mt.link_floor_mm(U[i], U[j]), LINK_MAX_MM + 1, step))
        return ([0] + out) if (not prev_zero and mt.crossing_ok(U[i], U[j])) else out

    def tips(i: int) -> List[int]:
        lo = mt.tip_floor_mm(U[i]) if buildable else 10
        return list(range(lo, LINK_MAX_MM + 1, step))

    def types(revolute_before: bool) -> List[Tuple[str, bool]]:
        """(type, revolute) choices for a joint whose predecessor is
        revolute or not (the first joint: no predecessor)."""
        out = [("hinge", True)] + [(t, False) for t in extra_types]
        if coupled and revolute_before:
            out.append(("coupled", True))
        return out

    @lru_cache(maxsize=None)
    def rec(left: int, i: int, prev_zero: bool, budget: int, revolute: bool) -> int:
        if left == 1:
            return sum(1 for t in tips(i) if t <= budget)
        total = 0
        for j in range(len(axes)):
            for L in links(i, j, prev_zero):
                if L > budget:
                    continue
                for _, rev in types(revolute):
                    total += rec(left - 1, j, L == 0, budget - L, rev)
        return total

    out = []
    for n in range(1, rules.max_joints + 1):
        total = 0
        for i in range(len(axes)):
            for _, rev in types(False):
                total += rec(n, i, False, rules.max_finger_length_mm, rev)
        out.append(total)
    return tuple(out)


def orientations(rules: Rules) -> List[int]:
    """Per base spot, its facings times the tilts."""
    from .operators import base_spots, facings_at

    tilts = len(range(TILT_RANGE_DEG[0], TILT_RANGE_DEG[1] + 1, GRID_DEG))
    return [len(facings_at(y, z, rules)) * tilts for (y, z) in base_spots(rules)]


def hands_log10(rules: Rules, axes: Tuple[Tuple[int, int], ...] = AXES_KINDS) -> Dict[int, float]:
    """log10 of the number of hands per finger count (no palm cut)."""
    per_finger = sum(chains(rules, axes))
    o = orientations(rules)
    return {n: math.log10(_elementary(o, n) * per_finger ** n)
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
            f"10^{total_a:.0f} with all 31 axis directions (before the clearance check)")
