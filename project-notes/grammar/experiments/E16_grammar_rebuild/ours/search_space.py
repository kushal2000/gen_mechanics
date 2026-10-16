"""How many hands the Feetech rules allow (2026-10-14; fingers at most 225 mm
since 2026-10-16), and how many of them
pass the clearance check (C1) and C2.

    PYTHONPATH=. python3 project-notes/grammar/experiments/E16_grammar_rebuild/ours/search_space.py [samples]

Counted exactly (no geometry): every finger is a base spot (20), a facing
(outward at that spot), a tilt (4: 0-90 degrees since 2026-10-16; 0 only
when the first joint is a bend) and a chain of 1-5 joints, each axis from
the axis set and each link from its allowed lengths (`hand.allowed_lengths`:
the servo floors to 90 mm, 0 for crossings, fingertips from their floor),
the finger at most the rules' cap (225 mm since 2026-10-16; 250 before). A hand is 2-6 such fingers on distinct spots
(finger order does not count). A palm cut multiplies a hand by at most 1 +
72 (12 directions x offsets 30-80 mm), counted separately.

Two axis sets: the three kinds (what random hands start from) and all 31
grid directions (what mutation reaches).

The count leaves out the centre rule (since 2026-10-16 the mount sits at the
palm's centre, within half a base step; it depends on the servos'
footprints). A uniform sample of the count (fixed finger number, no cut) is
checked for it, then with C1 and C2, which estimates how much of the space
is usable.
"""

from __future__ import annotations

import json
import math
import sys
from functools import lru_cache

import numpy as np

from hand_sampler.grammar import motor as mt
from hand_sampler.grammar import operators as ops
from hand_sampler.grammar import viability as V
from hand_sampler.grammar.hand import (FEETECH_RULES, KIND_AXIS, TILT_RANGE_DEG, Finger, Hand,
                                       Joint, axis_unit, canonical_axis, check)

R = FEETECH_RULES
SPOTS = ops.base_spots(R)
TILTS = list(range(TILT_RANGE_DEG[0], TILT_RANGE_DEG[1] + 1, 30))
FACINGS = [len(ops.facings_at(y, z, R)) for (y, z) in SPOTS]
BEND = KIND_AXIS["flexion"]
AXES_KINDS = [KIND_AXIS["flexion"], KIND_AXIS["abduction"], KIND_AXIS["roll"]]
AXES_ALL = sorted({canonical_axis(az, el) for az in range(0, 180, 30) for el in range(-90, 91, 30)})


def chain_counter(axes):
    U = [axis_unit(a) for a in axes]

    def lengths(i, j, prev_zero, turned_i):
        """(length, joint j's servo turned): a bend servo turns when its link is
        too short for it to lie along the link (`hand.turned_servo`)."""
        std = mt.link_floor_mm(U[i], U[j], turned_i, False)
        lo = mt.link_floor_mm(U[i], U[j], turned_i, True) if mt.is_bend(U[j]) else std
        out = [(L, L < std) for L in range(lo, R.max_link_mm + 1, 10)]
        if not prev_zero and mt.crossing_ok(U[i], U[j]):
            out = [(0, False)] + out
        return out

    def tips(i, turned_i):
        return list(range(mt.tip_floor_mm(U[i], turned_i), R.max_link_mm + 1, 10))

    @lru_cache(maxsize=None)
    def rec(left, i, prev_zero, budget, turned_i):
        """Ways to finish a finger whose current joint has axis i, `left`
        joints from it to the end; `turned_i`: its servo is turned."""
        if left == 1:
            return sum(1 for t in tips(i, turned_i) if t <= budget)
        total = 0
        for j in range(len(axes)):
            for L, tj in lengths(i, j, prev_zero, turned_i):
                if L <= budget:
                    total += rec(left - 1, j, L == 0, budget - L, tj)
        return total

    def tilts(i):
        """The tilts a finger whose first axis is axes[i] may take (a bend first: 0)."""
        return [0] if axes[i] == BEND else TILTS

    def count(n):
        """Chains of n joints, and chains times the tilts each may take."""
        c = [rec(n, i, False, R.max_finger_length_mm, False) for i in range(len(axes))]
        return sum(c), sum(ci * len(tilts(i)) for i, ci in enumerate(c))

    def sample(rng):
        """(tilt, chain) drawn uniformly among every chain of 1-5 joints and
        the tilts it may take."""
        opts = [(n, i) for n in range(1, 6) for i in range(len(axes))]
        w = np.array([rec(n, i, False, R.max_finger_length_mm, False) * len(tilts(i)) for n, i in opts], float)
        n, i = opts[int(rng.choice(len(opts), p=w / w.sum()))]
        tilt = tilts(i)[int(rng.integers(len(tilts(i))))]
        joints, prev_zero, budget, left, turned_i = [], False, R.max_finger_length_mm, n, False
        while left > 1:
            opts = [(j, L, tj) for j in range(len(axes)) for L, tj in lengths(i, j, prev_zero, turned_i) if L <= budget]
            w = np.array([rec(left - 1, j, L == 0, budget - L, tj) for j, L, tj in opts], float)
            j, L, tj = opts[int(rng.choice(len(opts), p=w / w.sum()))]
            joints.append(Joint("hinge", axes[i], L))
            prev_zero, budget, left, i, turned_i = L == 0, budget - L, left - 1, j, tj
        t = [t for t in tips(i, turned_i) if t <= budget]
        joints.append(Joint("hinge", axes[i], t[int(rng.integers(len(t)))]))
        return tilt, tuple(joints)

    return count, sample


def elementary(values, k):
    """e_k of `values` (exact integers)."""
    e = [1] + [0] * k
    for v in values:
        for m in range(k, 0, -1):
            e[m] += e[m - 1] * v
    return e[k]


def sample_spots(n, rng):
    """n distinct spots, drawn with probability proportional to the product
    of their facing counts (uniform over the counted hands: every spot has
    the same fingers per facing)."""
    left = list(range(len(SPOTS)))
    out = []
    for k in range(n, 0, -1):
        w = np.array([FACINGS[s] * elementary([FACINGS[t] for t in left if t > s], k - 1) for s in left], float)
        s = left[int(rng.choice(len(left), p=w / w.sum()))]
        out.append(s)
        left = [t for t in left if t > s]          # an unordered set: later picks come after s
    return out


def main(samples: int = 400) -> dict:
    result = {"spots": len(SPOTS), "facings_per_spot": sorted(set(FACINGS)), "tilts": TILTS,
              "cut_factor_at_most": 1 + 12 * 6}
    rng = np.random.default_rng(20261014)
    for name, axes in (("kinds", AXES_KINDS), ("all_31_axes", AXES_ALL)):
        count, sample = chain_counter(axes)
        by_joints = [count(n)[0] for n in range(1, 6)]
        per_facing = sum(count(n)[1] for n in range(1, 6))       # fingers per spot and facing
        hands = {n: elementary([fa * per_facing for fa in FACINGS], n) for n in range(2, 7)}
        total = sum(hands.values())
        rec = {"chains_by_joint_count": by_joints, "fingers_per_spot_and_facing": per_facing,
               "log10_hands_by_fingers": {n: round(math.log10(v), 2) for n, v in hands.items()},
               "log10_hands_total": round(math.log10(total), 2)}
        # a uniform sample of 4-finger hands, checked
        centred = c1 = c2 = 0
        for _ in range(samples):
            spots = sample_spots(4, rng)
            fingers = []
            for s in spots:
                y, z = SPOTS[s]
                fa = ops.facings_at(y, z, R)
                tilt, joints = sample(rng)
                fingers.append(Finger(y=y, z=z, facing=fa[int(rng.integers(len(fa)))], tilt=tilt, joints=joints))
            h = Hand(fingers=tuple(fingers))
            v = check(h, R)
            assert all(x.startswith("palm centre") for x in v), v      # only the centre rule is not counted
            if v:
                continue
            centred += 1
            if V.clear(h):
                c1 += 1
                c2 += V.c2_workspace_overlap(h).ok
        rec["four_finger_sample"] = {"samples": samples, "mount_at_centre": centred / samples,
                                     "and_pass_c1": c1 / samples, "and_pass_c1_and_c2": c2 / samples}
        p = c2 / samples
        rec["log10_usable_four_finger_hands"] = round(math.log10(hands[4] * p), 2) if p > 0 else None
        result[name] = rec
    return result


if __name__ == "__main__":
    print(json.dumps(main(int(sys.argv[1]) if len(sys.argv) > 1 else 400), indent=1))
