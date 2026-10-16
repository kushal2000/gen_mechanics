"""How many hands the Feetech rules allow (2026-10-14; fingers at most 225 mm
since 2026-10-16), and how many of them
pass the clearance check (C1) and C2.

    PYTHONPATH=. python3 project-notes/grammar/experiments/E16_grammar_rebuild/ours/search_space.py [samples]

Counted exactly (no geometry): every finger is a base spot (20), a facing
(outward at that spot), a tilt (5) and a chain of 1-5 joints, each axis from
the axis set and each link from its allowed lengths (`hand.allowed_lengths`:
the servo floors to 90 mm, 0 for crossings, fingertips from their floor),
the finger at most the rules' cap (225 mm since 2026-10-16; 250 before). A hand is 2-6 such fingers on distinct spots
(finger order does not count). A palm cut multiplies a hand by at most 1 +
72 (12 directions x offsets 30-80 mm), counted separately.

Two axis sets: the three kinds (what random hands start from) and all 31
grid directions (what mutation reaches).

Then a uniform sample of that count (fixed finger number, no cut) is checked
with C1 and C2, which estimates how much of the space is usable.
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
ORIENT = [len(ops.facings_at(y, z, R)) * len(TILTS) for (y, z) in SPOTS]
AXES_KINDS = [KIND_AXIS["flexion"], KIND_AXIS["abduction"], KIND_AXIS["roll"]]
AXES_ALL = sorted({canonical_axis(az, el) for az in range(0, 180, 30) for el in range(-90, 91, 30)})


def chain_counter(axes):
    U = [axis_unit(a) for a in axes]

    def lengths(i, j, prev_zero):
        out = list(range(mt.link_floor_mm(U[i], U[j]), R.max_link_mm + 1, 10))
        if not prev_zero and mt.crossing_ok(U[i], U[j]):
            out = [0] + out
        return out

    def tips(i):
        return list(range(mt.tip_floor_mm(U[i]), R.max_link_mm + 1, 10))

    @lru_cache(maxsize=None)
    def rec(left, i, prev_zero, budget):
        """Ways to finish a finger whose current joint has axis i, `left`
        joints from it to the end."""
        if left == 1:
            return sum(1 for t in tips(i) if t <= budget)
        total = 0
        for j in range(len(axes)):
            for L in lengths(i, j, prev_zero):
                if L <= budget:
                    total += rec(left - 1, j, L == 0, budget - L)
        return total

    def count(n):
        return sum(rec(n, i, False, R.max_finger_length_mm) for i in range(len(axes)))

    def sample(n, rng):
        """A chain drawn uniformly among the `count(n)` chains of n joints."""
        weights = [rec(n, i, False, R.max_finger_length_mm) for i in range(len(axes))]
        i = int(rng.choice(len(axes), p=np.array(weights, float) / sum(weights)))
        joints, prev_zero, budget, left = [], False, R.max_finger_length_mm, n
        while left > 1:
            opts = [(j, L) for j in range(len(axes)) for L in lengths(i, j, prev_zero) if L <= budget]
            w = np.array([rec(left - 1, j, L == 0, budget - L) for j, L in opts], float)
            j, L = opts[int(rng.choice(len(opts), p=w / w.sum()))]
            joints.append(Joint("hinge", axes[i], L))
            prev_zero, budget, left, i = L == 0, budget - L, left - 1, j
        t = [t for t in tips(i) if t <= budget]
        joints.append(Joint("hinge", axes[i], t[int(rng.integers(len(t)))]))
        return tuple(joints)

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
    of their orientation counts (uniform over the counted hands)."""
    left = list(range(len(SPOTS)))
    out = []
    for k in range(n, 0, -1):
        w = np.array([ORIENT[s] * elementary([ORIENT[t] for t in left if t > s], k - 1) for s in left], float)
        s = left[int(rng.choice(len(left), p=w / w.sum()))]
        out.append(s)
        left = [t for t in left if t > s]          # an unordered set: later picks come after s
    return out


def main(samples: int = 400) -> dict:
    result = {"spots": len(SPOTS), "orientations_per_spot": sorted(set(ORIENT)), "cut_factor_at_most": 1 + 12 * 6}
    rng = np.random.default_rng(20261014)
    for name, axes in (("kinds", AXES_KINDS), ("all_31_axes", AXES_ALL)):
        count, sample = chain_counter(axes)
        by_joints = [count(n) for n in range(1, 6)]
        chains = sum(by_joints)
        hands = {n: elementary(ORIENT, n) * chains ** n for n in range(2, 7)}
        total = sum(hands.values())
        rec = {"chains_by_joint_count": by_joints, "chains_per_spot_orientation": chains,
               "log10_hands_by_fingers": {n: round(math.log10(v), 2) for n, v in hands.items()},
               "log10_hands_total": round(math.log10(total), 2)}
        # a uniform sample of 4-finger hands, checked
        jw = np.array(by_joints, float)
        c1 = c2 = 0
        for _ in range(samples):
            spots = sample_spots(4, rng)
            fingers = []
            for s in spots:
                y, z = SPOTS[s]
                fa = ops.facings_at(y, z, R)
                n = int(rng.choice(5, p=jw / jw.sum())) + 1
                fingers.append(Finger(y=y, z=z, facing=fa[int(rng.integers(len(fa)))],
                                      tilt=TILTS[int(rng.integers(len(TILTS)))], joints=sample(n, rng)))
            h = Hand(fingers=tuple(fingers))
            assert check(h, R) == [], check(h, R)
            if V.clear(h):
                c1 += 1
                c2 += V.c2_workspace_overlap(h).ok
        rec["four_finger_sample"] = {"samples": samples, "pass_c1": c1 / samples, "pass_c1_and_c2": c2 / samples}
        p = c2 / samples
        rec["log10_usable_four_finger_hands"] = round(math.log10(hands[4] * p), 2) if p > 0 else None
        result[name] = rec
    return result


if __name__ == "__main__":
    print(json.dumps(main(int(sys.argv[1]) if len(sys.argv) > 1 else 400), indent=1))
