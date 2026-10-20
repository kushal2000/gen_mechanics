"""Ways to shrink the coarse stage further (2026-10-20, not adopted): hand
counts (log10, every finger count 2-6, before the centre rule and the
clearance check; exact unless marked approximate) under each narrowing of
`FEETECH_COARSE_RULES`. Notes: `project-notes/grammar/search-space-options.md`.

    PYTHONPATH=. python3 project-notes/grammar/experiments/E16_grammar_rebuild/ours/coarse_options.py
"""
import math
from functools import lru_cache
from hand_sampler.grammar import motor as mt, operators as ops
from hand_sampler.grammar.hand import FEETECH_COARSE_RULES as R, KIND_AXIS, axis_unit

KINDS = (KIND_AXIS["flexion"], KIND_AXIS["abduction"], KIND_AXIS["roll"])
BEND = KIND_AXIS["flexion"]
SPOTS = ops.base_spots(R)

def e_k(vals, k):
    e = [1] + [0] * k
    for v in vals:
        for m in range(k, 0, -1):
            e[m] += e[m - 1] * v
    return e[k]

def per_facing(axes=KINDS, step=20, top=90, max_joints=5, cap=225, tilts=4, sizes=None, one_length=False):
    """Fingers per spot and facing: joint chains x the tilts each may take."""
    U = [axis_unit(a) for a in axes]
    def lens(lo, std_turn=None):
        if sizes is not None:
            return [L for L in sizes if L >= lo]
        return list(range(lo, top + 1, step))
    @lru_cache(maxsize=None)
    def rec(left, i, prev_zero, budget, ti, Lfix):
        if left == 1:
            return sum(1 for t in lens(mt.tip_floor_mm(U[i], ti)) if t <= budget and (Lfix is None or t == Lfix))
        tot = 0
        for j in range(len(axes)):
            std = mt.link_floor_mm(U[i], U[j], ti, False)
            lo = mt.link_floor_mm(U[i], U[j], ti, True) if mt.is_bend(U[j]) else std
            opts = [(L, L < std) for L in lens(lo)] + ([(0, False)] if (not prev_zero and mt.crossing_ok(U[i], U[j])) else [])
            for L, tj in opts:
                if L <= budget and (Lfix is None or L in (0, Lfix)):
                    tot += rec(left - 1, j, L == 0, budget - L, tj, Lfix)
        return tot
    total = 0
    Ls = [None] if not one_length else list(range(30, top + 1, step))
    for Lfix in Ls:
        for n in range(1, max_joints + 1):
            for i in range(len(axes)):
                total += rec(n, i, False, cap, False, Lfix) * (1 if axes[i] == BEND else tilts)
    return total

def total(pf, radial=False, spots=SPOTS, fingers=range(2, 7)):
    fac = [1 if radial else len(ops.facings_at(y, z, R)) for (y, z) in spots]
    return math.log10(sum(e_k([f * pf for f in fac], n) for n in fingers))

pf0 = per_facing()
base = total(pf0)
print(f"per finger (one spot and facing): {pf0:,} options = 10^{math.log10(pf0):.1f}")
rows = [
    ("coarse stage now", base),
    ("at most 5 fingers", total(pf0, fingers=range(2, 6))),
    ("at most 4 fingers", total(pf0, fingers=range(2, 5))),
    ("at most 4 joints per finger", total(per_facing(max_joints=4))),
    ("at most 3 joints per finger", total(per_facing(max_joints=3))),
    ("links in 30 mm steps (e.g. 30, 60, 90)", total(per_facing(step=30))),
    ("one link length per finger (all links the same)", total(per_facing(one_length=True))),
    ("fingers point straight out (no facing choice)", total(pf0, radial=True)),
    ("tilt 0 only", total(per_facing(tilts=1))),
    ("no twist joints", total(per_facing(axes=KINDS[:2]))),
    ("only the outer ring of spots (12)", total(pf0, spots=[s for s in SPOTS if math.hypot(*s) >= 79]))]
for name, v in rows:
    print(f"{v:6.2f}  ({v - base:+.2f})  {name}")
# identical fingers: at most two finger designs per hand (a design: the chain and its tilt), placed freely
fac = [len(ops.facings_at(y, z, R)) for (y, z) in SPOTS]
ident = math.log10(sum(e_k(fac, n) * (pf0 ** 2) * (2 ** (n - 1)) for n in range(2, 7)))
print(f"{ident:6.2f}  ({ident - base:+.2f})  at most two finger designs per hand, each used by any number of fingers (approx.)")
ident1 = math.log10(sum(e_k(fac, n) * pf0 for n in range(2, 7)))
print(f"{ident1:6.2f}  ({ident1 - base:+.2f})  every finger the same design (approx.)")
