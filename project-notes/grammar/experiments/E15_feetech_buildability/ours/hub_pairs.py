"""Shortest link between consecutive joints under the hub-and-body model.
Convention P,P: each servo body sits on the link before its joint.
  link j-1: body1          link j: hub1 (shared), body2, hub2 (shared with j+1)
  - hub1, hub2 and body2 are on link j: hubs disjoint, body2 outside hub1 (>= 0 mm)
  - body1 moves about axis 1 relative to link j: >= 2 mm from body2 and hub2 over joint 1's range
"""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from hub_model import *

RANGES = {"bend": (-30, 90), "spread": (-30, 30), "twist": (-90, 90)}

def pair_gaps(a1, a2, L, v1, v2, s1=0.0, s2=0.0, rng1=(-30, 90), step=1.0, qstep=3.0):
    p1, p2 = np.zeros(3), L * X
    h1, h2 = Hub(p1, a1, s1), Hub(p2, a2, s2)
    b1, b2 = Body(p1, a1, v1, s1), Body(p2, a2, v2, s2)
    g_hubs = gap(h1, h2, step)
    g_b2h1 = gap(b2, h1, step)
    # body1 turns relative to link j by -q about axis 1
    g_sw = min(swept_gap(b2, b1, a1, p1, (-rng1[1], -rng1[0]), qstep, step),
               swept_gap(h2, b1, a1, p1, (-rng1[1], -rng1[0]), qstep, step))
    return g_hubs, g_b2h1, g_sw

def ok(g, clear=2.0):
    return g[0] >= 0 and g[1] >= 0 and g[2] >= clear

def min_L(a1, a2, v1, v2, s1=0.0, s2=0.0, rng1=(-30, 90), lo=0.0, hi=90.0, tol=0.25, step=1.0):
    """Bisection for the shortest feasible L, then a check that hi side stays feasible on a 1 mm scan."""
    if not ok(pair_gaps(a1, a2, hi, v1, v2, s1, s2, rng1, step)):
        return None
    if ok(pair_gaps(a1, a2, lo, v1, v2, s1, s2, rng1, step)):
        return lo
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ok(pair_gaps(a1, a2, mid, v1, v2, s1, s2, rng1, step)): hi = mid
        else: lo = mid
    return hi

if __name__ == "__main__":
    BACK, DOWN = -X, -Y
    cases = [
        ("bend->bend, bodies along the finger", Z, Z, BACK, BACK, "bend"),
        ("bend->bend, next body hangs to the back of the hand", Z, Z, BACK, DOWN, "bend"),
        ("spread->spread, along the finger", Y, Y, BACK, BACK, "spread"),
        ("bend->spread, along the finger", Z, Y, BACK, BACK, "bend"),
        ("spread->bend, along the finger", Y, Z, BACK, BACK, "spread"),
        ("spread->bend, next body hangs back", Y, Z, BACK, DOWN, "spread"),
        ("twist->bend, twist body hangs back", X, Z, DOWN, BACK, "twist"),
        ("bend->twist, twist body hangs back", Z, X, BACK, DOWN, "bend"),
        ("twist->twist", X, X, DOWN, DOWN, "twist"),
    ]
    for name, a1, a2, v1, v2, k in cases:
        L = min_L(a1, a2, v1, v2, rng1=RANGES[k])
        g = pair_gaps(a1, a2, L, v1, v2, rng1=RANGES[k]) if L is not None else None
        print(f"{name:55s} L_min {L if L is None else round(L, 1)}  gaps at L_min: {None if g is None else tuple(round(x, 2) for x in g)}", flush=True)
