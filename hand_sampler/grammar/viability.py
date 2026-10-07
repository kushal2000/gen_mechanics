"""The two viability checks (layer 3: what no generation rule can guarantee).

C1, self-overlap: no two bodies overlap by more than 3 mm, at the zero pose
and at the episode start pose (`derive.START_CURL`). Bodies are the
rounded-box links (core box grown by 6 mm, as the simulator builds them) and
the palm plates (sharp prisms); the overlap of two bodies is the sum of
their radii minus the distance between their cores (GJK). Bodies that touch
by construction are skipped: a link and the next link of its finger, links
joined through 0 mm links, a finger's first link(s) and the plate it sits
on, and a palm section and the main palm.

C2, fingertip workspaces meet above the palm: at least one pair of fingers
whose fingertips' reachable regions come within `C2_TOLERANCE_MM` of each
other above the palm. Each fingertip is sampled `C2_SAMPLES` times over its
own joints' ranges (its palm joint included); only samples above the plate's
top face and over the palm's footprint (all plates' outlines, grown by
`C2_FOOTPRINT_MARGIN_MM`) count. The tolerance and the sample count are
calibrated so every commercial hand passes (see the grammar README).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import derive as dv
from .hand import LINK_RADIUS_MM, PALM_THICKNESS_MM, Hand

MM = 1e-3
C1_MAX_OVERLAP_MM = 3.0
C2_SAMPLES = 1500
C2_TOLERANCE_MM = 20.0
C2_FOOTPRINT_MARGIN_MM = 20.0
C2_SEED = 0


# --------------------------------------------------------------------------
# GJK distance between convex point sets
# --------------------------------------------------------------------------


def _closest_segment(a: np.ndarray, b: np.ndarray):
    ab = b - a
    t = -float(a @ ab) / max(float(ab @ ab), 1e-30)
    if t <= 0.0:
        return a, [a]
    if t >= 1.0:
        return b, [b]
    return a + t * ab, [a, b]


def _closest_triangle(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Closest point to the origin on triangle abc (Ericson, Real-Time
    Collision Detection 5.1.5) and the vertices of the feature it lies on."""
    ab, ac, ap = b - a, c - a, -a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, [a]
    bp = -b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, [b]
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, [a, b]
    cp = -c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, [c]
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, [a, c]
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), [b, c]
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return a + ab * v + ac * w, [a, b, c]


def _closest_on_simplex(S: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """The point of the simplex S (1-4 points) closest to the origin, and the
    smallest sub-simplex containing it (a full tetrahedron: the origin is
    inside, distance 0)."""
    k = len(S)
    if k == 1:
        return S[0], [S[0]]
    if k == 2:
        return _closest_segment(S[0], S[1])
    if k == 3:
        return _closest_triangle(S[0], S[1], S[2])
    a, b, c, d = S
    best = None
    for tri, other in (((a, b, c), d), ((a, b, d), c), ((a, c, d), b), ((b, c, d), a)):
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        s_origin = float(-tri[0] @ n)
        s_other = float((other - tri[0]) @ n)
        if s_origin * s_other < 0.0 or abs(s_other) < 1e-18:      # the origin is outside this face
            p, sub = _closest_triangle(*tri)
            dd = float(p @ p)
            if best is None or dd < best[0]:
                best = (dd, p, sub)
    if best is None:
        return np.zeros(3), list(S)
    return best[1], best[2]


def gjk_distance(A: np.ndarray, B: np.ndarray, max_iter: int = 64) -> float:
    """Euclidean distance between the convex hulls of point sets `A` and `B`
    (0 when they intersect)."""
    v = A[0] - B[0]
    S = [v]
    for _ in range(max_iter):
        vv = float(v @ v)
        if vv < 1e-20:
            return 0.0
        w = A[int(np.argmax(A @ -v))] - B[int(np.argmax(B @ v))]
        if vv - float(v @ w) <= 1e-10 * max(vv, 1e-12):
            break
        S.append(w)
        v, S = _closest_on_simplex(S)
        if len(S) == 4 or float(v @ v) < 1e-20:
            return 0.0
    return math.sqrt(float(v @ v))


# --------------------------------------------------------------------------
# Bodies
# --------------------------------------------------------------------------


@dataclass
class Body:
    key: Tuple                # ("palm", -1 | k) or ("link", i, j)
    points: np.ndarray        # core corners, palm frame (m)
    radius: float             # m


def bodies_at(hand: Hand, q: Optional[np.ndarray] = None) -> List[Body]:
    pose = dv.fk(hand, q)
    out: List[Body] = []
    outlines = dv.palm_outlines_mm(hand)
    out.append(Body(("palm", -1), dv.plate_points_m(outlines[-1]), 0.0))
    for k in range(len(hand.palm_joints)):
        pts = dv.plate_points_m(outlines[k])
        T = pose.palm[k]
        out.append(Body(("palm", k), pts @ T[:3, :3].T + T[:3, 3], 0.0))
    r = LINK_RADIUS_MM * MM
    for i, f in enumerate(hand.fingers):
        for j in range(len(f.joints)):
            c, h = dv.link_core_box_m(hand, i, j)
            T = pose.links[i][j]
            pts = dv.box_corners(c, h)
            out.append(Body(("link", i, j), pts @ T[:3, :3].T + T[:3, 3], r))
    return out


def touching_pairs(hand: Hand) -> set:
    """Body-key pairs that touch by construction (both orders)."""
    out = set()

    def add(a, b):
        out.add((a, b))
        out.add((b, a))

    for k in range(len(hand.palm_joints)):
        add(("palm", -1), ("palm", k))
    for i, f in enumerate(hand.fingers):
        plate = ("palm", f.palm_joint)
        for j in range(len(f.joints)):
            me = ("link", i, j)
            # walk back toward the palm through 0 mm links
            prev = j - 1
            while True:
                other = plate if prev < 0 else ("link", i, prev)
                add(me, other)
                if prev < 0 or f.joints[prev].length != 0:
                    break
                prev -= 1
    return out


def overlaps(hand: Hand, q: Optional[np.ndarray] = None) -> List[Tuple[Tuple, Tuple, float]]:
    """`(body_a, body_b, overlap_mm)` for every pair not touching by
    construction whose rounded shapes overlap (overlap > 0)."""
    bodies = bodies_at(hand, q)
    skip = touching_pairs(hand)
    centres = [b.points.mean(axis=0) for b in bodies]
    reach = [float(np.max(np.linalg.norm(b.points - c, axis=1))) + b.radius for b, c in zip(bodies, centres)]
    out = []
    for a in range(len(bodies)):
        for b in range(a + 1, len(bodies)):
            A, B = bodies[a], bodies[b]
            if (A.key, B.key) in skip:
                continue
            if float(np.linalg.norm(centres[a] - centres[b])) > reach[a] + reach[b]:
                continue
            d = gjk_distance(A.points, B.points)
            ov = (A.radius + B.radius - d) / MM
            if ov > 1e-6:
                out.append((A.key, B.key, ov))
    return out


@dataclass
class C1Result:
    ok: bool
    worst_mm: float
    pairs: List[Tuple[Tuple, Tuple, float]]      # failing pairs (either pose)


def c1_self_overlap(hand: Hand, max_overlap_mm: float = C1_MAX_OVERLAP_MM) -> C1Result:
    bad = []
    worst = 0.0
    for q in (None, dv.start_q(hand)):
        for a, b, ov in overlaps(hand, q):
            worst = max(worst, ov)
            if ov > max_overlap_mm:
                bad.append((a, b, ov))
    return C1Result(ok=not bad, worst_mm=worst, pairs=bad)


# --------------------------------------------------------------------------
# C2
# --------------------------------------------------------------------------


@dataclass
class C2Result:
    ok: bool
    best_mm: float                 # the closest pair's distance between kept samples (inf: none)
    pair: Optional[Tuple[int, int]]
    kept: List[np.ndarray]         # per finger, kept fingertip samples (m)
    margin_mm: float               # tolerance - best (>= 0 passes)


def footprint_mm(hand: Hand) -> np.ndarray:
    outlines = dv.palm_outlines_mm(hand)
    return dv._hull2d(np.concatenate(list(outlines.values())))


def fingertip_samples(hand: Hand, n: int = C2_SAMPLES, seed: int = C2_SEED) -> List[np.ndarray]:
    """Per finger, `(n, 3)` fingertip positions (m) with that finger's joints
    (and its palm joint) uniform over their ranges and every other joint at 0."""
    rng = np.random.default_rng(seed)
    ds = dv.dofs(hand)
    out = []
    for i, f in enumerate(hand.fingers):
        lo = np.zeros(len(ds))
        hi = np.zeros(len(ds))
        for k, d in enumerate(ds):
            mine = d.finger == i or (d.type == "palm" and d.index == f.palm_joint)
            if mine:
                lo[k], hi[k] = d.limits
        q = dv.tie(hand, rng.uniform(lo, hi, size=(n, len(ds))), ds)
        out.append(dv.fk(hand, q).tips[:, i])
    return out


def c2_workspace_overlap(hand: Hand, n: int = C2_SAMPLES, tol_mm: float = C2_TOLERANCE_MM,
                         margin_mm: float = C2_FOOTPRINT_MARGIN_MM, seed: int = C2_SEED) -> C2Result:
    poly = footprint_mm(hand)
    top = PALM_THICKNESS_MM / 2.0 * MM
    kept = []
    for tips in fingertip_samples(hand, n, seed):
        m = (tips[:, 0] > top) & dv.point_in_polygon(tips[:, 1:] / MM, poly, margin_mm)
        kept.append(tips[m])
    best, pair = math.inf, None
    for a in range(len(kept)):
        for b in range(a + 1, len(kept)):
            A, B = kept[a], kept[b]
            if len(A) == 0 or len(B) == 0:
                continue
            d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
            d = math.sqrt(max(float(d2.min()), 0.0)) / MM
            if d < best:
                best, pair = d, (a, b)
    return C2Result(ok=best <= tol_mm, best_mm=best, pair=pair, kept=kept, margin_mm=tol_mm - best)


@dataclass
class Viability:
    c1: C1Result
    c2: C2Result

    @property
    def ok(self) -> bool:
        return self.c1.ok and self.c2.ok


def viability(hand: Hand) -> Viability:
    return Viability(c1=c1_self_overlap(hand), c2=c2_workspace_overlap(hand))


def is_viable(hand: Hand) -> bool:
    c1 = c1_self_overlap(hand)
    return c1.ok and c2_workspace_overlap(hand).ok
