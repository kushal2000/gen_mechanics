"""GJK distance between convex point sets, compiled with numba (6 Nov): derive.gjk_distance's algorithm step for step
(the same support points, simplex cases and thresholds, Ericson's closest-point-on-triangle), on a fixed 4 x 3
simplex instead of Python lists of arrays. derive.gjk_distance calls it; derive.gjk_distance_py keeps the original
for tests."""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def _triangle(a, b, c, sub):
    """Closest point to the origin on triangle abc; writes the feature's vertices into sub, returns (point, count)."""
    ab = b - a
    ac = c - a
    ap = -a
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        sub[0] = a
        return a.copy(), 1
    bp = -b
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        sub[0] = b
        return b.copy(), 1
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        sub[0] = a
        sub[1] = b
        return a + v * ab, 2
    cp = -c
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        sub[0] = c
        return c.copy(), 1
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        sub[0] = a
        sub[1] = c
        return a + w * ac, 2
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        sub[0] = b
        sub[1] = c
        return b + w * (c - b), 2
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    sub[0] = a
    sub[1] = b
    sub[2] = c
    return a + ab * v + ac * w, 3


@njit(cache=True)
def _simplex(S, k):
    """The point of simplex S[:k] closest to the origin; S is overwritten with the smallest sub-simplex containing it.
    Returns (point, new k); k 4 with the origin inside."""
    sub = np.empty((3, 3))
    if k == 1:
        return S[0].copy(), 1
    if k == 2:
        a = S[0].copy()
        b = S[1].copy()
        ab = b - a
        t = -_dot(a, ab) / max(_dot(ab, ab), 1e-30)
        if t <= 0.0:
            S[0] = a
            return a, 1
        if t >= 1.0:
            S[0] = b
            return b, 1
        return a + t * ab, 2
    if k == 3:
        p, n = _triangle(S[0].copy(), S[1].copy(), S[2].copy(), sub)
        for i in range(n):
            S[i] = sub[i]
        return p, n
    a = S[0].copy()
    b = S[1].copy()
    c = S[2].copy()
    d = S[3].copy()
    faces = ((a, b, c, d), (a, b, d, c), (a, c, d, b), (b, c, d, a))
    best_dd = -1.0
    best_p = np.zeros(3)
    best_sub = np.empty((3, 3))
    best_n = 0
    for f in range(4):
        t0, t1, t2, other = faces[f]
        e1 = t1 - t0
        e2 = t2 - t0
        nrm = np.array((e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]))
        s_origin = _dot(-t0, nrm)
        s_other = _dot(other - t0, nrm)
        if s_origin * s_other < 0.0 or abs(s_other) < 1e-18:
            p, n = _triangle(t0, t1, t2, sub)
            dd = _dot(p, p)
            if best_dd < 0.0 or dd < best_dd:
                best_dd = dd
                best_p = p
                best_n = n
                for i in range(n):
                    best_sub[i] = sub[i]
    if best_dd < 0.0:
        return np.zeros(3), 4
    for i in range(best_n):
        S[i] = best_sub[i]
    return best_p, best_n


@njit(cache=True)
def gjk_distance(A, B, max_iter=64):
    """Euclidean distance between the convex hulls of point sets A and B (n x 3, float64; 0 when they intersect)."""
    S = np.empty((4, 3))
    v = A[0] - B[0]
    S[0] = v
    k = 1
    for _ in range(max_iter):
        vv = _dot(v, v)
        if vv < 1e-20:
            return 0.0
        ia = 0
        best = -np.inf
        for i in range(A.shape[0]):
            s = -(A[i, 0] * v[0] + A[i, 1] * v[1] + A[i, 2] * v[2])
            if s > best:
                best = s
                ia = i
        ib = 0
        best = -np.inf
        for i in range(B.shape[0]):
            s = B[i, 0] * v[0] + B[i, 1] * v[1] + B[i, 2] * v[2]
            if s > best:
                best = s
                ib = i
        w = A[ia] - B[ib]
        if vv - _dot(v, w) <= 1e-10 * max(vv, 1e-12):
            break
        S[k] = w
        k += 1
        v, k = _simplex(S, k)
        if k == 4 or _dot(v, v) < 1e-20:
            return 0.0
    return math.sqrt(_dot(v, v))
