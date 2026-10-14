"""Feetech pair geometry, millimetres and degrees; NumPy only.

Run ``python3 verify.py`` for the calculations and checks used by REPORT.md.
Box columns are (u, v, w). The centre is p + s*u + 7.5*v;
half sizes are (14.5, 17, 10). The axis passes through p, not the box centre.
"""

from itertools import product
import numpy as np

H = np.array([14.5, 17.0, 10.0])
AXIAL = 14.5
FRONT = 9.5
BACK = 24.5
WIDTH = 10.0
GAP = 2.0
X, Y, Z = np.eye(3)
FRONT_RADIUS = np.hypot(FRONT, WIDTH)
BODY_RADIUS = np.linalg.norm([AXIAL, BACK, WIDTH])
KINDS = {"B": Z, "S": Y, "T": X}
LIMITS = {"B": (-30., 90.), "S": (-30., 30.), "T": (-90., 90.)}
SIGNS = np.array(list(product([-1., 1.], repeat=3)))
EDGE_IDS = np.array([(i, j) for i in range(8) for j in range(i+1, 8)
                     if np.count_nonzero(SIGNS[i] != SIGNS[j]) == 1])


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def direction(az, el):
    a, e = np.deg2rad([az, el])
    return np.array([np.sin(e), np.cos(e)*np.sin(a), np.cos(e)*np.cos(a)])


def kind(a):
    # The grammar breaks ties in favour of bend, then spread.
    i = max(range(3), key=lambda k: (round(abs(a[k]), 12), k))
    return {0: "T", 1: "S", 2: "B"}[i]


def rotation(a, angles):
    """Batched right-handed rotations. Angles are in degrees."""
    a = unit(a)
    q = np.deg2rad(np.asarray(angles))
    cross = np.array([[0., -a[2], a[1]], [a[2], 0., -a[0]],
                      [-a[1], a[0], 0.]])
    c, s = np.cos(q)[..., None, None], np.sin(q)[..., None, None]
    return c*np.eye(3) + (1-c)*np.outer(a, a) + s*cross


def frame(a, back=None, clock=0.):
    """Default back: proximal projection of -x; pure twist: -y (dorsal)."""
    a = unit(a)
    if back is None:
        back = -X if abs(a[0]) < 1-1e-12 else -Y
    v = np.asarray(back) - a*np.dot(a, back)
    if np.linalg.norm(v) < 1e-10:
        fallback = np.eye(3)[np.argmin(np.abs(a))]
        v = fallback - a*np.dot(a, fallback)
    v = rotation(a, clock) @ unit(v)
    return np.column_stack([a, v, np.cross(a, v)])


def box(p, axes, slide=0.):
    return np.asarray(p) + slide*axes[:, 0] + 7.5*axes[:, 1], axes


def support(axes, n, slide=0.):
    """Support relative to the *axis origin*, not the box centre."""
    c = slide*axes[:, 0] + 7.5*axes[:, 1]
    return np.dot(c, n) + np.abs(n @ axes) @ H


def pair_poses(a, b, length, qa, qb=0., attach="PP", slides=(0., 0.),
               frames=None):
    """P=body on parent, C=body on child. Inputs qa/qb broadcast together.

    Joint j's parent frame is the reference; earlier joints cancel out.
    The next joint's axis and origin rotate with joint j.
    """
    if frames is None:
        frames = (frame(a), frame(b))
    A, B = frames
    ca, _ = box(np.zeros(3), A, slides[0])
    cb, _ = box(np.zeros(3), B, slides[1])
    qa, qb = np.broadcast_arrays(qa, qb)
    Ra, Rb = rotation(a, qa), rotation(b, qb)
    Ia = Ra if attach[0] == "C" else np.broadcast_to(np.eye(3), Ra.shape)
    Ib = Ra @ Rb if attach[1] == "C" else Ra
    c1 = np.einsum("...ij,j->...i", Ia, ca)
    c2 = np.einsum("...ij,j->...i", Ra, length*X) + np.einsum("...ij,j->...i", Ib, cb)
    return c1, Ia @ A, c2, Ib @ B


def sat_margin(ca, A, cb, B):
    """Maximum separating gap over the 15 normalized SAT directions.

    Negative means overlap. Nonnegative is the exact intersection test;
    a positive margin is a LOWER BOUND on Euclidean clearance, not generally
    the distance. In particular, testing margin>=2 is conservative.
    """
    ca, cb = np.asarray(ca), np.asarray(cb)
    batch = np.broadcast_shapes(ca.shape[:-1], cb.shape[:-1], A.shape[:-2], B.shape[:-2])
    A = np.broadcast_to(A, batch+(3, 3))
    B = np.broadcast_to(B, batch+(3, 3))
    aa, bb = np.swapaxes(A, -1, -2), np.swapaxes(B, -1, -2)
    cross = np.cross(aa[..., :, None, :], bb[..., None, :, :]).reshape(batch+(9, 3))
    ns = np.concatenate([aa, bb, cross], axis=-2)
    norms = np.linalg.norm(ns, axis=-1)
    valid = norms > 1e-10
    ns = ns / np.maximum(norms[..., None], 1e-30)
    ra = np.abs(np.einsum("...ki,...ij->...kj", ns, A)) @ H
    rb = np.abs(np.einsum("...ki,...ij->...kj", ns, B)) @ H
    sep = np.abs(np.einsum("...ki,...i->...k", ns, cb-ca)) - ra - rb
    return np.max(np.where(valid, sep, -np.inf), axis=-1)


def vertices(c, axes):
    return np.asarray(c)[..., None, :] + np.einsum("...ij,kj->...ki", axes, SIGNS*H)


def point_box_d2(points, c, axes):
    local = np.einsum("...ki,...ij->...kj", points-np.asarray(c)[..., None, :], axes)
    return np.sum(np.maximum(np.abs(local)-H, 0.)**2, axis=-1)


def segment_distances_d2(p0, p1, q0, q1):
    """All pairs of edges, plus a leading pose batch.

    Interior/interior stationary points and all four boundary optima cover
    the convex two-parameter segment distance problem, including parallels.
    """
    p, q = p0[..., :, None, :], q0[..., None, :, :]
    u = (p1-p0)[..., :, None, :]
    v = (q1-q0)[..., None, :, :]
    d = p-q
    aa = np.sum(u*u, axis=-1)
    bb = np.sum(u*v, axis=-1)
    cc = np.sum(v*v, axis=-1)
    dd = np.sum(u*d, axis=-1)
    ee = np.sum(v*d, axis=-1)
    det = aa*cc - bb*bb
    nonparallel = det > 1e-10*aa*cc
    safe_det = np.where(nonparallel, det, 1.)
    ss = (bb*ee-cc*dd)/safe_det
    tt = (aa*ee-bb*dd)/safe_det
    interior = nonparallel & (ss >= 0) & (ss <= 1) & (tt >= 0) & (tt <= 1)
    val = np.sum((d+ss[..., None]*u-tt[..., None]*v)**2, axis=-1)
    best = np.where(interior, val, np.inf)
    for s in (0., 1.):
        t = np.clip((ee+s*bb)/cc, 0., 1.)
        val = np.sum((d+s*u-t[..., None]*v)**2, axis=-1)
        best = np.minimum(best, val)
    for t in (0., 1.):
        s = np.clip((t*bb-dd)/aa, 0., 1.)
        val = np.sum((d+s[..., None]*u-t*v)**2, axis=-1)
        best = np.minimum(best, val)
    return best


def box_distance(ca, A, cb, B):
    """Exact Euclidean distance of solid OBBs (floating-point tolerance).

    Disjoint boxes attain distance at a vertex/face or edge/edge pair.
    SAT handles intersections and containment first.
    """
    va, vb = vertices(ca, A), vertices(cb, B)
    d2 = np.minimum(np.min(point_box_d2(va, cb, B), axis=-1),
                    np.min(point_box_d2(vb, ca, A), axis=-1))
    ed2 = segment_distances_d2(va[..., EDGE_IDS[:, 0], :], va[..., EDGE_IDS[:, 1], :],
                              vb[..., EDGE_IDS[:, 0], :], vb[..., EDGE_IDS[:, 1], :])
    d2 = np.minimum(d2, np.min(ed2, axis=(-1, -2)))
    return np.where(sat_margin(ca, A, cb, B) <= 1e-10, 0., np.sqrt(np.maximum(d2, 0.)))


def samples(a, step=.25):
    lo, hi = LIMITS[kind(a)]
    return np.linspace(lo, hi, int(np.ceil((hi-lo)/step))+1)


def front_sweep(a):
    return (FRONT*np.cos(np.pi/6)+WIDTH*.5) if kind(a) == "S" else FRONT_RADIUS


def plane_rule(a, b, gap=GAP):
    """Continuous-range safe PP rule, centred and default clocking."""
    ta, tb = abs(a[0]), abs(b[0])
    return gap + AXIAL*(ta+tb) + front_sweep(a)*np.sqrt(max(0., 1-ta*ta)) + BACK*np.sqrt(max(0., 1-tb*tb))


def golden_min(f, lo, hi, iterations=65):
    """One-dimensional convex minimization, also for nondifferentiable f."""
    k = (np.sqrt(5.)-1)/2
    c, d = hi-k*(hi-lo), lo+k*(hi-lo)
    fc, fd = f(c), f(d)
    for _ in range(iterations):
        if fc <= fd:
            hi, d, fd = d, c, fc
            c = hi-k*(hi-lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo+k*(hi-lo)
            fd = f(d)
    z = .5*(lo+hi)
    return z, f(z)


def free_centered_min(a, b):
    """Global pair optimum with no slides, free clocks and attachments.

    The CP attachment is static and allows touching. Every box contains a
    cylinder of radius 9.5 and axial half-length 14.5 centred at the joint.
    Its support is the minimum achievable box support over the clock angle.
    A separating plane n=(1,y,z) gives the convex objective below. Nested
    scalar convex minimizations are sufficient. The bound follows from
    F(n)>=19*||n|| and F(x)<=sqrt(29**2+19**2).
    """
    a, b = unit(a), unit(b)
    def f(y, z):
        n = np.array([1., y, z])
        dots = np.array([a @ n, b @ n])
        perps = np.array([np.linalg.norm(np.cross(a, n)), np.linalg.norm(np.cross(b, n))])
        return AXIAL*np.abs(dots).sum() + FRONT*perps.sum()
    bound = 29./19.
    def inner(y):
        return golden_min(lambda z: f(y, z), -bound, bound)
    y, _ = golden_min(lambda y: inner(y)[1], -bound, bound)
    z, val = inner(y)
    n = unit([1., y, z])
    return val, n


def free_centered_frames(a, b, n):
    return frame(a, -n), frame(b, n)


def swept_min_distance(a, b, length, step=.25, **kw):
    q = samples(a, step)
    ds = box_distance(*pair_poses(a, b, length, q, **kw))
    i = np.argmin(ds)
    return float(ds[i]), float(q[i])


def pp_sampled_threshold(a, b, step=.25, iterations=22, gap=GAP):
    """Fixed centred default PP assembly, exact OBB distances, sampled q.

    Monotonic in L HERE: at L=0 both boxes contain the common origin in
    every pose; distance to each convex Minkowski difference is monotonic
    along this ray. Do not use this binary search for arbitrary slides.
    """
    lo, hi = 0., plane_rule(a, b, gap)
    for _ in range(iterations):
        mid = .5*(lo+hi)
        d, _ = swept_min_distance(a, b, mid, step)
        if d >= gap:
            hi = mid
        else:
            lo = mid
    d, q = swept_min_distance(a, b, hi, step)
    return hi, d, q


def slide_for_separator(a, b, length=0., gap=GAP, B=None):
    """Slide the upstream PP body by -S*a; separate on the invariant axis a."""
    B = frame(b) if B is None else B
    return max(0., AXIAL+gap+support(B, -a)-length*a[0])
