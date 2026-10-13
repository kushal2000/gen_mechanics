"""Hub-and-body model of a Feetech HL-3915 joint.

Hub: a cylinder around the joint axis that holds the servo's front end, horn,
idler and the printed bracket of the link the horn drives. Radius = the front
corners' swing (sqrt(9.5^2 + 10^2) = 13.8 mm) + 2 mm bracket = 15.8 mm;
half-length = 14.5 mm + 2 mm bracket = 16.5 mm. Rotationally symmetric, so it
does not move when its own joint turns.
Body: the servo box (|u| <= 14.5, -9.5 <= v <= 24.5, |w| <= 10) on one link.

Distances: exact point-to-solid distance from dense surface samples of the
other solid (overestimates by at most half the sample spacing).
"""
import numpy as np

R_FRONT = float(np.hypot(9.5, 10.0))
BRACKET = 2.0
HUB_R = R_FRONT + BRACKET          # 15.79
HUB_H = 14.5 + BRACKET             # 16.5
BOX_LO = np.array([-14.5, -9.5, -10.0])
BOX_HI = np.array([14.5, 24.5, 10.0])
X = np.array([1.0, 0.0, 0.0]); Y = np.array([0.0, 1.0, 0.0]); Z = np.array([0.0, 0.0, 1.0])


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def frame(u, v_hint):
    """Columns u, v, w with v the part of v_hint square to u."""
    u = unit(u); v = np.asarray(v_hint, float) - (np.asarray(v_hint, float) @ u) * u
    if np.linalg.norm(v) < 1e-9:
        raise ValueError("v_hint along the axis")
    v = unit(v); return np.stack([u, v, np.cross(u, v)], axis=1)


class Hub:
    def __init__(self, point, axis, slide=0.0):
        self.a = unit(axis); self.c = np.asarray(point, float) + slide * self.a

    def dist(self, P):
        d = P - self.c
        ax = d @ self.a
        rad = np.linalg.norm(d - ax[:, None] * self.a, axis=1)
        dr = np.maximum(rad - HUB_R, 0.0); da = np.maximum(np.abs(ax) - HUB_H, 0.0)
        out = np.hypot(dr, da)
        inside = (rad <= HUB_R) & (np.abs(ax) <= HUB_H)
        return np.where(inside, -1.0, out)

    def samples(self, step=1.0):
        a = self.a; e1 = frame(a, X if abs(a @ X) < 0.9 else Y)[:, 1]; e2 = np.cross(a, e1)
        nth = max(16, int(np.ceil(2 * np.pi * HUB_R / step)))
        th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
        ring = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
        zs = np.linspace(-HUB_H, HUB_H, max(2, int(np.ceil(2 * HUB_H / step)) + 1))
        side = (self.c + HUB_R * ring[None, :, :] + zs[:, None, None] * a).reshape(-1, 3)
        rs = np.linspace(0, HUB_R, max(2, int(np.ceil(HUB_R / step)) + 1))[:-1]
        caps = [self.c + s * HUB_H * a + rs[:, None, None] * ring[None, :, :] for s in (-1, 1)]
        return np.concatenate([side] + [c.reshape(-1, 3) for c in caps])

    def moved(self, Rm, about):
        h = Hub.__new__(Hub); h.a = Rm @ self.a; h.c = about + Rm @ (self.c - about); return h


class Body:
    def __init__(self, point, axis, v_hint, slide=0.0):
        self.R = frame(axis, v_hint); self.o = np.asarray(point, float) + slide * self.R[:, 0]

    def dist(self, P):
        loc = (P - self.o) @ self.R
        lo, hi = BOX_LO, BOX_HI
        d = np.maximum(np.maximum(lo - loc, loc - hi), 0.0)
        out = np.linalg.norm(d, axis=1)
        inside = np.all((loc >= lo) & (loc <= hi), axis=1)
        return np.where(inside, -1.0, out)

    def samples(self, step=1.0):
        lo, hi = BOX_LO, BOX_HI
        g = [np.linspace(lo[i], hi[i], max(2, int(np.ceil((hi[i] - lo[i]) / step)) + 1)) for i in range(3)]
        pts = []
        for i in range(3):
            j, k = [m for m in range(3) if m != i]
            A, B = np.meshgrid(g[j], g[k], indexing="ij")
            for val in (lo[i], hi[i]):
                P = np.zeros((A.size, 3)); P[:, i] = val; P[:, j] = A.ravel(); P[:, k] = B.ravel()
                pts.append(P)
        loc = np.concatenate(pts)
        return self.o + loc @ self.R.T

    def moved(self, Rm, about):
        b = Body.__new__(Body); b.R = Rm @ self.R; b.o = about + Rm @ (self.o - about); return b


def rot(axis, q):
    axis = unit(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)


def gap(A, B, step=1.0):
    """Distance between two solids (<= 0: touching or overlapping)."""
    d1 = B.dist(A.samples(step)).min()
    d2 = A.dist(B.samples(step)).min()
    return min(d1, d2)


def swept_gap(fixed, moving, axis, point, rng_deg, qstep=2.0, step=1.0):
    """Smallest gap while `moving` turns about the line (point, axis) over the range."""
    lo, hi = rng_deg
    qs = np.linspace(lo, hi, int(np.ceil((hi - lo) / qstep)) + 1)
    best = np.inf
    for q in qs:
        Rm = rot(axis, np.radians(q))
        best = min(best, gap(fixed, moving.moved(Rm, np.asarray(point, float)), step))
    return best
