"""Render a grammar hand (its solids from `derive.solids_at`) to PNG with matplotlib, palm down as in the task.

    from render import render_hand
    render_hand(hand, q, path, title, pen=(R, c))   # R, c: the pen's pose in the palm frame (mm)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from hand_sampler.grammar import derive as dv  # noqa: E402

COLORS = ["#d95f4c", "#4a90d9", "#5cb85c", "#e0a030", "#9a64c8", "#3cbcbc"]
# palm frame -> world (palm down): world z = -palm x, world x = palm z, world y = palm y
P2W = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])


def _box_faces(corners):
    c = corners
    # mt.box_corners order: itertools-like over (+-a, +-v, +-w); build faces from the hull
    from scipy.spatial import ConvexHull
    h = ConvexHull(c)
    return [c[s] for s in h.simplices]


def _capsule_pts(p0, p1, r, n=10):
    d = p1 - p0
    L = np.linalg.norm(d)
    a = d / L if L > 1e-9 else np.array([1.0, 0, 0])
    u = np.cross(a, [0, 0, 1.0] if abs(a[2]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    pts = []
    for th in np.linspace(0, math.pi, 6):
        for ph in np.linspace(0, 2 * math.pi, n, endpoint=False):
            for c, s in ((p0, -1), (p1, 1)):
                dirn = math.cos(th) * a * s + math.sin(th) * (math.cos(ph) * u + math.sin(ph) * v)
                if np.dot(dirn, a) * s >= -1e-9:
                    pts.append(c + r * dirn)
    return np.array(pts)


def hand_polys(hand, q=None):
    """[(faces (k, 3, 3) world mm, color)]"""
    out = []
    for s in dv.solids_at(hand, q):
        k = s.key
        if k[0] == "mount":
            continue
        if k[0] == "palm":
            pts = s.points / dv.MM
            col = "#b0b3bb"
        elif s.radius > 0:
            p0, p1 = s.points / dv.MM
            pts = _capsule_pts(p0, p1, s.radius / dv.MM)
            col = COLORS[k[1] % len(COLORS)]
        else:
            pts = s.points / dv.MM
            col = "#3a3a40" if k[0] == "motor" else COLORS[k[1] % len(COLORS)]
        w = pts @ P2W.T
        out.append((_box_faces(w), col))
    return out


def pen_polys(R, c):
    from itertools import product
    corners = np.array([R @ (np.array(s) * np.array([50, 15, 15])) + c for s in product((-1, 1), repeat=3)])
    return _box_faces(corners @ P2W.T)


def draw(ax, hand, q=None, pen=None, elev=20, azim=-60, lim=None, root_z=None):
    for faces, col in hand_polys(hand, q):
        pc = Poly3DCollection(faces, facecolor=col, edgecolor="none", alpha=0.85 if col != "#b0b3bb" else 0.5)
        ax.add_collection3d(pc)
    if pen is not None:
        ax.add_collection3d(Poly3DCollection(pen_polys(*pen), facecolor="#2860c8", edgecolor="#123", alpha=0.9,
                                             linewidths=0.3))
    lim = lim or 160
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-1.6 * lim, 0.4 * lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("mm")
    if root_z is not None:      # the table
        zt = -root_z
        xx, yy = np.meshgrid([-lim, lim], [-lim, lim])
        ax.plot_surface(xx, yy, np.full_like(xx, zt, dtype=float), color="#c8b89a", alpha=0.35)


def render_hand(hand, path, title="", poses=(), views=((20, -60), (0, -90), (90, -90))):
    """poses: list of (label, q, pen or None, root_z or None)."""
    poses = list(poses) or [("zero pose", None, None, None)]
    n = len(poses)
    fig = plt.figure(figsize=(4.2 * max(n, len(views)), 8.4 if n > 1 else 4.4))
    k = 1
    if n == 1:
        for (e, a) in views:
            ax = fig.add_subplot(1, len(views), k, projection="3d")
            label, q, pen, rz = poses[0]
            draw(ax, hand, q, pen, e, a, root_z=rz)
            ax.set_title(f"{label} (elev {e})", fontsize=9)
            k += 1
    else:
        for pose in poses:
            label, q, pen, rz = pose[:4]
            e, a = pose[4] if len(pose) > 4 else (12, -60)
            ax = fig.add_subplot(1, n, k, projection="3d")
            draw(ax, hand, q, pen, e, a, root_z=rz)
            ax.set_title(label, fontsize=9)
            k += 1
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
