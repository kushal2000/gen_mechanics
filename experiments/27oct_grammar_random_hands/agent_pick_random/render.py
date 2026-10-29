"""Render grammar hands to PNG (matplotlib, CPU): a view from below (the table's view of the palm) and a side view,
palm down, with every bend at `--curl` x its upper limit, and the pen at the screen's best centre (lying along the
axis that holds there, if given).

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_pick_random/render.py 038 075 --curl 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from hand_sampler.grammar import derive as dv  # noqa: E402
from hand_sampler.grammar.hand import hand_from_dict  # noqa: E402

ASSETS = REPO / "assets/urdf/27oct_grammar_random_hands"
PLOTS = REPO / "experiments/27oct_grammar_random_hands/plots/agent_pick_random"
COLS = ["#e65a46", "#4696e6", "#5abe64", "#e6aa32", "#a064d2", "#3cbebe"]


def capsule_points(p0, p1, r, n=10):
    d = p1 - p0
    L = np.linalg.norm(d)
    a = d / L if L > 1e-9 else np.array([1.0, 0, 0])
    ref = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, ref); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    pts = []
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        for ph in np.linspace(-np.pi / 2, np.pi / 2, 5):
            off = r * (np.cos(ph) * (np.cos(th) * u + np.sin(th) * v) + np.sin(ph) * a)
            pts.append((p1 if ph >= 0 else p0) + off)
    return np.array(pts)


def hull_faces(pts):
    h = ConvexHull(pts)
    return [pts[s] for s in h.simplices]


def pen_box(c, F):
    half = np.array([50.0, 15.0, 15.0])
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    return c + corners @ F.T


def draw(ax, H, q, pen=None, view=(0, 0)):
    for s in dv.solids_at(H, q):
        P = s.points / 1e-3
        if s.radius > 0:
            P = capsule_points(P[0], P[1], s.radius / 1e-3)
        k = s.key
        if k[0] in ("palm", "mount", "hinge"):
            col, al = "#9a9aa6", 0.35
        else:
            col, al = COLS[k[1] % 6], 0.85
        # world for drawing: X = palm y, Y = palm z, Z = -palm x (palm down, table below)
        W = np.stack([P[:, 1], P[:, 2], -P[:, 0]], 1)
        try:
            ax.add_collection3d(Poly3DCollection(hull_faces(W), facecolor=col, edgecolor="none", alpha=al))
        except Exception:
            pass
    if pen is not None:
        c, F = pen
        B = pen_box(np.array(c), F)
        W = np.stack([B[:, 1], B[:, 2], -B[:, 0]], 1)
        ax.add_collection3d(Poly3DCollection(hull_faces(W), facecolor="#3050c0", edgecolor="k", alpha=0.5, lw=0.3))
    lim = 190
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-2 * lim + 40, 40)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def curl_pose(H, curl):
    ds = dv.dofs(H)
    q = np.zeros(len(ds))
    for k, d in enumerate(ds):
        if d.kind == "flexion":
            q[k] = curl * d.limits[1]
    return q


def render(name, curl, metrics=None, out=None):
    H = hand_from_dict(json.load(open(ASSETS / name / f"{name}.design.json")))
    q = curl_pose(H, curl)
    fig = plt.figure(figsize=(12, 4.2))
    views = [(-90, -90, "from below (table's view)"), (0, -90, "side (palm y, down = table)"),
             (25, -50, "oblique")]
    for k, (el, az, lab) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        draw(ax, H, q, view=(el, az))
        ax.set_title(lab, fontsize=9)
    t = f"{name}: {len(H.fingers)} fingers, {len(dv.dofs(H))} joints, bends at {curl:.0%} of range"
    if metrics:
        t += "\n" + metrics
    fig.suptitle(t, fontsize=10)
    fig.tight_layout()
    out = out or PLOTS / f"{name}_curl{int(curl * 100)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("hands", nargs="+")
    ap.add_argument("--curl", type=float, default=0.5)
    a = ap.parse_args()
    for h in a.hands:
        nm = h if h.startswith("coarse") else f"coarse_s0_{h}"
        print(render(nm, a.curl))
