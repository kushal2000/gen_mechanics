"""Each fitted hand drawn on top of the vendor model it was fitted from.

A fit reports its error as numbers -- base slip, fingertip error, axis
obliquity -- and numbers do not say WHERE a hand went wrong. This draws the
grammar's version solid over the vendor's own geometry in ghost, in one frame,
so the difference is the thing you look at.

Run it:

    python -m hand_sampler.overlay_fits --out overlay.png

Not part of the viewer. The viewer is for walking the design space; this is a
one-shot picture for checking the imports, and it is the only thing here that
reads a vendor's MESHES rather than its joint origins.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hand_sampler import commercial as C
from hand_sampler import design_space as D
from hand_sampler import viewer

GHOST = "#7d838d"
GHOST_ALPHA = 0.10
CLUSTER_PITCH = 0.003
"""Decimation grid. LEAP's palm alone is 87,300 faces and matplotlib draws every
one as a depth-sorted polygon, so the vendor meshes are decimated first.

By vertex CLUSTERING -- snap to a 3 mm grid, re-index, drop the faces that
collapse -- which takes that palm to 8,194 faces in 0.05 s with its extents
unchanged to a millimetre. Written out because the alternatives here are worse:
trimesh's quadric decimation needs `fast_simplification` and its voxel remesh
needs `skimage`, neither of which is in the env, and the convex hull of that
same palm has three times its volume. Subsampling faces is what the first
version did, and it draws a cloud of disconnected triangles rather than a hand.
"""


# --- the vendor's own geometry ----------------------------------------------

def _link_visuals(root: ET.Element, urdf_dir: Path) -> dict[str, list[tuple]]:
    """``link name -> [(mesh path, 4x4 visual origin)]``, for links that have one."""
    out: dict[str, list[tuple]] = {}
    for link in root.findall("link"):
        items = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            name = mesh.get("filename")
            if name.startswith("package://"):
                continue                      # not vendored -- see MIDAS
            local = np.eye(4)
            origin = visual.find("origin")
            if origin is not None:
                local[:3, :3] = D.rpy_to_mat(
                    [float(v) for v in (origin.get("rpy") or "0 0 0").split()])
                local[:3, 3] = [float(v) for v in (origin.get("xyz") or "0 0 0").split()]
            items.append(((urdf_dir / name).resolve(), local))
        if items:
            out[link.get("name")] = items
    return out


def _link_world_frames(J: dict) -> dict[str, np.ndarray]:
    """Every link's pose in the URDF root's frame, at the REST pose.

    Every joint at zero, which is the pose the fit reads and the pose the
    grammar's own drawing is in. A joint's own axis never enters it.
    """
    children = {d["child"]: (n, d) for n, d in J.items()}
    roots = {d["parent"] for d in J.values()} - set(children)
    frames = {r: np.eye(4) for r in roots}
    pending = dict(J)
    while pending:
        progressed = False
        for n, d in list(pending.items()):
            if d["parent"] not in frames:
                continue
            step = np.eye(4)
            step[:3, :3] = D.rpy_to_mat(d["rpy"])
            step[:3, 3] = d["xyz"]
            frames[d["child"]] = frames[d["parent"]] @ step
            del pending[n]
            progressed = True
        if not progressed:                    # a loop closure, e.g. MIDAS's four-bars
            break
    return frames


def vendor_triangles(name: str) -> list[np.ndarray]:
    """The vendor's visual meshes at rest, as triangles in its own root frame.

    Empty when a vendor's meshes are not vendored -- MIDAS ships `package://`
    URIs and PROVENANCE.md says only its URDF was copied -- and the caller falls
    back to the skeleton.
    """
    import trimesh

    def cluster(mesh):
        q = np.round(np.asarray(mesh.vertices) / CLUSTER_PITCH).astype(np.int64)
        uniq, inv = np.unique(q, axis=0, return_inverse=True)
        f = inv[np.asarray(mesh.faces)]
        keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        return uniq.astype(float) * CLUSTER_PITCH, f[keep]

    path = C.urdf_of(name)
    root = ET.parse(path).getroot()
    frames = _link_world_frames(C._read(path))
    out = []
    for link, visuals in _link_visuals(root, path.parent).items():
        world = frames.get(link)
        if world is None:
            continue
        for mesh_path, local in visuals:
            if not mesh_path.exists():
                continue
            m = trimesh.load(mesh_path, force="mesh")
            verts, faces = cluster(m)
            if not len(faces):
                continue
            T = world @ local
            out.append((verts @ T[:3, :3].T + T[:3, 3])[faces])
    return out


def vendor_skeleton(name: str) -> list[np.ndarray]:
    """Each digit as a polyline through its joints to its tip, in the same frame.

    The fallback when the meshes are not here, and worth drawing even when they
    are: it is what the fit actually reads.
    """
    return [np.array(list(d.pos) + [d.tip]) for d in C.digits(name)]


# --- into the frame the fitted hand lives in --------------------------------

def vendor_to_palm(name: str) -> np.ndarray:
    """4x4 taking the vendor's root frame into the palm frame.

    The same two steps `commercial.fit` takes -- rotate onto the palm axes, then
    put the origin at the centroid of the digit bases -- plus PALM_CENTRE, since
    that is where the fit then places that centroid.
    """
    ds = [C._straighten(d) for d in C.digits(name)]
    row, thumb = C._split(ds)
    M = C._palm_axes(row)
    base = np.array([d.pos[0] for d in (row + [thumb])]) @ M
    origin = base.mean(axis=0)
    origin[0] = 0.0
    T = np.eye(4)
    T[:3, :3] = M.T                     # p @ M == M.T @ p, for a column vector
    T[:3, 3] = D.PALM_CENTRE - origin
    return T


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def alignment_error(name: str, hand) -> tuple[np.ndarray, np.ndarray]:
    """Per digit, vendor base to fitted mount: ``(in-plane, off-plane)`` in mm.

    The overlay is only worth looking at if the two hands are in the same frame,
    and the IN-PLANE number is the check: it is the fit's own reported base slip
    arrived at a second way, and the two agree to a tenth of a millimetre on
    every digit of every hand. If the transform above were wrong they would not.

    The OFF-PLANE number is x, the palm normal, and the fit does not report it
    at all -- a mount is a (y, z) offset, so every base is forced onto the
    midplane and whatever the vendor had in x is dropped. It is not small:
    6.3 mm across LEAP's row, 9.5 across MIDAS's, and 18.2 on Allegro's thumb.
    Drawing the two hands together is what made it visible.
    """
    T = vendor_to_palm(name)
    ds = [C._straighten(d) for d in C.digits(name)]
    row, thumb = C._split(ds)
    v = np.array([_apply(T, d.pos[0]) - D.mount_position(f.mount)
                  for d, f in zip(row + [thumb], hand.fingers)])
    return np.linalg.norm(v[:, 1:], axis=1) * 1000, np.abs(v[:, 0]) * 1000


# --- the picture -------------------------------------------------------------

def draw_overlay(ax, name: str, hand, meshes: bool = True,
                 alpha: float = GHOST_ALPHA) -> str:
    """The vendor in ghost, the fit solid on top. Returns what was drawn."""
    T = vendor_to_palm(name)
    tris = vendor_triangles(name) if meshes else []
    if tris:
        ax.add_collection3d(Poly3DCollection(
            np.vstack([_apply(T, t.reshape(-1, 3)).reshape(-1, 3, 3) for t in tris]),
            facecolor=GHOST, alpha=alpha, edgecolor="none", zorder=0))
        drew = "vendor meshes"
    else:
        drew = "vendor skeleton (its meshes are not vendored)"
    for poly in vendor_skeleton(name):
        p = _apply(T, poly)
        ax.plot(*p.T, color=GHOST, linewidth=2.0, alpha=0.9,
                solid_capstyle="round", zorder=1)
        ax.scatter(*p[:-1].T, s=8, c=GHOST, depthshade=False, zorder=1)
    return drew


def figure(names=None, out: str = "overlay.png", meshes: bool = True,
           span: float = 0.22, alpha: float = GHOST_ALPHA) -> str:
    names = list(names or C.HANDS)
    fig = plt.figure(figsize=(3.4 * len(names), 4.2), dpi=140)
    for i, name in enumerate(names):
        hand, _ = C.fit(name)
        ax = fig.add_subplot(1, len(names), i + 1, projection="3d")
        drew = draw_overlay(ax, name, hand, meshes=meshes, alpha=alpha)
        viewer.draw(ax, hand, "", 0.0)
        inplane, offplane = alignment_error(name, hand)
        print(f"  {name:8s} base in-plane " + " ".join(f"{e:5.1f}" for e in inplane)
              + "  |  off the midplane " + " ".join(f"{e:5.1f}" for e in offplane)
              + "  mm")
        # viewer.draw fixes a 0.16 m box around the design; the vendor hand is
        # bigger than the fit in places, so widen it or the ghost gets clipped.
        ax.set_xlim(-span / 2, span / 2)
        ax.set_ylim(-span / 2, span / 2)
        ax.set_zlim(-span / 4, 3 * span / 4)
        ax.set_title(f"{name}   {hand.n_fingers}f {hand.n_joints}j   "
                     f"curl {D.curl_score(hand):.2f}\n{drew}\n"
                     f"base in-plane {inplane.max():.0f} mm, "
                     f"off-midplane {offplane.max():.0f} mm (worst)",
                     fontsize=8, pad=-2)
    fig.suptitle("the fit, solid, over the vendor it came from, in ghost",
                 fontsize=11, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  ({len(names)} hands)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="overlay.png")
    ap.add_argument("--hands", default=None,
                    help="comma-separated; defaults to commercial.HANDS")
    ap.add_argument("--no-meshes", action="store_true",
                    help="skeletons only, which is what the fit actually reads")
    ap.add_argument("--span", type=float, default=0.22, help="box side, in metres")
    ap.add_argument("--alpha", type=float, default=GHOST_ALPHA,
                    help="ghost opacity. A mesh is many overlapping layers, so "
                         "this compounds -- 0.10 shows the fit through a hand")
    args = ap.parse_args()
    figure(args.hands.split(",") if args.hands else None,
           out=args.out, meshes=not args.no_meshes, span=args.span,
           alpha=args.alpha)


if __name__ == "__main__":
    main()
