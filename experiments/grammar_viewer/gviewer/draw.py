"""Meshes for the grammar viewer: rounded-box links, palm plates, joint
markers, and the commercial URDF overlay. Numpy + scipy; viser only through
the server handle passed in."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from hand_sampler.grammar import derive as gdv
from hand_sampler.grammar.hand import LINK_RADIUS_MM, Hand

MM = 1e-3
FINGER_RGB = [(230, 90, 70), (70, 150, 230), (90, 190, 100), (230, 170, 50), (160, 100, 210), (60, 190, 190)]
PALM_RGB = (175, 175, 182)
SECTION_RGB = (150, 150, 165)
JOINT_RGB = (40, 40, 48)
PALM_UP = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
"""Palm frame -> viewer world: palm normal up (+z), fingers along +x."""


def _icosphere(radius: float) -> np.ndarray:
    t = (1.0 + 5 ** 0.5) / 2.0
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=float)
    hull = ConvexHull(v)
    verts = list(v / np.linalg.norm(v, axis=1, keepdims=True))
    for a, b, c in hull.simplices:          # one subdivision: edge midpoints
        for i, j in ((a, b), (b, c), (c, a)):
            m = verts[i] + verts[j]
            verts.append(m / np.linalg.norm(m))
    return np.unique(np.round(np.array(verts), 9), axis=0) * radius


_SPHERE = _icosphere(1.0)


def hull_mesh(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(vertices float32, faces int32) of the convex hull, outward winding."""
    hull = ConvexHull(points)
    used = np.unique(hull.simplices)
    remap = {int(v): i for i, v in enumerate(used)}
    V = hull.points[used]
    F = np.array([[remap[int(i)] for i in tri] for tri in hull.simplices])
    c = V.mean(axis=0)
    for t in F:
        a, b, cc = V[t[0]], V[t[1]], V[t[2]]
        if np.dot(np.cross(b - a, cc - a), a - c) < 0:
            t[1], t[2] = t[2], t[1]
    return V.astype(np.float32), F.astype(np.int32)


@lru_cache(maxsize=512)
def rounded_box(x0_mm: float, x1_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """A link's rounded box (core x0..x1 along the link, the shared
    cross-section) in its link frame, metres."""
    centre = np.array([(x0_mm + x1_mm) / 2.0, 0.0, 0.0]) * MM
    half = np.array([(x1_mm - x0_mm) / 2.0, gdv.CORE_HALF_Y_MM, gdv.CORE_HALF_Z_MM]) * MM
    corners = gdv.box_corners(centre, half)
    pts = (corners[:, None, :] + _SPHERE[None] * LINK_RADIUS_MM * MM).reshape(-1, 3)
    return hull_mesh(pts)


def plate_mesh(outline_mm: np.ndarray, origin_mm=(0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    return hull_mesh(gdv.plate_points_m(outline_mm, origin_mm))


def _pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.array([w, x, y, z]), T[:3, 3].copy()


class HandDrawing:
    """One grammar hand in the viser scene: plates, links, joint markers.
    `build` makes the meshes for a hand; `pose` moves them to a joint
    vector."""

    def __init__(self, server, root: str = "/hand"):
        self.server = server
        self.root = root
        self.handles: List[object] = []
        self.links: Dict[Tuple[int, int], object] = {}
        self.sections: Dict[int, object] = {}
        self.markers: Dict[Tuple[int, int], object] = {}
        self.hand: Optional[Hand] = None
        self.frame = server.scene.add_frame(root, show_axes=False, wxyz=_pose(_pad(PALM_UP))[0])

    def clear(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self.handles, self.links, self.sections, self.markers = [], {}, {}, {}

    def build(self, hand: Hand, highlight: Sequence[Tuple] = ()) -> None:
        self.clear()
        self.hand = hand
        s = self.server.scene
        outlines = gdv.palm_outlines_mm(hand)
        V, F = plate_mesh(outlines[-1])
        self.handles.append(s.add_mesh_simple(f"{self.root}/palm", V, F, color=PALM_RGB, flat_shading=True))
        for k, p in enumerate(hand.palm_joints):
            V, F = plate_mesh(outlines[k], (p.y, p.z))
            h = s.add_mesh_simple(f"{self.root}/section{k}", V, F, color=SECTION_RGB, flat_shading=True)
            self.sections[k] = h
            self.handles.append(h)
        hot = {k for pair in highlight for k in pair}
        for i, f in enumerate(hand.fingers):
            rgb = FINGER_RGB[i % len(FINGER_RGB)]
            for j, jt in enumerate(f.joints):
                x0, x1 = gdv.link_core_x_mm(jt.length, j == len(f.joints) - 1)
                V, F = rounded_box(float(x0), float(x1))
                color = (220, 30, 30) if ("link", i, j) in hot else rgb
                h = s.add_mesh_simple(f"{self.root}/f{i}/l{j}", V, F, color=color)
                self.links[(i, j)] = h
                m = s.add_icosphere(f"{self.root}/f{i}/j{j}", radius=0.0045, color=JOINT_RGB)
                self.markers[(i, j)] = m
                self.handles += [h, m]

    def pose(self, q: np.ndarray) -> None:
        if self.hand is None:
            return
        p = gdv.fk(self.hand, q)
        for k, h in self.sections.items():
            T = p.palm[k] @ _translate(gdv.palm_hinge_m(self.hand, k))
            h.wxyz, h.position = _pose(T)
        for (i, j), h in self.links.items():
            T = p.links[i][j]
            h.wxyz, h.position = _pose(T)
            self.markers[(i, j)].position = T[:3, 3]

    def set_visible(self, visible: bool) -> None:
        for h in self.handles:
            h.visible = visible

    def bounds(self, q: np.ndarray) -> Tuple[np.ndarray, float]:
        """(centre, radius) of the hand in viewer world coordinates."""
        if self.hand is None:
            return np.zeros(3), 0.2
        p = gdv.fk(self.hand, q)
        pts = [p.tips.reshape(-1, 3), np.zeros((1, 3))] + [l[:, :3, 3] for l in p.links]
        pts = np.concatenate(pts) @ PALM_UP.T
        c = (pts.min(0) + pts.max(0)) / 2.0
        return c, float(np.linalg.norm(pts.max(0) - pts.min(0)) / 2.0 + 0.03)


def _translate(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = p
    return T


def _pad(R: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    return T


class MeshOverlay:
    """A commercial hand's URDF meshes, placed in the palm frame
    (`palm_T^-1` applied) and posed per URDF link."""

    def __init__(self, server, root: str = "/hand/urdf"):
        self.server = server
        self.root = root
        self.handles: Dict[str, List[object]] = {}
        self.visible = True

    def clear(self) -> None:
        for hs in self.handles.values():
            for h in hs:
                try:
                    h.remove()
                except Exception:  # noqa: BLE001
                    pass
        self.handles = {}

    def build(self, meshset, opacity: float = 0.35, color=(205, 205, 210)) -> None:
        self.clear()
        for link, pieces in meshset.pieces.items():
            self.handles[link] = [self.server.scene.add_mesh_simple(
                f"{self.root}/{link}/{n}", p.vertices, p.faces, color=color, opacity=opacity, side="double",
                visible=self.visible) for n, p in enumerate(pieces)]

    def update(self, link_poses: Dict[str, np.ndarray]) -> None:
        for link, hs in self.handles.items():
            T = link_poses.get(link)
            if T is None:
                continue
            wxyz, pos = _pose(T)
            for h in hs:
                h.wxyz, h.position = wxyz, pos

    def set_visible(self, visible: bool) -> None:
        self.visible = visible
        for hs in self.handles.values():
            for h in hs:
                h.visible = visible
