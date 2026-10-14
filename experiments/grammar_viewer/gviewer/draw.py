"""Meshes for the grammar viewer: servo-sized links, base motors, the mount,
the hinge motor, palm plates, joint markers, and the commercial URDF
overlay. Numpy + scipy; viser only through the server handle passed in."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from hand_sampler.grammar import derive as gdv
from hand_sampler.grammar.hand import Hand, finger_section

MM = 1e-3
FINGER_RGB = [(230, 90, 70), (70, 150, 230), (90, 190, 100), (230, 170, 50), (160, 100, 210), (60, 190, 190)]
PALM_RGB = (175, 175, 182)
SECTION_RGB = (150, 150, 165)
MOUNT_RGB = (110, 110, 118)
JOINT_RGB = (40, 40, 48)
HOT_RGB = (220, 30, 30)
OBJECT_RGB = (70, 110, 230)
PALM_UP = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
"""Palm frame -> viewer world: palm normal up (+z), +z of the palm along +x."""


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


@lru_cache(maxsize=1024)
def _box_mesh(key: Tuple[float, ...]) -> Tuple[np.ndarray, np.ndarray]:
    return hull_mesh(np.array(key, dtype=float).reshape(8, 3))


def box_mesh(corners_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """A box's mesh from its 8 corners (metres, in the frame it is drawn in)."""
    return _box_mesh(tuple(np.round(np.asarray(corners_m, dtype=float), 9).ravel().tolist()))


def plate_mesh(outline_mm: np.ndarray, origin_mm=(0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    return hull_mesh(gdv.plate_points_m(outline_mm, origin_mm))


def _darker(rgb: Tuple[int, int, int], f: float = 0.6) -> Tuple[int, int, int]:
    return tuple(int(c * f) for c in rgb)


def _pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.array([w, x, y, z]), T[:3, 3].copy()


class HandDrawing:
    """One grammar hand in the viser scene: one body per solid (the main
    plate, the mount, the hinge motor, the part beyond the cut, each base
    motor, each link) and joint markers. `build` makes the meshes for a
    hand; `pose` moves them to a joint vector. One colour per finger (its
    base motor darker), no other markers."""

    def __init__(self, server, root: str = "/hand"):
        self.server = server
        self.root = root
        self.handles: List[object] = []
        self.bodies: Dict[Tuple, object] = {}
        self.markers: Dict[Tuple[int, int], object] = {}
        self.object: Optional[object] = None
        self.hand: Optional[Hand] = None
        self.frame = server.scene.add_frame(root, show_axes=False, wxyz=_pose(_pad(PALM_UP))[0])

    def clear(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self.handles, self.bodies, self.markers, self.object = [], {}, {}, None

    def _add(self, key: Tuple, name: str, mesh, color, hot) -> None:
        V, F = mesh
        h = self.server.scene.add_mesh_simple(f"{self.root}/{name}", V, F, color=HOT_RGB if key in hot else color,
                                              flat_shading=True)
        self.bodies[key] = h
        self.handles.append(h)

    def build(self, hand: Hand, highlight: Sequence[Tuple] = (), object_half_m: Optional[float] = None) -> None:
        """Meshes for `hand`; with `object_half_m`, also the task's cube (that
        half size) where it starts (`derive.object_start_m`), half
        transparent."""
        self.clear()
        self.hand = hand
        hot = {k for pair in highlight for k in pair}
        outlines = gdv.palm_outlines_mm(hand)
        self._add(("palm", -1), "palm", plate_mesh(outlines[-1]), PALM_RGB, hot)
        self._add(("mount",), "mount", hull_mesh(gdv.mount_points_m()), MOUNT_RGB, hot)
        g = gdv.hand_geometry(hand)
        if hand.palm_joints:
            hinge = gdv.palm_hinge_m(hand) / MM
            self._add(("palm", 0), "section", plate_mesh(outlines[0], tuple(hinge[1:])), SECTION_RGB, hot)
            self._add(("hinge",), "hinge_motor", box_mesh(g.hinge_box.corners() * MM), MOUNT_RGB, hot)
        for i, f in enumerate(hand.fingers):
            rgb = FINGER_RGB[i % len(FINGER_RGB)]
            motor = gdv.base_motor_boxes_mm(f)[0].corners() * MM
            if finger_section(hand, i) == 0:          # drawn relative to the hinge, moved with the part beyond the cut
                motor = motor - gdv.palm_hinge_m(hand)
            self._add(("motor", i), f"f{i}/motor", box_mesh(motor), _darker(rgb), hot)
            for j in range(len(f.joints)):
                self._add(("link", i, j), f"f{i}/l{j}", box_mesh(gdv.link_points_m(hand, i, j)), rgb, hot)
                m = self.server.scene.add_icosphere(f"{self.root}/f{i}/j{j}", radius=0.0045, color=JOINT_RGB)
                self.markers[(i, j)] = m
                self.handles.append(m)
        if object_half_m is not None:
            c = gdv.object_start_m(hand, object_half_m)
            V, F = box_mesh(gdv.box_corners(c, np.full(3, object_half_m)))
            self.object = self.server.scene.add_mesh_simple(f"{self.root}/object", V, F, color=OBJECT_RGB,
                                                            opacity=0.4, flat_shading=True)
            self.handles.append(self.object)

    def pose(self, q: np.ndarray) -> None:
        if self.hand is None:
            return
        p = gdv.fk(self.hand, q)
        if self.hand.palm_joints:
            T = p.palm[0] @ _translate(gdv.palm_hinge_m(self.hand))
            wxyz, pos = _pose(T)
            self.bodies[("palm", 0)].wxyz, self.bodies[("palm", 0)].position = wxyz, pos
            for i in range(len(self.hand.fingers)):
                if finger_section(self.hand, i) == 0:
                    self.bodies[("motor", i)].wxyz, self.bodies[("motor", i)].position = wxyz, pos
        for (i, j), m in self.markers.items():
            T = p.links[i][j]
            h = self.bodies[("link", i, j)]
            h.wxyz, h.position = _pose(T)
            m.position = T[:3, 3]

    def set_visible(self, visible: bool) -> None:
        for h in self.handles:
            h.visible = visible

    def bounds(self, q: np.ndarray) -> Tuple[np.ndarray, float]:
        """(centre, radius) of the hand in viewer world coordinates."""
        if self.hand is None:
            return np.zeros(3), 0.2
        p = gdv.fk(self.hand, q)
        pts = [p.tips.reshape(-1, 3), gdv.mount_points_m()] + [l[:, :3, 3] for l in p.links]
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
