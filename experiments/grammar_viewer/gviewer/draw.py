"""Meshes for the grammar viewer: every Feetech servo with its output horn
and axis, the link boxes, the mount, palm plates, and the commercial URDF
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
PALM_OPACITY = 0.3          # the plates: just enough to see, so the servos in the palm show through
LINK_OPACITY = 0.25         # the link bodies by default: see-through, so the servos inside and the fingertips show
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
    return hull_mesh(np.array(key, dtype=float).reshape(-1, 3))


def box_mesh(corners_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The mesh of a box's 8 corners or a capsule's surface points (metres,
    in the frame it is drawn in)."""
    return _box_mesh(tuple(np.round(np.asarray(corners_m, dtype=float), 9).ravel().tolist()))


def plate_mesh(outline_mm: np.ndarray, origin_mm=(0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    return hull_mesh(gdv.plate_points_m(outline_mm, origin_mm))


def _darker(rgb: Tuple[int, int, int], f: float = 0.6) -> Tuple[int, int, int]:
    return tuple(int(c * f) for c in rgb)


def _pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.array([w, x, y, z]), T[:3, 3].copy()


HORN_RGB = (235, 235, 240)
AXIS_RGB = (25, 25, 30)
HORN_RADIUS_MM = 9.0          # the output horn's disc
HORN_THICK_MM = 3.0           # inside the 29 mm the servo box spans along its axis (horn and idler included)
HORN_ARM_MM = (14.0, 2.5)     # a radial arm (length, half width), so the horn's turn shows
AXIS_HALF_MM = 35.0
AXIS_RADIUS_MM = 0.8


def _prism(points2d: np.ndarray, u: np.ndarray, v: np.ndarray, n: np.ndarray, c: np.ndarray,
           h0: float, h1: float) -> np.ndarray:
    """Corners (mm) of a prism: the 2D outline in the (u, v) plane at centre
    c, from h0 to h1 along n."""
    out = []
    for h in (h0, h1):
        for x, y in points2d:
            out.append(c + x * u + y * v + h * n)
    return np.array(out)


def _join(meshes) -> Tuple[np.ndarray, np.ndarray]:
    Vs, Fs, k = [], [], 0
    for V, F in meshes:
        Vs.append(V)
        Fs.append(F + k)
        k += len(V)
    return np.concatenate(Vs).astype(np.float32), np.concatenate(Fs).astype(np.int32)


def horn_mesh(box, joint_mm: np.ndarray = np.zeros(3)) -> Tuple[np.ndarray, np.ndarray]:
    """The output horn of a servo (`derive.Box`, mm) whose axis passes through
    `joint_mm` (the frame's origin for a finger joint): a disc on the motor's
    +axis end, centred on the axis, and one radial arm along the motor's
    front (-back); metres."""
    a, v, w = box.axes[:, 0], box.axes[:, 1], box.axes[:, 2]
    joint_mm = np.asarray(joint_mm, dtype=float)
    c = joint_mm + float((box.centre - joint_mm) @ a) * a      # on the axis, at the motor's slide
    end = box.half[0]
    t = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    disc = np.stack([np.cos(t), np.sin(t)], axis=1) * HORN_RADIUS_MM
    D = _prism(disc, v, w, a, c, end - HORN_THICK_MM, end + 0.5)
    L, hw = HORN_ARM_MM
    arm2d = np.array([[0.0, -hw], [-L, -hw], [-L, hw], [0.0, hw]])
    A = _prism(arm2d, v, w, a, c, end - HORN_THICK_MM, end + 1.0)
    return _join([hull_mesh(D * MM), hull_mesh(A * MM)])


def axis_mesh(a: np.ndarray, c_mm: np.ndarray = np.zeros(3)) -> Tuple[np.ndarray, np.ndarray]:
    """A thin rod along the unit axis `a` through `c_mm` (metres)."""
    a = np.asarray(a, dtype=float)
    u = np.cross(a, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(a, [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    v = np.cross(a, u)
    t = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = np.stack([np.cos(t), np.sin(t)], axis=1) * AXIS_RADIUS_MM
    return hull_mesh(_prism(ring, u, v, a, np.asarray(c_mm, dtype=float), -AXIS_HALF_MM, AXIS_HALF_MM) * MM)


class HandDrawing:
    """One grammar hand in the viser scene: the main plate, the mount, the
    part beyond a cut, and every Feetech servo with its output horn and a rod
    along its axis (the hinge servo too); each link's box (the servo it
    carries and its yoke) at `link_opacity` (hidden at 0). `build` makes the
    meshes for a hand; `pose` moves them to a joint vector. One colour per
    finger; solids of a failing C1 pair in red (their link boxes shown)."""

    def __init__(self, server, root: str = "/hand"):
        self.server = server
        self.root = root
        self.handles: List[object] = []
        self.bodies: Dict[Tuple, object] = {}
        self.object: Optional[object] = None
        self.hand: Optional[Hand] = None
        self.hot: set = set()
        self.link_opacity = LINK_OPACITY
        self.frame = server.scene.add_frame(root, show_axes=False, wxyz=_pose(_pad(PALM_UP))[0])

    def clear(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self.handles, self.bodies, self.object = [], {}, None

    def _add(self, key: Tuple, name: str, mesh, color, opacity: Optional[float] = None, visible: bool = True) -> None:
        V, F = mesh
        hot = key in self.hot or (key[0] == "motor" and len(key) == 3 and key[2] == 0 and ("motor", key[1]) in self.hot)
        h = self.server.scene.add_mesh_simple(f"{self.root}/{name}", V, F, color=HOT_RGB if hot else color,
                                              opacity=opacity, flat_shading=True, visible=visible)
        self.bodies[key] = h
        self.handles.append(h)

    def _add_links(self) -> None:
        hand = self.hand
        for i, f in enumerate(hand.fingers):
            rgb = FINGER_RGB[i % len(FINGER_RGB)]
            for j in range(len(f.joints)):
                key = ("link", i, j)
                hot = key in self.hot
                opacity = 0.6 if hot else self.link_opacity
                self._add(key, f"f{i}/link{j}", box_mesh(gdv.link_points_m(hand, i, j)), rgb,
                          opacity=opacity if opacity < 1.0 else None, visible=hot or opacity > 0.0)

    def set_link_opacity(self, opacity: float, q: Optional[np.ndarray] = None) -> None:
        """Redraw the link boxes at `opacity` (viser cannot change a mesh's
        opacity in place)."""
        self.link_opacity = float(opacity)
        if self.hand is None:
            return
        for key in [k for k in self.bodies if k[0] == "link"]:
            h = self.bodies.pop(key)
            h.remove()
            self.handles = [x for x in self.handles if x is not h]
        self._add_links()
        if q is not None:
            self.pose(q)

    def build(self, hand: Hand, highlight: Sequence[Tuple] = (), object_half_m: Optional[float] = None,
              link_opacity: Optional[float] = None) -> None:
        """Meshes for `hand`; with `object_half_m`, also the task's cube (that
        half size) where it starts (`derive.object_start_m`), half
        transparent."""
        self.clear()
        self.hand = hand
        self.hot = {k for pair in highlight for k in pair}
        if link_opacity is not None:
            self.link_opacity = float(link_opacity)
        outlines = gdv.palm_outlines_mm(hand)
        self._add(("palm", -1), "palm", plate_mesh(outlines[-1]), PALM_RGB, opacity=PALM_OPACITY)
        self._add(("mount",), "mount", hull_mesh(gdv.mount_points_m()), MOUNT_RGB)
        g = gdv.hand_geometry(hand)
        if hand.palm_joints:
            hinge = gdv.palm_hinge_m(hand) / MM
            self._add(("palm", 0), "section", plate_mesh(outlines[0], tuple(hinge[1:])), SECTION_RGB,
                      opacity=PALM_OPACITY)
            hb = g.hinge_box
            self._add(("hinge",), "hinge_motor", box_mesh(hb.corners() * MM), MOUNT_RGB)
            # the horn and the rod turn with the part beyond the cut: drawn in the palm frame, posed with it
            self._add(("hinge_horn",), "hinge_horn", horn_mesh(hb, hinge), HORN_RGB)
            self._add(("hinge_axis",), "hinge_axis", axis_mesh(hb.axes[:, 0], hinge), AXIS_RGB)
        for i, f in enumerate(hand.fingers):
            rgb = FINGER_RGB[i % len(FINGER_RGB)]
            fg = gdv.finger_geometry(f)
            for j in range(len(f.joints)):
                box = fg.motors[j]
                self._add(("motor", i, j), f"f{i}/motor{j}", box_mesh(box.corners() * MM), _darker(rgb, 0.8))
                self._add(("horn", i, j), f"f{i}/horn{j}", horn_mesh(box), HORN_RGB)
                self._add(("axis", i, j), f"f{i}/axis{j}", axis_mesh(dv_axis(hand, i, j)), AXIS_RGB)
        self._add_links()
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

        def put(key, T):
            h = self.bodies.get(key)
            if h is not None:
                h.wxyz, h.position = _pose(T)

        if self.hand.palm_joints:
            put(("palm", 0), p.palm[0] @ _translate(gdv.palm_hinge_m(self.hand)))
            put(("hinge_horn",), p.palm[0])
            put(("hinge_axis",), p.palm[0])
        for i, f in enumerate(self.hand.fingers):
            for j in range(len(f.joints)):
                put(("motor", i, j), p.pre[i][j])
                put(("horn", i, j), p.links[i][j])
                put(("axis", i, j), p.links[i][j])
                put(("link", i, j), p.links[i][j])

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


def dv_axis(hand: Hand, i: int, j: int) -> np.ndarray:
    """Finger i's joint j axis (unit, its link frame)."""
    return gdv.direction(*hand.fingers[i].joints[j].axis)


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

    def build(self, meshset, opacity: float = 0.35, color=(205, 205, 210), scale: float = 1.0) -> None:
        """The meshes, `scale` x their size (a commercial hand is fitted at the
        scale it is buildable from Feetech servos)."""
        self.clear()
        for link, pieces in meshset.pieces.items():
            self.handles[link] = [self.server.scene.add_mesh_simple(
                f"{self.root}/{link}/{n}", p.vertices, p.faces, color=color, opacity=opacity, side="double",
                scale=float(scale), visible=self.visible) for n, p in enumerate(pieces)]

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
