"""viser scene drawing for one hand (and its ghost, gallery and spread copies).

Geometry is built once per design in each body's LOCAL frame; a pose change
only moves handles (`position`/`wxyz`), so dragging a slider sends transforms,
not meshes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .model import (
    CellMesh,
    ModelView,
    OVERLAP_RGB,
    PALM_RGB,
    Primitives,
    capsule_mesh_local,
    mat_to_wxyz,
)

TIP_RGB = (245, 215, 60)
TIP_REACH_RGB = (40, 200, 90)
TIP_MISS_RGB = (235, 110, 40)
SPAWN_RGB = (90, 160, 255)


@dataclass
class RenderOptions:
    capsules: bool = True
    palm_cells: bool = True
    palm_spines: bool = False
    joint_axes: bool = True
    tips: bool = True
    body_labels: bool = False
    joint_labels: bool = False
    root_frame: bool = True
    capsule_mode: str = "simulator"
    opacity: float = 1.0
    cell_opacity: float = 0.35
    tint: Optional[Tuple[int, int, int]] = None   # one colour for everything (ghost/gallery)
    axis_length: float = 0.025


def _pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return mat_to_wxyz(T[:3, :3]), np.asarray(T[:3, 3], dtype=float)


class HandRenderer:
    """All scene nodes for one hand live under `root` (a frame whose pose is
    the hand's orientation: identity, or the env's palm-up base rotation)."""

    def __init__(self, server, root: str):
        self.server = server
        self.root = root
        self.frame = server.scene.add_frame(root, show_axes=False)
        self.view: Optional[ModelView] = None
        self.opts = RenderOptions()
        self.caps: Dict[str, object] = {}
        self.cells: Dict[str, object] = {}
        self.base_rgb: Dict[str, Tuple[int, int, int]] = {}
        self.cur_rgb: Dict[str, Tuple[int, int, int]] = {}
        self.arrows = None
        self.tip_handles: List[object] = []
        self.tip_bodies: List[str] = []
        self.body_labels: Dict[str, object] = {}
        self.joint_labels: Dict[str, object] = {}
        self.root_axes = None
        self._others: List[object] = []

    # ---- lifecycle --------------------------------------------------------

    def clear(self) -> None:
        for h in list(self.caps.values()) + list(self.cells.values()) + self.tip_handles + \
                list(self.body_labels.values()) + list(self.joint_labels.values()) + self._others:
            try:
                h.remove()
            except Exception:  # noqa: BLE001 - already gone
                pass
        for h in (self.arrows, self.root_axes):
            if h is not None:
                try:
                    h.remove()
                except Exception:  # noqa: BLE001
                    pass
        self.caps, self.cells, self.base_rgb, self.cur_rgb = {}, {}, {}, {}
        self.tip_handles, self.tip_bodies, self.body_labels, self.joint_labels = [], [], {}, {}
        self.arrows = self.root_axes = None
        self._others = []
        self.view = None

    def remove(self) -> None:
        self.clear()
        try:
            self.frame.remove()
        except Exception:  # noqa: BLE001
            pass

    def set_frame_pose(self, wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0)) -> None:
        self.frame.wxyz = np.asarray(wxyz, dtype=float)
        self.frame.position = np.asarray(position, dtype=float)

    def set_visible(self, visible: bool) -> None:
        self.frame.visible = bool(visible)

    def build(self, view: ModelView, cells: Optional[Dict[str, CellMesh]], opts: RenderOptions,
              on_click: Optional[Callable] = None) -> None:
        self.clear()
        self.view = view
        self.opts = opts
        s = self.server.scene
        prims = view.primitives({})
        draw_spines = opts.palm_spines or not cells  # without cells, the palm is its capsules
        for name, b in prims.bodies.items():
            rgb = opts.tint or b.rgb
            if b.palm and not draw_spines:
                continue
            if b.palm:
                rgb = opts.tint or PALM_RGB
            V, F = capsule_mesh_local(b.length, b.radius, opts.capsule_mode)
            wxyz, pos = _pose(b.T)
            h = s.add_mesh_simple(f"{self.root}/caps/{name}", V, F, color=rgb,
                                  opacity=None if opts.opacity >= 1.0 else opts.opacity,
                                  wxyz=wxyz, position=pos, visible=opts.capsules)
            if on_click is not None:
                h.on_click(on_click)
            self.caps[name] = h
            self.base_rgb[name] = rgb
            self.cur_rgb[name] = rgb
        if cells:
            for name, c in cells.items():
                if name not in prims.bodies or len(c.faces) == 0:
                    continue
                wxyz, pos = _pose(prims.bodies[name].T)
                rgb = opts.tint or PALM_RGB
                h = s.add_mesh_simple(f"{self.root}/cell/{name}", c.vertices.astype(np.float32), c.faces,
                                      color=rgb, opacity=opts.cell_opacity if opts.tint is None else
                                      min(opts.cell_opacity, opts.opacity), side="double",
                                      wxyz=wxyz, position=pos, visible=opts.palm_cells)
                if on_click is not None:
                    h.on_click(on_click)
                self.cells[name] = h
                self.base_rgb["cell:" + name] = rgb
                self.cur_rgb["cell:" + name] = rgb
        if prims.joints and opts.joint_axes:
            pts, cols = self._arrow_points(prims)
            r = view.radius
            self.arrows = s.add_arrows(f"{self.root}/axes", pts, cols, shaft_radius=0.12 * r,
                                       head_radius=0.3 * r, head_length=0.5 * r)
        if opts.tips:
            for i, t in enumerate(prims.tips):
                h = s.add_icosphere(f"{self.root}/tip/{i}", radius=0.45 * view.radius,
                                    color=opts.tint or TIP_RGB, position=t.position, subdivisions=2)
                self.tip_handles.append(h)
                self.tip_bodies.append(t.body)
        if opts.root_frame:
            self.root_axes = s.add_frame(f"{self.root}/root_axes", axes_length=0.03, axes_radius=0.0012)
        self.set_labels(opts.body_labels, opts.joint_labels, prims)

    def set_labels(self, bodies: bool, joints: bool, prims: Optional[Primitives] = None) -> None:
        s = self.server.scene
        if self.view is None:
            return
        prims = prims or self.view.primitives({})
        for h in list(self.body_labels.values()) + list(self.joint_labels.values()):
            h.remove()
        self.body_labels, self.joint_labels = {}, {}
        if bodies:
            for name, b in prims.bodies.items():
                mid = (b.T @ np.array([0.0, 0.0, 0.5 * b.length, 1.0]))[:3]
                self.body_labels[name] = s.add_label(f"{self.root}/lbl/b/{name}", name, position=mid,
                                                     font_screen_scale=0.7)
        if joints:
            for jp in prims.joints:
                self.joint_labels[jp.name] = s.add_label(f"{self.root}/lbl/j/{jp.name}", f"{jp.name} ({jp.kind[:3]})",
                                                         position=jp.position, font_screen_scale=0.6,
                                                         anchor="bottom-left")
        self.opts.body_labels, self.opts.joint_labels = bodies, joints

    # ---- per pose --------------------------------------------------------

    def _arrow_points(self, prims: Primitives) -> Tuple[np.ndarray, np.ndarray]:
        L = self.opts.axis_length
        pts = np.zeros((len(prims.joints), 2, 3))
        cols = np.zeros((len(prims.joints), 3), dtype=np.uint8)
        for i, jp in enumerate(prims.joints):
            pts[i, 0] = jp.position - 0.5 * L * jp.axis
            pts[i, 1] = jp.position + 0.5 * L * jp.axis
            cols[i] = self.opts.tint or jp.rgb
        return pts, cols

    def update(self, prims: Primitives, highlight: Iterable[str] = (),
               tip_colors: Optional[Dict[str, Tuple[int, int, int]]] = None) -> None:
        hl = set(highlight)
        for name, h in self.caps.items():
            b = prims.bodies.get(name)
            if b is None:
                continue
            wxyz, pos = _pose(b.T)
            h.wxyz, h.position = wxyz, pos
            want = OVERLAP_RGB if name in hl else self.base_rgb[name]
            if self.cur_rgb.get(name) != want:
                h.color = want
                self.cur_rgb[name] = want
        for name, h in self.cells.items():
            T = prims.cell_poses.get(name)
            if T is None:
                continue
            wxyz, pos = _pose(T)
            h.wxyz, h.position = wxyz, pos
            key = "cell:" + name
            want = OVERLAP_RGB if (name in hl and name not in self.caps) else self.base_rgb[key]
            if self.cur_rgb.get(key) != want:
                h.color = want
                self.cur_rgb[key] = want
        if self.arrows is not None:
            pts, _cols = self._arrow_points(prims)
            self.arrows.points = pts
        tip_pos = {t.body: t.position for t in prims.tips}
        for body, h in zip(self.tip_bodies, self.tip_handles):
            if body in tip_pos:
                h.position = tip_pos[body]
            if tip_colors is not None and self.opts.tint is None:
                h.color = tip_colors.get(body, TIP_RGB)
        for name, h in self.body_labels.items():
            b = prims.bodies.get(name)
            if b is not None:
                h.position = (b.T @ np.array([0.0, 0.0, 0.5 * b.length, 1.0]))[:3]
        jpos = {jp.name: jp.position for jp in prims.joints}
        for name, h in self.joint_labels.items():
            if name in jpos:
                h.position = jpos[name]

    def set_layer_visibility(self, capsules: Optional[bool] = None, cells: Optional[bool] = None,
                             axes: Optional[bool] = None, tips: Optional[bool] = None,
                             root_frame: Optional[bool] = None) -> None:
        if capsules is not None:
            for h in self.caps.values():
                h.visible = capsules
            self.opts.capsules = capsules
        if cells is not None:
            for h in self.cells.values():
                h.visible = cells
            self.opts.palm_cells = cells
        if axes is not None and self.arrows is not None:
            self.arrows.visible = axes
        if tips is not None:
            for h in self.tip_handles:
                h.visible = tips
        if root_frame is not None and self.root_axes is not None:
            self.root_axes.visible = root_frame

    def add_extra(self, handle) -> None:
        """Track an extra node (label, marker) so `clear` removes it."""
        self._others.append(handle)


class MeshOverlay:
    """The original URDF's link meshes, posed per original link."""

    def __init__(self, server, root: str):
        self.server = server
        self.root = root
        self.handles: Dict[str, List[object]] = {}
        self.opacity = 0.35
        self.visible = True

    def clear(self) -> None:
        for hs in self.handles.values():
            for h in hs:
                try:
                    h.remove()
                except Exception:  # noqa: BLE001
                    pass
        self.handles = {}

    def build(self, meshset, opacity: float, visible: bool, color=(200, 200, 205)) -> None:
        self.clear()
        self.opacity, self.visible = opacity, visible
        for link, pieces in meshset.pieces.items():
            hs = []
            for i, p in enumerate(pieces):
                hs.append(self.server.scene.add_mesh_simple(
                    f"{self.root}/{link}/{i}", p.vertices, p.faces, color=color, opacity=opacity,
                    side="double", visible=visible))
            self.handles[link] = hs

    def update(self, link_poses: Dict[str, np.ndarray]) -> None:
        for link, hs in self.handles.items():
            T = link_poses.get(link)
            if T is None:
                continue
            wxyz, pos = _pose(T)
            for h in hs:
                h.wxyz, h.position = wxyz, pos

    def set_opacity(self, opacity: float) -> None:
        self.opacity = opacity
        for hs in self.handles.values():
            for h in hs:
                h.opacity = opacity

    def set_visible(self, visible: bool) -> None:
        self.visible = visible
        for hs in self.handles.values():
            for h in hs:
                h.visible = visible
