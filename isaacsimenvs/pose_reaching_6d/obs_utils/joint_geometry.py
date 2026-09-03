"""Kit-free construction of ordered SHARPA per-joint link boxes from its URDF.

The four returned points are expressed in the moving child-link frame of each
joint.  They are ordered as one corner and its three adjacent corners::

    p0 -> p1   proximal-to-distal link direction
    p0 -> p2   positive rotation (bending) direction
    p0 -> p3   signed joint-axis direction

The centre of the proximal face is the joint origin.  Consequently the points
encode link length/thickness, joint orientation and axis sign without a joint
name, role one-hot, or learned identity embedding.
"""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from hand_sampler.paths import resolve


def _values(node: ET.Element | None, key: str, default) -> np.ndarray:
    if node is None or node.get(key) is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in node.get(key).split()], dtype=np.float64)


def _origin(node: ET.Element | None) -> np.ndarray:
    """URDF origin as a homogeneous parent-from-child transform."""
    xyz = _values(node, "xyz", (0.0, 0.0, 0.0))
    r, p, y = _values(node, "rpy", (0.0, 0.0, 0.0))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rot = np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = xyz
    return out


def _box_corners(extents) -> np.ndarray:
    half = 0.5 * np.asarray(extents, dtype=np.float64)
    return np.asarray([
        (sx * half[0], sy * half[1], sz * half[2])
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ])


def _collision_vertices(collision: ET.Element, base_dir: Path) -> np.ndarray:
    geom = collision.find("geometry")
    if geom is None:
        return np.empty((0, 3), dtype=np.float64)
    box = geom.find("box")
    cylinder = geom.find("cylinder")
    sphere = geom.find("sphere")
    mesh = geom.find("mesh")
    if box is not None:
        vertices = _box_corners(_values(box, "size", (0.0, 0.0, 0.0)))
    elif cylinder is not None:
        radius = float(cylinder.get("radius"))
        length = float(cylinder.get("length"))
        # Its AABB is sufficient; the ordered box is not a collision proxy.
        vertices = _box_corners((2.0 * radius, 2.0 * radius, length))
    elif sphere is not None:
        radius = float(sphere.get("radius"))
        vertices = _box_corners((2.0 * radius,) * 3)
    elif mesh is not None:
        import trimesh

        filename = mesh.get("filename")
        if filename.startswith("package://"):
            raise ValueError(f"package URI is not resolvable offline: {filename}")
        loaded = trimesh.load(str(base_dir / filename), force="mesh", process=False)
        vertices = np.asarray(loaded.vertices, dtype=np.float64)
        if mesh.get("scale") is not None:
            vertices = vertices * _values(mesh, "scale", (1.0, 1.0, 1.0))
    else:
        return np.empty((0, 3), dtype=np.float64)

    transform = _origin(collision.find("origin"))
    return vertices @ transform[:3, :3].T + transform[:3, 3]


def _nearest_geometry(
    start_link: str,
    links: dict[str, ET.Element],
    outgoing: dict[str, list[ET.Element]],
    base_dir: Path,
) -> np.ndarray:
    """Nearest collision-bearing descendant, expressed in ``start_link``."""
    frontier = [(start_link, np.eye(4, dtype=np.float64))]
    seen: set[str] = set()
    while frontier:
        found: list[np.ndarray] = []
        following: list[tuple[str, np.ndarray]] = []
        for link_name, start_from_link in frontier:
            if link_name in seen:
                continue
            seen.add(link_name)
            link = links[link_name]
            pieces = [_collision_vertices(c, base_dir) for c in link.findall("collision")]
            pieces = [p for p in pieces if p.size]
            if pieces:
                vertices = np.concatenate(pieces, axis=0)
                found.append(
                    vertices @ start_from_link[:3, :3].T + start_from_link[:3, 3]
                )
                continue
            for joint in outgoing.get(link_name, ()):
                child = joint.find("child").get("link")
                following.append(
                    (child, start_from_link @ _origin(joint.find("origin")))
                )
        if found:
            return np.concatenate(found, axis=0)
        frontier = following
    return np.empty((0, 3), dtype=np.float64)


def _ordered_box(vertices: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, bool]:
    if not vertices.size:
        return np.zeros((4, 3), dtype=np.float32), False

    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    # The collision centroid normally points down the controlled segment.  If
    # the mesh is centred on the pivot, use its farthest transverse vertex.
    distal = vertices.mean(axis=0)
    distal = distal - axis * float(distal @ axis)
    if np.linalg.norm(distal) < 1e-8:
        transverse = vertices - np.outer(vertices @ axis, axis)
        distal = transverse[np.argmax(np.linalg.norm(transverse, axis=1))]
    if np.linalg.norm(distal) < 1e-8:
        basis = np.eye(3)[np.argmin(np.abs(axis))]
        distal = np.cross(axis, basis)
    distal /= np.linalg.norm(distal)

    bend = np.cross(axis, distal)  # positive angular motion of the distal ray
    bend /= max(float(np.linalg.norm(bend)), 1e-12)

    projected = np.stack(
        [vertices @ distal, vertices @ bend, vertices @ axis], axis=-1
    )
    spans = projected.max(axis=0) - projected.min(axis=0)
    # Preserve the collision dimensions but anchor the proximal face at the
    # joint.  Link meshes occasionally overhang slightly behind their pivot.
    length = max(float(spans[0]), float(projected[:, 0].max()), 1e-6)
    width = max(float(spans[1]), 1e-6)
    height = max(float(spans[2]), 1e-6)

    p0 = -0.5 * width * bend - 0.5 * height * axis
    return np.asarray([
        p0,
        p0 + length * distal,
        p0 + width * bend,
        p0 + height * axis,
    ], dtype=np.float32), True


def joint_link_boxes(
    urdf: str | Path | ET.Element,
    joint_names,
    *,
    base_dir: str | Path | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    """Return child bodies, ordered boxes, validity, and characteristic scale.

    ``boxes`` is ``(J, 4, 3)`` in each joint's child-link frame.  ``hand_scale``
    is the longest encoded link edge in metres and is used only for deterministic
    geometric normalization; it is also exposed to the policy explicitly.
    """
    if isinstance(urdf, ET.Element):
        root = urdf
        mesh_base = Path(base_dir or ".")
    else:
        path = Path(resolve(urdf))
        root = ET.parse(path).getroot()
        mesh_base = path.parent if base_dir is None else Path(base_dir)

    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    outgoing: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        outgoing.setdefault(joint.find("parent").get("link"), []).append(joint)

    child_links: list[str] = []
    boxes: list[np.ndarray] = []
    valid: list[bool] = []
    for name in joint_names:
        if name not in joints:
            raise KeyError(f"URDF has no joint {name!r}")
        joint = joints[name]
        child = joint.find("child").get("link")
        axis_node = joint.find("axis")
        axis = _values(axis_node, "xyz", (1.0, 0.0, 0.0))
        vertices = _nearest_geometry(child, links, outgoing, mesh_base)
        box, ok = _ordered_box(vertices, axis)
        child_links.append(child)
        boxes.append(box)
        valid.append(ok)

    boxes_np = np.stack(boxes).astype(np.float32)
    valid_np = np.asarray(valid, dtype=np.bool_)
    lengths = np.linalg.norm(boxes_np[:, 1] - boxes_np[:, 0], axis=-1)
    scale = float(lengths[valid_np].max()) if valid_np.any() else 1.0
    return child_links, boxes_np, valid_np, max(scale, 1e-6)


__all__ = ["joint_link_boxes"]
