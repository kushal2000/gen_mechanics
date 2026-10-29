"""A hand's URDF primitives placed onto the shared template tree (29 Oct). Kit-free; the authoring that needs
``pxr`` is inhand_olympics/author_template.py.

The template is one articulation every env of a ``multi:template:`` scene shares, so PhysX batches the whole scene
as one view (the generated-population path of September: 500k sim fps at 12k envs, where an articulation per hand
gives 46k). Its tree is fixed by the hand set's ``SlotGrid`` (finger_layout.py):

    wrist_base -x-> wrist_x_link -y-> wrist_y_link -z-> wrist_z_link -roll-> wrist_roll_link -pitch->
    wrist_pitch_link -yaw-> palm -s{palm slot}-> link_s{palm slot} (the palm section the palm joint moves)
    palm  -s{5f+0}-> link_s{5f+0} -s{5f+1}-> ... -s{5f+4}-> link_s{5f+4}      for a rigid finger slot f
    palm0 -s{5f+0}-> ...                                                        for a moving finger slot f

A hand fills its slots with its own links (the URDF's inertials and colliders, in the URDF's link frames, so joint
limits, masses and geometry are the converted asset's) and the rest with GHOSTS: a 1e-6 kg link with no collider on
a joint locked by limits (0, 1e-8), stacked at its parent's origin -- the population path's padding, which
``joint_enabled`` (read from the limits) masks out. A hand without a cut has a ghost palm0 locked to the palm; a cut
hand's moving fingers hang off its real section link.

Conventions match the URDF converter's assets (read off one, 29 Oct): link poses in the robot root frame at q = 0;
a revolute or prismatic joint as two local frames (body0's: the URDF origin times ``hinge_to_z``; body1's:
``hinge_to_z``) on the Z token axis; limits in radians here, degrees at authoring; a cylinder becomes a capsule with the
cylinder's length as its cylindrical part (hemispheres added beyond it), as the converter makes it.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hand_sampler import resolve
from hand_sampler.build import hinge_to_z
from hand_sampler.design_space import rpy_to_mat
from hand_sampler.robot_param_constants import VIRTUAL_LINK_INERTIA, VIRTUAL_LINK_MASS_KG
from isaacsimenvs.common.robots.finger_layout import FingerLayout
from isaacsimenvs.common.robots.multi_hand import PALM, slot_joint, slot_link

GHOST_LIMITS = (0.0, 1e-8)        # rad: locked, and not exactly coincident (a degenerate constraint)
WRIST_JOINTS = ("wrist_x", "wrist_y", "wrist_z", "wrist_roll", "wrist_pitch", "wrist_yaw")
WRIST_LINKS = ("wrist_base", "wrist_x_link", "wrist_y_link", "wrist_z_link", "wrist_roll_link", "wrist_pitch_link")
_WRIST_AXES = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1))


@dataclass
class Collider:
    kind: str                     # "box" (dims: lx, ly, lz), "capsule" (dims: cylinder height, radius), "mesh"
    dims: tuple[float, ...]
    pose: np.ndarray              # 4x4 in the link frame
    points: np.ndarray | None = None   # mesh: (V, 3) convex hull vertices in the link frame (pose applied)
    faces: list[tuple[int, int, int]] | None = None


@dataclass
class Link:
    name: str
    mass: float
    com: np.ndarray               # (3,) in the link frame
    inertia: np.ndarray           # (3, 3) about the centre of mass, link frame
    colliders: list[Collider] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    type: str                     # revolute, continuous, prismatic, fixed
    parent: str
    child: str
    origin: np.ndarray            # 4x4: the child's frame in the parent's, at q = 0
    axis: np.ndarray              # (3,) in the child frame
    lower: float
    upper: float
    effort: float
    velocity: float


@dataclass
class UrdfModel:
    links: dict[str, Link]
    joints: list[Joint]
    root: str

    def joint_to(self, link: str) -> Joint | None:
        return next((j for j in self.joints if j.child == link), None)

    def children(self, link: str) -> list[Joint]:
        return [j for j in self.joints if j.parent == link]

    def fk(self) -> dict[str, np.ndarray]:
        """Every link's frame in the root link's, at q = 0."""
        out = {self.root: np.eye(4)}
        pending = [self.root]
        while pending:
            p = pending.pop()
            for j in self.children(p):
                out[j.child] = out[p] @ j.origin
                pending.append(j.child)
        return out


def _xyz_rpy(el) -> np.ndarray:
    T = np.eye(4)
    if el is not None:
        T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
        T[:3, :3] = rpy_to_mat([float(v) for v in el.get("rpy", "0 0 0").split()])
    return T


def _hull(points: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    verts = points[hull.vertices]
    index = {v: i for i, v in enumerate(hull.vertices)}
    faces = []
    centre = verts.mean(axis=0)
    for tri, eq in zip(hull.simplices, hull.equations):
        a, b, c = (index[i] for i in tri)
        # outward winding: the face normal must point away from the hull's centre
        n = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        if np.dot(n, eq[:3]) < 0:
            b, c = c, b
        faces.append((a, b, c))
    return verts, faces


def _read_obj(path: Path) -> np.ndarray:
    pts = [tuple(float(v) for v in line.split()[1:4]) for line in path.read_text().splitlines()
           if line.startswith("v ")]
    if not pts:
        raise ValueError(f"{path}: no vertices")
    return np.asarray(pts, float)


def _collider(geom, origin: np.ndarray, urdf_dir: Path, capsules: bool) -> Collider:
    shape = next(iter(geom))
    if shape.tag == "box":
        return Collider("box", tuple(float(v) for v in shape.get("size").split()), origin)
    if shape.tag == "cylinder":
        r, L = float(shape.get("radius")), float(shape.get("length"))
        if not capsules:
            raise ValueError("a cylinder collider needs replace_cylinders_with_capsules (the converted assets' way)")
        # The URDF converter's replace_cylinders_with_capsules keeps the cylinder's LENGTH as the capsule's
        # cylindrical part and adds the hemispheres beyond it (read off a converted asset, 29 Oct: a 0.069 m, r 0.0165
        # cylinder became height 0.069, 0.102 m tip to tip). Match it, or every capsule fingertip is 2r short.
        return Collider("capsule", (L, r), origin)
    if shape.tag == "sphere":
        r = float(shape.get("radius"))
        return Collider("capsule", (0.0, r), origin)
    if shape.tag == "mesh":
        pts = _read_obj(urdf_dir / shape.get("filename"))
        scale = shape.get("scale")
        if scale:
            pts = pts * np.asarray([float(v) for v in scale.split()], float)
        pts = pts @ origin[:3, :3].T + origin[:3, 3]
        verts, faces = _hull(pts)
        return Collider("mesh", (), np.eye(4), points=verts, faces=faces)
    raise ValueError(f"unsupported collider <{shape.tag}>")


def load_urdf_model(urdf_path, capsules: bool) -> UrdfModel:
    """The URDF's links (mass, centre, inertia about it, colliders) and joints, fixed joints merged into their
    parents (as the converter's merge_fixed_joints does)."""
    path = resolve(urdf_path)
    root_el = ET.parse(path).getroot()
    links: dict[str, Link] = {}
    for el in root_el.findall("link"):
        inertial = el.find("inertial")
        if inertial is None:
            mass, com, I = 0.0, np.zeros(3), np.zeros((3, 3))
        else:
            T = _xyz_rpy(inertial.find("origin"))
            mass = float(inertial.find("mass").get("value"))
            i = inertial.find("inertia")
            m = np.array([[float(i.get("ixx")), float(i.get("ixy")), float(i.get("ixz"))],
                          [float(i.get("ixy")), float(i.get("iyy")), float(i.get("iyz"))],
                          [float(i.get("ixz")), float(i.get("iyz")), float(i.get("izz"))]])
            com, I = T[:3, 3].copy(), T[:3, :3] @ m @ T[:3, :3].T
        cols = [_collider(c.find("geometry"), _xyz_rpy(c.find("origin")), path.parent, capsules)
                for c in el.findall("collision")]
        links[el.get("name")] = Link(el.get("name"), mass, com, I, cols)
    joints = []
    for el in root_el.findall("joint"):
        lim = el.find("limit")
        ax = el.find("axis")
        joints.append(Joint(
            el.get("name"), el.get("type"), el.find("parent").get("link"), el.find("child").get("link"),
            _xyz_rpy(el.find("origin")),
            np.asarray([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()], float),
            float(lim.get("lower", 0.0)) if lim is not None else 0.0,
            float(lim.get("upper", 0.0)) if lim is not None else 0.0,
            float(lim.get("effort", 0.0)) if lim is not None else 0.0,
            float(lim.get("velocity", 0.0)) if lim is not None else 0.0))
    children = {j.child for j in joints}
    roots = [n for n in links if n not in children]
    if len(roots) != 1:
        raise ValueError(f"{path}: {len(roots)} root links {roots}")
    model = UrdfModel(links, joints, roots[0])
    _merge_fixed(model)
    return model


def _merge_fixed(model: UrdfModel) -> None:
    """Fold every fixed-joint child into its parent: colliders and inertia into the parent's frame."""
    while True:
        j = next((j for j in model.joints if j.type == "fixed"), None)
        if j is None:
            return
        parent, child = model.links[j.parent], model.links[j.child]
        T = j.origin
        for c in child.colliders:
            if c.kind == "mesh":
                c.points = c.points @ T[:3, :3].T + T[:3, 3]
            else:
                c.pose = T @ c.pose
            parent.colliders.append(c)
        parent.mass, parent.com, parent.inertia = merge_inertia(
            [(parent.mass, parent.com, parent.inertia), (child.mass, T[:3, 3] + T[:3, :3] @ child.com,
                                                          T[:3, :3] @ child.inertia @ T[:3, :3].T)])
        for jj in model.joints:
            if jj.parent == j.child:
                jj.parent, jj.origin = j.parent, T @ jj.origin
        model.joints.remove(j)
        del model.links[j.child]


def merge_inertia(parts) -> tuple[float, np.ndarray, np.ndarray]:
    """``(mass, centre, inertia about the centre)`` of parts given the same way, one frame (parallel axes)."""
    total = sum(m for m, _, _ in parts)
    if total <= 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    com = sum(m * np.asarray(c, float) for m, c, _ in parts) / total
    I = np.zeros((3, 3))
    for m, c, Ic in parts:
        d = np.asarray(c, float) - com
        I += np.asarray(Ic, float) + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return float(total), com, I


# --- the template ----------------------------------------------------------------------------------------------

@dataclass
class TBody:
    name: str
    parent: str | None            # body name; None for the root
    joint: str | None             # the joint from the parent
    pose: np.ndarray              # 4x4 in the robot root frame at q = 0
    mass: float
    com: np.ndarray
    inertia: np.ndarray           # (3, 3) about com, body frame
    colliders: list[Collider]
    real: bool                    # a hand's own link (ghost links carry no collider)
    urdf_link: str | None = None


@dataclass
class TJoint:
    name: str
    type: str                     # revolute | prismatic
    body0: str
    body1: str
    frame0: np.ndarray            # 4x4 in body0: the joint's frame, Z the hinge / slide axis
    rot1: np.ndarray              # 3x3 in body1
    lower: float                  # rad or m
    upper: float
    effort: float
    velocity: float               # rad/s or m/s
    real: bool


@dataclass
class TemplateRobot:
    bodies: list[TBody]
    joints: list[TJoint]
    body_of_link: dict[str, str]  # URDF link -> template body
    filtered_pairs: dict[str, list[str]]   # body -> bodies it must not collide with (the spec's adjacent links)

    def body(self, name: str) -> TBody:
        return next(b for b in self.bodies if b.name == name)


def _joint_frames(origin: np.ndarray, axis) -> tuple[np.ndarray, np.ndarray]:
    r = hinge_to_z(axis)
    f0 = origin.copy()
    f0[:3, :3] = origin[:3, :3] @ r
    return f0, r


def template_robot(spec, layout: FingerLayout, wrist_cfg, mount_rotation) -> TemplateRobot:
    """``spec``'s hand on the dummy wrist, placed on ``layout.grid`` (a FingerLayout on the hand set's grid).
    ``mount_rotation``: the hand root's 3x3 rotation in the wrist frame (inhand_olympics/wrist.py's
    ``palm_down_rotation``: palm down, fingers along +x); ``wrist_cfg`` the olympics WristCfg (ranges, limits,
    link mass)."""
    model = load_urdf_model(spec.urdf_path, spec.replace_cylinders_with_capsules)
    if model.root != spec.palm_body_name:
        raise ValueError(f"{spec.name}: URDF root {model.root} is not the palm body {spec.palm_body_name}")
    fk = model.fk()
    grid = layout.grid
    R = np.asarray(mount_rotation, float)
    mount = np.eye(4)
    mount[:3, :3] = R
    hand_joints = list(spec.hand_joint_names)
    own = [j for j in model.joints if j.name in hand_joints]
    ghost_effort = max(j.effort for j in own)            # a ghost's drive: the hand's own, holding it shut
    ghost_velocity = max(j.velocity for j in own)
    bodies: list[TBody] = []
    joints: list[TJoint] = []
    body_of_link: dict[str, str] = {}

    def ghost(name, parent, joint, pose):
        bodies.append(TBody(name, parent, joint, pose, VIRTUAL_LINK_MASS_KG, np.zeros(3),
                            np.eye(3) * VIRTUAL_LINK_INERTIA, [], real=False))

    def real(name, parent, joint, link: Link, pose):
        bodies.append(TBody(name, parent, joint, pose, link.mass, link.com, link.inertia, link.colliders, real=True,
                            urdf_link=link.name))
        body_of_link[link.name] = name

    def add_joint(name, jtype, body0, body1, origin, axis, lower, upper, effort, velocity, is_real):
        f0, r1 = _joint_frames(origin, axis)
        joints.append(TJoint(name, jtype, body0, body1, f0, r1, lower, upper, effort, velocity, is_real))

    # the wrist: six massive-enough dummy links at the origin (WristCfg.link_mass), world axes
    lim = [wrist_cfg.x_range, wrist_cfg.y_range, wrist_cfg.z_range, wrist_cfg.roll_range, wrist_cfg.pitch_range,
           wrist_cfg.yaw_range]
    m = float(wrist_cfg.link_mass)
    for k, name in enumerate(WRIST_LINKS):
        bodies.append(TBody(name, WRIST_LINKS[k - 1] if k else None, WRIST_JOINTS[k - 1] if k else None, np.eye(4),
                            m, np.zeros(3), np.eye(3) * m * 1e-4, [], real=True))
    for k, name in enumerate(WRIST_JOINTS):
        prismatic = k < 3
        child = WRIST_LINKS[k + 1] if k < 5 else PALM
        origin, axis = np.eye(4), np.asarray(_WRIST_AXES[k], float)
        if k == 5:                              # the mount rotation sits in the yaw joint; world z in the hand frame
            origin = mount.copy()
            axis = R.T @ axis
        add_joint(name, "prismatic" if prismatic else "revolute", WRIST_LINKS[k], child, origin, axis,
                  float(lim[k][0]), float(lim[k][1]),
                  float(wrist_cfg.prismatic_effort_limit if prismatic else wrist_cfg.revolute_effort_limit),
                  float(wrist_cfg.prismatic_velocity_limit if prismatic else wrist_cfg.revolute_velocity_limit), True)

    # the palm, and the section behind the palm joint (real for a cut hand, a locked ghost otherwise)
    palm = model.links[spec.palm_body_name]
    real(PALM, WRIST_LINKS[-1], WRIST_JOINTS[-1], palm, mount)
    palm_slot_name = slot_joint(grid.palm_slot)
    SECTION = slot_link(grid.palm_slot)           # the section body is the palm slot's link
    if layout.palm:
        pj = next(j for j in model.joints if j.name == hand_joints[layout.palm[0]])
        if pj.parent != spec.palm_body_name:
            raise ValueError(f"{spec.name}: palm joint {pj.name} does not hang off the palm body")
        real(SECTION, PALM, palm_slot_name, model.links[pj.child], mount @ fk[pj.child])
        add_joint(palm_slot_name, "revolute", PALM, SECTION, pj.origin, pj.axis, pj.lower, pj.upper, pj.effort,
                  pj.velocity, True)
        section_link = pj.child
    else:
        ghost(SECTION, PALM, palm_slot_name, mount)
        add_joint(palm_slot_name, "revolute", PALM, SECTION, np.eye(4), (0.0, 0.0, 1.0), *GHOST_LIMITS,
                  ghost_effort, ghost_velocity, False)
        section_link = None

    # the finger slots
    finger_of_slot = {s: f for f, s in enumerate(layout.finger_slot_of_finger)}
    for s in range(grid.n_fingers):
        section = grid.is_moving_slot(s)
        parent = SECTION if section else PALM
        finger = finger_of_slot.get(s)
        chain = [hand_joints[j] for j in layout.fingers[finger]] if finger is not None else []
        for k in range(grid.n_slots):
            slot = s * grid.n_slots + k
            name, jname = slot_link(slot), slot_joint(slot)
            if k < len(chain):
                j = next(jj for jj in model.joints if jj.name == chain[k])
                want_parent = section_link if (section and layout.palm and layout.moving[finger]) else spec.palm_body_name
                if k == 0 and j.parent != want_parent:
                    raise ValueError(f"{spec.name}: {j.name} hangs off {j.parent}, expected {want_parent}")
                if k and j.parent != model.joints[[jj.name for jj in model.joints].index(chain[k - 1])].child:
                    raise ValueError(f"{spec.name}: {j.name} does not follow {chain[k - 1]}")
                real(name, parent, jname, model.links[j.child], mount @ fk[j.child])
                add_joint(jname, "revolute", parent, name, j.origin, j.axis, j.lower, j.upper, j.effort, j.velocity,
                          True)
            else:
                ghost(name, parent, jname, bodies[-1].pose if k else next(b.pose for b in bodies if b.name == parent))
                add_joint(jname, "revolute", parent, name, np.eye(4), (0.0, 0.0, 1.0), *GHOST_LIMITS,
                          ghost_effort, ghost_velocity, False)
            parent = name

    missing = [n for n in model.links if n not in body_of_link]
    if missing:
        raise ValueError(f"{spec.name}: URDF links {missing} landed in no slot")
    pairs = {}
    for a, nbrs in spec.adjacent_links.items():
        if a in body_of_link:
            pairs[body_of_link[a]] = [body_of_link[b] for b in nbrs if b in body_of_link]
    return TemplateRobot(bodies, joints, body_of_link, pairs)


__all__ = ["Collider", "GHOST_LIMITS", "Joint", "Link", "TBody", "TJoint", "TemplateRobot", "UrdfModel",
           "WRIST_JOINTS", "WRIST_LINKS", "load_urdf_model", "merge_inertia", "template_robot"]
