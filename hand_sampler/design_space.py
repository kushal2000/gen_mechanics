"""The hand design space: what a hand is, and where its parts are."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, replace

import numpy as np

from hand_sampler import resolve

# --- palm ------------------------------------------------------------------- Mutable,...

PALM_QUANTUM = 0.005
"""Grid the palm dimensions must lie on."""

PALM_STEP = 0.010
"""How far one ``perturb_palm`` moves a dimension -- twice the grid."""
PALM_THICKNESS_RANGE = (0.015, 0.040)   # x -- NOT MUTATED, see below
PALM_WIDTH_RANGE = (0.040, 0.100)       # y
PALM_LENGTH_RANGE = (0.040, 0.100)      # z, wrist face at z = 0

# Thickness is seeded and never mutated: it is the dimension geometry cares least about, while...
MUTABLE_PALM_DIMS: tuple[str, ...] = ("width", "length")

# --- links ------------------------------------------------------------------

LINK_QUANTUM = 0.005
CAPSULE_RADIUS = 0.010
"""Fixed, on evidence: `radius_scale` scored Spearman -0.005 across a 2x range in the..."""

MIN_LINK_LENGTH = 0.015
"""The closest two joint axes can sit -- there is exactly one joint per link."""

MAX_LINK_LENGTH = 0.080
"""Deliberately loose."""

# --- joints -----------------------------------------------------------------

ANGLE_QUANTUM = math.radians(15.0)
"""Grid for every angle in the genotype: joint theta and offset."""

JOINT_LIMIT = (math.radians(-90.0), math.radians(90.0))
"""Symmetric, for every joint regardless of axis."""

# --- the envelope -----------------------------------------------------------

MIN_FINGERS = 2
"""A one-finger hand cannot oppose anything, so it is excluded rather than left for..."""

MAX_FINGERS = 5
MAX_JOINTS_PER_FINGER = 6
"""The articulation envelope: a HARD cap, not a rail."""

MIN_MOUNT_SEPARATION = 0.015
"""Centre-to-centre floor between mounts on DIFFERENT faces."""

MOUNT_EDGE_MARGIN = CAPSULE_RADIUS
"""How far a mount stays from its face boundary, or half the base capsule hangs off the..."""

MIN_SAME_FACE_SEPARATION = 2.5 * CAPSULE_RADIUS
"""Floor between mounts on the SAME face, where fingers run parallel and their base..."""

FINGER_FACES: tuple[str, ...] = ("+z", "+y", "-y")
"""The three THIN faces."""

GRASP_DIR = np.array([1.0, 0.0, 0.0])
"""Fingers curl toward the palm surface (+x)."""


# --- the tree ---------------------------------------------------------------

@dataclass(frozen=True)
class Joint:
    """One revolute DOF; ``theta`` and ``phi`` in radians, see kinematics.axis_of."""

    theta: float
    phi: float = math.pi / 2
    offset: float = 0.0
    # Set only by an imported hand, whose hinge is measured rather than drawn.
    axis_override: tuple[float, float, float] | None = None
    # Travel, in radians. None means the design space's JOINT_LIMIT.
    limits: tuple[float, float] | None = None
    # (effort N.m, velocity rad/s, stiffness, damping, armature). None means the
    # depth-indexed default, which is what a generated joint gets.
    drive: tuple[float, float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.limits is not None and self.limits[0] > self.limits[1]:
            raise ValueError(f"joint limits out of order: {self.limits}")
        if self.drive is not None and len(self.drive) != 5:
            raise ValueError(f"drive must be 5 numbers, got {self.drive}")
        if self.axis_override is not None and len(self.axis_override) != 3:
            raise ValueError(f"axis_override must be 3 numbers, got {self.axis_override}")
        if not math.isfinite(self.offset):
            raise ValueError(f"non-finite joint offset {self.offset}")
        if not math.isfinite(self.theta) or not math.isfinite(self.phi):
            raise ValueError(f"non-finite joint angles ({self.theta}, {self.phi})")


@dataclass(frozen=True)
class Segment:
    """A joint and the link distal to it."""

    joint: Joint
    length: float
    # (bend, axis) extents of the link box. None means the capsule a generated
    # design gets; an imported hand carries its measured cross-section instead.
    cross_section: tuple[float, float] | None = None
    # Collision/visual meshes, for a measured link. None authors a capsule.
    meshes: tuple[str, ...] | None = None
    # An imported hand's box, verbatim, in ITS child-link frame. A generated
    # hand has none: we author its USD, so its links lie along +x by
    # construction, and the box is derived. A measured hand's frames come from
    # its own asset and cannot be re-derived from length and axis alone.
    token_box: tuple | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.length) or self.length < 0.0:
            raise ValueError(f"bad link length {self.length}")
        if self.cross_section is not None and len(self.cross_section) != 2:
            raise ValueError(f"cross_section must be (bend, axis), got {self.cross_section}")

    @property
    def box(self) -> tuple[float, float, float]:
        """``(length, bend, axis)`` extents of this link, in metres."""
        if self.cross_section is None:
            return (self.length, 2.0 * CAPSULE_RADIUS, 2.0 * CAPSULE_RADIUS)
        return (self.length, self.cross_section[0], self.cross_section[1])


@dataclass(frozen=True)
class Mount:
    """Where a finger attaches to the palm."""

    face: str
    u: float
    v: float

    def __post_init__(self) -> None:
        if self.face not in FINGER_FACES:
            raise ValueError(f"{self.face!r} is not a finger face; use {FINGER_FACES}")
        for name in ("u", "v"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"non-finite mount {name}")


@dataclass(frozen=True)
class Finger:
    mount: Mount
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a finger needs at least one segment")

    @property
    def n_joints(self) -> int:
        return len(self.segments)

    @property
    def reach(self) -> float:
        """Fully-extended length from mount to tip."""
        return sum(s.length for s in self.segments)


@dataclass(frozen=True)
class Palm:
    thickness: float   # x
    width: float       # y
    length: float      # z

    @property
    def extents(self) -> tuple[float, float, float]:
        return (self.thickness, self.width, self.length)


@dataclass(frozen=True)
class Hand:
    palm: Palm
    fingers: tuple[Finger, ...]

    def __post_init__(self) -> None:
        if not self.fingers:
            raise ValueError("a hand needs at least one finger")

    @property
    def n_fingers(self) -> int:
        return len(self.fingers)

    @property
    def n_joints(self) -> int:
        return sum(f.n_joints for f in self.fingers)

    @property
    def n_motors(self) -> int:
        """One motor per joint -- couplings are deferred."""
        return self.n_joints


# --- complexity -------------------------------------------------------------

def complexity(hand: Hand) -> tuple[int, int]:
    """``(n_motors, n_joints)`` -- readable without touching a simulator."""
    return (hand.n_motors, hand.n_joints)


# --- small helpers ----------------------------------------------------------

def with_finger(hand: Hand, i: int, finger: Finger) -> Hand:
    """Replace finger ``i``. Frozen dataclasses, so every edit rebuilds."""
    fingers = list(hand.fingers)
    fingers[i] = finger
    return replace(hand, fingers=tuple(fingers))


# --- geometry --------------------------------------------------------------- Joint axes,...

_EPS = 1e-9


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation about ``axis`` by ``angle``. Axis need not be normalised."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


# --- joint axes -------------------------------------------------------------

def axis_of(joint: Joint) -> np.ndarray:
    """The hinge axis in the joint's own frame, where the link runs along +x."""
    if joint.axis_override is not None:
        a = np.asarray(joint.axis_override, dtype=float)
        return a / max(float(np.linalg.norm(a)), 1e-12)
    ct, st = math.cos(joint.theta), math.sin(joint.theta)
    cp, sp = math.cos(joint.phi), math.sin(joint.phi)
    return np.array([cp, sp * st, sp * ct])


# --- palm faces -------------------------------------------------------------

def face_frame(face: str, palm: Palm) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, float, float]:
    """``(centre, normal, t_u, t_v, span_u, span_v)`` for a palm face."""
    t, w, l = palm.thickness, palm.width, palm.length
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    table = {
        "+y": (np.array([0.0, w / 2, l / 2]), y, x, z, t, l),
        "-y": (np.array([0.0, -w / 2, l / 2]), -y, x, z, t, l),
        "+z": (np.array([0.0, 0.0, l]), z, x, y, t, w),
    }
    if face not in table:
        raise ValueError(f"{face!r} is not a finger face")
    return table[face]


def face_from_normal(n: np.ndarray) -> str | None:
    """Which finger face has this outward normal, if any."""
    axis = int(np.argmax(np.abs(n)))
    sign = "+" if n[axis] > 0 else "-"
    face = f"{sign}{'xyz'[axis]}"
    return face if face in FINGER_FACES else None


def mount_uv_bounds(face: str, palm: Palm) -> tuple[float, float, float, float]:
    """``(u_lo, u_hi, v_lo, v_hi)`` -- the normalised box a mount may occupy."""
    _, _, _, _, span_u, span_v = face_frame(face, palm)
    m = MOUNT_EDGE_MARGIN
    lo_u, hi_u = ((m / span_u, 1.0 - m / span_u) if span_u > 2 * m else (0.5, 0.5))
    lo_v, hi_v = ((m / span_v, 1.0 - m / span_v) if span_v > 2 * m else (0.5, 0.5))
    return lo_u, hi_u, lo_v, hi_v


def mount_position(mount: Mount, palm: Palm) -> np.ndarray:
    centre, _, t_u, t_v, span_u, span_v = face_frame(mount.face, palm)
    return (centre
            + (mount.u - 0.5) * span_u * t_u
            + (mount.v - 0.5) * span_v * t_v)


def mount_direction(mount: Mount, palm: Palm) -> np.ndarray:
    """The face normal: a finger leaves perpendicular to the face it sits on."""
    return face_frame(mount.face, palm)[1]


def _frame_from_axis(axis: np.ndarray) -> np.ndarray:
    """Orthonormal frame with ``axis`` as its first column."""
    a = axis / np.linalg.norm(axis)
    ref = GRASP_DIR
    if np.linalg.norm(np.cross(a, ref)) < 1e-6:
        ref = np.eye(3)[int(np.argmin(np.abs(a)))]
    fe = np.cross(a, ref)
    fe /= np.linalg.norm(fe)
    aa = np.cross(fe, a)
    return np.column_stack([a, aa, fe])


def mount_frame(mount: Mount, palm: Palm) -> tuple[np.ndarray, np.ndarray]:
    """``(position, R)`` for a finger's base, in the palm frame."""
    return mount_position(mount, palm), _frame_from_axis(mount_direction(mount, palm))


# --- forward kinematics -----------------------------------------------------

def forward_kinematics(finger: Finger, palm: Palm,
                       angles: dict[int, float] | None = None,
                       ) -> tuple[list[np.ndarray], list[tuple]]:
    """``(joint_positions, capsules)`` for one finger, in the palm frame."""
    angles = angles or {}
    p, R = mount_frame(finger.mount, palm)
    joints: list[np.ndarray] = []
    capsules: list[tuple] = []

    for i, seg in enumerate(finger.segments):
        joints.append(p.copy())
        R = R @ rodrigues(axis_of(seg.joint),
                          seg.joint.offset + angles.get(i, 0.0))
        nxt = p + R[:, 0] * seg.length
        if seg.length > _EPS:
            capsules.append((p.copy(), nxt.copy(), CAPSULE_RADIUS, i))
        p = nxt

    joints.append(p.copy())
    return joints, capsules


def joint_axes(finger: Finger, palm: Palm,
               angles: dict[int, float] | None = None) -> list[np.ndarray]:
    """Each joint's hinge axis as a unit vector in the palm frame."""
    angles = angles or {}
    _, R = mount_frame(finger.mount, palm)
    out: list[np.ndarray] = []
    for i, seg in enumerate(finger.segments):
        a = axis_of(seg.joint)
        out.append(R @ a)
        R = R @ rodrigues(a, seg.joint.offset + angles.get(i, 0.0))
    return out


def fingertip(finger: Finger, palm: Palm,
              angles: dict[int, float] | None = None) -> np.ndarray:
    return forward_kinematics(finger, palm, angles)[0][-1]


# --- the two measures that carry signal -------------------------------------

def segment_distance(p0: np.ndarray, p1: np.ndarray,
                     q0: np.ndarray, q1: np.ndarray) -> float:
    """Closest distance between two 3-D line segments, closed form."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = float(u @ u), float(u @ v), float(v @ v)
    d, e = float(u @ w), float(v @ w)
    det = a * c - b * b
    eps = 1e-12

    if det < eps:                                  # parallel or degenerate
        s_num, s_den, t_num, t_den = 0.0, 1.0, e, (c if c > eps else 1.0)
    else:
        s_den = t_den = det
        s_num, t_num = b * e - c * d, a * e - b * d
        if s_num < 0.0:
            s_num, t_num, t_den = 0.0, e, (c if c > eps else 1.0)
        elif s_num > s_den:
            s_num, t_num, t_den = s_den, e + b, (c if c > eps else 1.0)

    if t_num < 0.0:                                # re-solve s against t = 0
        t_num = 0.0
        if -d < 0.0:
            s_num, s_den = 0.0, 1.0
        elif -d > a:
            s_num, s_den = 1.0, 1.0
        else:
            s_num, s_den = -d, (a if a > eps else 1.0)
    elif t_num > t_den:                            # re-solve s against t = 1
        t_num = t_den
        if (-d + b) < 0.0:
            s_num, s_den = 0.0, 1.0
        elif (-d + b) > a:
            s_num, s_den = 1.0, 1.0
        else:
            s_num, s_den = -d + b, (a if a > eps else 1.0)

    sc = 0.0 if abs(s_num) < eps else s_num / s_den
    tc = 0.0 if abs(t_num) < eps else t_num / t_den
    return float(np.linalg.norm(w + sc * u - tc * v))


def base_capsules(hand: Hand) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each finger's proximal link at the rest pose, as a core segment."""
    out = []
    for f in hand.fingers:
        p0, R = mount_frame(f.mount, hand.palm)
        seg = f.segments[0]
        R = R @ rodrigues(axis_of(seg.joint), seg.joint.offset)
        out.append((p0, p0 + R[:, 0] * seg.length))
    return out


def mount_separations(hand: Hand) -> list[float]:
    """Pairwise distances between finger mounts, in metres."""
    pos = [mount_position(f.mount, hand.palm) for f in hand.fingers]
    return [float(np.linalg.norm(pos[i] - pos[j]))
            for i in range(len(pos)) for j in range(i + 1, len(pos))]


# --- rotation conventions, mass properties, and the scene a hand is judged in

from scipy.spatial.transform import Rotation


def rpy_to_mat(rpy) -> np.ndarray:
    """URDF RPY -> 3x3 rotation matrix."""
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=float)).as_matrix()


def mat_to_rpy(mat) -> tuple[float, float, float]:
    """3x3 rotation matrix -> URDF RPY."""
    r, p, y = Rotation.from_matrix(np.asarray(mat, dtype=float)[:3, :3]).as_euler("xyz")
    return float(r), float(p), float(y)


def rpy_to_quat_wxyz(rpy) -> tuple[float, float, float, float]:
    """URDF RPY -> (w, x, y, z)."""
    x, y, z, w = Rotation.from_euler("xyz", np.asarray(rpy, dtype=float)).as_quat()
    return float(w), float(x), float(y), float(z)


def rpy_to_rot6d(rpy) -> list[float]:
    """First two columns of the rotation matrix for an RPY triple."""
    m = rpy_to_mat(rpy)
    return [float(v) for v in m[:, 0]] + [float(v) for v in m[:, 1]]


def mat_to_pos_quat(m):
    """4x4 transform -> (translation, (w, x, y, z))."""
    m = np.asarray(m, dtype=float)
    x, y, z, w = Rotation.from_matrix(m[:3, :3]).as_quat()
    return tuple(float(v) for v in m[:3, 3]), (float(w), float(x), float(y), float(z))


__all__ = ["rpy_to_mat", "mat_to_rpy", "rpy_to_quat_wxyz", "rpy_to_rot6d",
           "mat_to_pos_quat"]


# --- mass properties ---



def compute_mass_and_inertia(scale, density: float):
    """Capsule-approximation for cylinders; exact for cuboids."""
    if len(scale) == 3:
        lx, ly, lz = scale
        v = lx * ly * lz
        m = v * density
        ixx = (1 / 12) * m * (ly * ly + lz * lz)
        iyy = (1 / 12) * m * (lx * lx + lz * lz)
        izz = (1 / 12) * m * (lx * lx + ly * ly)
        return m, ixx, iyy, izz
    if len(scale) == 2:
        h, d = scale[0], scale[1]
        r = d / 2
        # Capsule mass = cylinder + two hemispheres.
        m_c = density * math.pi * r * r * h
        m_h = density * (2 / 3) * math.pi * r ** 3
        m = m_c + 2 * m_h
        # Cylinder inertia about centroid (axis = z).
        i_c_axis = 0.5 * m_c * r * r
        i_c_perp = (1 / 12) * m_c * (3 * r * r + h * h)
        # Hemisphere inertia about its own centroid.
        i_h_axis = (2 / 5) * m_h * r * r
        i_h_perp = (83 / 320) * m_h * r * r
        d_com = (h / 2) + (3 * r / 8)
        izz = i_c_axis + 2 * i_h_axis
        ixx = iyy = i_c_perp + 2 * (i_h_perp + m_h * d_com * d_com)
        return m, ixx, iyy, izz
    raise ValueError(f"Invalid scale: {scale}")


__all__ = ["compute_mass_and_inertia"]


# --- task geometry: the scene a hand is judged in ---


from hand_sampler import resolve as resolve_repo_path


# Task geometry, from ResetCfg.
GOAL_VOLUME_MINS = (-0.35, -0.2, 0.6)
GOAL_VOLUME_MAXS = (0.35, 0.2, 0.95)
TABLE_Z = 0.38
TABLE_URDF = "assets/urdf/table_narrow.urdf"


def table_extents() -> tuple[float, float, float]:
    """Box dimensions read from the actual table asset."""
    import xml.etree.ElementTree as ET

    root = ET.parse(resolve_repo_path(TABLE_URDF)).getroot()
    for link in root.findall("link"):
        for coll in link.findall("collision"):
            box = coll.find("geometry/box")
            if box is not None:
                return tuple(float(v) for v in box.get("size").split())
    raise RuntimeError(f"no box collision geometry in {TABLE_URDF}")


def _hull_collision_scene(urdf) -> int:
    """Replace each collision mesh with its convex hull, in place."""
    import trimesh

    scene = getattr(urdf, "collision_scene", None)
    if scene is None:
        return 0
    n = 0
    for key, geom in list(scene.geometry.items()):
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 3:
            try:
                scene.geometry[key] = geom.convex_hull
                n += 1
            except Exception:
                pass
    return n


__all__ = ["GOAL_VOLUME_MINS", "GOAL_VOLUME_MAXS", "TABLE_Z", "TABLE_URDF",
           "table_extents", "_hull_collision_scene"]


# --- joint tokens: the one description a policy reads, for any hand ---------

def joint_boxes(hand: "Hand") -> tuple[np.ndarray, np.ndarray, float]:
    """``(boxes, valid, hand_scale)`` -- the per-joint link box of every joint."""
    boxes, valid = [], []
    for finger in hand.fingers:
        for seg in finger.segments:
            if seg.token_box is not None:
                boxes.append(np.asarray(seg.token_box, dtype=np.float32).reshape(4, 3))
                valid.append(True)
                continue
            length, width, height = seg.box
            axis = axis_of(seg.joint)
            distal = np.array([1.0, 0.0, 0.0])          # the link runs along +x
            distal = distal - axis * float(distal @ axis)
            n = float(np.linalg.norm(distal))
            if n < 1e-8:                                 # a roll joint: pick any transverse ray
                distal = np.cross(axis, np.eye(3)[int(np.argmin(np.abs(axis)))])
                n = float(np.linalg.norm(distal))
            distal = distal / max(n, 1e-12)
            bend = np.cross(axis, distal)
            bend = bend / max(float(np.linalg.norm(bend)), 1e-12)
            p0 = -0.5 * width * bend - 0.5 * height * axis
            boxes.append(np.asarray([p0, p0 + length * distal,
                                     p0 + width * bend, p0 + height * axis], dtype=np.float32))
            valid.append(length > 0.0)
    if not boxes:
        return np.zeros((0, 4, 3), np.float32), np.zeros((0,), bool), 0.0
    arr = np.stack(boxes)
    scale = float(max(np.linalg.norm(arr[:, i] - arr[:, 0], axis=-1).max() for i in (1, 2, 3)))
    return arr, np.asarray(valid, dtype=bool), scale


# --- importing a measured hand: its URDF read once, never at runtime ---





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
    """Return child bodies, ordered boxes, validity, and characteristic scale."""
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


def hand_from_urdf(urdf, joint_names, *, base_dir=None, palm=None) -> "Hand":
    """Import a measured hand -- SHARPA -- as a ``Hand``."""
    if isinstance(urdf, ET.Element):
        root, meshes = urdf, base_dir or "."
    else:
        path = resolve(urdf)
        root, meshes = ET.parse(path).getroot(), base_dir or path.parent
    _bodies, boxes, valid, _scale = joint_link_boxes(root, joint_names, base_dir=meshes)
    joints = {j.get("name"): j for j in root.findall("joint")}
    parent_of = {n: j.find("parent").get("link") for n, j in joints.items()}
    child_of = {n: j.find("child").get("link") for n, j in joints.items()}
    wanted = list(joint_names)
    in_hand = set(wanted)
    joint_making = {child_of[n]: n for n in joints}          # child link -> the joint above it

    # A finger is a chain of controlled joints. Fixed joints sit between them in
    # the URDF, so the predecessor is the nearest CONTROLLED ancestor, not the
    # joint immediately above.
    def predecessor(name):
        link = parent_of[name]
        while link in joint_making:
            up = joint_making[link]
            if up in in_hand:
                return up
            link = parent_of[up]
        return None

    prev = {n: predecessor(n) for n in wanted}
    nxt = {}
    for n, pr in prev.items():
        if pr is not None:
            nxt.setdefault(pr, []).append(n)
    chains = []
    for head in [n for n in wanted if prev[n] is None]:
        chain, cur = [head], head
        while nxt.get(cur):
            cur = nxt[cur][0]        # one controlled child per link in a finger
            chain.append(cur)
        chains.append(chain)

    index = {n: i for i, n in enumerate(wanted)}
    fingers, faces = [], list(FINGER_FACES)
    the_palm = palm or Palm(thickness=0.020, width=0.060, length=0.060)
    for k, chain in enumerate(chains):
        segs = []
        for n in chain:
            i = index[n]
            box = np.asarray(boxes[i], dtype=float)
            length = float(np.linalg.norm(box[1] - box[0]))
            segs.append(Segment(joint=Joint(theta=0.0), length=max(length, 1e-6),
                                token_box=tuple(map(tuple, box))))
        face = faces[k % len(faces)]
        u0, u1, v0, v1 = mount_uv_bounds(face, the_palm)
        fingers.append(Finger(mount=Mount(face=face, u=0.5 * (u0 + u1), v=0.5 * (v0 + v1)),
                              segments=tuple(segs)))
    return Hand(palm=the_palm, fingers=tuple(fingers))
