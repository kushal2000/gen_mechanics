"""From a grammar hand to its printable parts: where every servo, yoke,
housing and part goes, as OpenSCAD statements over the library
`scad/feetech_hand.scad` (Martin, 2026-10-23: the grammar drives, OpenSCAD
builds).

No new kinematics: every frame comes from the grammar (`derive.fk` at the
zero pose, `derive.finger_geometry`, `derive.hand_geometry`). Millimetres
throughout; a part's frame is its rigid body's frame (the palm frame for the
palm, link j's frame for finger link j).

Each joint has a parent (the part holding the servo's case: the palm or
link j - 1) and a child (link j, holding the horns in a yoke); a flipped
servo (`hand.servo_poses`) the other way round, its case in link j and its
horns in a yoke on link j - 1, which turns about it. Every part is
its positive pieces, less the keep-outs of the servos it holds, less
- as a child: what its joint's servo case (and plug) sweeps over the joint's
  range relative to it;
- as a parent: what its child sweeps relative to it (every convex piece of
  the child, its horns and the servo it carries, grown by `PART_CLEAR`).
The child keeps its shape near a joint and the parent gives way.

The palm is a shell (`_palm`): a plate whose face stands a few mm over the
base servos' tops, a rim down from its edge, every base servo screwed
straight into its housing under the face (the case screws' screwdrivers
pass under the plate), and on the main palm a ring round the mount's column
screws with a spoke out to each housing.

A servo's frame S (servo.py): X along its axis toward the driven horn. It
is the grammar's motor frame (a, v, w), flipped to (-a, v, -w) where the
driven horn must face the other way: along the finger for a single-arm yoke.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from hand_sampler.grammar import derive as dv
from hand_sampler.grammar import motor as mt
from hand_sampler.grammar.hand import PLATE_HALF_MM, SEAM_MM, Hand, cut_side_mm, finger_section

SINGLE_ARM_LEAN = 0.5
"""A joint whose axis leans this much along the finger (|a . x| >= 0.5:
twist, and 30 degrees of elevation or more) gets a one-arm yoke on its
driven horn (the idler horn comes off): a U's rear arm would need room
behind the servo that the link floors do not budget."""

BRIDGE_HALF_MM = 9.5         # a U's bridge, either side of the axis across the finger: as wide as its arms' discs
ARM_X0 = 14.6                                # an arm's inner face from the servo's mid-plane (horn face 14.5 + 0.1)
ARM_DRIVEN, ARM_IDLER = 3.0, 2.0
CASE_CLEAR, PART_CLEAR, WALL, FIT = 0.5, 1.0, 2.0, 0.2
HORN_HOLES_DRIVEN = (45.0, 135.0, 225.0, 315.0)   # degrees from +Z about X (the 12 mm circle)
HORN_HOLES_IDLER = (45.0, 135.0, 225.0, 315.0)     # four screws on every horn (Martin, 10/26)
PLUG = ((-9.2, -2.9), (4.8, 13.2), 9.0)          # (X range, Y range, length beyond the case face)
CASE_Z = 10.0
YOKE_REACH_MM = 20.0          # how far a yoke turning round its axis reaches from it (bridge corners and a gap)
FACE_TRIM_MM = BRIDGE_HALF_MM + PART_CLEAR    # the face stops this far behind a lying base servo's axis (its arms turning)
ARM_R_MM = 9.5                # an arm's disc round the axis (library ARM_R)
REACH_MM = 1.2                # pieces that join reach this far into each other (library REACH)
ROUND_MM = 0.8                # every part's outer edges, rounded on its whole body (render.round_body)
BLOCK_MIN_MM = 6.0            # a right-angle connector at least this thick (Martin, 10/26: not a thin plate)
GUSSET_MM = 16.0              # a gusset's legs (library GUSSET)
HOUSING_MM = (13.6, 26.6, 11.3)   # a housing's outer faces in its servo's frame: side plates (X), back (Y), walls (Z)
SOCKET_Y0_MM = 17.0               # where a cradle's socket starts (library SOCKET_Y0)
CRADLE_BACK_MM = (2 * HOUSING_MM[0], 2 * HOUSING_MM[2])     # a cradle's back face (along its servo's X, Z)
CRADLE_SIDE_MM = (2 * HOUSING_MM[0], HOUSING_MM[1] - SOCKET_Y0_MM)   # a socket's side face (X, Y)
CRADLE_SIDE_Y = (HOUSING_MM[1] + SOCKET_Y0_MM) / 2.0


def housing_corners(S: np.ndarray, grow: float = 0.0, y0: float = -9.5) -> np.ndarray:
    """A housing's (or, from `y0` = SOCKET_Y0_MM, a socket's) corners placed
    by the servo frame S, grown by `grow`."""
    hx, hy, hz = HOUSING_MM
    C = mt.box_corners(np.array([0.0, (y0 + hy) / 2.0, 0.0]), np.eye(3),
                       np.array([hx + grow, (hy - y0) / 2.0 + grow, hz + grow]))
    return C @ S[:3, :3].T + S[:3, 3]
PALM_TOP_MM = PLATE_HALF_MM - mt.WALL_MM   # the base servos' tops (the grammar's palm face less a wall)
FACE_RAISE_MM = 3.5
"""The palm's face stands this far over the base servos' tops, its plate's
underside just above them (Martin, 10/29): the screwdrivers for their case
screws pass under the plate, never through the face."""
PALM_FACE_MM = PALM_TOP_MM + FACE_RAISE_MM   # the palm's face (library PALM_FACE)
PALM_PLATE_MM = 3.0           # the palm's plate, under its face
RIM_MM = 2.5                  # the tray's wall round the face's edge, and a pocket's walls
CRADLE_CHANNEL_MM = 14.0      # a socket's servo needs only its own front clear (it is open on two more sides)
CLIP_MM = 2.4
"""Every link and fingertip is clipped to its simulator body grown by this
much (Martin, 2026-10-23: small overhangs are fine; the checks allow 2.5)."""

Piece = Callable[[str], str]     # growth expression -> a convex OpenSCAD piece in its part's frame


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def T(R: Optional[np.ndarray] = None, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    M[:3, 3] = np.asarray(t, dtype=float)
    return M


def inv(M: np.ndarray) -> np.ndarray:
    R, t = M[:3, :3], M[:3, 3]
    return T(R.T, -R.T @ t)


def mm(M: np.ndarray) -> str:
    """`multmatrix(...)` for a 4x4 (mm)."""
    rows = ",".join("[" + ",".join(f"{v:.5f}" for v in r) + "]" for r in M)
    return f"multmatrix([{rows}])"


def _mat(M: np.ndarray) -> str:
    return "[" + ",".join("[" + ",".join(f"{v:.5f}" for v in r) + "]" for r in M) + "]"


def v3(p) -> str:
    return "[" + ",".join(f"{float(x):.4f}" for x in p) + "]"


def _pts(P) -> str:
    """Points as an OpenSCAD list (for `hull_pts`)."""
    return "[" + ",".join(v3(q[:3]) for q in P) + "]"


def _seg_dist(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Each point's distance to the segment a-b."""
    e = b - a
    t = np.clip((P - a) @ e / max(float(e @ e), 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * e), axis=1)



# --------------------------------------------------------------------------
# servos
# --------------------------------------------------------------------------


@dataclass
class ServoSpot:
    """One servo: its joint, its frame and how it is held and driven."""

    key: Tuple                    # ("f", i, j) or ("hinge",)
    parent: Tuple                 # body holding its case: ("palm", k) or ("link", i, j - 1) (flipped: link j)
    child: Tuple                  # body holding its horns: ("link", i, j) or ("palm", 0) (flipped: link j - 1)
    S_child: np.ndarray           # 4x4: the servo frame in the child's frame (q = 0)
    S_parent: np.ndarray          # 4x4: the servo frame in the parent's frame (q = 0)
    rot_deg: Tuple[float, float]  # the child's turn about the servo's X over the joint's range
    yoke: str                     # "u", "single" or "crossing"
    cable: int                    # the Z face whose socket takes the cable (+1 / -1)
    turned: bool = False
    flip: bool = False
    sign: float = 1.0             # + joint angle turns the driven horn this way about S's X (servo's own sign)
    across: bool = False              # lies across the finger (its 23 x 34 face toward the palm): a square connector
    flange: bool = False              # held by its idler face in a flange cup (its idler horn off: a one-arm yoke)
    flipped: bool = False             # lies along the link after its joint, which holds its case (`hand.servo_poses`)

    @property
    def idler_horn(self) -> bool:
        return self.yoke == "u"


def _servo_frame(a: np.ndarray, v: np.ndarray, origin: np.ndarray, flip: bool) -> np.ndarray:
    a, v = np.asarray(a, float), np.asarray(v, float)
    w = np.cross(a, v)
    s = -1.0 if flip else 1.0
    return T(np.column_stack([s * a, v, s * w]), origin)


def _cable_side(S: np.ndarray, dorsal: np.ndarray, back: np.ndarray) -> int:
    """The Z face facing most toward the finger's back (dorsal), ties toward
    the palm (`back`)."""
    z = S[:3, 2]
    score = float(z @ dorsal) * 10.0 + float(z @ back)
    return 1 if score >= 0.0 else -1


def finger_servos(hand: Hand, i: int) -> List[ServoSpot]:
    f = hand.fingers[i]
    fg = dv.finger_geometry(f)
    pose = dv.fk(hand)
    sec = finger_section(hand, i)
    out = []
    for j, jt in enumerate(f.joints):
        a = fg.axes[j]
        M = fg.motors[j].axes                      # (a, v, w), joint j frame
        turned, flipped = fg.poses[j] == "turned", fg.poses[j] == "flipped"
        crossing = j < len(f.joints) - 1 and jt.length == 0
        single = abs(float(a[0])) >= SINGLE_ARM_LEAN - 1e-9 and not crossing
        # a one-arm yoke's driven horn faces the part carrying it: the fingertip's way (a flipped servo: the palm's)
        flip = single and (float(a[0]) > 0.0 if flipped else float(a[0]) < 0.0)
        origin = fg.slides_mm[j] * a
        S_joint = _servo_frame(M[:, 0], M[:, 1], origin, flip)       # at joint j, in link j's frame at q = 0
        if j == 0:
            Tpj = pose.pre[i][0].copy()
            Tpj[:3, 3] *= 1e3
            before = ("palm", sec)                 # beyond a cut, that part's own frame is the palm frame at q = 0
        else:
            Tpj = T(None, (float(f.joints[j - 1].length), 0.0, 0.0))
            before = ("link", i, j - 1)
        sgn = dv.joint_sign(hand, i, j) * (-1.0 if flip else 1.0)
        if flipped:
            # its case turns with link j and its horns stay with the link before: that link turns about the servo,
            # back by the joint's angle
            parent, child, S_parent, S_child, sgn = ("link", i, j), before, S_joint, Tpj @ S_joint, -sgn
        else:
            parent, child, S_parent, S_child = before, ("link", i, j), Tpj @ S_joint, S_joint
        lo, hi = (math.degrees(x) for x in dv.joint_limits(hand, i, j))
        rot = tuple(sorted((sgn * lo, sgn * hi)))
        dorsal = -np.array([0.0, 1.0, 0.0])        # the link frame's -y
        cable = _cable_side(S_child, dorsal, -np.array([1.0, 0.0, 0.0]))
        yoke = "crossing" if crossing else ("single" if single else "u")
        sp = ServoSpot(("f", i, j), parent, child, S_child, S_parent, rot, yoke, cable, turned, flip, sgn,
                       flipped=flipped)
        if parent[0] == "link":
            # a servo lying across the finger (its 23 x 34 face toward the palm: a turned bend, a twist) gets a square
            # connector (Martin, 10/24); one whose idler horn is off (a one-arm yoke on its output horn) is held by its
            # idler face in a flange cup, nothing round its horn (10/26)
            sp.across = yoke != "crossing" and abs(float(S_parent[0, 1])) < 0.7
            sp.flange = yoke == "single"
        out.append(sp)
    return out


def hinge_servo(hand: Hand) -> Optional[ServoSpot]:
    if not hand.palm_joints:
        return None
    g = dv.hand_geometry(hand)
    hb = g.hinge_box
    S = _servo_frame(hb.axes[:, 0], hb.axes[:, 1], g.hinge_mm, False)
    sgn = float(np.sign(dv.palm_axis(hand) @ hb.axes[:, 0])) or 1.0
    lo, hi = (math.degrees(x) for x in dv.palm_limits())
    rot = tuple(sorted((sgn * lo, sgn * hi)))
    cable = _cable_side(S, np.array([-1.0, 0.0, 0.0]), np.zeros(3))
    return ServoSpot(("hinge",), ("palm", -1), ("palm", 0), S, S.copy(), rot, "u", cable, False, False, sgn)


# --------------------------------------------------------------------------
# parts
# --------------------------------------------------------------------------


@dataclass
class Part:
    name: str
    body: Tuple                                # ("palm", k) or ("link", i, j)
    frame: np.ndarray                          # 4x4: the body frame in the palm frame at q = 0 (mm)
    positive: List[str] = field(default_factory=list)
    negative: List[str] = field(default_factory=list)
    pieces: List[Piece] = field(default_factory=list)   # convex pieces (for the parent's sweep)
    holds: List[ServoSpot] = field(default_factory=list)  # servos whose case it holds
    carries: List[ServoSpot] = field(default_factory=list)  # servos whose horns it holds (two where the next is flipped)
    clip: Optional[str] = None                 # a convex body the part stays inside (its simulator body, grown)
    swept_by: Dict[int, List[str]] = field(default_factory=dict)   # palm parts: each finger's sweeps
    kept: List[Tuple[int, str]] = field(default_factory=list)      # palm parts: (finger, pocket walls kept from other fingers)
    tunnels: List[str] = field(default_factory=list)                # screwdrivers' ways in (never through a kept housing)
    face: Optional[np.ndarray] = None                                # palm parts: the face's outline (y, z)

    def scad(self) -> str:
        """Three modules: `pieces_` (its positive pieces, sharp, within its
        clip), `body_` (those rounded as a whole: imported from bodies/,
        made by `render.round_body`) and `part_` (the body less everything
        it gives way to)."""
        n = self.name
        pieces = "union() {\n        " + ("\n        ".join(self.positive) or "cube(0.001);") + "\n        }"
        if self.clip:
            pieces = f"intersection() {{\n    {pieces}\n    {self.clip}\n    }}"
        neg = "\n    ".join(self.negative + [x for xs in self.swept_by.values() for x in xs] + self.tunnels)
        body = f"difference() {{\n        body_{n}();\n    {neg}\n    }}"
        if self.kept:
            # a servo's housing keeps its walls against other fingers' sweeps (fingers colliding in
            # motion is left to the physics and the policy, as in the simulator), and gives way to
            # everything else: its own finger's sweeps, the cut's hinge, every servo's keep-outs
            keep = []
            for i, walls in self.kept:
                rest = (self.negative + self.swept_by.get(i, [])
                        + (self.swept_by.get(-1, []) if i != -1 else []))
                keep.append(f"difference() {{ intersection() {{ body_{n}(); {walls} }}\n        "
                            + "\n        ".join(rest) + " }")
            body = f"union() {{\n    {body}\n    " + "\n    ".join(keep) + "\n    }"
        return (f"module pieces_{n}() {{\n    {pieces}\n}}\n"
                f"module body_{n}() if (ROUNDED) import(\"bodies/{n}.stl\"); else pieces_{n}();\n"
                f"module part_{n}() {{\n    {body}\n}}\n")


def _bridge_frame(origin: np.ndarray, front: np.ndarray, a: np.ndarray, x_b: float) -> np.ndarray:
    """A U's bridge frame: x along `front` (the way the yoke leaves the
    joint), from `origin` at x_b, y along the joint axis (canonical), z
    across."""
    a = np.asarray(a, float)
    ex = np.asarray(front, float) / np.linalg.norm(front)
    ay = a - ex * float(a @ ex)
    ay /= np.linalg.norm(ay)
    return T(np.column_stack([ex, ay, np.cross(ex, ay)]), np.asarray(origin, float) + x_b * ex)


def _yoke_pieces(sp: ServoSpot, a: np.ndarray, front: np.ndarray, x_b: float) -> Tuple[List[Piece], Piece]:
    """The yoke's convex pieces in the child's frame, and the piece the rest
    of the part joins to (the bridge, or the one arm). A U's bridge lies
    `x_b` along `front` from the joint, past the case's swing."""
    S = mm(sp.S_child)
    if sp.yoke == "u":
        B = _bridge_frame(sp.S_child[:3, 3], front, a, x_b)
        # the arms along the canonical axis: the driven side is +a unless flipped
        s_pos = -1.0 if sp.flip else 1.0
        a_lo = -(ARM_X0 + (ARM_DRIVEN if s_pos < 0 else ARM_IDLER))
        a_hi = ARM_X0 + (ARM_DRIVEN if s_pos > 0 else ARM_IDLER)
        Bm = mm(B)

        def bridge(g: str, Bm=Bm, a_lo=a_lo, a_hi=a_hi) -> str:
            return (f"{Bm} box_lh([-({g}), {a_lo:.3f} - ({g}), {-BRIDGE_HALF_MM:.3f} - ({g})], "
                    f"[{WALL:.3f} + ({g}), {a_hi:.3f} + ({g}), {BRIDGE_HALF_MM:.3f} + ({g})]);")

        def arm(s: int) -> Piece:
            return lambda g: (f"hull() {{ {S} arm_disc({s}, {g}); intersection() {{ {bridge(g)} {S} arm_slab({s}, {g}); }} }}")

        return [arm(1), arm(-1), bridge], bridge
    if sp.yoke == "single":
        arm1: Piece = lambda g: f"{S} arm_disc(1, {g});"
        return [arm1], arm1
    # crossing: one plate on the near (driven) horn, as thick as the gap to the next case
    raise ValueError("crossing yokes are built with their next housing")


def _horn_pieces(sp: ServoSpot) -> List[Piece]:
    """The horns and their centre screws, fixed to the child."""
    S = mm(sp.S_child)
    out: List[Piece] = [lambda g: f"{S} cyl_x(HORN_R + ({g}), DRIVEN_X[0], HEAD_X + ({g}));"]
    if sp.idler_horn:
        out.append(lambda g: f"{S} cyl_x(HORN_R + ({g}), -HEAD_X - ({g}), IDLER_X[1]);")
    return out


def _case_pieces(S_in_body: np.ndarray, cable: int, idler_horn: bool = True) -> List[Piece]:
    """A servo's case, its cable's plug (on the cable side) and, with the
    idler horn off, the idler stub: convex, in the body holding it."""
    S = mm(S_in_body)
    (x0, x1), (y0, y1), out = PLUG
    z0, z1 = (CASE_Z, CASE_Z + out) if cable > 0 else (-CASE_Z - out, -CASE_Z)
    pieces: List[Piece] = [lambda g: f"{S} case_box({g});",
                           lambda g: (f"{S} box_lh([{x0} - ({g}), {y0} - ({g}), {z0} - ({g})], "
                                      f"[{x1} + ({g}), {y1} + ({g}), {z1} + ({g})]);")]
    if not idler_horn:
        pieces.append(lambda g: f"{S} cyl_x(STUB_R + ({g}), STUB_X - ({g}), 0);")
    return pieces


def _clip_body(f, j: int) -> str:
    """Link j's simulator body grown by `CLIP_MM`, as a convex OpenSCAD body:
    the link's box, or the fingertip's hull (its capsule and yoke)."""
    fg = dv.finger_geometry(f)
    if j < len(f.joints) - 1:
        b = fg.links[j].grown(CLIP_MM)
        lo, hi = b.centre - b.half, b.centre + b.half
        return f"box_lh({v3(lo)}, {v3(hi)});"
    pts = fg.tip_hull.points          # the simulator's own points (its capsule is a 12-sided polygon)
    return f"minkowski() {{ hull_pts({_pts(pts)}); sphere(r = {CLIP_MM}, $fn = 16); }}"


def along_only(eS: np.ndarray) -> bool:
    """A finger leaving its base servo along the servo's axis (a twist)."""
    return bool(abs(float(eS[0])) > max(abs(float(eS[1])), abs(float(eS[2]))))


def build_design(hand: Hand) -> "Design":
    return Design(hand)


class Design:
    """Every part of `hand`, its servos and screws."""

    def __init__(self, hand: Hand, more_tubes: Optional[set] = None):
        self.hand = hand
        self.more_tubes = set(more_tubes or ())     # links whose cradle gets a tube to each face it can take
        self.tubes: Dict[Tuple[int, int], np.ndarray] = {}    # each link's connector: its 8 corners (link frame)
        self.reach: Dict[Tuple[int, int], List[np.ndarray]] = {}   # and where its gussets may reach (the same, moved)
        self.servos: Dict[Tuple, ServoSpot] = {}
        self.parts: Dict[Tuple, Part] = {}
        pose = dv.fk(hand)
        for i, f in enumerate(hand.fingers):
            for sp in finger_servos(hand, i):
                self.servos[sp.key] = sp
            for j in range(len(f.joints)):
                Tl = pose.links[i][j].copy()
                Tl[:3, 3] *= 1e3
                self.parts[("link", i, j)] = Part(f"f{i}_l{j}" if j < len(f.joints) - 1 else f"f{i}_tip",
                                                  ("link", i, j), Tl)
        self.parts[("palm", -1)] = Part("palm", ("palm", -1), np.eye(4))
        self.extras: List[Part] = []          # parts fixed to the main palm (the bay and the mount: later)
        h = hinge_servo(hand)
        if h is not None:
            self.servos[h.key] = h
            self.parts[("palm", 0)] = Part("palm_beyond", ("palm", 0), np.eye(4))
        for sp in self.servos.values():
            self.parts[sp.parent].holds.append(sp)
            self.parts[sp.child].carries.append(sp)
        for i, f in enumerate(hand.fingers):
            for j in range(len(f.joints)):
                self._link(i, j)
        self._palm(-1)
        if h is not None:
            self._palm(0)
        self._sweeps()

    # ---------------------------------------------------------------- links

    def _link(self, i: int, j: int) -> None:
        """Link j: at its joint the yoke on servo j's horns (a U, one arm, or a
        crossing's plate) or, where servo j is flipped, its case in a cradle;
        at the next joint servo j + 1's case in a cradle or, where that one is
        flipped, a yoke on its horns; a connector from one end's face to the
        other's, with gussets in the L pieces' inside corners. The fingertip:
        its yoke or cradle and a rounded end."""
        f = self.hand.fingers[i]
        fg = dv.finger_geometry(f)
        part = self.parts[("link", i, j)]
        sp = self.servos[("f", i, j)]
        a = fg.axes[j]
        last = j == len(f.joints) - 1
        nxt = None if last else self.servos[("f", i, j + 1)]
        X = np.array([1.0, 0.0, 0.0])
        yokes: List[Tuple[ServoSpot, np.ndarray, np.ndarray]] = []    # (servo, its axis, the way its yoke leaves it)
        cases: List[ServoSpot] = []                                  # servos whose case this link holds
        if sp.yoke == "crossing":
            # the plate on servo j's near horn, as far as the next servo's case
            S = mm(sp.S_child)
            Sn = mm(nxt.S_parent)
            gap = self._crossing_gap(sp, nxt)
            plate: Piece = lambda g, S=S, Sn=Sn, gap=gap: (
                f"hull() {{ {S} cyl_x(ARM_R + ({g}), ARM_X0 - ({g}), {gap:.3f} + ({g})); {Sn} housing_box({g}); }}")
            pieces = [plate]
            near = None
        elif sp.flipped:
            cases.append(sp)
            near = ("case", sp)
            pieces = []
        else:
            yokes.append((sp, a, X))
            near = ("yoke", sp, a, X)
            pieces = []
        far = None
        if nxt is not None and sp.yoke != "crossing":
            if nxt.flipped:
                far = ("yoke", nxt, fg.axes[j + 1], -X)
                yokes.append((nxt, fg.axes[j + 1], -X))
            else:
                far = ("case", nxt)
                cases.append(nxt)
        elif nxt is not None:
            cases.append(nxt)
        # each end's pieces and the face a connector meets
        ends = []
        for e in (near, far):
            if e is None:
                ends.append(None)
            elif e[0] == "yoke":
                _, y, ax, front = e
                x_b = mt.swept_front_mm(ax, y.turned) + CASE_CLEAR
                yp, jp = _yoke_pieces(y, ax, front, x_b)
                A, sa = self._yoke_face(y, ax, front)
                ends.append({"kind": "yoke", "sp": y, "axis": ax, "front": front, "pieces": yp, "joint": jp,
                             "face": A, "size": sa, "reach": REACH_MM if y.yoke == "u" else 0.0})
            else:
                c = e[1]
                Sc = mm(c.S_parent)
                if c.flange:
                    cradle: Piece = lambda g, Sc=Sc: f"{Sc} flange_cup({g});"
                else:
                    cradle = lambda g, Sc=Sc: f"{Sc} cradle_socket({g});"
                ends.append({"kind": "case", "sp": c, "pieces": [cradle], "joint": cradle, "reach": REACH_MM})
        for e in ends:
            if e is not None:
                pieces += e["pieces"]
        if last:
            e = ends[0]
            tip = fg.tip
            p1, r = tip.p1, tip.radius
            # the rounded end only ahead of the yoke or the case (behind it, round the servo, the fingertip is cut
            # away anyway; the part before it should not give way to it)
            if e["kind"] == "case":
                x_cut = float((sp.S_parent @ np.array([0.0, HOUSING_MM[1], 0.0, 1.0]))[0])
            elif sp.yoke == "u":
                x_cut = mt.swept_front_mm(a, sp.turned) + CASE_CLEAR
            else:
                th = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
                disc = np.array([[x, 9.5 * math.cos(t), 9.5 * math.sin(t)] for x in (ARM_X0, ARM_X0 + ARM_DRIVEN) for t in th])
                x_cut = float((disc @ sp.S_child[:3, :3].T + sp.S_child[:3, 3])[:, 0].min())
            body: Piece = lambda g, jp=e["joint"], p1=p1, r=r, xc=x_cut: (
                f"hull() {{ {jp(g)} intersection() {{ translate({v3(p1)}) sphere(r = {r:.3f} + ({g}), $fn = 48); "
                f"translate([{xc:.3f} - ({g}), -100, -100]) cube([300, 200, 200]); }} }}")
            pieces = pieces + [body]
        elif far is not None:
            # a bracket frame: a connector from the near end's whole face (a bridge's, a single arm's horn face, a
            # cradle's) to the far end's (`_connector`)
            n0, n1 = ends
            if n0["kind"] == "case":
                n0["face"], n0["size"], n0["side"] = self._tube_end(n0["sp"], self._end_centre(n1))
            if n1["kind"] == "case":
                n1["face"], n1["size"], n1["side"] = self._tube_end(n1["sp"], n0["face"][:3, 3])
            square = any(e["kind"] == "case" and e["sp"].across for e in ends)
            # a single arm's face leads: a quarter-turned connector's block then stands out from the arm, along its
            # servo's axis, never round the front of that servo's case (the part holding the case gives way to it)
            arm = [e for e in ends if e["kind"] == "yoke" and e["sp"].yoke == "single"]
            lead, other = (n1, n0) if arm and arm[0] is n1 and len(arm) == 1 else (n0, n1)
            A, sa, B, sb = self._connector(lead["face"], lead["size"], other["face"], other["size"], square=square)
            self.tubes[(i, j)] = np.array([(F @ np.array([0.0, sy * sz_[0] / 2.0, sz * sz_[1] / 2.0, 1.0]))[:3]
                                           for F, sz_ in ((A, sa), (B, sb)) for sy in (-1, 1) for sz in (-1, 1)])
            for e in ends:
                if e["kind"] == "case":
                    side = e["side"]
                    if side in (1, -1) and e["sp"].cable == side:
                        e["sp"].cable = -side                    # the tube lands on that side: the cable takes the other
            # its ends reach into a bridge and a cradle (a single arm's face stays on its horn's face: nothing round
            # a horn)
            tube: Piece = lambda g, A=A, B=B, sa=sa, sb=sb, ea=lead["reach"], eb=other["reach"]: (
                f"tube({_mat(A)}, {list(sa)}, {_mat(B)}, {list(sb)}, {g}, {ea}, {eb});")
            extra: List[Piece] = []
            if ("link", i, j) in self.more_tubes:
                # this link came out in pieces (something turns through its tube): a tube to every face of a cradle
                for k, e in enumerate(ends):
                    if e["kind"] != "case":
                        continue
                    o = ends[1 - k]
                    for Bx, sbx, fx in self._tube_ends(e["sp"]):
                        if fx != e["side"] and not (fx != 0 and fx == e["sp"].cable):
                            Ax, sax, Bx, sbx = self._connector(o["face"], o["size"], Bx, sbx, square=square)
                            extra.append(lambda g, A=Ax, B=Bx, sa=sax, sb=sbx, ea=o["reach"]:
                                         f"tube({_mat(A)}, {list(sa)}, {_mat(B)}, {list(sb)}, {g}, {ea}, {REACH_MM});")

            def along(o: Piece, v: np.ndarray) -> Piece:
                """The connector and its copy moved `GUSSET_MM` along v: where a gusset may stand on it."""
                sh = mm(T(None, tuple(float(c) for c in v * GUSSET_MM)))
                self.reach.setdefault((i, j), []).append(self.tubes[(i, j)] + v * GUSSET_MM)
                return lambda g: f"hull() {{ {o(g)} {sh} {o(g)} }}"

            # gussets in the L pieces' inside corners (Martin, 10/26), each kept to where the connector is (none
            # round a flange cup: lying along the servo's sides with nothing to hold, they only read as ears; 10/28)
            gussets: List[Piece] = []
            for e in ends:
                if e["kind"] == "case":
                    c, side = e["sp"], e["side"]
                    if not c.flange and side in (1, -1):
                        Sc = mm(c.S_parent)
                        on = along(tube, -side * c.S_parent[:3, 2])        # from the side it lands on, into the socket
                        gussets += [lambda g, on=on, Sc=Sc, f=side, s=s: (
                            f"intersection() {{ {Sc} socket_gusset({f}, {s}, GUSSET, {g}); {on(g)} }}") for s in (1, -1)]
                elif e["sp"].yoke == "u":
                    gussets += self._arm_gussets(e["sp"], e["axis"], along(tube, -e["face"][:3, 0]), e["front"])
                else:
                    # a single arm and its connector are one plate: the arm takes in the connector's footprint in
                    # its own layer, so where the connector leaves the horn's face at a slant they still meet
                    # through the arm's whole thickness (not a sliver the body's rounding would take away)
                    S0 = mm(e["sp"].S_child)
                    arm1 = e["pieces"][0]
                    merged: Piece = lambda g, arm1=arm1, S0=S0: (
                        f"hull() {{ {arm1(g)} intersection() {{ {tube(g)} {S0} arm_slab(1, {g}); }} }}")
                    pieces = [merged if q is arm1 else q for q in pieces]
            pieces = pieces + [tube] + gussets + extra
        part.pieces = pieces
        part.positive = [p("0") for p in pieces]
        part.clip = _clip_body(f, j)
        # what this part leaves empty: the horns of the servos it carries and their yokes' screws, the keep-outs of
        # the servos it holds
        for y, _, _ in yokes + ([(sp, a, X)] if sp.yoke == "crossing" else []):
            S = mm(y.S_child)
            part.negative.append(f"{S} cyl_x(HORN_R + CASE_CLEAR, DRIVEN_X[0] - FIT, DRIVEN_X[1]);")
            if y.idler_horn:
                part.negative.append(f"{S} cyl_x(HORN_R + CASE_CLEAR, IDLER_X[0], -(DRIVEN_X[0] - FIT));")
            if y.yoke == "u":
                part.negative += [f"{S} arm_keepout(1, {list(HORN_HOLES_DRIVEN)});",
                                  f"{S} arm_keepout(-1, {list(HORN_HOLES_IDLER)});"]
            else:
                part.negative.append(f"{S} arm_keepout(1, {list(HORN_HOLES_DRIVEN)});")
        for c in cases:
            Sc = mm(c.S_parent)
            part.tunnels.append(f"{Sc} case_screw_tunnels(40, {str(c.flange).lower()});")
            if c.flange:
                part.negative.append(f"{Sc} flange_keepout({c.cable});")
            else:
                ch = 80 if c.yoke == "crossing" else CRADLE_CHANNEL_MM
                part.negative.append(f"{Sc} housing_keepout({str(c.yoke != 'u').lower()}, {c.cable}, {ch});")

    @staticmethod
    def _end_centre(e: dict) -> np.ndarray:
        """Where a link's end is (its face's centre; a cradle's servo's back)."""
        if "face" in e:
            return e["face"][:3, 3]
        return (e["sp"].S_parent @ np.array([0.0, HOUSING_MM[1], 0.0, 1.0]))[:3]

    @staticmethod
    def _arm_gussets(sp: ServoSpot, a: np.ndarray, on: Piece, front: np.ndarray) -> List[Piece]:
        """A U's gussets: in each arm's layer, the corners between the arm's
        edges and the connector's back face, where the connector is wider
        than the bridge (`on`: where they may stand). Outboard of the case,
        so they turn clear of the part holding the servo."""
        x_b = mt.swept_front_mm(a, sp.turned) + CASE_CLEAR
        B = _bridge_frame(sp.S_child[:3, 3], front, a, x_b)
        s_pos = -1.0 if sp.flip else 1.0
        a_lo = -(ARM_X0 + (ARM_DRIVEN if s_pos < 0 else ARM_IDLER))
        a_hi = ARM_X0 + (ARM_DRIVEN if s_pos > 0 else ARM_IDLER)
        x0, L, h = WALL - REACH_MM, GUSSET_MM, BRIDGE_HALF_MM
        out: List[Piece] = []
        for y0, y1 in ((a_lo, -ARM_X0), (ARM_X0, a_hi)):
            for zs in (1.0, -1.0):
                # the triangle reaches REACH back into the arm and the connector (one solid with both)
                R = REACH_MM
                pts = [(B @ np.array([x0 - u, y, zs * (h + v), 1.0]))[:3]
                       for u, v in ((-R, -R), (L + R, -R), (-R, L + R)) for y in (y0, y1)]
                out.append(lambda g, pts=_pts(pts), on=on: f"intersection() {{ hull_pts({pts}, {g}); {on(g)} }}")
        return out

    @staticmethod
    def _connector(A: np.ndarray, sa, B: np.ndarray, sb, square: bool = False) -> Tuple[
            np.ndarray, Tuple[float, float], np.ndarray, Tuple[float, float]]:
        """The two rectangles a link's connector joins (frames with the
        rectangle in their y-z plane, and sizes). Where the cradle face is
        turned a quarter turn against the bridge's (a spread and a bend at
        one knuckle), a hull between them twists: then the connector is a
        straight block in the bridge's frame, wide enough for both faces,
        from the bridge to the cradle face's far edge (Martin, 10/24: work
        in right angles there). `square`: a block whatever the faces (a
        servo lying across the finger)."""
        if not square and abs(float(B[:3, 1] @ A[:3, 1])) >= 0.5:
            return A, tuple(sa), B, tuple(sb)
        cb = np.array([B @ np.array([0.0, sy * sb[0] / 2.0, sz * sb[1] / 2.0, 1.0]) for sy in (-1, 1) for sz in (-1, 1)])
        loc = (cb[:, :3] - A[:3, 3]) @ A[:3, :3]
        ys = np.concatenate([loc[:, 1], [-sa[0] / 2.0, sa[0] / 2.0]])
        zs = np.concatenate([loc[:, 2], [-sa[1] / 2.0, sa[1] / 2.0]])
        yc, zc = (ys.min() + ys.max()) / 2.0, (zs.min() + zs.max()) / 2.0
        size = (float(ys.max() - ys.min()), float(zs.max() - zs.min()))
        x_end = max(float(loc[:, 0].max()), BLOCK_MIN_MM)
        return A @ T(None, (0.0, yc, zc)), size, A @ T(None, (x_end, yc, zc)), size

    def _yoke_face(self, sp: ServoSpot, a: np.ndarray, front: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Where a link's connector leaves a yoke whose bridge lies along
        `front` from the joint (link frame): the bridge's whole outer face; a
        single arm's horn face, as wide as the arm's disc (the connector and
        the arm are one plate, Martin 10/26)."""
        if sp.yoke == "u":
            x_b = mt.swept_front_mm(a, sp.turned) + CASE_CLEAR
            B = _bridge_frame(sp.S_child[:3, 3], front, a, x_b)
            s_pos = -1.0 if sp.flip else 1.0
            a_lo = -(ARM_X0 + (ARM_DRIVEN if s_pos < 0 else ARM_IDLER))
            a_hi = ARM_X0 + (ARM_DRIVEN if s_pos > 0 else ARM_IDLER)
            return B @ T(None, (WALL, (a_lo + a_hi) / 2.0, 0.0)), (a_hi - a_lo, 2.0 * BRIDGE_HALF_MM)
        return sp.S_child @ T(None, (ARM_X0, 0.0, 0.0)), (2.0 * ARM_R_MM, 2.0 * ARM_R_MM)

    @staticmethod
    def _flange_face(nxt: ServoSpot) -> Tuple[np.ndarray, Tuple[float, float]]:
        """A flange cup's plate, seen from behind (link frame): its frame
        (the plate in the frame's y-z plane, x out of it) and size."""
        S = nxt.S_parent
        R = S[:3, :3]
        y0, y1 = -9.5 - FIT - 1.2, 24.5 + FIT + 1.2               # library FLANGE_Y
        F = np.column_stack([-R[:, 0], R[:, 1], -R[:, 2]])
        return T(F, R @ np.array([-HOUSING_MM[0], (y0 + y1) / 2.0, 0.0]) + S[:3, 3]), (y1 - y0, 2.0 * HOUSING_MM[2])

    def _tube_ends(self, nxt: ServoSpot) -> List[Tuple[np.ndarray, Tuple[float, float], int]]:
        """Every cradle face a tube can arrive at: the back plate and both
        side walls (a flange cup: its plate)."""
        if nxt.flange:
            F, size = self._flange_face(nxt)
            return [(F, size, 2)]
        S = nxt.S_parent
        R = S[:3, :3]
        out = []
        hz = HOUSING_MM[2]
        for side, c, F in ((0, np.array([0.0, HOUSING_MM[1], 0.0]), np.column_stack([R[:, 1], -R[:, 0], R[:, 2]])),
                           (1, np.array([0.0, CRADLE_SIDE_Y, hz]), np.column_stack([R[:, 2], R[:, 0], R[:, 1]])),
                           (-1, np.array([0.0, CRADLE_SIDE_Y, -hz]), np.column_stack([-R[:, 2], -R[:, 0], R[:, 1]]))):
            out.append((T(F, R @ c + S[:3, 3]), CRADLE_BACK_MM if side == 0 else CRADLE_SIDE_MM, side))
        return out

    def _tube_end(self, nxt: ServoSpot, start: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], int]:
        """The cradle face a tube arrives at (link frame): its back plate or
        a side wall (Z face), whichever faces the tube's start; the face's
        Z side (0 for the back plate). A flange cup: its plate (side 2)."""
        if nxt.flange:
            F, size = self._flange_face(nxt)
            return F, size, 2
        S = nxt.S_parent
        R = S[:3, :3]
        hz = HOUSING_MM[2]
        faces = [(0, R[:, 1], np.array([0.0, HOUSING_MM[1], 0.0]), np.column_stack([R[:, 1], -R[:, 0], R[:, 2]])),
                 (1, R[:, 2], np.array([0.0, CRADLE_SIDE_Y, hz]), np.column_stack([R[:, 2], R[:, 0], R[:, 1]])),
                 (-1, -R[:, 2], np.array([0.0, CRADLE_SIDE_Y, -hz]), np.column_stack([-R[:, 2], -R[:, 0], R[:, 1]]))]
        best = None
        for side, n, c, F in faces:
            p = S[:3, :3] @ c + S[:3, 3]
            d = start - p
            score = float(n @ d) / (np.linalg.norm(d) + 1e-9)
            if best is None or score > best[0] + 1e-6:
                best = (score, side, T(F, p))
        _, side, M = best
        # the whole face: the back plate, or an open side across both side plates
        return M, (CRADLE_BACK_MM if side == 0 else CRADLE_SIDE_MM), side

    def _crossing_gap(self, sp: ServoSpot, nxt: ServoSpot) -> float:
        """How far from servo j's mid-plane (along its X) the next servo's
        case starts, less the fit: where the crossing plate ends."""
        C = mt.box_corners(np.array([0.0, 7.5, 0.0]), np.eye(3), np.array([11.5, 17.0, 10.0]))
        P = C @ nxt.S_parent[:3, :3].T + nxt.S_parent[:3, 3]       # in the child (link j) frame
        X = (P - sp.S_child[:3, 3]) @ sp.S_child[:3, 0]
        return float(X.min()) - FIT

    # ---------------------------------------------------------------- palm

    def _palm(self, sec: int) -> None:
        """A palm part, a tray designed face first (Martin, 10/28-29): a plate
        whose face stands `FACE_RAISE_MM` over the base servos' tops, over the
        palm's outline less what lies ahead of a lying base servo's yoke (no
        cut-outs round the U outputs: `_face_outline`), and one wall round the
        whole outline down to the palm's back face, which stays open (servos
        and cables go in from behind; the arm and the board come later). Each
        base servo screws straight into its housing under the face and its
        finger turns in a pocket (`_pocket`): a clean box round what turns at
        the finger's base, walled where it lies inside the palm, open where
        the finger leaves it (through the wall for a servo lying at the edge,
        through the face for one under it). Past its pocket a finger folding
        onto the palm stops on it, as in the simulator, whose palm is a solid
        plate (Martin, 10/29)."""
        g = dv.hand_geometry(self.hand)
        part = self.parts[("palm", sec)]
        x0, x1 = PALM_FACE_MM - PALM_PLATE_MM, PALM_FACE_MM          # the plate's underside and face
        keep: List[np.ndarray] = []
        bridge_up: List[Piece] = []
        if sec == 0 and part.carries:
            # past a cut the hinge yoke carries the part: the face reaches over it, and its arms and bridge run
            # straight up into the plate
            h = part.carries[0]
            B = _bridge_frame(h.S_child[:3, 3], -h.S_child[:3, 1], h.S_child[:3, 0], mt.FRONT_SWEEP_MM + CASE_CLEAR)
            C = mt.box_corners(np.array([WALL / 2.0, 0.0, 0.0]), np.eye(3), np.array([WALL / 2.0, ARM_X0, BRIDGE_HALF_MM]))
            Cp = C @ B[:3, :3].T + B[:3, 3]
            keep += list(Cp[:, 1:])
            lift = (x0 + x1) / 2.0 - float(Cp[:, 0].max())
            yp, _ = _yoke_pieces(h, h.S_child[:3, 0], -h.S_child[:3, 1], mt.FRONT_SWEEP_MM + CASE_CLEAR)
            bridge_up += [lambda gg, q=q, lift=lift: f"hull() {{ {q(gg)} translate([{lift:.3f}, 0, 0]) {q(gg)} }}"
                          for q in yp]
        outline = self._face_outline(part, g.outlines[sec], x0, keep)
        part.face = outline
        o = "[" + ",".join(f"[{y:.3f},{z:.3f}]" for y, z in outline) + "]"
        plate: Piece = lambda gg, o=o: f"plate({o}, {x0} - ({gg}), {x1} + ({gg}));"
        pieces: List[Piece] = [plate]
        part.clip = f"translate([-500, -500, -500]) cube([{500 + x1}, 1000, 1000]);"    # nothing above the face
        for sp in part.holds:
            Sp = mm(sp.S_parent)
            crossing = sp.yoke == "crossing"
            at_plate = self._at_plate(sp)
            pocket = self._pocket(sp) if sp.key[0] == "f" else None
            along_axis = pocket is not None and pocket[1] == 1        # the finger leaves along the servo's axis
            ch, back = (80, 0) if (crossing or not at_plate) else (CRADLE_CHANNEL_MM, 0)
            if crossing and float(sp.S_parent[0, 1]) < -0.7:
                ch, back = 0, 80        # a crossing's servo standing on its back end: in from the open back, under the face
            keepout = f"{Sp} housing_keepout({str(sp.yoke != 'u').lower()}, {sp.cable}, {ch}, {back});"
            part.negative.append(keepout)
            if at_plate or crossing or along_axis:
                # its socket (a full housing for a crossing's servo, or one whose finger leaves along its axis),
                # its outline straight up into the plate (the case's keep-out leaves the walls)
                y0 = SOCKET_Y0_MM if (at_plate and not crossing and not along_axis) else -9.5
                body = f"box_lh([{-HOUSING_MM[0]} - ({{g}}), {y0} - ({{g}}), {-HOUSING_MM[2]} - ({{g}})], " \
                       f"[{HOUSING_MM[0]} + ({{g}}), {HOUSING_MM[1]} + ({{g}}), {HOUSING_MM[2]} + ({{g}})])"
                up = housing_corners(sp.S_parent, y0=y0)
                up[:, 0] = (x0 + x1) / 2.0
                ped: Piece = lambda gg, Sp=Sp, body=body, pts=_pts(up): (
                    f"hull() {{ {Sp} {body.format(g=gg)}; hull_pts({pts}); }}")
                pieces.append(ped)
                housing = [lambda gg, ped=ped, Sp=Sp: f"intersection() {{ {ped(gg)} {Sp} housing_box({gg}); }}"]
            else:
                # standing under the face: its housing from its pocket's floor down (the floor and the pocket's
                # walls carry it up to the face)
                y0 = pocket[0][0][1][1]
                box: Piece = lambda gg, Sp=Sp, y0=y0: (
                    f"{Sp} box_lh([{-HOUSING_MM[0]} - ({gg}), {y0:.3f} - ({gg}), {-HOUSING_MM[2]} - ({gg})], "
                    f"[{HOUSING_MM[0]} + ({gg}), {HOUSING_MM[1]} + ({gg}), {HOUSING_MM[2]} + ({gg})]);")
                pieces.append(box)
                housing = [box]
            if pocket is not None:
                # the pocket: emptied of everything but the servos' housings (`kept`), walled where it is inside
                # the palm and down to the servo's housing (a crossing's servo slid out under the palm's back)
                top = float(housing_corners(sp.S_parent, y0=-9.5 if crossing else SOCKET_Y0_MM)[:, 0].max())
                x_bot = min(-PLATE_HALF_MM, top - RIM_MM)
                if crossing:
                    # a crossing's servo slid out under the palm: its pocket's floor (on the housing) and walls
                    # reach down to it, a cup under the tray
                    lo, hi = pocket[0][0]
                    t = RIM_MM
                    floor = np.array([[lo[0] - t, y, z, 1.0] for y in (-9.5 - t, hi[1] + t) for z in (lo[2] - t, hi[2] + t)])
                    x_bot = min(x_bot, float((floor @ sp.S_parent.T)[:, 0].min()) - REACH_MM)
                for lo, hi in pocket[0]:
                    part.swept_by.setdefault(sp.key[1], []).append(f"{Sp} box_lh({v3(lo)}, {v3(hi)});")
                    pieces += self._pocket_walls(sp.S_parent, lo, hi, 0, f"plate({o}, {x_bot:.3f}, {x1 + REACH_MM});")
                if pocket[2] is not None:
                    part.swept_by[sp.key[1]].append(f"{Sp} hull_pts({_pts(pocket[2])}, {CLIP_MM + PART_CLEAR});")
            part.tunnels += self._palm_tunnels(sp.S_parent, x0, x1)
            part.tunnels += self._arm_access(sp)
            if sp.key[0] in ("f", "hinge"):
                # its housing keeps its walls against other fingers' pockets (the hinge's: against every finger)
                walls = " ".join(c("0") for c in housing)
                part.kept.append((sp.key[1] if sp.key[0] == "f" else -1,
                                  f"difference() {{ union() {{ {walls} }} {keepout} }}"))
        pieces += self._rim(outline, x0, sec)
        pieces += bridge_up
        if sec == 0 and part.carries:                  # the hinge's yoke on the part beyond the cut
            sp = part.carries[0]
            front = -sp.S_child[:3, 1]                 # ahead of the hinge servo: across the cut
            yp, _ = _yoke_pieces(sp, sp.S_child[:3, 0], front, mt.FRONT_SWEEP_MM + CASE_CLEAR)
            pieces += yp
            S = mm(sp.S_child)
            part.negative += [f"{S} arm_keepout(1, {list(HORN_HOLES_DRIVEN)});",
                              f"{S} arm_keepout(-1, {list(HORN_HOLES_IDLER)});"]
        part.pieces = pieces
        part.positive = [p("0") for p in pieces]

    def _pocket(self, sp: ServoSpot) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int, Optional[np.ndarray]]:
        """The pocket a base servo's finger turns in: boxes in the servo's
        frame (lo, hi) round what turns at the finger's base (`_turn_extent`),
        run out 200 mm along the axis the finger leaves the palm by most (its
        front, -Y; along the axis past the driven horn for a twist); whether
        it leaves along the axis (1, else -2); and the corners of a convex
        pocket where the base link is in the palm ahead of the boxes, turned
        up to 10 degrees either way (or None). Behind a yoke the box stops
        `FACE_TRIM_MM` behind the axis (the arms' discs and a gap) over the
        case, or behind where a single arm's connector turns (it stands out
        behind the axis as far as the arm's disc), with the turning reach
        beside the case in a single arm's layer; a crossing's runs along the
        axis from the near horn past the next servo and the next link's yoke."""
        S = sp.S_parent
        e = dv.finger_geometry(self.hand.fingers[sp.key[1]]).B[:, 0]
        eS = S[:3, :3].T @ e                                     # the finger's direction, servo frame
        hx, r = self._turn_extent(sp)
        twist_up = along_only(eS) and abs(float(S[0, 0])) > 0.7       # a twist whose axis stands out of the face
        if sp.yoke == "crossing":
            jp = (inv(sp.S_child) @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]     # the joints' common point
            far = max(hx, float(jp[0]) + YOKE_REACH_MM + PART_CLEAR)
            r = max(r, YOKE_REACH_MM)
            lo, hi = np.array([ARM_X0 - PART_CLEAR, -r, -r]), np.array([far, r, r])
        elif along_only(eS):
            # past the driven horn (the idler side: the stub); a twist out of the face from there on, any other
            # (along the palm) in the arm's layer beside the case
            x_in = ARM_X0 - PART_CLEAR
            x_out = x_in if twist_up else hx
            lo, hi = (np.array([x_in, -r, -r]), np.array([x_out, r, r])) if eS[0] > 0.0 else \
                (np.array([-x_out, -r, -r]), np.array([-x_in, r, r]))
        else:
            lo, hi = np.array([-hx, -r, -r]), np.array([hx, FACE_TRIM_MM, r])
        hull = None
        if sp.yoke != "crossing":
            # the base link where it is in the palm (below the face): a twist turns all of it round the axis (its
            # simulator body); any other's turned up to 10 degrees either way, so the finger moves at least that far
            # before it stops on the palm (Martin, 10/29)
            f = self.hand.fingers[sp.key[1]]
            fg = dv.finger_geometry(f)
            m = CLIP_MM + PART_CLEAR
            x_top = (S[0, :3], PALM_FACE_MM + m - S[0, 3])         # below the face (servo frame: n.p < d)
            qs = [-10.0, -5.0, 0.0, 5.0, 10.0]
            if twist_up:
                body = fg.tip_hull.aabb() if len(f.joints) == 1 else fg.links[0]
                loc = self._in_palm(sp, self._link_frame_points(sp.key[1], body.corners()), [0.0], [x_top])
                if len(loc):
                    r = max(r, float(np.linalg.norm(loc[:, 1:], axis=1).max()) + m)
                    lo[1:], hi[1:] = -r, r
            elif along_only(eS):
                # along the palm: the printed link where it is beside the case, near rest
                sg = 1.0 if eS[0] > 0.0 else -1.0
                beside = (np.array([-sg, 0.0, 0.0]), -(HOUSING_MM[0] + CASE_CLEAR) - m)
                loc = self._in_palm(sp, self._link0_corners(sp.key[1]), qs, [x_top, beside])
                hull = loc if len(loc) >= 4 else None
            else:
                if sp.yoke == "single" and (sp.key[1], 0) in self.tubes:
                    # a single arm's connector stands out behind the axis as far as the arm's disc, all along it:
                    # the face stops behind where it turns, as far along the axis as it reaches there
                    behind = (np.array([0.0, -1.0, 0.0]), PART_CLEAR - FACE_TRIM_MM)
                    loc = self._in_palm(sp, self.tubes[(sp.key[1], 0)], qs, [x_top, behind])
                    if len(loc):
                        lo = np.minimum(lo, loc.min(axis=0) - PART_CLEAR)
                        hi = np.maximum(hi, loc.max(axis=0) + PART_CLEAR)
                # the printed link ahead of the face's trim line, where it is: a convex pocket of its own, grown
                # by m in the palm (a box round it would take a slanting finger's whole corner of the palm)
                loc = self._in_palm(sp, self._link0_corners(sp.key[1]), qs,
                                    [x_top, (np.array([0.0, 1.0, 0.0]), float(hi[1]) - m)])
                hull = loc if len(loc) >= 4 else None
        boxes = [(lo, hi)]
        if sp.yoke == "single" and not along_only(eS):
            # the driven side (+X), past the case: the one arm turning behind the axis
            boxes.append((np.array([HOUSING_MM[0] + CASE_CLEAR, lo[1], -r]), np.array([hx, r, r])))
        k = int(np.argmax(np.abs(eS)))                           # out of the palm the way the finger goes
        for lo, hi in boxes:
            if along_only(eS) and not twist_up:
                break                                            # along the palm: it stops on it past its pocket
            if eS[k] > 0.0:
                hi[k] = 200.0
            else:
                lo[k] = -200.0
        return boxes, 1 if along_only(eS) else -2, hull

    def _link0_corners(self, i: int) -> np.ndarray:
        """Finger i's printed base link as corners in its frame: its
        connector's, where its gussets may reach, and those of what its far
        end holds (the next servo's housing, or a yoke round its horns where
        that servo is flipped). A fingertip, or a link with no connector: its
        simulator body."""
        f = self.hand.fingers[i]
        if len(f.joints) == 1 or (i, 0) not in self.tubes:
            fg = dv.finger_geometry(f)
            body = fg.tip_hull.aabb() if len(f.joints) == 1 else fg.links[0]
            return self._link_frame_points(i, body.corners())
        nxt = self.servos[("f", i, 1)]
        if nxt.flipped:
            far = mt.box_corners(np.zeros(3), np.eye(3), np.array([ARM_X0 + ARM_DRIVEN, YOKE_REACH_MM, YOKE_REACH_MM]))
            far = far @ nxt.S_child[:3, :3].T + nxt.S_child[:3, 3]
        else:
            far = housing_corners(nxt.S_parent)
        return np.vstack([self.tubes[(i, 0)], far] + self.reach.get((i, 0), []))

    def _link_frame_points(self, i: int, P: np.ndarray) -> np.ndarray:
        """A finger's simulator-body corners (finger frame, as
        `finger_geometry` gives them) in its base link's frame."""
        fg = dv.finger_geometry(self.hand.fingers[i])
        sp = self.servos[("f", i, 0)]
        W = P @ fg.B.T + fg.base_mm                               # the palm's frame, at rest
        M = sp.S_child @ inv(sp.S_parent)                         # palm -> link (at rest)
        return W @ M[:3, :3].T + M[:3, 3]

    @staticmethod
    def _in_palm(sp: ServoSpot, P: np.ndarray, qs, planes) -> np.ndarray:
        """Where a convex body (its corners `P`, the base link's frame), turned
        by each angle in `qs` (degrees), is on the inner side of every plane
        (n, d: n.p < d, the servo's frame): the corners of that part of its
        hull (servo frame)."""
        loc0 = (P - sp.S_child[:3, 3]) @ sp.S_child[:3, :3]
        out = []
        for q in qs:
            c, s_ = math.cos(math.radians(q)), math.sin(math.radians(q))
            L = loc0.copy()
            L[:, 1], L[:, 2] = loc0[:, 1] * c - loc0[:, 2] * s_, loc0[:, 1] * s_ + loc0[:, 2] * c
            for n, d in planes:
                h = L @ n - d
                cut = [L[a] + h[a] / (h[a] - h[b]) * (L[b] - L[a])
                       for a, b in itertools.combinations(range(len(L)), 2) if h[a] * h[b] < 0.0]
                L = np.array(list(L[h <= 0.0]) + cut).reshape(-1, 3)
                if len(L) >= 4:
                    try:
                        L = L[ConvexHull(L).vertices]
                    except QhullError:
                        pass
            out += list(L)
        return np.array(out).reshape(-1, 3)

    @staticmethod
    def _pocket_walls(S: np.ndarray, lo: np.ndarray, hi: np.ndarray, side: int, tray: str) -> List[Piece]:
        """A pocket's walls (`RIM_MM`) on each of its box's faces but the open
        one, outside the box, within the tray (the palm's outline from its back
        face up into the plate): convex pieces."""
        Sm = mm(S)
        out: List[Piece] = []
        t = RIM_MM
        for k in range(3):
            for s in (-1, 1):
                if s * 200.0 == (hi[k] if s > 0 else lo[k]):
                    continue                                     # the open side, out of the palm
                a, b = lo.copy() - t, hi.copy() + t
                if s < 0:
                    a[k], b[k] = lo[k] - t, lo[k]
                else:
                    a[k], b[k] = hi[k], hi[k] + t
                for m in range(3):                               # no further than the box on its open side
                    if lo[m] <= -199.0:
                        a[m] = lo[m]
                    if hi[m] >= 199.0:
                        b[m] = hi[m]
                out.append(lambda gg, a=v3(a), b=v3(b): (f"intersection() {{ {Sm} box_lh({a}, {b}); {tray} }}"))
        return out

    def _rim(self, outline: np.ndarray, x0: float, sec: int) -> List[Piece]:
        """The tray's wall: `RIM_MM` thick inside the face's edge, from the
        palm's back face up into the plate; one convex piece per edge (its
        strip within the rounded plate). None along a cut, where the two
        parts turn against each other. The pockets cut it where fingers leave
        the palm."""
        P = np.asarray(outline, float)
        o = "[" + ",".join(f"[{y:.3f},{z:.3f}]" for y, z in P) + "]"
        xb, xt = -PLATE_HALF_MM, x0 + REACH_MM
        cut = self.hand.palm_joints[0] if self.hand.palm_joints else None
        out: List[Piece] = []
        n = len(P)
        for k in range(n):
            a, b = P[k], P[(k + 1) % n]
            e = b - a
            L = float(np.linalg.norm(e))
            if L < 1.0:
                continue
            if cut is not None and all(abs(cut_side_mm(cut, *q)) < SEAM_MM + RIM_MM for q in (a, b)):
                continue
            u = e / L
            nin = np.array([-u[1], u[0]])                     # counter-clockwise: the inside is on the left
            pts = [np.array([x, *(q + w * nin)]) for q in (a - u, b + u) for w in (-1.0, RIM_MM) for x in (xb, xt)]
            out.append(lambda gg, pts=_pts(pts), o=o: (f"intersection() {{ hull_pts({pts}, {gg}); "
                                                       f"plate({o}, {xb} - ({gg}), {xt} + ({gg})); }}"))
        return out

    @staticmethod
    def _at_plate(sp: ServoSpot) -> bool:
        """A base servo whose socket reaches up to the plate (the rest stand
        on their back ends under it, or a crossing slid them down)."""
        Pc = housing_corners(sp.S_parent, y0=-9.5 if sp.yoke == "crossing" else SOCKET_Y0_MM)
        return float(Pc[:, 0].max()) >= PALM_TOP_MM - PALM_PLATE_MM - 3.0

    def _lying(self, sp: ServoSpot) -> bool:
        """A base servo lying in the palm under the plate: its back axis in
        the plate's plane (its yoke at the palm's edge, ahead of it)."""
        return sp.key[0] == "f" and self._at_plate(sp) and abs(float(sp.S_parent[0, 1])) < 0.7

    def _face_outline(self, part: "Part", outline: np.ndarray, x0: float, keep=()) -> np.ndarray:
        """The face's outline: the palm's outline less what lies ahead of a
        lying base servo's trim line (`FACE_TRIM_MM` behind its axis, where its
        arms turning stand): the hull of the outline's corners not ahead of
        one, those servos' housings behind it, every other base servo's
        housing and the points `keep` (what hangs from the face)."""
        lying = [sp.S_parent for sp in part.holds if self._lying(sp)]
        reach = ARM_X0 + ARM_DRIVEN + PART_CLEAR + 2.0

        def ahead(q) -> bool:
            for S in lying:
                loc = inv(S) @ np.array([x0, q[0], q[1], 1.0])
                if loc[1] < FACE_TRIM_MM and abs(loc[0]) <= reach:
                    return True
            return False

        pts = [np.asarray(q, float) for q in outline if not ahead(q)] + [np.asarray(q, float) for q in keep]
        hx, hy, hz = HOUSING_MM
        for S in lying:
            C = mt.box_corners(np.array([0.0, (FACE_TRIM_MM + hy) / 2.0, 0.0]), np.eye(3),
                               np.array([hx, (hy - FACE_TRIM_MM) / 2.0, hz]))
            pts += list((C @ S[:3, :3].T + S[:3, 3])[:, 1:])
        # every other base servo's housing under the face too (one standing under it, a crossing's slid servo)
        for sp in part.holds:
            if sp.key[0] == "f" and not self._lying(sp):
                y0 = -9.5 if sp.yoke == "crossing" else FACE_TRIM_MM
                pts += list(housing_corners(sp.S_parent, y0=y0)[:, 1:])
        return dv._hull2d(np.array(pts))

    def _turn_extent(self, sp: ServoSpot) -> Tuple[float, float]:
        """How far what a base servo turns reaches from its axis, with a gap:
        along the axis and round it. A yoke: its arms' span and its bridge's
        corners; a crossing: its next servo's housing, turning with it."""
        if sp.yoke != "crossing":
            return ARM_X0 + ARM_DRIVEN + PART_CLEAR + 1.0, YOKE_REACH_MM
        nxt = self.servos[(sp.key[0], sp.key[1], sp.key[2] + 1)]
        loc = (housing_corners(nxt.S_parent) - sp.S_child[:3, 3]) @ sp.S_child[:3, :3]
        along = max(float(np.abs(loc[:, 0]).max()), ARM_X0 + ARM_DRIVEN) + PART_CLEAR + 1.0
        return along, float(np.linalg.norm(loc[:, 1:], axis=1).max()) + PART_CLEAR + 1.0

    @staticmethod
    def _palm_tunnels(S: np.ndarray, x0: float, x1: float, reach: float = 30.0) -> List[str]:
        """A screwdriver's way to each of a palm-held servo's case screws (as
        the library's `case_screw_tunnels`): under the plate for a servo lying
        level (the face stands over the servos' tops), through it only where
        the axis tilts up into it."""
        out = []
        hx = HOUSING_MM[0]
        for yy, zz in ((22.5, -8.0), (22.5, 8.0)):                  # the far holes (library FAR_HOLES)
            for sgn in (1.0, -1.0):
                a, b = (hx, hx + reach) if sgn > 0 else (-hx - reach, -hx)
                out.append(f"{mm(S)} cyl_x_at({yy}, {zz}, 1.9, {a}, {b});")
        return out

    @staticmethod
    def _arm_access(sp: ServoSpot, reach: float = 27.0) -> List[str]:
        """A screwdriver's way to each arm screw of a palm-held base servo's
        yoke at the zero pose, through whatever of the palm is in its way but
        the housings (the yoke goes on after the servo is in)."""
        out = []
        S = sp.S_parent
        sides = [(1.0, ARM_X0 + ARM_DRIVEN)] + ([(-1.0, ARM_X0 + ARM_IDLER)] if sp.yoke == "u" else [])
        for sgn, xa in sides:
            a, b = (xa, xa + reach) if sgn > 0 else (-xa - reach, -xa)
            for ang in HORN_HOLES_DRIVEN:
                yy, zz = 6.0 * math.sin(math.radians(ang)), 6.0 * math.cos(math.radians(ang))
                out.append(f"{mm(S)} cyl_x_at({yy:.3f}, {zz:.3f}, 2.2, {a:.3f}, {b:.3f});")
        return out

    @staticmethod
    def _dist_out(p: np.ndarray, poly: np.ndarray) -> float:
        """Signed distance from p to a convex counter-clockwise polygon (> 0 outside)."""
        best = -math.inf
        n = len(poly)
        for k in range(n):
            a, b = poly[k], poly[(k + 1) % n]
            e = b - a
            nrm = np.array([e[1], -e[0]]) / (np.linalg.norm(e) + 1e-12)
            best = max(best, float((p - a) @ nrm))
        return best

    # --------------------------------------------------------------- sweeps

    def _sweeps(self) -> None:
        for sp in self.servos.values():
            child, parent = self.parts[sp.child], self.parts[sp.parent]
            lo, hi = sp.rot_deg
            # the child gives room to the servo case (and plug) it turns against
            Sc = sp.S_child
            for pc in _case_pieces(np.eye(4), sp.cable, sp.idler_horn):
                child.negative.append(f"{mm(Sc)} sweep_x({-hi:.3f}, {-lo:.3f}) {pc('CASE_CLEAR')}")
            # the parent gives room to everything the child carries: its pieces, its horns,
            # the servos it holds (case and plug), swept about this joint
            Sp = sp.S_parent
            to_child = inv(Sc)                      # child frame -> servo frame
            occupants = list(child.pieces) + _horn_pieces(sp)
            for h in child.holds:
                occupants += _case_pieces(h.S_parent, h.cable, h.idler_horn)
            if sp.parent[0] == "palm" and sp.key[0] == "f":
                # a finger's base turns in its pocket in the palm (`_pocket`); past it the finger stops on
                # the palm, as in the simulator
                occupants = []
            for pc in occupants:
                stmt = f"{mm(Sp)} sweep_x({lo:.3f}, {hi:.3f}) {mm(to_child)} {pc('PART_CLEAR')}"
                if sp.parent[0] == "palm":
                    parent.swept_by.setdefault(sp.key[1] if sp.key[0] == "f" else -1, []).append(stmt)
                else:
                    parent.negative.append(stmt)
            if sp.yoke == "crossing":
                self._crossing_sweeps(sp)

    def _crossing_sweeps(self, sp: ServoSpot) -> None:
        """Across a 0 mm link (joints j and j + 1 at one point): the part
        before joint j and link j + 1 turn against each other about both
        joints."""
        _, i, j = sp.key
        nxt = self.servos[("f", i, j + 1)]
        grand, far = self.parts[sp.parent], self.parts[nxt.child]
        lo1, hi1 = sp.rot_deg
        lo2, hi2 = nxt.rot_deg
        # the parent gives room to link j + 1, its horns and what it holds (the palm: its pocket does)
        Tn = inv(sp.S_child) @ nxt.S_parent          # servo j + 1's frame in servo j's
        occupants = list(far.pieces) + _horn_pieces(nxt) if sp.parent[0] != "palm" else []
        for h in far.holds if sp.parent[0] != "palm" else []:
            occupants += _case_pieces(h.S_parent, h.cable, h.idler_horn)
        for pc in occupants:
            stmt = (f"{mm(sp.S_parent)} sweep_x2({_mat(Tn)}, {lo1:.3f}, {hi1:.3f}, {lo2:.3f}, "
                    f"{hi2:.3f}) {mm(inv(nxt.S_child))} {pc('PART_CLEAR')}")
            if sp.parent[0] == "palm":
                grand.swept_by.setdefault(i, []).append(stmt)
            else:
                grand.negative.append(stmt)
        # link j + 1 gives room to servo j's case, turning back about both joints: the
        # convex hull of what it takes from each piece, since the case grazes a face
        # in lobes there (Martin, 10/26: a bridge's edge looked bitten)
        Tb = inv(nxt.S_parent) @ sp.S_child          # servo j's frame in servo j + 1's
        for pc in _case_pieces(np.eye(4), sp.cable, sp.idler_horn):
            sw = (f"{mm(nxt.S_child)} sweep_x2({_mat(Tb)}, {-hi2:.3f}, {-lo2:.3f}, {-hi1:.3f}, "
                  f"{-lo1:.3f}) {pc('CASE_CLEAR')}")
            far.negative += [f"hull() intersection() {{ {q} {sw} }}" for q in far.positive]

    # --------------------------------------------------------------- output

    def part_list(self) -> List[Part]:
        return list(self.parts.values()) + list(self.extras)
