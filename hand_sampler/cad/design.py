"""From a grammar hand to its printable parts: where every servo, yoke,
housing and part goes, as OpenSCAD statements over the library
`scad/feetech_hand.scad` (Martin, 2026-10-23: the grammar drives, OpenSCAD
builds).

No new kinematics: every frame comes from the grammar (`derive.fk` at the
zero pose, `derive.finger_geometry`, `derive.hand_geometry`). Millimetres
throughout; a part's frame is its rigid body's frame (the palm frame for the
palm, link j's frame for finger link j).

Each joint has a parent (the part holding the servo's case: the palm or
link j - 1) and a child (link j, holding the horns in a yoke). Every part is
its positive pieces, less the keep-outs of the servos it holds, less
- as a child: what its joint's servo case (and plug) sweeps over the joint's
  range relative to it;
- as a parent: what its child sweeps relative to it (every convex piece of
  the child, its horns and the servo it carries, grown by `PART_CLEAR`).
The child keeps its shape near a joint and the parent gives way.

A base servo whose cradle can take two pads on the plate stands in a cradle
of its own (`_cradle_pads`, `_cradle_parts`): a part in the palm's frame,
screwed to the servo on the bench, standing in a window of the plate and
screwed to it from behind. Its servo's `holder` names it; `parent` stays
the palm, the rigid body.

A servo's frame S (servo.py): X along its axis toward the driven horn. It
is the grammar's motor frame (a, v, w), flipped to (-a, v, -w) where the
driven horn must face the other way: toward the fingertip for a wrap yoke (a twist).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from hand_sampler.grammar import derive as dv
from hand_sampler.grammar import motor as mt
from hand_sampler.grammar.hand import PLATE_HALF_MM, Hand, finger_section, turned_servo

ARM_R_MM = 9.5                # an arm's disc round the axis (library ARM_R)

BRIDGE_HALF_MM = float(mt.YOKE_HALF_MM[2])   # a U's bridge, either side of the axis across the finger (12)
ARM_X0 = 14.6                                # an arm's inner face from the servo's mid-plane (horn face 14.5 + 0.1)
ARM_DRIVEN, ARM_IDLER = 3.0, 2.0
CASE_CLEAR, PART_CLEAR, WALL, FIT = 0.5, 1.0, 2.0, 0.2
HORN_HOLES_DRIVEN = (45.0, 135.0, 225.0, 315.0)   # degrees from +Z about X (the 12 mm circle)
HORN_HOLES_IDLER = (90.0, 270.0)
PLUG = ((-9.2, -2.9), (4.8, 13.2), 9.0)          # (X range, Y range, length beyond the case face)
CASE_Z = 10.0
BASE_BEAM_MM = 38.0
"""A housing slid out behind the palm (a crossing at the base) hangs from a
beam this long toward the palm centre, under the sweep of the crossing's
next servo (its back corners turn on about 30 mm), then a post up into the
plate."""
PALM_BOLTS_MAX = 6
PALM_BOLT_CLEAR_MM = 5.0      # and this far from any base sweep
PALM_BOLT_SPACING_MM = 18.0
BORE_R_MM = 7.0               # the cables' bore (library BORE_R)
COLUMN_SCREW_R_MM = 16.5
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
PALM_PLATE_MM = 3.0           # the palm's plate, from its back face up
CRADLE_CHANNEL_MM = 14.0      # a cradle's servo needs only its own front clear (it is open on two more sides)
PAD_R_MM, PAD_H_MM = 4.5, 7.0
"""A base servo's cradle is its own part: the servo is screwed into it on the
bench, then it stands in a window of the plate and two pads (this radius and
height) on the plate's front face take an M3 x 8 each from behind."""
PAD_SWEEP_MARGIN_MM = 3.5     # pads keep this far from the links' simulator bodies turning (clip 2.4 + 1)
PAD_COLUMN_R_MM = 24.0        # off the mount's column (20) behind the main plate, by a screw head
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


# --------------------------------------------------------------------------
# servos
# --------------------------------------------------------------------------


@dataclass
class ServoSpot:
    """One servo: its joint, its frame and how it is held and driven."""

    key: Tuple                    # ("f", i, j) or ("hinge",)
    parent: Tuple                 # body holding its case: ("palm", k) or ("link", i, j - 1)
    child: Tuple                  # body holding its horns: ("link", i, j) or ("palm", 0)
    S_child: np.ndarray           # 4x4: the servo frame in the child's frame (q = 0)
    S_parent: np.ndarray          # 4x4: the servo frame in the parent's frame (q = 0)
    rot_deg: Tuple[float, float]  # the child's turn about the servo's X over the joint's range
    yoke: str                     # "u", "wrap" (a U round a twist, its bridge past the case's front) or "crossing"
    cable: int                    # the Z face whose socket takes the cable (+1 / -1)
    turned: bool = False
    flip: bool = False
    sign: float = 1.0             # + joint angle turns the driven horn this way about S's X (servo's own sign)
    holder: Optional[Tuple] = None    # the part holding its case when that is not the parent (a palm cradle)
    full_plates: bool = False         # plates over its whole horn faces, near holes screwed too (a knuckle)

    @property
    def idler_horn(self) -> bool:
        return self.yoke in ("u", "wrap")


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
        turned = turned_servo(f, j)
        crossing = j < len(f.joints) - 1 and jt.length == 0
        wrap = mt.is_wrap(a, crossing)
        flip = wrap and float(a[0]) < 0.0          # a wrap's driven horn faces the fingertip
        origin = fg.slides_mm[j] * a
        S_child = _servo_frame(M[:, 0], M[:, 1], origin, flip)
        if j == 0:
            Tpj = pose.pre[i][0].copy()
            Tpj[:3, 3] *= 1e3
            if sec >= 0:                           # the part beyond the cut: its own frame is the palm frame at q = 0
                pass
            parent = ("palm", sec)
        else:
            Tpj = T(None, (float(f.joints[j - 1].length), 0.0, 0.0))
            parent = ("link", i, j - 1)
        S_parent = Tpj @ S_child
        sgn = dv.joint_sign(hand, i, j) * (-1.0 if flip else 1.0)
        lo, hi = (math.degrees(x) for x in dv.joint_limits(hand, i, j))
        rot = tuple(sorted((sgn * lo, sgn * hi)))
        dorsal = -np.array([0.0, 1.0, 0.0])        # the link frame's -y
        cable = _cable_side(S_child, dorsal, -np.array([1.0, 0.0, 0.0]))
        out.append(ServoSpot(("f", i, j), parent, ("link", i, j), S_child, S_parent, rot,
                             "crossing" if crossing else ("wrap" if wrap else "u"),
                             cable, turned, flip, sgn))
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
    carries: Optional[ServoSpot] = None        # the servo whose horns it holds
    clip: Optional[str] = None                 # a convex body the part stays inside (its simulator body, grown)
    swept_by: Dict[int, List[str]] = field(default_factory=dict)   # palm parts: each finger's sweeps
    kept: List[Tuple[int, str]] = field(default_factory=list)      # palm parts: (finger, pocket walls kept from other fingers)
    tunnels: List[str] = field(default_factory=list)                # screwdrivers' ways in (never through a kept cradle)
    windows: List[str] = field(default_factory=list)                # palm parts: room for the cradles standing in the plate
    pads: List[Tuple[float, float]] = field(default_factory=list)   # cradles: (y, z) of the screws into the plate

    def scad(self) -> str:
        pos = "\n        ".join(self.positive) or "cube(0.001);"
        neg = "\n    ".join(self.negative + [n for ns in self.swept_by.values() for n in ns] + self.tunnels
                               + self.windows)
        body = f"difference() {{\n        union() {{\n        {pos}\n        }}\n    {neg}\n    }}"
        if self.kept:
            # a servo's cradle keeps its walls against other fingers' sweeps (fingers colliding in
            # motion is left to the physics and the policy, as in the simulator), and gives way to
            # everything else: its own finger's sweeps, the cut's hinge, every servo's keep-outs
            keep = []
            for i, walls in self.kept:
                rest = (self.negative + self.windows + self.swept_by.get(i, [])
                        + (self.swept_by.get(-1, []) if i != -1 else []))
                keep.append(f"difference() {{ {walls}\n        " + "\n        ".join(rest) + " }")
            body = f"union() {{\n    {body}\n    " + "\n    ".join(keep) + "\n    }"
        if self.clip:
            body = f"intersection() {{\n    {body}\n    {self.clip}\n    }}"
        return f"module part_{self.name}() {{\n    {body}\n}}\n"


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
    of the part joins to (a U's bridge; a wrap's driven arm). The bridge
    lies `x_b` along `front` from the joint, past the case's swing."""
    S = mm(sp.S_child)
    if sp.yoke in ("u", "wrap"):
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

        return [arm(1), arm(-1), bridge], (arm(1) if sp.yoke == "wrap" else bridge)
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
    return ("minkowski() { hull() { " + " ".join(f"translate({v3(p)}) cube(0.01, center = true);" for p in pts)
            + f" }} sphere(r = {CLIP_MM}, $fn = 16); }}")


def build_design(hand: Hand) -> "Design":
    return Design(hand)


class Design:
    """Every part of `hand`, its servos and screws."""

    def __init__(self, hand: Hand, more_tubes: Optional[set] = None):
        self.hand = hand
        self.more_tubes = set(more_tubes or ())     # links whose cradle gets a tube to each face it can take
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
        self.extras: List[Part] = []          # parts fixed to the main palm: the bay and the mount
        self._cradles: List[dict] = []        # base servos on their own cradles (`_palm`, then `_cradle_parts`)
        h = hinge_servo(hand)
        if h is not None:
            self.servos[h.key] = h
            self.parts[("palm", 0)] = Part("palm_beyond", ("palm", 0), np.eye(4))
        for sp in self.servos.values():
            self.parts[sp.parent].holds.append(sp)
            self.parts[sp.child].carries = sp
        for i, f in enumerate(hand.fingers):
            for j in range(len(f.joints)):
                self._link(i, j)
        self._palm(-1)
        if h is not None:
            self._palm(0)
        self._sweeps()
        self._palm_fixings()
        self._bay()
        self._cradle_parts()

    # ---------------------------------------------------------------- links

    def _link(self, i: int, j: int) -> None:
        f = self.hand.fingers[i]
        fg = dv.finger_geometry(f)
        part = self.parts[("link", i, j)]
        sp = self.servos[("f", i, j)]
        a = fg.axes[j]
        last = j == len(f.joints) - 1
        nxt = None if last else self.servos[("f", i, j + 1)]
        if sp.yoke == "crossing":
            # the plate on servo j's near horn, as far as the next servo's case
            S = mm(sp.S_child)
            Sn = mm(nxt.S_parent)
            gap = self._crossing_gap(sp, nxt)
            plate: Piece = lambda g, S=S, Sn=Sn, gap=gap: (
                f"hull() {{ {S} cyl_x(ARM_R + ({g}), ARM_X0 - ({g}), {gap:.3f} + ({g})); {Sn} housing_box({g}); }}")
            pieces = [plate]
            joint_piece = plate
        elif sp.yoke == "wrap":
            # both horns held, the bridge past the case's front (its swing over the joint's range)
            pieces, joint_piece = _yoke_pieces(sp, a, -sp.S_child[:3, 1], mt.FRONT_SWEEP_MM + CASE_CLEAR)
        else:
            x_b = mt.swept_front_mm(a, sp.turned) + CASE_CLEAR
            pieces, joint_piece = _yoke_pieces(sp, a, np.array([1.0, 0.0, 0.0]), x_b)
        if last:
            tip = fg.tip
            p1 = tip.p1
            r = tip.radius
            # the rounded end only ahead of the yoke (behind it, round the servo's front, the
            # fingertip is cut away anyway; the part before it should not give way to it)
            if sp.yoke == "u":
                x_cut = mt.swept_front_mm(a, sp.turned) + CASE_CLEAR
            else:
                th = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
                disc = np.array([[x, 9.5 * math.cos(t), 9.5 * math.sin(t)] for x in (ARM_X0, ARM_X0 + ARM_DRIVEN) for t in th])
                x_cut = float((disc @ sp.S_child[:3, :3].T + sp.S_child[:3, 3])[:, 0].min())
            body: Piece = lambda g, jp=joint_piece, p1=p1, r=r, xc=x_cut: (
                f"hull() {{ {jp(g)} intersection() {{ translate({v3(p1)}) sphere(r = {r:.3f} + ({g}), $fn = 48); "
                f"translate([{xc:.3f} - ({g}), -100, -100]) cube([300, 200, 200]); }} }}")
            pieces = pieces + [body]
        elif sp.yoke != "crossing":
            # a bracket frame: the next servo in a cradle (side plates and back plate), joined to the
            # yoke by a hollow connector from the bridge's (a wrap's: its driven arm's) whole face to the whole
            # cradle face toward it (`_connector`)
            Sn = mm(nxt.S_parent)
            A, sa = self._tube_start(sp, a)
            B, sb, face = self._tube_end(nxt, A[:3, 3])
            # a servo lying across the finger (its 23 x 34 face toward the palm, as a turned bend near the
            # tip or at a knuckle) is held by plates over its whole horn faces, its near holes screwed too;
            # servos stacked lengthwise along the finger keep the short socket (Martin, 10/24)
            nxt.full_plates = nxt.yoke != "crossing" and abs(float(nxt.S_parent[0, 1])) < 0.7
            A2, sa2, B, sb = self._connector(A, sa, B, sb, square=nxt.full_plates)
            if face != 0 and nxt.cable == face:
                nxt.cable = -face                        # the tube lands on that side: the cable takes the other
            tube: Piece = lambda g, A=A2, B=B, sa=sa2, sb=sb: f"tube({_mat(A)}, {list(sa)}, {_mat(B)}, {list(sb)}, {g});"
            extra: List[Piece] = []
            if ("link", i, j) in self.more_tubes:
                # this link came out in pieces (something turns through its tube): a tube to every face
                for Bx, sbx, fx in self._tube_ends(nxt):
                    if fx != face and not (fx != 0 and fx == nxt.cable):
                        Ax, sax, Bx, sbx = self._connector(A, sa, Bx, sbx, square=nxt.full_plates)
                        extra.append(lambda g, A=Ax, B=Bx, sa=sax, sb=sbx:
                                     f"tube({_mat(A)}, {list(sa)}, {_mat(B)}, {list(sb)}, {g});")
            if nxt.yoke == "crossing":
                # a crossing's first servo: its near horn carries the crossing plate, so its cradle is a
                # full housing (the plate's swing opens the near side)
                pieces = pieces + [lambda g, Sn=Sn: f"{Sn} housing_box({g});", tube] + extra
            else:
                plates = ([lambda g, Sn=Sn: f"{Sn} cradle_plate(1, {g});", lambda g, Sn=Sn: f"{Sn} cradle_plate(-1, {g});"]
                          if nxt.full_plates else [])
                pieces = pieces + [lambda g, Sn=Sn: f"{Sn} cradle_socket({g});"] + plates + [tube] + extra
        part.pieces = pieces
        part.positive = [p("0") for p in pieces]
        part.clip = _clip_body(f, j)
        # what this part leaves empty: its own servo's horns, its yoke's screws, the next
        # servo's housing keep-outs
        S = mm(sp.S_child)
        part.negative.append(f"{S} cyl_x(HORN_R + CASE_CLEAR, DRIVEN_X[0] - FIT, DRIVEN_X[1]);")
        if sp.idler_horn:
            part.negative.append(f"{S} cyl_x(HORN_R + CASE_CLEAR, IDLER_X[0], -(DRIVEN_X[0] - FIT));")
        if sp.idler_horn:
            part.negative += [f"{S} arm_keepout(1, {list(HORN_HOLES_DRIVEN)});",
                              f"{S} arm_keepout(-1, {list(HORN_HOLES_IDLER)});"]
        else:
            part.negative.append(f"{S} arm_keepout(1, {list(HORN_HOLES_DRIVEN)});")
        if nxt is not None:
            full = str(nxt.full_plates).lower()
            part.tunnels.append(f"{mm(nxt.S_parent)} case_screw_tunnels(40, {full});")
            if nxt.yoke == "wrap":           # the next link's idler arm faces this part: its screws driven through it
                part.tunnels.append(f"{mm(nxt.S_parent)} arm_screw_tunnels(-1, {list(HORN_HOLES_IDLER)});")
            ch = 80 if nxt.yoke == "crossing" else CRADLE_CHANNEL_MM
            part.negative.append(f"{mm(nxt.S_parent)} housing_keepout({str(not nxt.idler_horn).lower()}, {nxt.cable}, "
                                 f"{ch}, {full});")

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
        x_end = max(float(loc[:, 0].max()), 1.0)
        return A @ T(None, (0.0, yc, zc)), size, A @ T(None, (x_end, yc, zc)), size

    def _tube_start(self, sp: ServoSpot, a: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Where a link's connector leaves its yoke (link frame): a U's
        bridge's whole outer face; a wrap's driven arm's outer face, from its
        disc to the bridge."""
        if sp.yoke == "u":
            x_b = mt.swept_front_mm(a, sp.turned) + CASE_CLEAR
            B = _bridge_frame(sp.S_child[:3, 3], np.array([1.0, 0.0, 0.0]), a, x_b)
            s_pos = -1.0 if sp.flip else 1.0
            a_lo = -(ARM_X0 + (ARM_DRIVEN if s_pos < 0 else ARM_IDLER))
            a_hi = ARM_X0 + (ARM_DRIVEN if s_pos > 0 else ARM_IDLER)
            return B @ T(None, (WALL, (a_lo + a_hi) / 2.0, 0.0)), (a_hi - a_lo, 2.0 * BRIDGE_HALF_MM)
        bot = mt.FRONT_SWEEP_MM + CASE_CLEAR + WALL           # the bridge's outer face, past the case's front
        return (sp.S_child @ T(None, (ARM_X0 + ARM_DRIVEN, (ARM_R_MM - bot) / 2.0, 0.0)),
                (ARM_R_MM + bot, 2.0 * ARM_R_MM))

    @staticmethod
    def _faces(nxt: ServoSpot) -> List[Tuple[int, np.ndarray, np.ndarray, Tuple[float, float]]]:
        """The faces of a cradle a connector can arrive at, in the link's
        frame: (side, outward normal, the face's frame (its rectangle in the
        frame's y-z plane), size). Side 0 is the back plate, +/-1 the open Z
        sides (across the socket), +/-2 the full-face plates' outer faces."""
        S = nxt.S_parent
        R, o = S[:3, :3], S[:3, 3]
        hx, hy, hz = HOUSING_MM
        out = [(0, R[:, 1], T(np.column_stack([R[:, 1], R[:, 0], R[:, 2]]), R @ np.array([0.0, hy, 0.0]) + o), CRADLE_BACK_MM),
               (1, R[:, 2], T(np.column_stack([R[:, 2], R[:, 0], R[:, 1]]), R @ np.array([0.0, CRADLE_SIDE_Y, hz]) + o),
                CRADLE_SIDE_MM),
               (-1, -R[:, 2], T(np.column_stack([-R[:, 2], -R[:, 0], R[:, 1]]), R @ np.array([0.0, CRADLE_SIDE_Y, -hz]) + o),
                CRADLE_SIDE_MM)]
        if nxt.full_plates:
            y0 = -9.5 - 2.0                           # the plates reach 2 mm past the case's front
            c = (y0 + hy) / 2.0
            out += [(2, R[:, 0], T(np.column_stack([R[:, 0], R[:, 1], R[:, 2]]), R @ np.array([hx, c, 0.0]) + o),
                     (hy - y0, 2.0 * hz)),
                    (-2, -R[:, 0], T(np.column_stack([-R[:, 0], R[:, 1], -R[:, 2]]), R @ np.array([-hx, c, 0.0]) + o),
                     (hy - y0, 2.0 * hz))]
        return out

    def _tube_ends(self, nxt: ServoSpot) -> List[Tuple[np.ndarray, Tuple[float, float], int]]:
        """Every cradle face a connector can arrive at (`_faces`)."""
        return [(F, size, side) for side, _, F, size in self._faces(nxt)]

    def _tube_end(self, nxt: ServoSpot, start: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], int]:
        """The cradle face a connector arrives at (link frame): whichever of
        `_faces` faces the connector's start most squarely; its frame, size
        and side."""
        best = None
        for side, n, F, size in self._faces(nxt):
            d = start - F[:3, 3]
            score = float(n @ d) / (np.linalg.norm(d) + 1e-9)
            if best is None or score > best[0] + 1e-6:
                best = (score, F, size, side)
        _, F, size, side = best
        return F, size, side

    def _crossing_gap(self, sp: ServoSpot, nxt: ServoSpot) -> float:
        """How far from servo j's mid-plane (along its X) the next servo's
        case starts, less the fit: where the crossing plate ends."""
        C = mt.box_corners(np.array([0.0, 7.5, 0.0]), np.eye(3), np.array([11.5, 17.0, 10.0]))
        P = C @ nxt.S_parent[:3, :3].T + nxt.S_parent[:3, 3]       # in the child (link j) frame
        X = (P - sp.S_child[:3, 3]) @ sp.S_child[:3, 0]
        return float(X.min()) - FIT

    # ---------------------------------------------------------------- palm

    def _palm(self, sec: int) -> None:
        """A palm part as a frame: a plate over the palm's outline (its back
        `PALM_PLATE_MM`), each base servo screwed into a cradle on it; a servo
        slid out behind the palm (a crossing at the base) hangs from beams."""
        g = dv.hand_geometry(self.hand)
        part = self.parts[("palm", sec)]
        outline = g.outlines[sec]
        o = "[" + ",".join(f"[{y:.3f},{z:.3f}]" for y, z in outline) + "]"
        x0, x1 = -PLATE_HALF_MM, -PLATE_HALF_MM + PALM_PLATE_MM
        plate: Piece = lambda gg, o=o: f"plate({o}, {x0} - ({gg}), {x1} + ({gg}));"
        pieces: List[Piece] = [plate]
        loose = set()                  # pieces of the cradles that are parts of their own
        for sp in part.holds:
            Sp = mm(sp.S_parent)
            cradle = ([lambda gg, Sp=Sp: f"{Sp} housing_box({gg});"] if sp.yoke == "crossing" else
                      [lambda gg, Sp=Sp: f"{Sp} cradle_socket({gg});"])
            P = housing_corners(sp.S_parent)
            standing = float(P[:, 0].max()) >= x0 + 0.5
            ch = 80 if (sp.yoke == "crossing" or not standing) else CRADLE_CHANNEL_MM
            part.negative.append(f"{Sp} housing_keepout({str(not sp.idler_horn).lower()}, {sp.cable}, {ch});")
            if sp.yoke == "wrap":
                part.negative.append(f"{Sp} arm_screw_tunnels(-1, {list(HORN_HOLES_IDLER)});")
            # the socket (a crossing's servo: its full housing) on a pedestal: its hull with its outline
            # straight down to the plate, so a servo standing off the plate is joined to it (the case's
            # keep-out leaves the pedestal's walls; a servo lying on the plate gains nothing)
            body = "housing_box" if sp.yoke == "crossing" else "cradle_socket"
            Pc = housing_corners(sp.S_parent, y0=-9.5 if sp.yoke == "crossing" else SOCKET_Y0_MM)

            def pedestal(x_to: float, Sp=Sp, body=body, Pc=Pc) -> Piece:
                down = Pc.copy()
                down[:, 0] = x_to
                pts = " ".join(f"translate({v3(q)}) cube(0.01, center = true);" for q in down)
                return lambda gg: f"hull() {{ {Sp} {body}({gg}); {pts} }}"

            pads = self._cradle_pads(sp, sec, outline, x0, x1, Pc) if standing else None
            if pads is not None:
                # its own part: the servo is screwed in on the bench (nothing in a screwdriver's way), then
                # the cradle stands in a window of the plate, its pads on the plate's front face, and two
                # M3 x 8 go in from behind
                cradle = [pedestal(x1)]
                pad_pieces, heads = [], []
                for pd in pads:
                    y, z = pd["at"]
                    ny, nz = pd["near"]
                    pad_pieces.append(lambda gg, y=y, z=z, ny=ny, nz=nz, w=cradle[0]: (
                        f"hull() {{ translate([0, {y:.3f}, {z:.3f}]) cyl_x({PAD_R_MM} + ({gg}), {x1} - ({gg}), "
                        f"{x1 + PAD_H_MM} + ({gg})); intersection() {{ {w(gg)} translate([{x1} - ({gg}), "
                        f"{ny - 6.0:.3f}, {nz - 6.0:.3f}]) cube([{PAD_H_MM} + 2 * ({gg}), 12, 12]); }} }}"))
                    heads.append(lambda gg, y=y, z=z: f"translate([0, {y:.3f}, {z:.3f}]) cyl_x(3.2 + ({gg}), "
                                                      f"{x0 - 2.6} - ({gg}), {x0} + ({gg}));")
                    part.negative.append(f"translate([0, {y:.3f}, {z:.3f}]) cyl_x(M3_CLEAR / 2, {x0 - 1}, {x1 + 0.01});")
                # the plate inside the cradle's box is the cradle's floor (so no piece of plate is left
                # between its walls), the plate keeps the rest
                floor: Piece = lambda gg, Sp=Sp, o=o: (f"intersection() {{ {Sp} housing_box({gg}); "
                                                       f"plate({o}, {x0} - ({gg}), {x1} + ({gg})); }}")
                own = cradle + [floor] + pad_pieces + heads
                pieces += own
                loose.update(id(q) for q in own)
                part.windows += [f"{Sp} housing_box(FIT);", cradle[0]("FIT")]
                # and gives way to its pads above the plate's front face
                part.windows += [f"intersection() {{ {pc('FIT')} translate([{x1}, -500, -500]) cube([500, 1000, 1000]); }}"
                                 for pc in pad_pieces]
                self._cradles.append({"sp": sp, "sec": sec, "walls": cradle + [floor] + pad_pieces,
                                      "pads": [pd["at"] for pd in pads]})
                continue
            if standing:
                pieces.append(pedestal(x0))
                # kept against other fingers only round the servo (as far as its housing box): a tilted
                # servo's pedestal reaching out to the plate gives way to every finger
                cradle = [lambda gg, ped=pieces[-1], Sp=Sp: f"intersection() {{ {ped(gg)} {Sp} housing_box({gg}); }}"]
            else:
                cradle = [lambda gg, Sp=Sp: f"{Sp} housing_box({gg});"]
                # slid out behind the plate: a full housing, joined to the plate by several ways, each cut
                # back where something turns through it: straight up over its footprint, and beams under
                # what the finger's first links sweep toward the palm centre and 60 degrees either side
                pieces.append(lambda gg, Sp=Sp: f"{Sp} housing_box({gg});")
                oo = sp.S_parent[:3, 3]
                up = P.copy()
                up[:, 0] = x1
                pts_u = " ".join(f"translate({v3(q)}) cube(0.01, center = true);" for q in np.concatenate([P, up]))
                pieces.append(lambda gg, pts_u=pts_u: f"hull() {{ {pts_u} }}")
                d0 = -np.array([0.0, oo[1], oo[2]])
                d0 = d0 / np.linalg.norm(d0) if np.linalg.norm(d0) > 1e-6 else np.array([0.0, 1.0, 0.0])
                for turn in (0.0, 60.0, -60.0):
                    c, s_ = math.cos(math.radians(turn)), math.sin(math.radians(turn))
                    d = np.array([0.0, c * d0[1] - s_ * d0[2], s_ * d0[1] + c * d0[2]])
                    end_ = np.array([0.0, oo[1], oo[2]]) + BASE_BEAM_MM * d
                    side = np.cross([1.0, 0.0, 0.0], d)
                    beam = [end_ + s1 * 10.0 * side + np.array([x, 0.0, 0.0]) for s1 in (-1, 1)
                            for x in (x0 - 10.0, x0 - 3.0)]
                    post = [end_ + s1 * 10.0 * side + s2 * 6.0 * d + np.array([x, 0.0, 0.0]) for s1 in (-1, 1)
                            for s2 in (-1, 1) for x in (x0 - 10.0, x1)]
                    pts_b = " ".join(f"translate({v3(q)}) cube(0.01, center = true);" for q in beam)
                    pts_p = " ".join(f"translate({v3(q)}) cube(0.01, center = true);" for q in post)
                    pieces.append(lambda gg, Sp=Sp, pts_b=pts_b: f"hull() {{ {Sp} housing_box({gg}); {pts_b} }}")
                    pieces.append(lambda gg, pts_p=pts_p: f"hull() {{ {pts_p} }}")
            part.tunnels.append(f"{Sp} case_screw_tunnels(30);")
            if sp.key[0] in ("f", "hinge"):
                # its cradle keeps its walls against other fingers' sweeps (the hinge's: against every finger)
                walls = " ".join(c("0") for c in cradle)
                part.kept.append((sp.key[1] if sp.key[0] == "f" else -1, f"difference() {{ union() {{ {walls} }} "
                                             f"{Sp} housing_keepout({str(not sp.idler_horn).lower()}, {sp.cable}, {ch}); }}"))
        if sec == 0 and part.carries is not None:      # the hinge's yoke on the part beyond the cut
            sp = part.carries
            front = -sp.S_child[:3, 1]                 # ahead of the hinge servo: across the cut
            yp, _ = _yoke_pieces(sp, sp.S_child[:3, 0], front, mt.FRONT_SWEEP_MM + CASE_CLEAR)
            pieces += yp
            S = mm(sp.S_child)
            part.negative += [f"{S} arm_keepout(1, {list(HORN_HOLES_DRIVEN)});",
                              f"{S} arm_keepout(-1, {list(HORN_HOLES_IDLER)});"]
        part.pieces = pieces
        part.positive = [p("0") for p in pieces if id(p) not in loose]

    @staticmethod
    def _slab_yz(S: np.ndarray, lo, hi, xa: float, xb: float) -> Optional[np.ndarray]:
        """The box [lo, hi] (servo frame), placed by S, within xa <= x <= xb
        of the palm frame, seen along x: a counter-clockwise (y, z) polygon,
        or None."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        C = mt.box_corners((lo + hi) / 2.0, np.eye(3), (hi - lo) / 2.0)
        P = C @ S[:3, :3].T + S[:3, 3]
        pts = [q for q in P if xa - 1e-9 <= q[0] <= xb + 1e-9]
        for a in range(8):
            for b in range(a + 1, 8):
                if int(np.sum(np.abs(C[a] - C[b]) > 1e-9)) != 1:
                    continue
                for xp in (xa, xb):
                    da, db = P[a, 0] - xp, P[b, 0] - xp
                    if da * db < 0:
                        pts.append(P[a] + da / (da - db) * (P[b] - P[a]))
        if len(pts) < 3:
            return None
        poly = dv._hull2d(np.array(pts)[:, 1:])
        return poly if len(poly) >= 3 else None

    def _sweep_hulls(self) -> List[np.ndarray]:
        """Every finger's links at its base turning over their joints' ranges
        (as `derive.base_sweep_mm`, without the motors): one hull's plane
        equations per pose and link (palm frame, mm, unit normals)."""
        if getattr(self, "_hulls", None) is None:
            from scipy.spatial import ConvexHull

            out, owner = [], []
            for fi, f in enumerate(self.hand.fingers):
                hand = Hand(fingers=(f,))
                fg = dv.finger_geometry(f)
                shapes = [fg.tip_hull if j == len(f.joints) - 1 else fg.links[j] for j in fg.base_joints]
                grids = []
                for j in fg.base_joints:
                    lo, hi = dv.joint_limits(hand, 0, j)
                    if f.joints[j].type == "sliding":
                        grids.append([0.0])
                        continue
                    n = max(2, int(math.ceil((hi - lo) / (5.0 * dv.DEG))) + 1)
                    grids.append(list(np.linspace(lo, hi, n)) if j == fg.base_joints[0] else [lo, 0.0, hi])
                axes = [dv.joint_axis(hand, 0, j) for j in fg.base_joints]
                for qs in itertools.product(*grids):
                    R = fg.B.copy()
                    for k, j in enumerate(fg.base_joints):
                        R = R @ dv._rot(axes[k], qs[k])
                        out.append(ConvexHull(shapes[k].corners() @ R.T + fg.base_mm).equations)
                        owner.append(fi)
            self._hulls, self._hull_owner = out, owner
        return self._hulls

    def _cradle_pads(self, sp: ServoSpot, sec: int, outline: np.ndarray, x0: float, x1: float,
                     Pc: np.ndarray) -> Optional[List[dict]]:
        """Two pads for a base servo's cradle on the plate's front face: next
        to a side or back plate (where the cradle has a wall to join), on the
        plate and off the window it stands in, clear of every finger's links
        turning at its base, of the other servos' housings and of the pads
        already placed; as far apart as they go, toward the servo's back
        (away from its own link). None if two do not fit."""
        S = sp.S_parent
        Si = inv(S)
        hx, hy1, hz = HOUSING_MM
        top = dv._hull2d(Pc[:, 1:])          # the pedestal's outline on the plate (`Pc`: its socket's corners)
        win = self._slab_yz(S, (-hx - 0.2, -9.7, -hz - 0.2), (hx + 0.2, hy1 + 0.2, hz + 0.2), x0, x1)
        if len(top) < 3:
            return None
        others = [o for o in self.servos.values() if o.parent[0] == "palm" and o is not sp]
        taken = [np.array(c) for cr in self._cradles for c in cr["pads"]]
        # a screwdriver from behind the plate passes no servo slid out behind it, nor its beams (`_palm`)
        behind = []
        for o in others:
            Po = housing_corners(o.S_parent)
            if float(Po[:, 0].max()) >= x0 + 0.5:
                continue
            oo = o.S_parent[:3, 3][1:]
            d0 = -oo / np.linalg.norm(oo) if np.linalg.norm(oo) > 1e-6 else np.array([1.0, 0.0])
            pts2 = list(Po[:, 1:])
            for turn in (0.0, 60.0, -60.0):
                c_, s_ = math.cos(math.radians(turn)), math.sin(math.radians(turn))
                d = np.array([c_ * d0[0] - s_ * d0[1], s_ * d0[0] + c_ * d0[1]])
                side = np.array([-d[1], d[0]])
                pts2 += [oo + BASE_BEAM_MM * d + k * 10.0 * side for k in (-1, 1)]
            behind.append(dv._hull2d(np.array(pts2)))
        hulls = self._sweep_hulls()
        # its own link, turning less than a right angle either way, comes behind the servo's axis only with
        # its arms' discs (the links' simulator boxes are far bigger there): those hulls are tested only
        # in front of that
        lo, hi = sp.rot_deg
        own = (sp.key[1] if sp.key[0] == "f" and sp.yoke != "crossing" and max(abs(lo), abs(hi)) <= 90.0 else None)
        behind_mm = 9.5 + PAD_SWEEP_MARGIN_MM          # ARM_R in the library
        ring = [np.array([0.0, math.cos(t), math.sin(t)]) for t in np.linspace(0, 2 * math.pi, 8, endpoint=False)]

        def nearest(q):
            best, bd = None, math.inf
            for k in range(len(top)):
                a, b = top[k], top[(k + 1) % len(top)]
                e = b - a
                t = float(np.clip((q - a) @ e / max(float(e @ e), 1e-12), 0.0, 1.0))
                c = a + t * e
                d = float(np.linalg.norm(q - c))
                if d < bd:
                    best, bd = c, d
            return best

        cands = []
        n = len(top)
        for k in range(n):
            a, b = top[k], top[(k + 1) % n]
            e = b - a
            L = float(np.linalg.norm(e))
            if L < 1e-6:
                continue
            nrm = np.array([e[1], -e[0]]) / L
            for t in (0.15, 0.32, 0.5, 0.68, 0.85):
                cands.append(a + t * e + (PAD_R_MM + 0.6) * nrm)
        out = []
        for q in cands:
            if self._dist_out(q, outline) > -(PAD_R_MM + 1.0):
                continue
            if win is not None and self._dist_out(q, win) < 3.5:
                continue
            if sec < 0 and float(np.linalg.norm(q)) < PAD_COLUMN_R_MM:
                continue
            if any(float(np.linalg.norm(q - c)) < 2 * PAD_R_MM + 1.0 for c in taken):
                continue
            if any(self._dist_out(q, b) < 3.3 for b in behind):
                continue
            c = nearest(q)
            # it joins the pedestal where that has a wall: outside the case's keep-out
            ps = Si @ np.array([x1 + PAD_H_MM / 2.0, c[0], c[1], 1.0])
            if not (abs(ps[0]) >= 11.7 or abs(ps[2]) >= 10.2 or ps[1] >= 24.7):
                continue
            pts = np.array([np.array([x, q[0], q[1]]) + r * u for x in (x1 + 0.5, x1 + PAD_H_MM / 2, x1 + PAD_H_MM - 0.5)
                            for r, u in [(0.0, ring[0])] + [(PAD_R_MM, u) for u in ring]])
            front = pts[((pts - S[:3, 3]) @ S[:3, 1]) < behind_mm]
            if any(np.all(P_ @ eq[:, :3].T + eq[:, 3] <= PAD_SWEEP_MARGIN_MM, axis=1).any()
                   for eq, P_ in ((eq, front if self._hull_owner[k] == own else pts) for k, eq in enumerate(hulls))
                   if len(P_)):
                continue
            hit = False
            for o in others:
                loc = (pts - o.S_parent[:3, 3]) @ o.S_parent[:3, :3]
                if np.any((np.abs(loc[:, 0]) <= hx + 1.0) & (loc[:, 1] >= -9.5 - 1.0 - CRADLE_CHANNEL_MM)
                          & (loc[:, 1] <= hy1 + 1.0) & (np.abs(loc[:, 2]) <= hz + 1.0)):
                    hit = True
                    break
            if hit:
                continue
            back = float((Si @ np.array([x1, q[0], q[1], 1.0]))[1])
            out.append({"at": (float(q[0]), float(q[1])), "near": (float(c[0]), float(c[1])), "back": back})
        best, score = None, -math.inf
        for a in range(len(out)):
            for b in range(a + 1, len(out)):
                d = float(np.linalg.norm(np.subtract(out[a]["at"], out[b]["at"])))
                if d < 8.0:                     # two screw heads apart (the window keeps the cradle square)
                    continue
                sc = d + 0.3 * (out[a]["back"] + out[b]["back"])
                if sc > score:
                    best, score = [out[a], out[b]], sc
        return best

    def _cradle_parts(self) -> None:
        """Each cradle set apart in `_palm` as a part of its own (in its palm
        part's frame): its socket on its pedestal, its floor and pads, less everything its palm
        part gives way to but the other fingers' sweeps (as a kept cradle
        wall), less the cradles made before it, with a pilot hole in each
        pad."""
        done: List[Piece] = []
        for cr in self._cradles:
            sp, sec = cr["sp"], cr["sec"]
            palm = self.parts[("palm", sec)]
            i = sp.key[1] if sp.key[0] == "f" else -1
            key = ("cradle", i)
            part = Part(f"f{i}_cradle" if i >= 0 else "hinge_cradle", ("palm", sec), palm.frame)
            part.positive = [w("0") for w in cr["walls"]]
            x1 = -PLATE_HALF_MM + PALM_PLATE_MM
            part.negative = list(palm.negative) + [w("FIT") for w in done]
            part.negative += [f"translate([0, {y:.3f}, {z:.3f}]) cyl_x(M3_PILOT / 2, {x1 - 1}, {x1 + PAD_H_MM - 1.5});"
                              for y, z in cr["pads"]]
            part.swept_by = {i: list(palm.swept_by.get(i, []))}
            if i != -1:
                part.swept_by[-1] = list(palm.swept_by.get(-1, []))
            part.holds = [sp]
            part.pads = list(cr["pads"])
            sp.holder = key
            self.parts[key] = part
            done += cr["walls"]

    # ------------------------------------------------- palm fixings, bay, mount

    def _keep_clear_yz(self, sec: int) -> List[np.ndarray]:
        """(y, z) outlines on plate `sec` that bolts and screws keep clear of:
        each of its fingers' base sweeps, the cradles that are parts of their
        own (on before the column's screws, a screwdriver's width more than
        their walls), and the hinge servo with its yoke."""
        out = []
        for i in range(len(self.hand.fingers)):
            if finger_section(self.hand, i) == sec:
                out.append(dv._hull2d(dv.base_sweep_mm(self.hand, i)[:, 1:]))
        for cr in self._cradles:
            if cr["sec"] == sec:
                S = cr["sp"].S_parent
                out.append(dv._hull2d(housing_corners(S, grow=1.0)[:, 1:]))
        h = self.servos.get(("hinge",))
        if h is not None:
            c = h.S_parent[:3, 3]
            ring = np.array([[math.cos(t), math.sin(t)] for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)])
            out.append(c[1:] + 24.0 * ring)
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

    def _palm_fixings(self) -> None:
        g = dv.hand_geometry(self.hand)
        for sec in ([-1, 0] if self.hand.palm_joints else [-1]):
            part = self.parts[("palm", sec)]
            outline = g.outlines[sec]
            clear = self._keep_clear_yz(sec)
            screws: List[Tuple[float, float]] = []
            if sec < 0:
                # the mount's column: its screws from the split plane, the cables' bore through the back half
                screws = self._column_screws(clear)
                part.column_screws = screws
                for y, z in screws:
                    part.negative.append(f"column_screw({y:.3f}, {z:.3f});")
                self.bore = self._bore_spot(screws)
                part.negative.append(f"translate([0, {self.bore[0]:.3f}, {self.bore[1]:.3f}]) "
                                     f"cyl_x(BORE_R, {-PLATE_HALF_MM - 1}, {-PLATE_HALF_MM + PALM_PLATE_MM + 1});")

    def _bore_spot(self, screws: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Where the cables' bore goes through the palm's back half: inside the
        column, off its screws, as far as it goes from every servo pocket
        reaching into the back half."""
        boxes = []
        for sp in self.parts[("palm", -1)].holds:
            C = mt.box_corners(np.array([0.0, 7.5, 0.0]), np.eye(3), np.array([13.9, 19.4, 12.4]))
            P = C @ sp.S_parent[:3, :3].T + sp.S_parent[:3, 3]
            if float(P[:, 0].min()) < 0.0:
                boxes.append(dv._hull2d(P[:, 1:]))
        best, best_d = (0.0, 0.0), -math.inf
        for y in np.arange(-9.0, 9.01, 1.5):
            for z in np.arange(-9.0, 9.01, 1.5):
                q = np.array([y, z])
                if float(np.linalg.norm(q)) > 20.0 - BORE_R_MM - 2.5:
                    continue
                if any(float(np.linalg.norm(q - np.array(c))) < BORE_R_MM + 3.5 for c in screws):
                    continue
                d = min((self._dist_out(q, b) for b in boxes), default=math.inf) - BORE_R_MM
                if d > best_d + 1e-9 or (abs(d - best_d) < 1e-9 and float(np.linalg.norm(q)) < float(np.linalg.norm(best))):
                    best, best_d = (float(y), float(z)), d
        return best

    def _column_screws(self, clear: List[np.ndarray]) -> List[Tuple[float, float]]:
        r = COLUMN_SCREW_R_MM
        for n in (4, 3):
            for k in range(0, 360 // n, 15):
                pts = [np.array([r * math.cos(math.radians(k + 360 / n * m)), r * math.sin(math.radians(k + 360 / n * m))])
                       for m in range(n)]
                if all(self._dist_out(q, c) >= 2.5 for q in pts for c in clear):
                    return [tuple(map(float, q)) for q in pts]
        return []

    def _bay(self) -> None:
        g = dv.hand_geometry(self.hand)
        bay = g.bay
        u, v = bay.axes[:, 1], bay.axes[:, 2]
        B = T(np.column_stack([np.array([1.0, 0.0, 0.0]), u, v]), (-PLATE_HALF_MM - dv.MOUNT_DEPTH_MM, 0.0, 0.0))
        cy, cz = float(bay.centre @ u), float(bay.centre @ v)
        self.bay_frame, self.bay_centre = B, (cy, cz)
        screws = getattr(self.parts[("palm", -1)], "column_screws", [])
        sb = [(float(np.array([0.0, y, z]) @ u), float(np.array([0.0, y, z]) @ v)) for y, z in screws]
        cup = Part("bay", ("palm", -1), B)
        cup.positive = [f"bay_cup({cy:.3f}, {cz:.3f});"]
        bore = getattr(self, "bore", (0.0, 0.0))
        bb = (float(np.array([0.0, *bore]) @ u), float(np.array([0.0, *bore]) @ v))
        mount = Part("mount", ("palm", -1), B)
        mount.positive = [f"mount({cy:.3f}, {cz:.3f}, {[list(q) for q in sb]}, {list(bb)});"]
        # both give way to everything of the main palm but its plate (housings are 2.2 mm walls past
        # the bare servo the grammar keeps the bay a wall from; hanging housings' joining paths)
        Binv = mm(inv(B))
        for pc in self.parts[("palm", -1)].pieces[1:]:
            for part in (cup, mount):
                part.negative.append(f"{Binv} {pc('FIT + 0.3')}")
        # the column is on before the palm's own servos: a screwdriver reaches a base wrap's idler arm through it
        for sp in self.parts[("palm", -1)].holds:
            if sp.yoke == "wrap" and sp.holder is None:
                mount.negative.append(f"{Binv} {mm(sp.S_parent)} arm_screw_tunnels(-1, {list(HORN_HOLES_IDLER)});")
        self.extras = [cup, mount]

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
        # the parent gives room to link j + 1, its horns and what it holds
        Tn = inv(sp.S_child) @ nxt.S_parent          # servo j + 1's frame in servo j's
        occupants = list(far.pieces) + _horn_pieces(nxt)
        for h in far.holds:
            occupants += _case_pieces(h.S_parent, h.cable, h.idler_horn)
        for pc in occupants:
            stmt = (f"{mm(sp.S_parent)} sweep_x2({_mat(Tn)}, {lo1:.3f}, {hi1:.3f}, {lo2:.3f}, "
                    f"{hi2:.3f}) {mm(inv(nxt.S_child))} {pc('PART_CLEAR')}")
            if sp.parent[0] == "palm":
                grand.swept_by.setdefault(i, []).append(stmt)
            else:
                grand.negative.append(stmt)
        # link j + 1 gives room to servo j's case, turning back about both joints
        Tb = inv(nxt.S_parent) @ sp.S_child          # servo j's frame in servo j + 1's
        for pc in _case_pieces(np.eye(4), sp.cable, sp.idler_horn):
            far.negative.append(f"{mm(nxt.S_child)} sweep_x2({_mat(Tb)}, {-hi2:.3f}, {-lo2:.3f}, {-hi1:.3f}, "
                                f"{-lo1:.3f}) {pc('CASE_CLEAR')}")

    # --------------------------------------------------------------- output

    def part_list(self) -> List[Part]:
        return list(self.parts.values()) + list(self.extras)
