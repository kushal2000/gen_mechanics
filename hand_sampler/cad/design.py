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

The palm is a shell: a plate whose face is level with the base servos'
tops, their cradles hanging under it, posts down to the mount's column. A
base servo whose cradle can take two pads under the plate has a cradle of
its own (`_cradle_pads`, `_cradle_parts`): a part in the palm's frame,
screwed to the servo on the bench, pushed up into a window of the plate and
screwed to it from the palm's face. Its servo's `holder` names it; `parent`
stays the palm, the rigid body.

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

from hand_sampler.grammar import derive as dv
from hand_sampler.grammar import motor as mt
from hand_sampler.grammar.hand import PLATE_HALF_MM, Hand, finger_section, turned_servo

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
BASE_BEAM_MM = 38.0
"""A base servo hanging too deep for its socket to reach the palm's plate
(standing on its back end, its yoke turning round its top end, or slid down
by a crossing) hangs in a full housing from beams this long toward the palm
centre and 60 degrees either side, past its yoke's sweep, each with a post
up into the plate."""
YOKE_REACH_MM = 20.0          # how far a yoke turning round its axis reaches from it (bridge corners and a gap)
SLING_POST_MM = 10.0          # a sling's posts and bars (square)
SLING_POST_OFF_MM = 12.0      # a sling's posts this far beside the axis: off the arm screws' circle (6), within the case
SLING_BAR_MM = 7.0            # a sling's bars: how deep
FACE_TRIM_MM = BRIDGE_HALF_MM + PART_CLEAR    # the face stops this far behind a lying base servo's axis (its arms turning)
PALM_BOLTS_MAX = 6
PALM_BOLT_CLEAR_MM = 5.0      # and this far from any base sweep
PALM_BOLT_SPACING_MM = 18.0
BORE_R_MM = 7.0               # the cables' bore (library BORE_R)
COLUMN_SCREW_R_MM = 16.5
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
PALM_TOP_MM = PLATE_HALF_MM - mt.WALL_MM   # the palm's face: level with the base servos' tops (library PALM_TOP)
PALM_PLATE_MM = 3.0           # the palm's plate, under its face
POST_R_MM = 4.5               # a post round each of the mount's column screws (library POST_R)
POST_CLEAR_MM = 6.0           # a post's screw this far from every base sweep and cradle
CRADLE_CHANNEL_MM = 14.0      # a cradle's servo needs only its own front clear (it is open on two more sides)
PAD_R_MM, PAD_H_MM = 4.5, 7.0
"""A base servo's cradle is its own part: the servo is screwed into it on the
bench, then it goes up into a window of the plate from below, and two pads
(this radius and height) under the plate take a countersunk M3 x 10 each
from the palm's face."""
PAD_SWEEP_MARGIN_MM = 3.5     # pads keep this far from the links' simulator bodies turning (clip 2.4 + 1)
PAD_COLUMN_R_MM = 24.0        # off the posts round the mount's column screws (16.5 + 4.5 + a gap)
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
    parent: Tuple                 # body holding its case: ("palm", k) or ("link", i, j - 1)
    child: Tuple                  # body holding its horns: ("link", i, j) or ("palm", 0)
    S_child: np.ndarray           # 4x4: the servo frame in the child's frame (q = 0)
    S_parent: np.ndarray          # 4x4: the servo frame in the parent's frame (q = 0)
    rot_deg: Tuple[float, float]  # the child's turn about the servo's X over the joint's range
    yoke: str                     # "u", "single" or "crossing"
    cable: int                    # the Z face whose socket takes the cable (+1 / -1)
    turned: bool = False
    flip: bool = False
    sign: float = 1.0             # + joint angle turns the driven horn this way about S's X (servo's own sign)
    holder: Optional[Tuple] = None    # the part holding its case when that is not the parent (a palm cradle)
    across: bool = False              # lies across the finger (its 23 x 34 face toward the palm): a square connector
    flange: bool = False              # held by its idler face in a flange cup (its idler horn off: a one-arm yoke)

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
        turned = turned_servo(f, j)
        crossing = j < len(f.joints) - 1 and jt.length == 0
        single = abs(float(a[0])) >= SINGLE_ARM_LEAN - 1e-9 and not crossing
        flip = single and float(a[0]) < 0.0        # a one-arm yoke's driven horn faces the fingertip
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
                             "crossing" if crossing else ("single" if single else "u"),
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
        neg = "\n    ".join(self.negative + [x for xs in self.swept_by.values() for x in xs] + self.tunnels
                               + self.windows)
        body = f"difference() {{\n        body_{n}();\n    {neg}\n    }}"
        if self.kept:
            # a servo's cradle keeps its walls against other fingers' sweeps (fingers colliding in
            # motion is left to the physics and the policy, as in the simulator), and gives way to
            # everything else: its own finger's sweeps, the cut's hinge, every servo's keep-outs
            keep = []
            for i, walls in self.kept:
                rest = (self.negative + self.windows + self.swept_by.get(i, [])
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
        self._posts: set = set()              # the palm's posts down to the mount's column (`_palm_fixings`)
        self._slung: set = set()              # base servos hanging in a sling (`_sling`)
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
            # yoke by a hollow connector from the bridge's (or the one arm's) whole face to the whole
            # cradle face toward it (`_connector`)
            Sn = mm(nxt.S_parent)
            # a servo lying across the finger (its 23 x 34 face toward the palm: a turned bend, a twist) gets
            # a square connector (Martin, 10/24); a servo whose idler horn is off (a one-arm yoke on its
            # output horn) is held by its idler face in a flange cup, nothing round its horn (10/26)
            nxt.across = nxt.yoke != "crossing" and abs(float(nxt.S_parent[0, 1])) < 0.7
            nxt.flange = nxt.yoke == "single"
            A, sa = self._tube_start(sp, a)
            B, sb, face = self._tube_end(nxt, A[:3, 3])
            A2, sa2, B, sb = self._connector(A, sa, B, sb, square=nxt.across)
            if face != 0 and nxt.cable == face:
                nxt.cable = -face                        # the tube lands on that side: the cable takes the other
            # its ends reach into the bridge and the cradle (a single arm's start stays on its horn's face:
            # nothing round a horn)
            ea = REACH_MM if sp.yoke == "u" else 0.0
            tube: Piece = lambda g, A=A2, B=B, sa=sa2, sb=sb, ea=ea: (
                f"tube({_mat(A)}, {list(sa)}, {_mat(B)}, {list(sb)}, {g}, {ea}, {REACH_MM});")
            extra: List[Piece] = []
            if ("link", i, j) in self.more_tubes:
                # this link came out in pieces (something turns through its tube): a tube to every face
                for Bx, sbx, fx in self._tube_ends(nxt):
                    if fx != face and not (fx != 0 and fx == nxt.cable):
                        Ax, sax, Bx, sbx = self._connector(A, sa, Bx, sbx, square=nxt.across)
                        extra.append(lambda g, A=Ax, B=Bx, sa=sax, sb=sbx, ea=ea:
                                     f"tube({_mat(A)}, {list(sa)}, {_mat(B)}, {list(sb)}, {g}, {ea}, {REACH_MM});")
            if nxt.yoke == "crossing":
                # a crossing's first servo: its near horn carries the crossing plate, so its cradle is a
                # full housing (the plate's swing opens the near side)
                cradle: Piece = lambda g, Sn=Sn: f"{Sn} housing_box({g});"
            elif nxt.flange:
                cradle = lambda g, Sn=Sn: f"{Sn} flange_cup({g});"
            else:
                cradle = lambda g, Sn=Sn: f"{Sn} cradle_socket({g});"
            # gussets in the L pieces' inside corners (Martin, 10/26), each kept to where the connector is
            outer = tube

            def along(o: Piece, v: np.ndarray) -> Piece:
                """The connector and its copy moved `GUSSET_MM` along v: where a gusset may stand on it."""
                sh = mm(T(None, tuple(float(c) for c in v * GUSSET_MM)))
                return lambda g: f"hull() {{ {o(g)} {sh} {o(g)} }}"

            # (none round a flange cup: lying along the servo's sides with nothing to hold, they only read as ears;
            # Martin, 10/28)
            gussets: List[Piece] = []
            if not nxt.flange and nxt.yoke != "crossing" and face in (1, -1):
                on = along(outer, -face * nxt.S_parent[:3, 2])           # from the side it lands on, into the socket
                gussets += [lambda g, on=on, Sn=Sn, f=face, s=s: f"intersection() {{ {Sn} socket_gusset({f}, {s}, GUSSET, {g}); {on(g)} }}"
                            for s in (1, -1)]
            if sp.yoke == "u":
                gussets += self._arm_gussets(sp, a, along(outer, -A[:3, 0]))
            else:
                # a single arm and its connector are one plate: the arm takes in the connector's footprint in
                # its own layer, so where the connector leaves the horn's face at a slant they still meet
                # through the arm's whole thickness (not a sliver the body's rounding would take away)
                S0 = mm(sp.S_child)
                arm1 = pieces[0]
                pieces = [lambda g, arm1=arm1, S0=S0, tube=tube: (
                    f"hull() {{ {arm1(g)} intersection() {{ {tube(g)} {S0} arm_slab(1, {g}); }} }}")] + pieces[1:]
            pieces = pieces + [cradle, tube] + gussets + extra
        part.pieces = pieces
        part.positive = [p("0") for p in pieces]
        part.clip = _clip_body(f, j)
        # what this part leaves empty: its own servo's horns, its yoke's screws, the next
        # servo's housing keep-outs
        S = mm(sp.S_child)
        part.negative.append(f"{S} cyl_x(HORN_R + CASE_CLEAR, DRIVEN_X[0] - FIT, DRIVEN_X[1]);")
        if sp.idler_horn:
            part.negative.append(f"{S} cyl_x(HORN_R + CASE_CLEAR, IDLER_X[0], -(DRIVEN_X[0] - FIT));")
        if sp.yoke == "u":
            part.negative += [f"{S} arm_keepout(1, {list(HORN_HOLES_DRIVEN)});",
                              f"{S} arm_keepout(-1, {list(HORN_HOLES_IDLER)});"]
        else:
            part.negative.append(f"{S} arm_keepout(1, {list(HORN_HOLES_DRIVEN)});")
        if nxt is not None:
            part.tunnels.append(f"{mm(nxt.S_parent)} case_screw_tunnels(40, {str(nxt.flange).lower()});")
            if nxt.flange:
                part.negative.append(f"{mm(nxt.S_parent)} flange_keepout({nxt.cable});")
            else:
                ch = 80 if nxt.yoke == "crossing" else CRADLE_CHANNEL_MM
                part.negative.append(f"{mm(nxt.S_parent)} housing_keepout({str(nxt.yoke != 'u').lower()}, {nxt.cable}, "
                                     f"{ch});")

    @staticmethod
    def _arm_gussets(sp: ServoSpot, a: np.ndarray, on: Piece) -> List[Piece]:
        """A U's gussets: in each arm's layer, the corners between the arm's
        edges and the connector's back face, where the connector is wider
        than the bridge (`on`: where they may stand). Outboard of the case,
        so they turn clear of the part holding the servo."""
        x_b = mt.swept_front_mm(a, sp.turned) + CASE_CLEAR
        B = _bridge_frame(sp.S_child[:3, 3], np.array([1.0, 0.0, 0.0]), a, x_b)
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

    def _tube_start(self, sp: ServoSpot, a: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Where a link's connector leaves its yoke (link frame): the bridge's
        whole outer face; a single arm's horn face, as wide as the arm's disc
        (the connector and the arm are one plate, Martin 10/26)."""
        if sp.yoke == "u":
            x_b = mt.swept_front_mm(a, sp.turned) + CASE_CLEAR
            B = _bridge_frame(sp.S_child[:3, 3], np.array([1.0, 0.0, 0.0]), a, x_b)
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
        for side, c, F in ((0, np.array([0.0, HOUSING_MM[1], 0.0]), np.column_stack([R[:, 1], R[:, 0], R[:, 2]])),
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
        faces = [(0, R[:, 1], np.array([0.0, HOUSING_MM[1], 0.0]), np.column_stack([R[:, 1], R[:, 0], R[:, 2]])),
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
        """A palm part, face first (Martin, 10/28: the face is the first thing
        in the design, the servos mounted under it): a plate whose face is
        level with the base servos' tops (`PALM_TOP_MM`), over the palm's
        outline less what lies ahead of a lying base servo's yoke (no cut-outs
        round the U outputs: `_face_outline`), each base servo hanging under
        it: a lying one in its socket under the face (in a cradle of its own
        where two pads fit), one standing on its back end or slid down by a
        crossing in a sling (`_sling`), the face open over its yoke."""
        g = dv.hand_geometry(self.hand)
        part = self.parts[("palm", sec)]
        x0, x1 = PALM_TOP_MM - PALM_PLATE_MM, PALM_TOP_MM          # the plate's underside and face
        # the servos standing under the face hang in slings; the face reaches over their posts (and, past a
        # cut, over the hinge yoke's bridge, which carries it)
        slings, keep = {}, []
        for sp in part.holds:
            if not self._at_plate(sp):
                slings[sp.key], tops = self._sling(sp, sec, g.outlines[sec], x1)
                keep += tops
        if sec == 0 and part.carries is not None:
            h = part.carries
            B = _bridge_frame(h.S_child[:3, 3], -h.S_child[:3, 1], h.S_child[:3, 0], mt.FRONT_SWEEP_MM + CASE_CLEAR)
            C = mt.box_corners(np.array([WALL / 2.0, 0.0, 0.0]), np.eye(3), np.array([WALL / 2.0, ARM_X0, BRIDGE_HALF_MM]))
            keep += list((C @ B[:3, :3].T + B[:3, 3])[:, 1:])
        outline = self._face_outline(part, g.outlines[sec], x0, keep)
        part.face = outline
        o = "[" + ",".join(f"[{y:.3f},{z:.3f}]" for y, z in outline) + "]"
        plate: Piece = lambda gg, o=o: f"plate({o}, {x0} - ({gg}), {x1} + ({gg}));"
        pieces: List[Piece] = [plate]
        loose = set()                  # pieces of the cradles that are parts of their own
        part.clip = f"translate([-500, -500, -500]) cube([{500 + x1}, 1000, 1000]);"    # nothing above the face
        for sp in part.holds:
            Sp = mm(sp.S_parent)
            cradle = ([lambda gg, Sp=Sp: f"{Sp} housing_box({gg});"] if sp.yoke == "crossing" else
                      [lambda gg, Sp=Sp: f"{Sp} cradle_socket({gg});"])
            body = "housing_box" if sp.yoke == "crossing" else "cradle_socket"
            Pc = housing_corners(sp.S_parent, y0=-9.5 if sp.yoke == "crossing" else SOCKET_Y0_MM)
            # its socket reaches up to the plate (a servo lying in the palm), or the servo stands on its
            # back end under it, its yoke turning round its top end
            hanging = self._at_plate(sp)
            ch = 80 if (sp.yoke == "crossing" or not hanging) else CRADLE_CHANNEL_MM
            part.negative.append(f"{Sp} housing_keepout({str(sp.yoke != 'u').lower()}, {sp.cable}, {ch});")
            # the socket (a crossing's servo: its full housing) hangs from the plate: its hull with its
            # outline straight up to the plate (the case's keep-out leaves the walls)

            def pedestal(x_to: float, Sp=Sp, body=body, Pc=Pc) -> Piece:
                up = Pc.copy()
                up[:, 0] = x_to
                return lambda gg, pts=_pts(up): f"hull() {{ {Sp} {body}({gg}); hull_pts({pts}); }}"

            pads = self._cradle_pads(sp, sec, outline, x0, x1, Pc) if hanging else None
            if pads is not None:
                # its own part: the servo is screwed in on the bench (nothing in a screwdriver's way),
                # then the cradle goes up into a window of the plate from below, its top flush with the
                # face and its pads under the plate, and two countersunk M3 go in from the face
                cradle = [pedestal(x1)]
                pad_pieces = []
                for pd in pads:
                    y, z = pd["at"]
                    ny, nz = pd["near"]
                    pad_pieces.append(lambda gg, y=y, z=z, ny=ny, nz=nz, w=cradle[0]: (
                        f"hull() {{ translate([0, {y:.3f}, {z:.3f}]) cyl_x({PAD_R_MM} + ({gg}), "
                        f"{x0 - PAD_H_MM} - ({gg}), {x0} + ({gg})); intersection() {{ {w(gg)} "
                        f"translate([{x0 - PAD_H_MM} - ({gg}), {ny - 6.0:.3f}, {nz - 6.0:.3f}]) "
                        f"cube([{PAD_H_MM} + 2 * ({gg}), 12, 12]); }} }}"))
                    part.negative.append(f"translate([0, {y:.3f}, {z:.3f}]) cs_m3({x1}, {PALM_PLATE_MM + 1});")
                # the plate inside the cradle's box is the cradle's (its top round the servo), the plate
                # keeps the rest
                floor: Piece = lambda gg, Sp=Sp, o=o: (f"intersection() {{ {Sp} housing_box({gg}); "
                                                       f"plate({o}, {x0} - ({gg}), {x1} + ({gg})); }}")
                own = cradle + [floor] + pad_pieces
                pieces += own
                loose.update(id(q) for q in own)
                part.windows += [f"{Sp} housing_box(FIT);", cradle[0]("FIT")]
                # and gives way to its pads under the plate
                part.windows += [f"intersection() {{ {pc('FIT')} translate([-500, -500, -500]) cube([{500 + x0}, "
                                 f"1000, 1000]); }}" for pc in pad_pieces]
                self._cradles.append({"sp": sp, "sec": sec, "walls": cradle + [floor] + pad_pieces,
                                      "pads": [pd["at"] for pd in pads]})
                continue
            if hanging:
                pieces.append(pedestal((x0 + x1) / 2.0))     # into the plate
                # kept against other fingers only round the servo (as far as its housing box): a tilted
                # servo's pedestal reaching up to the plate gives way to every finger
                cradle = [lambda gg, ped=pieces[-1], Sp=Sp: f"intersection() {{ {ped(gg)} {Sp} housing_box({gg}); }}"]
            else:
                cradle = [lambda gg, Sp=Sp: f"{Sp} housing_box({gg});"]
                # standing on its back end under the face, or slid down by a crossing: a full housing in a
                # sling (`_sling`), and the face open over its yoke
                pieces.append(lambda gg, Sp=Sp: f"{Sp} housing_box({gg});")
                pieces += slings[sp.key]
            part.tunnels += self._palm_tunnels(sp.S_parent, x0, x1)
            part.negative += self._arm_access(sp, x0)
            if sp.key[0] in ("f", "hinge"):
                # its cradle keeps its walls against other fingers' sweeps (the hinge's: against every finger)
                walls = " ".join(c("0") for c in cradle)
                part.kept.append((sp.key[1] if sp.key[0] == "f" else -1, f"difference() {{ union() {{ {walls} }} "
                                             f"{Sp} housing_keepout({str(sp.yoke != 'u').lower()}, {sp.cable}, {ch}); }}"))
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
        one, those servos' housings behind it and the points `keep` (what
        hangs from the face), within the outline."""
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

    def _beam_fan(self, sp: ServoSpot, sec: int, outline: np.ndarray, x1: float) -> Tuple[List[Piece], List[np.ndarray]]:
        """A servo slid down by a crossing whose axis leans toward the palm's
        normal: what turns on it sweeps all round above it, so its housing
        hangs from beams under that (past a cut, just under the plate),
        toward the middle of its own plate (the palm's centre on the main
        palm) and 60 degrees either side, each as long as it goes (up to
        `BASE_BEAM_MM`) with its post inside its part's outline and off the
        mount's column, and from its footprint straight up (cut back where
        something turns)."""
        S = sp.S_parent
        Sp = mm(S)
        P = housing_corners(S)
        oo, a = S[:3, 3], S[:3, 0]
        if sec < 0:
            # under what turns on it: its reach below a leaning axis, the housing's top under a steep one
            below = float(oo[0]) - YOKE_REACH_MM if abs(float(a[0])) < 0.7 else float(P[:, 0].max())
            b1 = below - 0.5
            b0 = max(float(P[:, 0].min()), b1 - 7.0)
        else:
            # past a cut the part swings on the hinge: just under its plate (deeper beams would swing
            # through the main palm's mount)
            b0, b1 = x1 - PALM_PLATE_MM - 7.0, x1 - PALM_PLATE_MM
        face = np.asarray(outline, float)
        mid = np.zeros(2) if sec < 0 else np.mean(face, axis=0)
        d0 = np.array([0.0, mid[0] - oo[1], mid[1] - oo[2]])
        d0 = d0 / np.linalg.norm(d0) if np.linalg.norm(d0) > 1e-6 else np.array([0.0, 1.0, 0.0])
        up = P.copy()
        up[:, 0] = x1
        out: List[Piece] = [lambda gg, pts_u=_pts(np.concatenate([P, up])): f"hull_pts({pts_u});"]
        tops: List[np.ndarray] = []
        beams, spare = [], None
        for turn in (0.0, 60.0, -60.0):
            c, s_ = math.cos(math.radians(turn)), math.sin(math.radians(turn))
            d = np.array([0.0, c * d0[1] - s_ * d0[2], s_ * d0[1] + c * d0[2]])
            for length in (BASE_BEAM_MM, 32.0, 26.0):
                end_ = np.array([0.0, oo[1], oo[2]]) + length * d
                if sec < 0 and float(np.linalg.norm(end_)) < dv.MOUNT_RADIUS_MM + 16.0:
                    continue
                out_by = self._dist_out(end_[1:], face) + 12.0
                if out_by <= 0.0:
                    beams.append((d, end_))
                    break
                if spare is None or out_by < spare[0]:
                    spare = (out_by, d, end_)
        if not beams and spare is not None:
            beams = [spare[1:]]
        for d, end_ in beams:
            side = np.cross([1.0, 0.0, 0.0], d)
            beam = [end_ + s1 * 10.0 * side + np.array([x, 0.0, 0.0]) for s1 in (-1, 1) for x in (b0, b1)]
            post = [end_ + s1 * 10.0 * side + s2 * 6.0 * d + np.array([x, 0.0, 0.0]) for s1 in (-1, 1)
                    for s2 in (-1, 1) for x in (b0, x1)]
            out.append(lambda gg, Sp=Sp, pts_b=_pts(beam): f"hull() {{ {Sp} housing_box({gg}); hull_pts({pts_b}); }}")
            out.append(lambda gg, pts_p=_pts(post): f"hull_pts({pts_p});")
            tops += [q[1:] for q in post]
        return out, tops

    def _sling(self, sp: ServoSpot, sec: int, outline: np.ndarray, x1: float) -> Tuple[List[Piece], List[np.ndarray]]:
        """A base servo standing on its back end under the face (its axis
        level, its yoke turning round its top end) hangs in its housing from
        a sling: straight bars below what turns on it, out along the axis to
        straight posts up to the plate, past the yoke's width (the link folds
        across the axis, never out along it) and off the arm screws' circle;
        one at each end of the axis, on opposite sides of it, where both
        posts fall inside the palm's outline and off the mount's column, else
        the one that does (the nearest if none). A steeper axis: `_beam_fan`.
        Its pieces, and its posts' corners (y, z): the face reaches over
        them."""
        S = sp.S_parent
        P = housing_corners(S)
        oo, a = S[:3, 3], S[:3, 0]
        hx, r = self._turn_extent(sp)
        b1 = float(oo[0]) - r - 0.5
        if abs(float(a[0])) >= 0.45 or b1 < float(P[:, 0].min()) + 3.0:
            return self._beam_fan(sp, sec, outline, x1)          # the bars would miss the housing
        self._slung.add(sp.key)
        b0 = max(float(P[:, 0].min()), b1 - SLING_BAR_MM)
        n = np.array([1.0, 0.0, 0.0])
        ax = a - float(a @ n) * n
        ax = ax / np.linalg.norm(ax)                             # along the axis, in the plate's plane
        d = np.cross(n, ax)                                      # across it (where the link folds)
        h = SLING_POST_MM / 2.0
        along, off = hx + h + 1.0, SLING_POST_OFF_MM
        face = np.asarray(outline, float)
        c0 = np.array([0.0, oo[1], oo[2]])

        def fits(c) -> float:
            out = self._dist_out(c[1:], face) + SLING_POST_MM + 2.0
            if sec < 0:
                out = max(out, dv.MOUNT_RADIUS_MM + SLING_POST_MM + 4.0 - float(np.linalg.norm(c[1:])))
            return out

        # beyond the yoke's width along the axis (the link folds across it, never there), off the arm
        # screws' circle; one post on each end of the axis, on opposite sides of it
        posts = {(sa, sd): c0 + sa * along * ax + sd * off * d for sa in (1.0, -1.0) for sd in (1.0, -1.0)}
        good = []
        for pair in (((1.0, 1.0), (-1.0, -1.0)), ((1.0, -1.0), (-1.0, 1.0))):
            if all(fits(posts[k]) <= 0.0 for k in pair):
                good = list(pair)
                break
        if not good:
            ok = [k for k in posts if fits(posts[k]) <= 0.0]
            good = ok[:1] if ok else [min(posts, key=lambda k: fits(posts[k]))]
        out: List[Piece] = []
        tops: List[np.ndarray] = []
        for sa, sd in good:
            c = posts[(sa, sd)]
            bar = [c0 + sd * off * d + t * sa * ax + u * d + np.array([x, 0.0, 0.0]) for t in (0.0, along + h)
                   for u in (-h, h) for x in (b0, b1)]
            post = [c + t * ax + u * d + np.array([x, 0.0, 0.0]) for t in (-h, h) for u in (-h, h) for x in (b0, x1)]
            out.append(lambda gg, pts=_pts(bar): f"hull_pts({pts}, {gg});")
            out.append(lambda gg, pts=_pts(post): f"hull_pts({pts}, {gg});")
            tops += [q[1:] for q in post]
        return out, tops

    @staticmethod
    def _palm_tunnels(S: np.ndarray, x0: float, x1: float, reach: float = 30.0) -> List[str]:
        """A screwdriver's way to each of a palm-held servo's case screws (as
        the library's `case_screw_tunnels`), each opened up through the plate
        where it would pass just under the palm's face (a slot, not a skin)."""
        out = []
        hx = HOUSING_MM[0]
        for yy, zz in ((22.5, -8.0), (22.5, 8.0)):                  # the far holes (library FAR_HOLES)
            for sgn in (1.0, -1.0):
                a, b = (hx, hx + reach) if sgn > 0 else (-hx - reach, -hx)
                tunnel = f"{mm(S)} cyl_x_at({yy}, {zz}, 1.9, {a}, {b});"
                ends = np.array([S @ np.array([x, yy, zz, 1.0]) for x in (a, b)])[:, :3]
                r_x = 1.9 * math.sqrt(max(0.0, 1.0 - float(S[0, 0]) ** 2))   # the cylinder's reach along the normal
                top = float(ends[:, 0].max()) + r_x
                if top > x0 - 0.5:
                    lift = x1 + 1.0 - (float(ends[:, 0].min()) - r_x)
                    tunnel = f"hull() {{ {tunnel} translate([{lift:.3f}, 0, 0]) {tunnel} }}"
                out.append(tunnel)
        return out

    @staticmethod
    def _arm_driver_paths(sp: ServoSpot, reach: float = 27.0) -> List[Tuple[np.ndarray, np.ndarray]]:
        """A screwdriver's way to each arm screw of a base servo's yoke at the
        zero pose (palm frame): segments along its axis, out from each arm."""
        S = sp.S_parent
        sides = [(1.0, ARM_X0 + ARM_DRIVEN)] + ([(-1.0, ARM_X0 + ARM_IDLER)] if sp.yoke == "u" else [])
        out = []
        for sgn, xa in sides:
            for ang in HORN_HOLES_DRIVEN:
                yy, zz = 6.0 * math.sin(math.radians(ang)), 6.0 * math.cos(math.radians(ang))
                a, b = (xa, xa + reach) if sgn > 0 else (-xa - reach, -xa)
                out.append(tuple((S @ np.array([x, yy, zz, 1.0]))[:3] for x in (a, b)))
        return out

    @staticmethod
    def _arm_access(sp: ServoSpot, x0: float, reach: float = 27.0) -> List[str]:
        """A screwdriver's way to each arm screw of a palm-held base servo's
        yoke, at the zero pose, where it would pass the plate (an axis
        tilting up toward the palm's face): the yoke goes on after the servo
        is in, and nothing can slide in from above the plate."""
        out = []
        S = sp.S_parent
        sides = [(1.0, ARM_X0 + ARM_DRIVEN)] + ([(-1.0, ARM_X0 + ARM_IDLER)] if sp.yoke == "u" else [])
        for sgn, xa in sides:
            a, b = (xa, xa + reach) if sgn > 0 else (-xa - reach, -xa)
            for ang in HORN_HOLES_DRIVEN:
                yy, zz = 6.0 * math.sin(math.radians(ang)), 6.0 * math.cos(math.radians(ang))
                top = max(float((S @ np.array([x, yy, zz, 1.0]))[0]) for x in (a, b)) + 2.2
                if top > x0 - 0.5:
                    out.append(f"{mm(S)} cyl_x_at({yy:.3f}, {zz:.3f}, 2.2, {a:.3f}, {b:.3f});")
        return out

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
        """Two pads for a base servo's cradle under the plate: next to a side
        or back plate (where the cradle has a wall to join), under the plate
        and off the window it stands in, clear of every finger's links turning
        at its base, of the other servos' housings (and the beams of those
        hanging deep) and of the pads already placed; as far apart as they
        go, toward the servo's back (away from its own link). None if two do
        not fit."""
        S = sp.S_parent
        Si = inv(S)
        hx, hy1, hz = HOUSING_MM
        top = dv._hull2d(Pc[:, 1:])          # the pedestal's outline on the plate (`Pc`: its socket's corners)
        win = self._slab_yz(S, (-hx - 0.2, -9.7, -hz - 0.2), (hx + 0.2, hy1 + 0.2, hz + 0.2), x0, x1)
        if len(top) < 3:
            return None
        others = [o for o in self.servos.values() if o.parent[0] == "palm" and o is not sp]
        taken = [np.array(c) for cr in self._cradles for c in cr["pads"]]
        # no pad over the beams and posts of a servo hanging too deep to reach the plate (`_palm`)
        behind = []
        for o in others:
            Po = housing_corners(o.S_parent)
            if float(Po[:, 0].max()) >= x0 - 3.0:
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
        # every finger's links turning at its base, its own too: under the plate a pad is above the servo's
        # axis, where a link turning a right angle brings its sides round behind the axis
        hulls = self._sweep_hulls()
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
            ps = Si @ np.array([x0 - PAD_H_MM / 2.0, c[0], c[1], 1.0])
            if not (abs(ps[0]) >= 11.7 or abs(ps[2]) >= 10.2 or ps[1] >= 24.7):
                continue
            pts = np.array([np.array([x, q[0], q[1]]) + r * u for x in (x0 - 0.5, x0 - PAD_H_MM / 2, x0 - PAD_H_MM + 0.5)
                            for r, u in [(0.0, ring[0])] + [(PAD_R_MM, u) for u in ring]])
            if any(np.all(pts @ eq[:, :3].T + eq[:, 3] <= PAD_SWEEP_MARGIN_MM, axis=1).any() for eq in hulls):
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
            # off every other base servo's arm-screw driver paths (the yokes go on after the cradles)
            if any(_seg_dist(pts, a_, b_).min() < 2.0 + 1.0 for o in others for a_, b_ in self._arm_driver_paths(o)):
                continue
            # off its own yoke turning round its axis (the hinge's: the part past the cut)
            loc = (pts - S[:3, 3]) @ S[:3, :3]
            yx, yr = self._turn_extent(sp)
            if np.any((np.abs(loc[:, 0]) <= yx + 1.0) & (np.linalg.norm(loc[:, 1:], axis=1) <= yr + 1.0)):
                continue
            back = float((Si @ np.array([x0, q[0], q[1], 1.0]))[1])
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
        part's frame): its socket on its pedestal, its top in the plate and
        its pads under it, less everything its palm part gives way to but the
        other fingers' sweeps (as a kept cradle wall), less the cradles made
        before it, with a pilot hole in each pad."""
        done: List[Piece] = []
        for cr in self._cradles:
            sp, sec = cr["sp"], cr["sec"]
            palm = self.parts[("palm", sec)]
            i = sp.key[1] if sp.key[0] == "f" else -1
            key = ("cradle", i)
            part = Part(f"f{i}_cradle" if i >= 0 else "hinge_cradle", ("palm", sec), palm.frame)
            part.positive = [w("0") for w in cr["walls"]]
            part.clip = palm.clip
            x0 = PALM_TOP_MM - PALM_PLATE_MM               # the plate's underside: the pads' tops
            part.negative = list(palm.negative) + [w("FIT") for w in done]
            part.negative += [f"translate([0, {y:.3f}, {z:.3f}]) cyl_x(M3_PILOT / 2, {x0 - PAD_H_MM + 1.5}, {x0 + 1});"
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
        their walls) and their pads, and the hinge servo with its yoke."""
        out = []
        for i in range(len(self.hand.fingers)):
            if finger_section(self.hand, i) == sec:
                out.append(dv._hull2d(dv.base_sweep_mm(self.hand, i)[:, 1:]))
        ring = np.array([[math.cos(t), math.sin(t)] for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)])
        for cr in self._cradles:
            if cr["sec"] == sec:
                S = cr["sp"].S_parent
                out.append(dv._hull2d(housing_corners(S, grow=1.0)[:, 1:]))
                out += [np.asarray(q) + (PAD_R_MM + 1.0) * ring for q in cr["pads"]]
        h = self.servos.get(("hinge",))
        if h is not None:
            c = h.S_parent[:3, 3]
            out.append(c[1:] + 24.0 * ring)
        # every base servo's arm-screw driver paths (a post stands the plate's whole depth)
        for sp in self.servos.values():
            if sp.parent == ("palm", sec):
                for a_, b_ in self._arm_driver_paths(sp):
                    out.append(dv._hull2d(np.array([q[1:] + 3.0 * u for q in (a_, b_) for u in ring])))
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
                # the mount's column under the palm's back: a post from it up to the plate round each of its
                # screws (countersunk, from the palm's face); the cables run under the plate, between the
                # posts and down the column's bore
                screws = self._column_screws(clear)
                part.column_screws = screws
                for y, z in screws:
                    part.negative.append(f"column_screw({y:.3f}, {z:.3f});")
                    post: Piece = lambda gg, y=y, z=z: (f"translate([0, {y:.3f}, {z:.3f}]) cyl_x(POST_R + ({gg}), "
                                                        f"{-PLATE_HALF_MM} - ({gg}), {PALM_TOP_MM} + ({gg}));")
                    self._posts.add(id(post))
                    part.pieces.append(post)
                    part.positive.append(post("0"))
                self.bore = self._bore_spot(screws)

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
        """Four (or three) screws on the column's circle, each with room for
        its post (`POST_CLEAR_MM` from every outline in `clear`), or failing
        that for the screw alone."""
        r = COLUMN_SCREW_R_MM
        for gap in (POST_CLEAR_MM, 2.5):
            for n in (4, 3):
                for k in range(0, 360 // n, 15):
                    pts = [np.array([r * math.cos(math.radians(k + 360 / n * m)),
                                     r * math.sin(math.radians(k + 360 / n * m))]) for m in range(n)]
                    if all(self._dist_out(q, c) >= gap for q in pts for c in clear):
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
        # both give way to everything of the main palm but its plate and its posts (housings are 2.2 mm
        # walls past the bare servo the grammar keeps the bay a wall from; deep housings' beams)
        Binv = mm(inv(B))
        for pc in [q for q in self.parts[("palm", -1)].pieces[1:] if id(q) not in self._posts]:
            for part in (cup, mount):
                part.negative.append(f"{Binv} {pc('FIT + 0.3')}")
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
        # the face open over a servo standing under it on a level axis (`_sling`): the convex hull of what its
        # finger sweeps through the plate round its yoke, a clean opening rather than the sweeps' lobes
        # (Martin, 10/28)
        x0, x1 = PALM_TOP_MM - PALM_PLATE_MM, PALM_TOP_MM
        for sp in self.servos.values():
            if sp.key in self._slung:
                stmts = self.parts[sp.parent].swept_by.get(sp.key[1], [])
                hx, r = self._turn_extent(sp)
                yoke = (f"intersection() {{ {mm(sp.S_parent)} box_lh([{-hx:.3f}, -100, {-r - 1.0:.3f}], "
                        f"[{hx:.3f}, 100, {r + 1.0:.3f}]); translate([{x0 - 1.0}, -500, -500]) "
                        f"cube([{x1 - x0 + 2.0}, 1000, 1000]); }}")
                if stmts:
                    self.parts[sp.parent].negative.append(
                        f"hull() intersection() {{ {yoke} union() {{ " + " ".join(stmts) + " } }")

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
