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

A servo's frame S (servo.py): X along its axis toward the driven horn. It
is the grammar's motor frame (a, v, w), flipped to (-a, v, -w) where the
driven horn must face the other way: along the finger for a single-arm yoke.
"""

from __future__ import annotations

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
PALM_BOLT_EDGE_MM = 6.0       # a palm bolt this far inside the plate's outline
PALM_BOLT_CLEAR_MM = 5.0      # and this far from any base sweep
PALM_BOLT_SPACING_MM = 18.0
BORE_CLEAR_MM = 11.5          # palm bolts stay off the column's cable bore (7 mm) by a nut
COLUMN_SCREW_R_MM = 16.5
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
    yoke: str                     # "u", "single" or "crossing"
    clamped: bool                 # held in a palm pocket (between the palm's halves), not by case screws
    cable: int                    # the Z face whose socket takes the cable (+1 / -1)
    turned: bool = False
    flip: bool = False
    sign: float = 1.0             # + joint angle turns the driven horn this way about S's X (servo's own sign)

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
        clamped = j == 0 and _crosses_mid_plane(S_parent)
        out.append(ServoSpot(("f", i, j), parent, ("link", i, j), S_child, S_parent, rot,
                             "crossing" if crossing else ("single" if single else "u"),
                             clamped, cable, turned, flip, sgn))
    return out


def _crosses_mid_plane(S_palm: np.ndarray) -> bool:
    """Whether a base servo's case straddles the palm's mid-plane (x = 0), so
    the palm's two halves can clamp it."""
    C = mt.box_corners(np.array([0.0, 7.5, 0.0]), np.eye(3), np.array([11.5, 17.0, 10.0]))
    x = (C @ S_palm[:3, :3].T + S_palm[:3, 3])[:, 0]
    return float(x.min()) < -1.0 and float(x.max()) > 1.0


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
    return ServoSpot(("hinge",), ("palm", -1), ("palm", 0), S, S.copy(), rot, "u", True, cable, False, False, sgn)


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
    split_x: Optional[float] = None            # cut into front/back halves at this x (palm parts)
    clip: Optional[str] = None                 # a convex body the part stays inside (its simulator body, grown)

    def scad(self) -> str:
        pos = "\n        ".join(self.positive) or "cube(0.001);"
        neg = "\n    ".join(self.negative)
        body = f"difference() {{\n        union() {{\n        {pos}\n        }}\n    {neg}\n    }}"
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
    return ("minkowski() { hull() { " + " ".join(f"translate({v3(p)}) cube(0.01, center = true);" for p in pts)
            + f" }} sphere(r = {CLIP_MM}, $fn = 16); }}")


def build_design(hand: Hand) -> "Design":
    return Design(hand)


class Design:
    """Every part of `hand`, its servos and screws."""

    def __init__(self, hand: Hand):
        self.hand = hand
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
        self.parts[("palm", -1)] = Part("palm", ("palm", -1), np.eye(4), split_x=0.0)
        self.extras: List[Part] = []          # parts fixed to the main palm: the bay and the mount
        h = hinge_servo(hand)
        if h is not None:
            self.servos[h.key] = h
            self.parts[("palm", 0)] = Part("palm_beyond", ("palm", 0), np.eye(4), split_x=0.0)
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
            body: Piece = lambda g, jp=joint_piece, p1=p1, r=r: (
                f"hull() {{ {jp(g)} translate({v3(p1)}) sphere(r = {r:.3f} + ({g}), $fn = 48); }}")
            pieces = pieces + [body]
        elif sp.yoke != "crossing":
            Sn = mm(nxt.S_parent)
            body = lambda g, jp=joint_piece, Sn=Sn: f"hull() {{ {jp(g)} {Sn} housing_box({g}); }}"
            pieces = pieces + [body]
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
            part.negative.append(f"{mm(nxt.S_parent)} housing_keepout({str(nxt.yoke != 'u').lower()}, {nxt.cable});")

    def _crossing_gap(self, sp: ServoSpot, nxt: ServoSpot) -> float:
        """How far from servo j's mid-plane (along its X) the next servo's
        case starts, less the fit: where the crossing plate ends."""
        C = mt.box_corners(np.array([0.0, 7.5, 0.0]), np.eye(3), np.array([11.5, 17.0, 10.0]))
        P = C @ nxt.S_parent[:3, :3].T + nxt.S_parent[:3, 3]       # in the child (link j) frame
        X = (P - sp.S_child[:3, 3]) @ sp.S_child[:3, 0]
        return float(X.min()) - FIT

    # ---------------------------------------------------------------- palm

    def _palm(self, sec: int) -> None:
        g = dv.hand_geometry(self.hand)
        part = self.parts[("palm", sec)]
        outline = g.outlines[sec]
        o = "[" + ",".join(f"[{y:.3f},{z:.3f}]" for y, z in outline) + "]"
        plate: Piece = lambda gg, o=o: f"plate({o}, {-PLATE_HALF_MM} - ({gg}), {PLATE_HALF_MM} + ({gg}));"
        pieces: List[Piece] = [plate]
        for sp in part.holds:
            Sp = mm(sp.S_parent)
            if sp.clamped:
                pieces.append(lambda gg, Sp=Sp: f"{Sp} pocket_box({gg});")
                part.negative.append(f"{Sp} pocket_keepout();")
            else:
                # a servo slid out behind the plate (a crossing at the base): its housing, joined
                # to the plate's back face by a block over its footprint (what turns past it is
                # cut away later)
                pieces.append(lambda gg, Sp=Sp: f"{Sp} housing_box({gg});")
                # a beam from it toward the palm centre, under what link 0 sweeps (that reaches
                # about 30 mm from the axis, down to 2.4 mm behind the plate), then up into the plate
                o = sp.S_parent[:3, 3]
                d = -np.array([0.0, o[1], o[2]])
                d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else np.array([0.0, 1.0, 0.0])
                end = np.array([0.0, o[1], o[2]]) + BASE_BEAM_MM * d
                side = np.cross([1.0, 0.0, 0.0], d)
                xb = -PLATE_HALF_MM
                beam = [end + s1 * 10.0 * side + np.array([x, 0.0, 0.0]) for s1 in (-1, 1)
                        for x in (xb - 10.0, xb - 3.0)]
                post = [end + s1 * 10.0 * side + s2 * 6.0 * d + np.array([x, 0.0, 0.0]) for s1 in (-1, 1)
                        for s2 in (-1, 1) for x in (xb - 10.0, xb + 1.0)]
                pts_b = " ".join(f"translate({v3(q)}) cube(0.01, center = true);" for q in beam)
                pts_p = " ".join(f"translate({v3(q)}) cube(0.01, center = true);" for q in post)
                pieces.append(lambda gg, Sp=Sp, pts_b=pts_b: f"hull() {{ {Sp} housing_box({gg}); {pts_b} }}")
                pieces.append(lambda gg, pts_p=pts_p: f"hull() {{ {pts_p} }}")
                part.negative.append(f"{Sp} housing_keepout({str(sp.yoke != 'u').lower()}, {sp.cable});")
        if sec == 0 and part.carries is not None:      # the hinge's yoke on the part beyond the cut
            sp = part.carries
            front = -sp.S_child[:3, 1]                 # ahead of the hinge servo: across the cut
            yp, _ = _yoke_pieces(sp, sp.S_child[:3, 0], front, mt.FRONT_SWEEP_MM + CASE_CLEAR)
            pieces += yp
            S = mm(sp.S_child)
            part.negative += [f"{S} arm_keepout(1, {list(HORN_HOLES_DRIVEN)});",
                              f"{S} arm_keepout(-1, {list(HORN_HOLES_IDLER)});"]
        part.pieces = pieces
        part.positive = [p("0") for p in pieces]

    # ------------------------------------------------- palm bolts, bay, mount

    def _keep_clear_yz(self, sec: int) -> List[np.ndarray]:
        """(y, z) outlines on plate `sec` that bolts and screws keep clear of:
        each of its fingers' base sweeps, and the hinge servo with its yoke."""
        out = []
        for i in range(len(self.hand.fingers)):
            if finger_section(self.hand, i) == sec:
                out.append(dv._hull2d(dv.base_sweep_mm(self.hand, i)[:, 1:]))
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
                part.negative.append(f"cyl_x(BORE_R, {-PLATE_HALF_MM - 1}, 0.01);")
            part.bolts = []
            # cable holes: each servo plug held in this plate reaches the back face
            for sp in part.holds:
                if not sp.clamped:
                    continue
                (x0, x1), (y0, y1), out = PLUG
                z0, z1 = (CASE_Z, CASE_Z + out) if sp.cable > 0 else (-CASE_Z - out, -CASE_Z)
                C = mt.box_corners(np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]), np.eye(3),
                                   np.array([(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2]))
                P = C @ sp.S_parent[:3, :3].T + sp.S_parent[:3, 3]
                if float(P[:, 0].min()) > -PLATE_HALF_MM - 0.5:
                    down = P.copy()
                    down[:, 0] = -PLATE_HALF_MM - 2.0
                    pts = " ".join(f"translate({v3(q)}) cube(0.01, center = true);" for q in np.concatenate([P, down]))
                    part.negative.append(f"hull() {{ {pts} }}")

    def bolt_candidates(self, sec: int) -> List[Tuple[float, float]]:
        """Where a bolt across plate `sec`'s halves may go before looking at
        the material: on a 3 mm grid inside the outline, off the column's
        bore and screws (the rendered plate then decides, `choose_bolts`)."""
        g = dv.hand_geometry(self.hand)
        outline = g.outlines[sec]
        screws = getattr(self.parts[("palm", sec)], "column_screws", [])
        lo, hi = outline.min(axis=0), outline.max(axis=0)
        out = []
        for y in np.arange(lo[0], hi[0] + 1e-9, 3.0):
            for z in np.arange(lo[1], hi[1] + 1e-9, 3.0):
                q = np.array([y, z])
                if self._dist_out(q, outline) > -PALM_BOLT_EDGE_MM:
                    continue
                if sec < 0 and float(np.linalg.norm(q)) < BORE_CLEAR_MM:
                    continue
                if any(float(np.linalg.norm(q - np.array(c))) < 7.5 for c in screws):
                    continue
                out.append((float(y), float(z)))
        return out

    def set_bolts(self, sec: int, bolts: List[Tuple[float, float]]) -> None:
        part = self.parts[("palm", sec)]
        part.negative = [n for n in part.negative if not n.startswith("palm_bolt(")]
        part.bolts = list(bolts)
        for y, z in bolts:
            part.negative.append(f"palm_bolt({y:.3f}, {z:.3f});")

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
        mount = Part("mount", ("palm", -1), B)
        mount.positive = [f"mount({cy:.3f}, {cz:.3f}, {[list(q) for q in sb]});"]
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
                parent.negative.append(f"{mm(Sp)} sweep_x({lo:.3f}, {hi:.3f}) {mm(to_child)} {pc('PART_CLEAR')}")
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
            grand.negative.append(f"{mm(sp.S_parent)} sweep_x2({_mat(Tn)}, {lo1:.3f}, {hi1:.3f}, {lo2:.3f}, "
                                  f"{hi2:.3f}) {mm(inv(nxt.S_child))} {pc('PART_CLEAR')}")
        # link j + 1 gives room to servo j's case, turning back about both joints
        Tb = inv(nxt.S_parent) @ sp.S_child          # servo j's frame in servo j + 1's
        for pc in _case_pieces(np.eye(4), sp.cable, sp.idler_horn):
            far.negative.append(f"{mm(nxt.S_child)} sweep_x2({_mat(Tb)}, {-hi2:.3f}, {-lo2:.3f}, {-hi1:.3f}, "
                                f"{-lo1:.3f}) {pc('CASE_CLEAR')}")

    # --------------------------------------------------------------- output

    def part_list(self) -> List[Part]:
        return list(self.parts.values()) + list(self.extras)
