"""LEAP as a design the grammar would produce.

LEAP is the one commercial hand this grammar targets. It fits because its own
geometry is already what the grammar says a hand is: every joint axis is a
coordinate direction, every fixed rotation a multiple of 90 degrees, and -- once
the kinematically free slides are taken out -- every link straight.

Two things about the vendor URDF have to be handled or the fit comes out wrong,
and both produced confident nonsense before being caught:

* The joint NUMBERING is actuator ID, not kinematic order. The index chain is
  ``palm -> j1 -> j0 -> j2 -> j3``. Follow parent/child, never the names.
* The joint ORIGINS sit at bracket corners, so the raw offset between two of
  them cuts the corner of an L and looks diagonal. Sliding a joint's origin
  along its OWN axis changes nothing a joint does -- rotating about a line does
  not care which point of the line is called the origin -- so the slides are
  free, and taking them out leaves straight links. ``_straighten`` solves for
  them.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from hand_sampler import design_space as D
from hand_sampler import validate_design

_URDF_DIR = (Path(__file__).resolve().parents[1]
             / "assets/urdf/unified_dynamics_commercial_hands")

HANDS: tuple[str, ...] = ("leap", "wuji2", "sharpa")
"""The vendor hands fitted into this grammar, in the order they were done.

LEAP first because it fits almost exactly -- its own geometry is already what
the grammar says a hand is. The others are harder and say so in their notes.
"""


def urdf_of(name: str) -> Path:
    return _URDF_DIR / name / f"{name}_left.urdf"


URDF = urdf_of("leap")

ROW_FACE = "+z"
THUMB_FACE = "-y"


@dataclass(frozen=True)
class Digit:
    """One finger of the vendor hand, in the vendor's own frame."""
    name: str
    joints: tuple[str, ...]
    pos: np.ndarray        # (n, 3) joint origins
    axis: np.ndarray       # (n, 3) unit hinge axes
    tip: np.ndarray        # (3,)


# --- reading the vendor hand ------------------------------------------------

def _read(path: Path) -> dict:
    root = ET.parse(path).getroot()
    out = {}
    for j in root.findall("joint"):
        o, a = j.find("origin"), j.find("axis")
        out[j.get("name")] = dict(
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            xyz=np.array([float(v) for v in (o.get("xyz", "0 0 0")).split()]),
            rpy=np.array([float(v) for v in (o.get("rpy", "0 0 0")).split()]),
            axis=np.array([float(v) for v in
                           ((a.get("xyz", "0 0 1") if a is not None else "0 0 1")).split()]),
            type=j.get("type"))
    return out


def digits(name_or_path: "str | Path" = "leap") -> list[Digit]:
    """Every digit, in KINEMATIC order, with the palm as root."""
    path = (urdf_of(name_or_path) if isinstance(name_or_path, str)
            else name_or_path)
    J = _read(path)
    by_parent: dict[str, list[str]] = {}
    for n, d in J.items():
        by_parent.setdefault(d["parent"], []).append(n)

    palm = max(by_parent, key=lambda k: sum(J[n]["type"] == "revolute"
                                            for n in by_parent[k]))
    out = []
    for root_joint in [n for n in by_parent[palm] if J[n]["type"] == "revolute"]:
        chain, cur = [], root_joint
        while True:
            chain.append(cur)
            nxt = [x for x in by_parent.get(J[cur]["child"], [])
                   if J[x]["type"] == "revolute"]
            if not nxt:
                break
            cur = nxt[0]
        T = np.eye(4)
        pos, axis = [], []
        for n in chain:
            d = J[n]
            M = np.eye(4)
            M[:3, :3] = R.from_euler("xyz", d["rpy"]).as_matrix()
            M[:3, 3] = d["xyz"]
            T = T @ M
            pos.append(T[:3, 3].copy())
            axis.append(T[:3, :3] @ d["axis"])
        # The fingertip can be SEVERAL fixed joints out, and a link can carry
        # more than one: SHARPA goes DP -> elastomer -> fingertip, and wuji2
        # hangs both a tip and a zero-offset sensor frame off its distal link.
        # Follow every fixed branch and keep the point furthest from the last
        # joint, which is the one that is actually a fingertip.
        tip = T[:3, 3].copy()
        stack = [(T, J[chain[-1]]["child"])]
        while stack:
            frame, link = stack.pop()
            for x in by_parent.get(link, []):
                if J[x]["type"] != "fixed":
                    continue
                d = J[x]
                M = np.eye(4)
                M[:3, :3] = R.from_euler("xyz", d["rpy"]).as_matrix()
                M[:3, 3] = d["xyz"]
                nxt = frame @ M
                if np.linalg.norm(nxt[:3, 3] - T[:3, 3]) > np.linalg.norm(tip - T[:3, 3]):
                    tip = nxt[:3, 3].copy()
                stack.append((nxt, d["child"]))
        out.append(Digit(root_joint, tuple(chain), np.array(pos), np.array(axis), tip))
    return out


# --- taking out the free slides ---------------------------------------------

def _straighten(d: Digit) -> Digit:
    """Slide each joint origin along its own axis until the links are straight.

    Kinematically exact: verified to 0.000 microns over random poses. What it
    changes is only which point of each axis LINE the frame is hung from.
    """
    P, A = d.pos, d.axis
    n = len(A)

    floor = D.MIN_LINK_LENGTH

    span0 = sum(float(np.linalg.norm(q - p))
                for p, q in zip(list(P) + [d.tip], list(P)[1:] + [d.tip]))

    def cost(s):
        Q = P + s[:, None] * A
        pts = list(Q) + [d.tip]
        c = 0.0
        # Keep the chain from folding back on itself. A 180 degree turn is
        # perfectly axis-aligned, so the straightness term alone is happy to
        # double a link back the way it came; what gives it away is the total
        # length, which only grows when the chain doubles back.
        total = sum(float(np.linalg.norm(pts[i + 1] - pts[i]))
                    for i in range(len(pts) - 1))
        c += 0.3 * (total / span0) ** 2
        for i in range(len(pts) - 1):
            v = pts[i + 1] - pts[i]
            L = float(np.linalg.norm(v))
            # A link under the floor is not buildable, and for the thumb there is
            # a choice: the slides can put its two base axes at ONE point (a
            # universal joint, one zero-length link) or hold them apart with a 90
            # degree turn between them. Both are the same knuckle; only the second
            # is spellable with every link above the floor, so it is what the
            # penalty picks. The weight is deliberately small: straightness wins
            # outright, and the floor only breaks ties between solutions that are
            # equally straight. LEAP's index has a 15 mm link that NO slide can
            # lengthen without bending the finger, so there the floor loses and
            # the snap below rounds it up to 20.
            if L < floor:
                c += 0.05 * (floor - L) ** 2 / floor ** 2
            if L < 1e-6:
                continue
            u = v / L
            c += float(np.sum(u ** 2) - np.max(u ** 2))   # angle only, length-free
        return c

    best = None
    for seed in range(16):
        s0 = np.random.default_rng(seed).uniform(-0.05, 0.05, n)
        r = minimize(cost, s0, bounds=[(-0.08, 0.08)] * n)
        if best is None or r.fun < best.fun:
            best = r
    return replace(d, pos=P + best.x[:, None] * A)


# --- the palm frame ---------------------------------------------------------

def _first_dir(d: Digit) -> np.ndarray:
    """Which way the digit leaves its base.

    Skips degenerate links: SHARPA's MCP_FE and MCP_AA are coincident, so its
    first link has no length and no direction. Taking it anyway divided by zero
    and handed every digit a NaN, which made the thumb search pick whichever
    digit happened to come first.
    """
    for q in list(d.pos[1:]) + [d.tip]:
        v = q - d.pos[0]
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            return v / n
    return np.array([1.0, 0.0, 0.0])


def _split(ds: list[Digit]) -> tuple[list[Digit], Digit]:
    """The row, and the thumb -- the digit pointing least like the others."""
    dirs = [_first_dir(d) for d in ds]
    score = [sum(float(a @ b) for b in dirs) for a in dirs]
    thumb = int(np.argmin(score))
    return [d for i, d in enumerate(ds) if i != thumb], ds[thumb]


def _palm_axes(row: list[Digit]) -> np.ndarray:
    """Columns ``(x, y, z)`` of the grammar's palm frame, in vendor coordinates.

    ``z`` is forward, along the row's own pointing direction. ``y`` runs through
    the knuckles -- the ROW's line, taken from the base positions and squared to
    z, because the knuckle line is what a palm's width means. ``x`` closes it.
    """
    z = np.mean([_first_dir(d) for d in row], axis=0)
    z = z / float(np.linalg.norm(z))
    base = np.array([d.pos[0] for d in row])
    y = base[-1] - base[0]
    y = y - z * float(y @ z)
    y = y / float(np.linalg.norm(y))
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def _lean_for(direction: np.ndarray) -> int:
    """The lean closest to ``direction``, a unit vector in the parent's frame."""
    return int(np.argmax([float(direction @ np.asarray(v)) for v in D.LEANS]))


def _kind_for(axis: np.ndarray) -> int:
    """Which of the three kinds of hinge ``axis`` is, in the joint's own frame.

    An axis and its negative are the same hinge, so the comparison is on the
    absolute dot product: there is one spelling per joint and no sign to pick.
    """
    return max(D.JOINT_KINDS,
               key=lambda k: abs(float(axis @ D.axis_of(D.Joint(k)))))


# --- the fit ----------------------------------------------------------------

def _snap_length(metres: float) -> float:
    q = D.LINK_QUANTUM
    v = round(metres / q) * q
    return min(max(v, D.MIN_LINK_LENGTH), D.MAX_LINK_LENGTH)


def _v_for(face: str, palm: D.Palm, target: np.ndarray) -> float:
    """The ``v`` on ``face`` whose mount sits nearest ``target``, clamped to the
    face's own margins."""
    centre, _, _, t_v, _, span_v = D.face_frame(face, palm)
    v = 0.5 + float((target - centre) @ t_v) / span_v
    _, _, lo, hi = D.mount_uv_bounds(face, palm)
    return min(max(v, lo), hi)


def _push_apart(ys: np.ndarray, floor: float) -> np.ndarray:
    """Move values apart until adjacent ones clear ``floor``, as little as possible.

    Sweep up enforcing the gap, sweep back down, then re-centre on where the row
    started. Order is preserved and a row already clear of the floor is returned
    untouched, so only hands that need spreading get spread.
    """
    order = np.argsort(ys)
    v = ys[order].astype(float).copy()
    for i in range(1, len(v)):
        v[i] = max(v[i], v[i - 1] + floor)
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], v[i + 1] - floor)
    v += float(np.mean(ys)) - float(np.mean(v))
    out = np.empty_like(v)
    out[order] = v
    return out


def fit(name: str = "leap", spread: bool = True) -> tuple[D.Hand, list[str]]:
    """A vendor hand as a Hand, plus notes on every way it is not the vendor's."""
    ds = [_straighten(d) for d in digits(name)]
    row, thumb = _split(ds)
    M = _palm_axes(row)

    # into the palm frame, with the row's knuckle plane forward and the row
    # centred across the width
    base = np.array([d.pos[0] for d in row])
    q_base = base @ M
    centre_y = float(np.mean(q_base[:, 1]))
    front_z = float(np.mean(q_base[:, 2]))

    # The knuckles may sit closer than a motor allows: wuji2 packs its row at
    # 19-24 mm centres and SHARPA at 17-20, against a 35 mm floor set by two 30
    # mm capsules plus clearance. Spreading the row is the only way to make such
    # a hand buildable, and it is a real distortion -- the notes say how much.
    want_y = q_base[:, 1].copy()
    row_y = _push_apart(want_y, D.MIN_MOUNT_SEPARATION) if spread else want_y
    centre_y = float(np.mean(row_y))
    half = float(np.max(np.abs(row_y - centre_y)))
    width = D.PALM_QUANTUM * math.ceil(
        (2.0 * (half + D.MOUNT_EDGE_MARGIN)) / D.PALM_QUANTUM)
    width = min(max(width, D.PALM_WIDTH_RANGE[0]), D.PALM_WIDTH_RANGE[1])

    q_thumb = thumb.pos[0] @ M
    depth = front_z - float(q_thumb[2])
    length = D.PALM_QUANTUM * math.ceil(
        (depth + 2 * D.MOUNT_EDGE_MARGIN) / D.PALM_QUANTUM)
    length = min(max(length, D.PALM_LENGTH_RANGE[0]), D.PALM_LENGTH_RANGE[1])

    palm = D.Palm(D.PALM_THICKNESS, width, length)
    # origin: the row's knuckle plane lands on the +z face, the row on the midline
    origin = np.array([0.0, centre_y, front_z - length])

    def to_palm(p):
        return (p @ M) - origin

    notes: list[str] = []
    moved = {id(d): float(row_y[i] - want_y[i]) for i, d in enumerate(row)}
    fingers = []
    for d, face in [(x, ROW_FACE) for x in row] + [(thumb, THUMB_FACE)]:
        pts = [to_palm(p) for p in d.pos] + [to_palm(d.tip)]
        axes = [M.T @ a for a in d.axis]
        want_base = pts[0]
        shift = moved.get(id(d), 0.0)
        if abs(shift) > 1e-4:
            want_base = want_base + np.array([0.0, shift, 0.0])
            notes.append(f"{d.name}: knuckle moved {abs(shift)*1000:.0f} mm "
                         f"across the row to clear the "
                         f"{D.MIN_MOUNT_SEPARATION*1000:.0f} mm motor floor")
        v = _v_for(face, palm, want_base)
        mount = D.Mount(face, 0.5, v)
        got_base = D.mount_position(mount, palm)
        slip = float(np.linalg.norm(got_base - want_base)) * 1000
        if slip > 1.0:
            notes.append(f"{d.name}: base {slip:.0f} mm off -- a mount lives on a "
                         f"palm FACE and this one does not")

        _, R = D.mount_frame(mount, palm)
        segs = []
        for i in range(len(d.pos)):
            v_link = pts[i + 1] - pts[i]
            L = float(np.linalg.norm(v_link))
            lean = _lean_for(R.T @ (v_link / max(L, 1e-12)))
            R = R @ D.lean_rot(lean)
            kind = _kind_for(R.T @ axes[i])
            snapped = _snap_length(L)
            if abs(snapped - L) > D.LINK_QUANTUM / 2 + 1e-9:
                notes.append(f"{d.name}: link {i} {L*1000:.1f} -> {snapped*1000:.0f} mm "
                             f"(the {D.MIN_LINK_LENGTH*1000:.0f} mm floor)")
            segs.append(D.Segment(D.Joint(kind), snapped, lean=lean))
        fingers.append(D.Finger(mount, tuple(segs)))

    hand = D.Hand(palm, tuple(fingers))
    for reason in validate_design.check(hand):
        notes.append(f"NOT A LEGAL DESIGN: {reason}")
    return hand, notes
