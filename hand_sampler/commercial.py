"""Fit commercial hands into the design space.

The point is not a replica -- the grammar has one fixed motor, a quantised grid,
three mount faces and a box palm, and no real hand obeys all four. The point is
the closest hand the grammar CAN express, so the vendors' designs can be run as
baselines against generated ones in the same simulator, through the same build
path, with the same articulation envelope.

Each finger is fitted in its OWN base frame, so nothing depends on the vendor's
global convention: the roll about a finger's axis is gauge in this design space
(absorbed into ``theta``), and using the same frame recipe for the real hand and
the generated one makes the fit independent of it.

What the fit cannot preserve is reported per hand rather than hidden -- see
``fit_report``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hand_sampler import design_space as D
from hand_sampler import validate_design

HANDS: tuple[str, ...] = ("sharpa", "shadow", "wuji2", "tesollo", "leap")
_URDF_DIR = Path(__file__).resolve().parents[1] / "assets/urdf/unified_dynamics_commercial_hands"


# --- reading a URDF ---------------------------------------------------------

def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp,     cp * sr,                cp * cr]])


@dataclass(frozen=True)
class Chain:
    """One finger of a real hand, at its zero pose, in the URDF root frame."""
    positions: tuple          # joint origins, one per joint
    axes: tuple               # hinge axis per joint, unit
    tip: np.ndarray           # end of the last link


def read_chains(path: Path) -> list[Chain]:
    """Every serial chain of movable joints hanging off the root.

    Transforms are composed through FIXED joints too: a vendor that inserts one
    between two movable joints would otherwise have its link lengths measured
    from the wrong frame.
    """
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0").split()
                                           if o is not None else "0 0 0".split())])
        rpy = [float(v) for v in (o.get("rpy", "0 0 0").split()
                                  if o is not None else "0 0 0".split())]
        ax = j.find("axis")
        axis = np.array([float(v) for v in (ax.get("xyz", "1 0 0").split()
                                            if ax is not None else [1, 0, 0])])
        joints[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"),
                                     child=j.find("child").get("link"), xyz=xyz,
                                     R=_rpy(*rpy), axis=axis)

    by_parent: dict[str, list[str]] = {}
    for n, j in joints.items():
        by_parent.setdefault(j["parent"], []).append(n)
    links = {j["child"] for j in joints.values()} | {j["parent"] for j in joints.values()}
    base = next(l for l in links if l not in {j["child"] for j in joints.values()})

    out: list[Chain] = []
    movable = ("revolute", "continuous", "prismatic")

    def walk(link, T, pos, axes):
        kids = by_parent.get(link, [])
        if not kids and pos:
            out.append(Chain(tuple(pos), tuple(axes), T[:3, 3].copy()))
            return
        for n in kids:
            j = joints[n]
            T2 = T.copy()
            T2[:3, 3] = T[:3, 3] + T[:3, :3] @ j["xyz"]
            T2[:3, :3] = T[:3, :3] @ j["R"]
            if j["type"] in movable:
                a = T2[:3, :3] @ j["axis"]
                walk(j["child"], T2, pos + [T2[:3, 3].copy()],
                     axes + [a / np.linalg.norm(a)])
            else:
                walk(j["child"], T2, pos, axes)
        if not kids and not pos:
            return

    walk(base, np.eye(4), [], [])
    # one chain per finger: keep the deepest reached through each root joint
    best: dict[tuple, Chain] = {}
    for c in out:
        if not c.positions:
            continue
        key = tuple(np.round(c.positions[0], 6))
        if key not in best or len(c.positions) > len(best[key].positions):
            best[key] = c
    return list(best.values())


# --- fitting ----------------------------------------------------------------

def _snap_angle(x: float) -> float:
    return round(x / D.ANGLE_QUANTUM) * D.ANGLE_QUANTUM


def _fit_finger(chain: Chain, face: str, v: float, palm: D.Palm,
                curl: float = 1.0) -> tuple[D.Finger, list[str]]:
    """Lengths snapped to the grid; theta and offset chosen per joint by search.

    Sequential and greedy: joint i's frame depends on every joint before it, so
    the frames are built as the fit proceeds. At each joint the whole (theta,
    offset) grid is scored -- 12 x 13 pairs, cheap -- against two targets, the
    real hinge axis and the real direction of the link leaving that joint.
    """
    notes: list[str] = []
    n = len(chain.positions)
    pts = list(chain.positions) + [chain.tip]
    raw = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(n)]
    dirs = []
    for i in range(n):
        d = pts[i + 1] - pts[i]
        nd = float(np.linalg.norm(d))
        dirs.append(d / nd if nd > 1e-9 else chain.axes[i])

    lengths = []
    for i, L in enumerate(raw):
        floor = D.MIN_DISTAL_LINK_LENGTH if i == n - 1 else D.MIN_LINK_LENGTH
        q = min(max(round(L / D.LINK_QUANTUM) * D.LINK_QUANTUM, floor), D.MAX_LINK_LENGTH)
        if abs(q - L) > D.LINK_QUANTUM:
            notes.append(f"link {i}: {L*1000:.1f} -> {q*1000:.0f} mm")
        lengths.append(q)

    # the real chain expressed in its own base frame, so the vendor's global
    # convention drops out
    R_real = D._frame_from_axis(dirs[0])
    tgt_axis = [R_real.T @ a for a in chain.axes]
    tgt_dir = [R_real.T @ d for d in dirs]

    # a RELATIVE accumulator: the fit matches the finger's internal shape, and
    # the mount supplies the world frame when the finger is placed
    R_fit = np.eye(3)
    segments = []
    lo, hi = D.JOINT_LIMIT
    n_off = int(round((hi - lo) / D.ANGLE_QUANTUM))
    for i in range(n):
        best = None
        for k in range(int(round(math.pi / D.ANGLE_QUANTUM))):
            th = k * D.ANGLE_QUANTUM
            a_local = D.axis_of(D.Joint(th, math.pi / 2))
            e_axis = 1.0 - abs(float(np.dot(R_fit @ a_local, tgt_axis[i])))
            # The BASE joint's offset is not a property of the finger: it aims the
            # whole finger, and where it should aim is decided by the face we mount
            # it on, not by where the vendor's palm happened to point. Fitting it
            # reproduces the vendor's rest pose on OUR layout, which lands fingers
            # inside one another. Every other offset is the finger's own curl and
            # is fitted.
            offs = (0.0,) if i == 0 else tuple(
                _snap_angle((lo + m * D.ANGLE_QUANTUM) * curl)
                for m in range(n_off + 1))
            for off in offs:
                Rn = R_fit @ D.rodrigues(a_local, off)
                e_dir = 0.0 if i == 0 else 1.0 - float(np.dot(Rn[:, 0], tgt_dir[i]))
                err = 2.0 * e_axis + e_dir
                if best is None or err < best[0]:
                    best = (err, th, off, Rn)
        _, th, off, R_fit = best
        segments.append(D.Segment(D.Joint(th, math.pi / 2, off), lengths[i]))
    return D.Finger(D.Mount(face, 0.5, v), tuple(segments)), notes


def _slots(palm: D.Palm) -> dict:
    """Legal mount positions per face: the v grid the separation actually allows.

    ``u`` is pinned to the midplane, so a face is a line, and the mounts on it are
    whatever fits between the two edge margins at MIN_MOUNT_SEPARATION.
    """
    out = {}
    for face in D.FINGER_FACES:
        _, _, _, _, _, span_v = D.face_frame(face, palm)
        lo, hi = D.mount_uv_bounds(face, palm)[2:]
        usable = (hi - lo) * span_v
        n = int(usable // D.MIN_MOUNT_SEPARATION) + 1
        out[face] = [lo if n == 1 else lo + i * (hi - lo) / (n - 1) for i in range(n)]
    return out


def _layout(k: int, palm: D.Palm) -> list[tuple[str, float]]:
    """Where k fingers mount: one opposed on -y, the rest filling +z then +y.

    A thumb-and-fingers arrangement, not the vendor's exact placement -- mounts
    here are confined to three faces, a midplane and a 35 mm separation, so an
    exact copy is not available at any palm size. Candidates are taken in that
    order and kept only if they clear every mount already placed, so the result
    is legal by construction rather than by retry.
    """
    slots = _slots(palm)
    order = ([("-y", v) for v in slots["-y"][len(slots["-y"]) // 2:]]
             + [("+z", v) for v in slots["+z"]]
             + [("+y", v) for v in slots["+y"]]
             + [("-y", v) for v in slots["-y"]])
    placed: list[tuple[str, float]] = []
    pos: list[np.ndarray] = []
    for face, v in order:
        p = D.mount_position(D.Mount(face, 0.5, v), palm)
        if all(np.linalg.norm(p - q) >= D.MIN_MOUNT_SEPARATION - 1e-9 for q in pos):
            placed.append((face, v)); pos.append(p)
        if len(placed) == k:
            break
    return placed


def fit(name: str) -> tuple[D.Hand, list[str]]:
    """The closest hand in the design space, and what the fit cost."""
    chains = read_chains(_URDF_DIR / name / f"{name}_left.urdf")
    chains.sort(key=lambda c: -len(c.positions))
    notes: list[str] = []

    if len(chains) > D.MAX_FINGERS:
        notes.append(f"{len(chains)} fingers -> {D.MAX_FINGERS}")
        chains = chains[:D.MAX_FINGERS]
    trimmed = [c for c in chains if len(c.positions) > D.MAX_JOINTS_PER_FINGER]
    if trimmed:
        notes.append(f"{len(trimmed)} finger(s) truncated to "
                     f"{D.MAX_JOINTS_PER_FINGER} joints")
    chains = [Chain(c.positions[:D.MAX_JOINTS_PER_FINGER],
                    c.axes[:D.MAX_JOINTS_PER_FINGER],
                    c.positions[D.MAX_JOINTS_PER_FINGER]
                    if len(c.positions) > D.MAX_JOINTS_PER_FINGER else c.tip)
              for c in chains]

    palm = D.Palm(D.PALM_THICKNESS, 0.100, 0.100)
    places = _layout(len(chains), palm)
    if len(places) < len(chains):
        notes.append(f"palm holds only {len(places)} mounts; "
                     f"{len(chains) - len(places)} finger(s) dropped")
        chains = chains[:len(places)]

    # Mount separation is a rule about where fingers START. Two fingers 35 mm
    # apart whose links curl inward still converge, and the validator checks
    # every link pair, not just the proximal ones. So the curl is relaxed only
    # as far as it has to be, and how far is reported.
    for curl in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
        fingers, fnotes = [], []
        for c, (face, v) in zip(chains, places):
            f, fn = _fit_finger(c, face, v, palm, curl)
            fingers.append(f)
            fnotes += [f"finger {len(fingers)-1}: {m}" for m in fn]
        hand = D.Hand(palm, tuple(fingers))
        bad = validate_design.check(hand)
        if not bad:
            if curl < 1.0:
                notes.append(f"rest curl relaxed to {curl:.0%} to clear "
                             f"finger-finger contact")
            return hand, notes + fnotes
    notes.append(f"INVALID even with no curl: {bad[0]}")
    return hand, notes + fnotes


def fit_report() -> str:
    rows = []
    for name in HANDS:
        hand, notes = fit(name)
        ok = "ok" if not any(n.startswith("INVALID") for n in notes) else "INVALID"
        rows.append(f"{name:<9}{hand.n_fingers:>3}f {hand.n_joints:>3}j  {ok:<8}"
                    f"{len([n for n in notes if 'link' in n]):>3} links resized")
    return "\n".join(rows)
