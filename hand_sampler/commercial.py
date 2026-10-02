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
from collections import Counter
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

def _frame_from(axis: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``design_space._frame_from_axis`` with the roll reference supplied.

    The roll about a finger's own axis is gauge in this design space, absorbed
    into theta -- but only per finger. Across a ROW of fingers it is the thing
    that decides whether they flex in one plane or four, so the real hand needs
    a frame whose roll is tied to its palm rather than to GRASP_DIR, which means
    nothing in the vendor's coordinates.
    """
    a = axis / np.linalg.norm(axis)
    if np.linalg.norm(np.cross(a, ref)) < 1e-6:
        ref = np.eye(3)[int(np.argmin(np.abs(a)))]
    fe = np.cross(a, ref)
    fe /= np.linalg.norm(fe)
    return np.column_stack([a, np.cross(fe, a), fe])


def _row_frame(chains: list, idx: list, sign: float) -> np.ndarray:
    """ONE frame for every finger in the row.

    Fitting each finger in its own base frame -- which is what this did first --
    is correct per finger and wrong for the hand: each vendor finger leaves at a
    slightly different angle, so the same physical hinge axis lands on a
    different theta, and fingers that flex in one plane in reality end up flexing
    in four. Measured on sharpa and shadow, one finger per hand came out 45 deg
    out of plane and swept straight through its neighbour.

    The roll is tied to the palm: the row of bases gives the spread direction,
    and its cross product with the pointing direction is the palm normal, which
    is what GRASP_DIR means here. Its sign is not recoverable from geometry
    alone, so the caller tries both.
    """
    fwd = np.mean([_base_dir(chains[i]) for i in idx], axis=0)
    fwd /= np.linalg.norm(fwd)
    base = np.array([chains[i].positions[0] for i in idx])
    spread = np.linalg.svd(base - base.mean(axis=0))[2][0]
    n = np.cross(spread, fwd)
    return _frame_from(fwd, sign * n / max(float(np.linalg.norm(n)), 1e-12))


def _base_dir(chain: Chain) -> np.ndarray:
    """Which way the finger leaves its base.

    The FIRST non-degenerate link, not simply the first: sharpa and shadow place
    two joints at the same point to make a universal joint, so their base link
    has zero length and no direction at all. Normalising it gives a frame built
    from noise, and every finger sharing that frame inherits it.
    """
    pts = list(chain.positions) + [chain.tip]
    for i in range(len(pts) - 1):
        d = pts[i + 1] - pts[i]
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            return d / n
    return np.array([1.0, 0.0, 0.0])


def _snap_angle(x: float) -> float:
    return round(x / D.ANGLE_QUANTUM) * D.ANGLE_QUANTUM


def _fit_finger(chain: Chain, face: str, u: float, v: float, palm: D.Palm,
                curl: float, R_real: np.ndarray,
                pin_base: bool = True) -> tuple[D.Finger, list[str], float]:
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

    # expressed in the frame the caller supplies -- shared across a row, so
    # fingers that flex together in reality flex together here
    tgt_axis = [R_real.T @ a for a in chain.axes]
    tgt_dir = [R_real.T @ d for d in dirs]

    # a RELATIVE accumulator: the fit matches the finger's internal shape, and
    # the mount supplies the world frame when the finger is placed
    R_fit = np.eye(3)
    segments = []
    lo, hi = D.JOINT_LIMIT
    n_off = int(round((hi - lo) / D.ANGLE_QUANTUM))
    err = 0.0
    for i in range(n):
        R_prev, best = R_fit, None
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
            offs = (0.0,) if (i == 0 and pin_base) else tuple(
                _snap_angle((lo + m * D.ANGLE_QUANTUM) * curl)
                for m in range(n_off + 1))
            for off in offs:
                Rn = R_fit @ D.rodrigues(a_local, off)
                e_dir = 0.0 if i == 0 else 1.0 - float(np.dot(Rn[:, 0], tgt_dir[i]))
                err = 2.0 * e_axis + e_dir
                if best is None or err < best[0]:
                    best = (err, th, off, Rn)
        _, th, off, R_fit = best
        # WORST joint, not the sum: a variant that fixes one axis and spoils three
        # has a better sum and a worse hand. This is the quantity "do the axes
        # point the same way" actually asks about.
        err = max(err, 1.0 - abs(float(np.dot(
            R_prev @ D.axis_of(D.Joint(th, math.pi / 2)), tgt_axis[i]))))
        segments.append(D.Segment(D.Joint(th, math.pi / 2, off), lengths[i]))
    return D.Finger(D.Mount(face, u, v), tuple(segments)), notes, err


def _thumb_index(chains: list) -> int:
    """Which digit is the thumb: the one pointing furthest from the rest.

    DIRECTION, not base position. An earlier version took the base furthest off
    the line the other bases form, which sounds equivalent and is not: a thumb's
    base often sits close to that line, and on four of the five hands here the
    positional test picked an ordinary finger. Leap was the clearest -- its three
    fingers deviate 19.5 deg from the mean direction and its thumb 91, while by
    base position the thumb looked unremarkable, so the thumb was fitted into the
    row and a finger onto the thumb's face.

    Joint count would also work for sharpa and shadow, whose thumbs carry an
    extra joint, and fail for tesollo and wuji2 where every digit has four.
    """
    dirs = np.array([_base_dir(c) for c in chains])
    mean = dirs.mean(axis=0)
    mean /= max(float(np.linalg.norm(mean)), 1e-12)
    return int(np.argmax(np.arccos(np.clip(dirs @ mean, -1.0, 1.0))))


def _row_order(chains: list, thumb: int) -> list:
    """The non-thumb digits, index first and little finger last.

    Three things have to be right at once, and each was wrong in turn:

    * WHICH POINT. Not the chain's base. A digit carrying an extra proximal
      joint -- sharpa's pinky CMC, shadow's little-finger metacarpal -- starts
      its chain at a different depth from its neighbours, and shadow's lands
      65 mm behind the other knuckles. Chains are aligned from the TIP instead,
      so every digit is compared at the same anatomical joint.
    * WHICH AXIS. The principal direction through those knuckles, not distance
      from the thumb: on shadow and tesollo the digits are not equidistant and
      a distance sort scrambles them.
    * WHICH WAY. An SVD axis has an arbitrary sign, so the row came out mirrored
      about half the time. It is signed to point away from the thumb.

    Low ``v`` on +z is the side the thumb's -y face is on, so slot 0 is the
    digit nearest the thumb.
    """
    idx = [i for i in range(len(chains)) if i != thumb]
    modal = Counter(len(chains[i].positions) for i in idx).most_common(1)[0][0]
    knuckle = np.array([chains[i].positions[len(chains[i].positions) - modal]
                        for i in idx])
    centre = knuckle.mean(axis=0)
    d = np.linalg.svd(knuckle - centre)[2][0]
    if (centre - chains[thumb].positions[0]) @ d < 0:
        d = -d
    return [idx[k] for k in np.argsort((knuckle - centre) @ d)]


def _layout(n_row: int, palm: D.Palm) -> tuple[list, tuple]:
    """A row of mounts across +z, plus one opposed on -y.

    This is the arrangement every hand in the set actually has, so it is built
    rather than searched: a packer that merely satisfies the separation rule
    spreads fingers over three faces pointing three different ways, which is
    legal and looks nothing like a hand.
    """
    lo_v, hi_v = D.mount_uv_bounds("+z", palm)[2:]
    vs = ([0.5] if n_row == 1
          else [lo_v + i * (hi_v - lo_v) / (n_row - 1) for i in range(n_row)])
    return [("+z", v) for v in vs], ("-y", 0.5)


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

    # Wide enough for the row, on the grid, within the range.
    n_row = len(chains) - 1
    need = (n_row - 1) * D.MIN_MOUNT_SEPARATION + 2 * D.MOUNT_EDGE_MARGIN
    width = min(max(math.ceil(need / D.PALM_QUANTUM) * D.PALM_QUANTUM,
                    D.PALM_WIDTH_RANGE[0]), D.PALM_WIDTH_RANGE[1])
    # Length from the vendor, not a constant: it is how far the thumb sits from
    # the row, which is the only palm dimension the hand's own geometry fixes.
    # Width cannot be taken the same way -- vendors space fingers 20-25 mm and
    # MIN_MOUNT_SEPARATION is 35, so the row sets the width whatever they did.
    rowc = np.mean([chains[i].positions[0] for i in range(len(chains))
                    if i != _thumb_index(chains)], axis=0)
    reach = float(np.linalg.norm(chains[_thumb_index(chains)].positions[0] - rowc))
    length = min(max(round(reach / D.PALM_QUANTUM) * D.PALM_QUANTUM,
                     D.PALM_LENGTH_RANGE[0]), D.PALM_LENGTH_RANGE[1])
    if width < need - 1e-9:
        notes.append(f"row needs {need*1000:.0f} mm of palm, capped at {width*1000:.0f}")

    thumb = _thumb_index(chains)
    order = _row_order(chains, thumb)

    # Three nested searches, outermost first.
    #
    # LENGTH starts at the vendor's own thumb-to-row distance and grows only if
    # it has to. It usually has to: the vendor puts its thumb closer to the row
    # than 35 mm mount separation and 30 mm link clearance allow, so a palm that
    # is faithful in length leaves the thumb intersecting the fingers.
    #
    # SIGN is the palm normal, which the vendor geometry does not determine.
    #
    # CURL is relaxed last and least, because a row of parallel fingers curling
    # together stays clear -- a hand that needs heavy relaxation is usually
    # telling you its axes are wrong, not that its pose is impossible.
    best, grown = None, 0
    while best is None and length <= D.PALM_LENGTH_RANGE[1] + 1e-9:
        palm = D.Palm(D.PALM_THICKNESS, width, length)
        row_places, thumb_place = _layout(len(order), palm)
        for sign in (1.0, -1.0):
            R_row = _row_frame(chains, order, sign)
            R_thumb = _frame_from(_base_dir(chains[thumb]), R_row[:, 2])
            for curl in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
                fingers, fnotes = [], []
                for ci, (face, v), R in [(c, pl, R_row)
                                         for c, pl in zip(order, row_places)]:
                    f, fn, _ = _fit_finger(chains[ci], face, 0.5, v, palm, curl, R)
                    fingers.append(f)
                    fnotes += [f"finger {len(fingers)-1}: {m}" for m in fn]
                # The thumb's base offset is the one place pinning is wrong. A row
                # finger should leave perpendicular to its face; a thumb has to be
                # AIMED, and the base joint's zero offset is the grammar's only
                # mechanism for that. Which option wins is not predictable per
                # hand, so both are fitted and the lower axis error kept.
                # The thumb's base offset is the one place pinning is wrong. A
                # row finger should leave perpendicular to its face; a thumb has
                # to be AIMED, and the base joint's zero offset is the grammar's
                # only mechanism for that.
                pface, vv = thumb_place
                picked = None
                for pin in (True, False):
                    f, fn, e = _fit_finger(chains[thumb], pface, 0.5, vv, palm,
                                           curl, R_thumb, pin)
                    if validate_design.check(D.Hand(palm, tuple(fingers) + (f,))):
                        continue
                    if picked is None or e < picked[0]:
                        picked = (e, f, fn, pface, pin)
                if picked is None:
                    continue
                _, f, fn, pface, pin = picked
                fingers.append(f)
                fnotes += [f"thumb: {m}" for m in fn]
                if not pin:
                    fnotes.append("thumb aimed by its base offset")
                hand = D.Hand(palm, tuple(fingers))
                if not validate_design.check(hand):
                    fnotes.append(f"palm-normal sign {sign:+.0f}")
                    best = (curl, hand, fnotes)
                    break
            if best:
                break
        if best is None:
            length += D.PALM_QUANTUM
            grown += 1

    if best is None:
        notes.append("no palm length in range gave a valid hand")
        return hand, notes + fnotes
    curl, hand, fnotes = best
    if grown:
        notes.append(f"palm lengthened {grown * D.PALM_QUANTUM * 1000:.0f} mm past the "
                     f"vendor's thumb offset to clear the row")
    if curl < 1.0:
        notes.append(f"rest curl relaxed to {curl:.0%} to clear finger-finger contact")
    return hand, notes + fnotes


def fit_report() -> str:
    rows = []
    for name in HANDS:
        hand, notes = fit(name)
        ok = "ok" if not any(n.startswith("INVALID") for n in notes) else "INVALID"
        rows.append(f"{name:<9}{hand.n_fingers:>3}f {hand.n_joints:>3}j  {ok:<8}"
                    f"{len([n for n in notes if 'link' in n]):>3} links resized")
    return "\n".join(rows)
