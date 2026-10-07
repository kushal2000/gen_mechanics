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

HANDS: tuple[str, ...] = ("leap", "wuji2", "midas", "allegro")
"""The vendor hands fitted into this grammar, in the order they were done.

LEAP first because it fits almost exactly -- its own geometry is already what
the grammar says a hand is. wuji2 is harder and says so in its notes. Allegro
fits best of all by base placement, 1 to 3 mm, because it IS a stack of servos:
three identical straight fingers, no bracket anywhere in the row.

SHARPA was fitted and then dropped. It needed the most distortion of any hand
tried: knuckles at 17-20 mm centres spread to 35, joint axes 7.7 degrees off a
coordinate direction at the median and 47.6 at worst, and a coincident MCP pair
whose first link had to be invented at 20 mm. The result validated but did not
look like SHARPA. hand_sampler.sharpa_capsule is the hand-built stand-in that
the trained runs use; fitting the real one is unfinished work, not a solved
problem.
"""


def urdf_of(name: str) -> Path:
    """Most vendors here ship a left hand; MIDAS declares no handedness."""
    for stem in (f"{name}_left", name):
        p = _URDF_DIR / name / f"{stem}.urdf"
        if p.exists():
            return p
    raise FileNotFoundError(f"no URDF for {name!r} under {_URDF_DIR}")


TIP_FALLBACK: dict[str, dict[str, tuple[float, float, float]]] = {
    "midas": {
        "thumb_dip_joint": (-0.004, 0.023, -0.017),
        "index_dip_joint": (0.0016, 0.021, -0.0085),
        "middle_dip_joint": (0.0016, 0.021, -0.0085),
        "ring_dip_joint": (0.0016, 0.021, -0.0085),
    },
}
"""Where a finger ENDS, for vendors whose URDF does not say.

MIDAS's chains stop at the last joint -- no tip link, no fixed frame -- so a
finger would come out with a zero-length distal link. These offsets are the
contact geometry from the vendor's own MuJoCo model, which is where it says the
finger touches things: the `fingertip_contact` class for the three fingers and
`thumb_dip_contact` for the thumb, both in the distal joint's frame.

Kept here rather than patched into the vendored URDF so that file stays
byte-identical to upstream. Estimating a tip from the link's centre of mass was
tried instead and is not good enough -- against the hands that DO carry tip
frames it is 14 mm short on wuji2 and 11 mm long on LEAP.
"""


URDF = urdf_of("leap")



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


def _longest_chain(J: dict, by_parent: dict, root: str) -> list[str]:
    """The finger, when a link carries more than one revolute child.

    MIDAS hangs a four-bar off each MCP pitch link, so that link has two
    revolute children -- the PIP, which continues the finger, and a linkage
    joint that closes a loop and goes nowhere. Taking the first child followed
    whichever the file happened to list first. The finger is the longest way
    down, and a closing linkage is always the shorter one.
    """
    best: list[str] = []
    stack = [[root]]
    while stack:
        path = stack.pop()
        kids = [x for x in by_parent.get(J[path[-1]]["child"], [])
                if J[x]["type"] == "revolute" and x not in path]
        if not kids:
            if len(path) > len(best):
                best = path
            continue
        for k in kids:
            stack.append(path + [k])
    return best


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
        chain = _longest_chain(J, by_parent, root_joint)
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
        fallback = TIP_FALLBACK.get(
            path.parent.name, {}).get(chain[-1])
        if fallback is not None:
            tip = (T @ np.append(np.asarray(fallback, dtype=float), 1.0))[:3]
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

def _worst_kink(d: Digit) -> float:
    """The sharpest turn between two consecutive links, in radians.

    How bent the chain IS, as against how far its links sit from a coordinate
    axis -- which is what _straighten's cost term measures, and not the same
    thing at all.
    """
    pts = list(d.pos) + [d.tip]
    vs = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
    out = 0.0
    for a, b in zip(vs, vs[1:]):
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            continue
        out = max(out, math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))
    return out


def _straighten(d: Digit) -> Digit:
    """Slide each joint origin along its own axis until the links are straight.

    Kinematically exact: verified to 0.000 microns over random poses. What it
    changes is only which point of each axis LINE the frame is hung from.

    A chain that is ALREADY straight is returned untouched. The cost below is
    axis ALIGNMENT -- how near each link lies to a coordinate axis of the
    digit's own frame -- which is the right proxy for an L-bracket and the wrong
    one for a finger that is already a straight rod set at an ANGLE. wuji2's
    whole row is exactly that, each finger straight to within 4.7 degrees but
    mounted at -9, 0, +8 and +15, and the slide read the angle as something to
    remove: it bent the row to as much as 16 degrees to buy 0.4 mm of first
    link. Worse, the fit reads a mount's FACING off this first link, so the row
    came out parallel and the hand lost the splay that is the first thing you
    see on the real one.

    The threshold is the grammar's own. Under half a LEAN_QUANTUM a chain
    already spells as straight, so a slide has nothing to win and whatever it
    does is loss; above it there is a real bend to place, and sliding is what
    finds the straight description of it -- LEAP's 79 degree brackets come down
    to 0.6 this way. Nothing sits near the line: wuji2's row is at 4.7 degrees
    and the next digit up, MIDAS's fingers, at 29.6.

    Deliberately NOT the wider rule "keep whichever is less bent". That one also
    refuses the slide on MIDAS's thumb -- 55 degrees raw against 69 slid -- and
    costs its fingertip 12 mm, because at that angle what decides is how the
    chain lands on the lean grid, not how bent it is.

    Refusing is always safe: both descriptions are the same mechanism.
    """
    if _worst_kink(d) < D.LEAN_QUANTUM / 2.0:
        return d

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

def _snap(value: float, quantum: float) -> float:
    return round(value / quantum) * quantum


def _snap_length(metres: float) -> float:
    q = D.LINK_QUANTUM
    v = round(metres / q) * q
    return min(max(v, D.MIN_LINK_LENGTH), D.MAX_LINK_LENGTH)


def _separate_on_grid(mounts: list, floor: float, tries: int = 400) -> list:
    """Nudge snapped mounts apart until every pair clears ``floor``.

    Done AFTER snapping, not before, because the snap itself can close a gap:
    a bearing rounded to 15 degrees moves a base by r times 0.13, which is 6.5 mm
    out at r = 50. Pushing in continuous space and then snapping left wuji2 and
    MIDAS a few tenths short of the floor.

    Each step takes the closest offending pair and moves the OUTER one of the
    two one grid step, in whichever of the four directions helps most and stays
    legal. Small moves, and only where needed.
    """
    out = list(mounts)
    q = D.PALM_QUANTUM
    for _ in range(tries):
        pos = [D.mount_position(m) for m in out]
        worst, pair = None, None
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                d = float(np.linalg.norm(pos[i] - pos[j]))
                if d < floor and (worst is None or d < worst):
                    worst, pair = d, (i, j)
        if pair is None:
            return out
        i, j = pair
        k = i if out[i].radius >= out[j].radius else j      # move the outer one
        best = None
        for dy, dz in ((q, 0.0), (-q, 0.0), (0.0, q), (0.0, -q),
                       (q, q), (q, -q), (-q, q), (-q, -q)):
            cand = replace(out[k], y=out[k].y + dy, z=out[k].z + dz)
            if not (D.PALM_MIN_RADIUS - 1e-9 <= cand.radius
                    <= D.MAX_MOUNT_RADIUS + 1e-9):
                continue
            trial = list(out)
            trial[k] = cand
            tp = [D.mount_position(m) for m in trial]
            gap = min(float(np.linalg.norm(tp[x] - tp[y]))
                      for x in range(len(trial)) for y in range(x + 1, len(trial)))
            if best is None or gap > best[0]:
                best = (gap, cand)
        if best is None:
            return out
        out[k] = best[1]
    return out


def _push_apart_2d(pts: np.ndarray, floor: float, rounds: int = 200) -> np.ndarray:
    """Spread points in the plane until every pair clears ``floor``.

    Relaxation rather than a row sweep: a radial palm has no row to push along,
    so each pair that is too close pushes both of its points apart along the
    line between them, repeatedly, until none is. The centroid is restored each
    round so the hand does not drift off its own centre.
    """
    q = pts.astype(float).copy()
    centre = q.mean(axis=0)
    for _ in range(rounds):
        worst = 0.0
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                v = q[j] - q[i]
                d = float(np.linalg.norm(v))
                if d >= floor:
                    continue
                if d < 1e-9:              # exactly coincident: push along y
                    v, d = np.array([1.0, 0.0]), 1e-9
                push = (floor - d) / 2.0
                step = v / d * push
                q[i] -= step
                q[j] += step
                worst = max(worst, push)
        q += centre - q.mean(axis=0)
        if worst < 1e-9:
            break
    return q


def fit(name: str = "leap", spread: bool = True) -> tuple[D.Hand, list[str]]:
    """A vendor hand as a Hand, plus notes on every way it is not the vendor's."""
    ds = [_straighten(d) for d in digits(name)]
    row, thumb = _split(ds)
    M = _palm_axes(row)

    # Into the palm frame, with the centre of the digit bases as the origin:
    # a palm has no faces to anchor to any more, so the centre is simply where
    # the fingers balance.
    base = np.array([d.pos[0] for d in (row + [thumb])]) @ M
    origin = base.mean(axis=0)
    origin[0] = 0.0                       # the plate's own plane

    def to_palm(p):
        return (p @ M) - origin

    # A base may sit closer to another than a motor allows. Push the whole set
    # apart in the plane rather than along a row axis, because a radial palm has
    # no row to push along.
    want = np.array([to_palm(d.pos[0])[1:] for d in (row + [thumb])])
    digits_ = row + [thumb]
    prepared = [( d, [to_palm(q) for q in d.pos] + [to_palm(d.tip)],
                  [M.T @ a for a in d.axis] ) for d in digits_]

    def place(floor: float) -> tuple[list, list[D.Mount], np.ndarray]:
        """Mounts for a given separation floor, snapped and then separated."""
        spread_pts = _push_apart_2d(want, floor + 1e-4) if spread else want
        raw = []
        for k, (d, pts, _axes) in enumerate(prepared):
            y, z = float(spread_pts[k][0]), float(spread_pts[k][1])
            gy, gz = _snap(y, D.PALM_QUANTUM), _snap(z, D.PALM_QUANTUM)
            # The ring is a bound on the distance, so a snap that lands outside
            # it is pushed back along its own radius and re-snapped.
            r = math.hypot(gy, gz)
            if r < D.PALM_MIN_RADIUS or r > D.MAX_MOUNT_RADIUS:
                onto = min(max(r, D.PALM_MIN_RADIUS), D.MAX_MOUNT_RADIUS)
                scale = onto / max(r, 1e-12)
                gy = _snap(gy * scale, D.PALM_QUANTUM)
                gz = _snap(gz * scale, D.PALM_QUANTUM)
            out_dir = pts[1] - pts[0] if len(pts) > 1 else np.array([0.0, y, z])
            if float(np.hypot(out_dir[1], out_dir[2])) < 1e-9:
                out_dir = np.array([0.0, y, z])
            facing = _snap(D.bearing_of(float(out_dir[1]), float(out_dir[2])),
                           D.ANGLE_QUANTUM) % (2.0 * math.pi)
            raw.append(D.Mount(y=gy, z=gz, facing=facing))
        return raw, (_separate_on_grid(raw, floor) if spread else raw), spread_pts

    def build(mounts: list) -> tuple[D.Hand, list[str]]:
        notes: list[str] = []
        fingers = []
        for (d, pts, axes), mount in zip(prepared, mounts):
            _, R = D.mount_frame(mount)
            segs = []
            for i in range(len(d.pos)):
                v_link = pts[i + 1] - pts[i]
                L = float(np.linalg.norm(v_link))
                lean = _lean_for(R.T @ (v_link / max(L, 1e-12)))
                R = R @ D.lean_rot(lean)
                kind = _kind_for(R.T @ axes[i])
                snapped = _snap_length(L)
                if abs(snapped - L) > D.LINK_QUANTUM / 2 + 1e-9:
                    notes.append(f"{d.name}: link {i} {L*1000:.1f} -> "
                                 f"{snapped*1000:.0f} mm "
                                 f"(the {D.MIN_LINK_LENGTH*1000:.0f} mm floor)")
                segs.append(D.Segment(D.Joint(kind), snapped, lean=lean))
            fingers.append(D.Finger(mount, tuple(segs)))
        return D.Hand(D.Palm(D.PALM_THICKNESS), tuple(fingers)), notes

    # A margin ladder. MIN_MOUNT_SEPARATION keeps two PARALLEL base capsules
    # apart, which is not enough when a hand splays its fingers -- wuji2's links
    # still meet at 27 mm with its bases a legal 35 apart. Widen the floor until
    # the whole hand clears, and stop at the first one that does.
    best = None
    for extra in (0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030):
        floor = D.MIN_MOUNT_SEPARATION + extra
        raw, mounts, spread_pts = place(floor)
        hand, notes = build(mounts)
        reasons = validate_design.check(hand)
        if best is None:
            best = (hand, notes, reasons, raw, mounts, spread_pts, floor)
        if not reasons:
            best = (hand, notes, reasons, raw, mounts, spread_pts, floor)
            break

    hand, notes, reasons, raw, mounts, spread_pts, floor = best
    if any("capsules intersect" in r for r in reasons):
        notes.insert(0, f"this hand's own LINKS pass closer than two "
                        f"{2*D.CAPSULE_RADIUS*1000:.0f} mm capsules allow, so no "
                        f"spreading of the BASES makes it buildable with this "
                        f"motor")
    if floor > D.MIN_MOUNT_SEPARATION + 1e-9:
        notes.insert(0, f"bases spread to {floor*1000:.0f} mm rather than the "
                        f"{D.MIN_MOUNT_SEPARATION*1000:.0f} mm floor: at the floor "
                        f"this hand's LINKS still met")
    for k, ((d, _pts, _axes), mount) in enumerate(zip(prepared, mounts)):
        # `want` is about the base centroid, which is where PALM_CENTRE
        # goes -- so take the standoff back out before comparing.
        here = D.mount_position(mount)[1:] - D.PALM_CENTRE[1:]
        slip = float(np.linalg.norm(here - want[k])) * 1000
        if slip > 1.0:
            notes.append(f"{d.name}: base {slip:.0f} mm from where the vendor "
                         f"puts it, after the grid and the motor floor")
        # Every mount is on the palm's midplane, so whatever the vendor had
        # ACROSS it is dropped -- silently, until hand_sampler.overlay_fits drew
        # the two hands together. LEAP's row is 6 mm off it, MIDAS's 10, and
        # Allegro's thumb 18.
        off = abs(float(to_palm(d.pos[0])[0])) * 1000
        if off > 1.0:
            notes.append(f"{d.name}: base {off:.0f} mm off the palm midplane, "
                         f"which a mount cannot express")
    for reason in reasons:
        notes.append(f"NOT A LEGAL DESIGN: {reason}")
    return hand, notes
