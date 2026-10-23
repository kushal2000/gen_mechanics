"""The two viability checks (layer 3: what no generation rule can guarantee).

C1, clearance: every two solids that do not touch by construction keep one
printed wall (2 mm) apart, at the zero pose and at the episode start pose
(`derive.START_CURL`; since 2026-10-20 the fingers start open, so that is the
zero pose); every base motor's housing stays out of a palm cut's seam;
and the board bay keeps a wall from every finger's base links over their
whole ranges (`derive.base_sweep_mm`; Martin, 2026-10-23: the built hand
must hold its board). Solids are `derive.solids_at`: the plates, the mount,
the board bay, the hinge motor, each finger's base motor and each link (a box around
its yoke and the next motor; the fingertip the hull of a capsule and its
yoke). Distances are exact (GJK on the corners, or a capsule's segment less
its radius).

Touching by construction (skipped):
- the main plate with the mount, the hinge motor and the part beyond the
  cut; the bay with the mount (on its column); the hinge motor with that
  part;
- a plate with the base motors on it (they sit in pockets);
- within a finger, each joint's two sides: its plate and base motor with
  link 0, link j - 1 with link j, and across a 0 mm link the links on either
  side of it.
One exception keeps a link from lying in the palm: at the zero pose a plate
is checked against the far end (the next motor, or the pad) of a link hinged
to it, when that end is away from the base point. (At the start pose the
link's near end swings through the plate's edge, where a built palm has a
pocket.)

Collisions while the fingers move are left to the physics and the policy.

C2, fingertip workspaces meet above the palm: at least one pair of fingers
whose fingertips' reachable regions come within `C2_TOLERANCE_MM` of each
other above the palm. Each fingertip is sampled `C2_SAMPLES` times over its
own joints' ranges (the palm joint included); fingers beyond the cut share
one hinge angle per comparison. Only samples above the plate's front face
and over the palm's footprint (all plates' outlines, grown by
`C2_FOOTPRINT_MARGIN_MM`) count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import derive as dv
from . import motor as mt
from .derive import gjk_distance  # noqa: F401  (re-exported: the checks' distance)
from .hand import PLATE_HALF_MM, SEAM_MM, Hand, cut_side_mm, finger_section

MM = 1e-3
CLEARANCE_MM = dv.CLEARANCE_MM          # one printed wall
C1_MAX_OVERLAP_MM = 0.0                 # how far two solids may come inside the wall (none)
C1_TOLERANCE_MM = 1e-3                  # numerical slack (a micron)
C2_SAMPLES = 1500
C2_TOLERANCE_MM = 20.0
C2_FOOTPRINT_MARGIN_MM = 20.0
C2_HINGE_LEVELS = 7                     # hinge angles shared by the fingers beyond a cut
C2_SEED = 0


# --------------------------------------------------------------------------
# Which solids touch by construction
# --------------------------------------------------------------------------


def touching_pairs(hand: Hand) -> Set[Tuple[Tuple, Tuple]]:
    """Solid-key pairs that touch by construction (both orders)."""
    out: Set[Tuple[Tuple, Tuple]] = set()

    def add(a, b):
        out.add((a, b))
        out.add((b, a))

    add(("palm", -1), ("mount",))
    add(("mount",), ("bay",))
    if hand.palm_joints:
        add(("palm", -1), ("palm", 0))
        add(("hinge",), ("palm", -1))
        add(("hinge",), ("palm", 0))
    for i, f in enumerate(hand.fingers):
        plate, motor = ("palm", finger_section(hand, i)), ("motor", i)
        add(plate, motor)
        for j in range(len(f.joints)):
            me = ("link", i, j)
            prev = j - 1
            while True:
                for other in ((plate, motor) if prev < 0 else (("link", i, prev),)):
                    add(me, other)
                if prev < 0 or f.joints[prev].length != 0:
                    break
                prev -= 1
    return out


def plate_far_pairs(hand: Hand) -> Set[Tuple[Tuple, Tuple]]:
    """(plate, link) pairs checked against the link's far end although the
    link is hinged to the plate: links reached from the base through 0 mm
    links whose own length is not 0."""
    out = set()
    for i, f in enumerate(hand.fingers):
        plate = ("palm", finger_section(hand, i))
        for j, jt in enumerate(f.joints):
            if jt.length > 0:
                out.add((plate, ("link", i, j)))
            if jt.length != 0:
                break
    return out


# --------------------------------------------------------------------------
# C1
# --------------------------------------------------------------------------


def _finger_of(key: Tuple) -> int:
    return key[1] if key[0] in ("motor", "link") else -1


def overlaps(hand: Hand, q: Optional[np.ndarray] = None, focus_finger: Optional[int] = None
             ) -> List[Tuple[Tuple, Tuple, float]]:
    """`(solid_a, solid_b, overlap_mm)` for every pair not touching by
    construction that comes closer than the wall (overlap = wall - distance,
    so 2 mm when they intersect). `q` None is the zero pose (with the
    plate-against-far-end exception). `focus_finger`: only pairs involving
    that finger or a plate (what changes when a finger is added)."""
    solids = dv.solids_at(hand, q)
    skip = touching_pairs(hand)
    far = plate_far_pairs(hand) if q is None else set()
    centres = [s.points.mean(axis=0) for s in solids]
    reach = [float(np.max(np.linalg.norm(s.points - c, axis=1))) + s.radius for s, c in zip(solids, centres)]
    margin = CLEARANCE_MM * MM
    out = []
    for a in range(len(solids)):
        for b in range(a + 1, len(solids)):
            A, B = solids[a], solids[b]
            if focus_finger is not None and focus_finger not in (_finger_of(A.key), _finger_of(B.key)) \
                    and A.key[0] != "palm" and B.key[0] != "palm":
                continue
            PA, PB, rA, rB = A.points, B.points, A.radius, B.radius
            if (A.key, B.key) in skip:
                if (A.key, B.key) in far:
                    PB, rB = dv.shape_points(B.far)
                    PB, rB = PB * MM, rB * MM
                elif (B.key, A.key) in far:
                    PA, rA = dv.shape_points(A.far)
                    PA, rA = PA * MM, rA * MM
                else:
                    continue
            if float(np.linalg.norm(centres[a] - centres[b])) > reach[a] + reach[b] + margin:
                continue
            d = max(0.0, gjk_distance(PA, PB) - rA - rB)      # intersecting solids: 0, as for boxes
            ov = CLEARANCE_MM - d / MM
            if ov > C1_TOLERANCE_MM:
                out.append((A.key, B.key, ov))
    return out


def seam_overlaps(hand: Hand) -> List[Tuple[Tuple, Tuple, float]]:
    """`(("motor", i), ("seam",), mm)` for every finger whose base motor
    housings reach into a palm cut's seam (the band half a seam either side
    of the line), by how much."""
    if not hand.palm_joints:
        return []
    p = hand.palm_joints[0]
    out = []
    for i, f in enumerate(hand.fingers):
        sec = finger_section(hand, i)
        worst = -math.inf
        for b in dv.base_motor_boxes_mm(f):
            for y, z in b.grown(mt.WALL_MM).corners()[:, 1:]:
                s = cut_side_mm(p, y, z)
                worst = max(worst, SEAM_MM / 2.0 - s if sec == 0 else s + SEAM_MM / 2.0)
        if worst > C1_TOLERANCE_MM:
            out.append((("motor", i), ("seam",), float(worst)))
    return out


def bay_overlaps(hand: Hand) -> List[Tuple[Tuple, Tuple, float]]:
    """`(("bay",), ("base sweep",), mm)` when no turn and shift of the board
    bay keeps a wall from every finger's base links over their ranges
    (`derive.base_sweep_mm`) and off a cut, by how much."""
    gap = dv.hand_geometry(hand).bay_gap_mm
    if gap >= CLEARANCE_MM - C1_TOLERANCE_MM:
        return []
    return [(("bay",), ("base sweep",), float(CLEARANCE_MM - max(gap, 0.0)))]


@dataclass
class C1Result:
    ok: bool
    worst_mm: float
    pairs: List[Tuple[Tuple, Tuple, float]]      # failing pairs (either pose)


def c1_self_overlap(hand: Hand, max_overlap_mm: float = C1_MAX_OVERLAP_MM) -> C1Result:
    bad = []
    worst = 0.0
    found = seam_overlaps(hand) + bay_overlaps(hand)
    for q in dv.check_poses(hand):
        found += overlaps(hand, q)
    for a, b, ov in found:
        worst = max(worst, ov)
        if ov > max_overlap_mm + C1_TOLERANCE_MM:
            bad.append((a, b, ov))
    return C1Result(ok=not bad, worst_mm=worst, pairs=bad)


def clear(hand: Hand, focus_finger: Optional[int] = None) -> bool:
    """C1 as a yes/no (`focus_finger`: only what involves that finger or a
    plate, for building a hand finger by finger)."""
    if seam_overlaps(hand) or bay_overlaps(hand):
        return False
    for q in dv.check_poses(hand):
        if overlaps(hand, q, focus_finger):
            return False
    return True


# --------------------------------------------------------------------------
# C2
# --------------------------------------------------------------------------


@dataclass
class C2Result:
    ok: bool
    best_mm: float                 # the closest pair's distance between kept samples (inf: none)
    pair: Optional[Tuple[int, int]]
    kept: List[np.ndarray]         # per finger, kept fingertip samples (m)
    margin_mm: float               # tolerance - best (>= 0 passes)


def footprint_mm(hand: Hand) -> np.ndarray:
    outlines = dv.palm_outlines_mm(hand)
    return dv._hull2d(np.concatenate(list(outlines.values())))


def fingertip_samples(hand: Hand, n: int = C2_SAMPLES, seed: int = C2_SEED) -> List[np.ndarray]:
    """Per finger, `(n, 3)` fingertip positions (m) with that finger's joints
    uniform over their ranges, every other joint at 0, and the palm joint (for
    a finger beyond the cut) at sample k's shared level (`C2_HINGE_LEVELS`
    angles over its range, sample k at level k mod levels)."""
    rng = np.random.default_rng(seed)
    ds = dv.dofs(hand)
    lo_p, hi_p = dv.palm_limits()
    levels = np.linspace(lo_p, hi_p, C2_HINGE_LEVELS)
    hinge = levels[np.arange(n) % C2_HINGE_LEVELS]
    out = []
    for i, f in enumerate(hand.fingers):
        lo = np.zeros(len(ds))
        hi = np.zeros(len(ds))
        for k, d in enumerate(ds):
            if d.finger == i:
                lo[k], hi[k] = d.limits
        q = rng.uniform(lo, hi, size=(n, len(ds)))
        if finger_section(hand, i) == 0:
            q[:, 0] = hinge
        q = dv.tie(hand, q, ds)
        out.append(dv.fk(hand, q).tips[:, i])
    return out


def c2_workspace_overlap(hand: Hand, n: int = C2_SAMPLES, tol_mm: float = C2_TOLERANCE_MM,
                         margin_mm: float = C2_FOOTPRINT_MARGIN_MM, seed: int = C2_SEED) -> C2Result:
    poly = footprint_mm(hand)
    top = PLATE_HALF_MM * MM
    kept, idx = [], []
    for tips in fingertip_samples(hand, n, seed):
        m = (tips[:, 0] > top) & dv.point_in_polygon(tips[:, 1:] / MM, poly, margin_mm)
        kept.append(tips[m])
        idx.append(np.nonzero(m)[0])
    moving = [finger_section(hand, i) == 0 for i in range(len(hand.fingers))]
    best, pair = math.inf, None
    for a in range(len(kept)):
        for b in range(a + 1, len(kept)):
            A, B = kept[a], kept[b]
            if len(A) == 0 or len(B) == 0:
                continue
            d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
            if moving[a] and moving[b]:          # one hinge angle for both
                same = (idx[a] % C2_HINGE_LEVELS)[:, None] == (idx[b] % C2_HINGE_LEVELS)[None, :]
                d2 = np.where(same, d2, np.inf)
            d = math.sqrt(max(float(d2.min()), 0.0)) / MM
            if d < best:
                best, pair = d, (a, b)
    return C2Result(ok=best <= tol_mm, best_mm=best, pair=pair, kept=kept, margin_mm=tol_mm - best)


@dataclass
class Viability:
    c1: C1Result
    c2: C2Result

    @property
    def ok(self) -> bool:
        return self.c1.ok and self.c2.ok


def viability(hand: Hand) -> Viability:
    return Viability(c1=c1_self_overlap(hand), c2=c2_workspace_overlap(hand))


def is_viable(hand: Hand) -> bool:
    c1 = c1_self_overlap(hand)
    return c1.ok and c2_workspace_overlap(hand).ok
