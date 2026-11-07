"""The v3 grammar (7 Nov 2026): link geometry only -- no servos -- and a generator that draws hands INSIDE a given v3
MAP-Elites cell (coevolution/evolution/descriptors_v3.py) instead of drawing at random and rejecting.

Geometry (the user's calls, 7 Nov: no real-world servo constraints, "just put link geometry"; links 2 cm wide;
fingers straight; links boxes, the fingertip a capsule):
  * every link a box LINK_WIDTH_MM x LINK_WIDTH_MM in section, from its joint to the next joint; the last link a
    capsule of the same width ending at the fingertip pad; a co-located knuckle's 0 mm link has no solid;
  * at q = 0 a finger is one straight line along its facing (tilted out of the plate where the tilt rule allows);
  * the palm a convex plate PALM_THICKNESS_MM thick: the hull of the finger bases grown by half a link width; the
    root frame (where the wrist attaches) at the palm's centroid;
  * clearance (q = 0): the convex distance (GJK) between every two solids that are not joint neighbours (nor the palm
    with a finger's first two bodies, nor two links across a co-located knuckle).

Rules:
  * 2-6 fingers; every finger 2-4 joints; hinges of two kinds only, flexion and abduction (no roll); after the base
    each joint's kind is flexion or abduction 50 / 50;
  * links of 30 / 40 / 50 mm; a 0 mm link only for a co-located base knuckle (a finger's first two joints a flexion
    and an abduction at the same point, in either order);
  * every finger of a hand has the same base type: 1-DOF, FE->AbAd (flexion first) or AbAd->FE;
  * a finger may tilt out of the plate (0 / 30 / 60 deg, never 90: a finger straight up has no direction in the plate
    and no layout) only on a 1-DOF base whose first joint is an abduction;
  * a finger may ROLL about its own length (ROLLS: the full circle on the 30 deg grid -- the axis direction sets which
    way flexion bends; seeds draw SEED_ROLLS, -90..90, the grasp-side half; 7 Nov, the user's call: opposable
    thumbs -- Wuji's is rolled ~60 deg): every joint axis turns about the link by the roll, so a joint's kind is
    relative to it -- flexion at the roll angle, abduction 90 deg from it (kind_of). The roll lives on v3.Finger and
    in the design dict ("roll"; v3.hand_to_dict / hand_from_dict -- the grammar's own serialiser drops it);
  * facings on the 30 deg grid; bases on a 20 mm grid within BASE_DISTANCE_MAX_MM of the palm centre; a finger may
    point any way in the plate (7 Nov, the user's call: no outward-facing rule -- an opposable thumb points across the
    palm; the clearance check rejects any finger that runs into the palm or another finger). The seed generator still
    draws outward-facing fingers; mutation may turn them inward.

`random_hand_in_cell(rng, cell)` fixes each descriptor where the hand is built: the finger count; per-finger joint
counts whose sum is in the cell's motor bin; the base type; finger facings with exactly the cell's largest angular gap;
bases (radial, or side by side on an edge for clustered facings); link lengths whose straight-finger reach lands in the
cell's reach bin. `build(hand, name, out_dir)` writes the URDF + spec the template scene loads; the seed script
re-checks every built hand's cell from its URDF (descriptors_v3.features), the ground truth.
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import hand as gh

LINK_WIDTH_MM = 20.0
TIP_RADIUS_MM = LINK_WIDTH_MM / 2.0
PALM_THICKNESS_MM = 20.0
LINKS_MM = (30, 40, 50)
KNUCKLE_LINK_MM = 0
MIN_JOINTS, MAX_JOINTS = 2, 4
FLEX, ABD = (0, 0), (90, 0)
BASE_TYPES = ("kn1dof", "knFA", "knAF")
BASE_PITCH_MM = 20
BASE_RADII_MM = (20, 40, 60, 80)
EDGE_SPACINGS_MM = (20, 30, 40)
BASE_DISTANCE_MAX_MM = 90
KIND_RANGE_DEG = {"flex": (-30.0, 90.0), "abd": (-30.0, 30.0)}
ROLLS = tuple(range(-150, 181, 30))             # the full circle: the axis DIRECTION sets which way +q bends
SEED_ROLLS = (-90, -60, -30, 0, 30, 60, 90)     # seeds: the half that bends toward the grasp side


def wrap_roll(r: int) -> int:
    """A roll into (-180, 180]."""
    r = ((int(r) + 180) % 360) - 180
    return 180 if r == -180 else r


@dataclass(frozen=True)
class Finger(gh.Finger):
    """A grammar finger plus its roll about its own length (deg)."""
    roll: int = 0


def roll_of(f) -> int:
    return int(getattr(f, "roll", 0))


def axis_for(kind: str, roll: int) -> Tuple[int, int]:
    """A joint axis (az, el) in the link frame: flexion at the roll angle about the link, abduction 90 deg from it
    (gh.axis_unit: az turns the axis from z toward y about the link)."""
    return (int(roll) + (0 if kind == "flex" else 90), 0)


def kind_of(f, jt) -> str:
    """'flex' / 'abd' relative to the finger's roll, or 'bad' (off the two lines, or tipped toward the link)."""
    if jt.axis[1] != 0:
        return "bad"
    rel = (int(jt.axis[0]) - roll_of(f)) % 180
    return "flex" if rel == 0 else ("abd" if rel == 90 else "bad")


def as_v3(f, roll: Optional[int] = None) -> Finger:
    return Finger(y=f.y, z=f.z, facing=f.facing, tilt=f.tilt, joints=f.joints,
                  roll=roll_of(f) if roll is None else int(roll))


def hand_to_dict(hand: gh.Hand) -> dict:
    d = gh.hand_to_dict(hand)
    for fd, f in zip(d["fingers"], hand.fingers):
        fd["roll"] = roll_of(f)
    return d


def hand_from_dict(d: dict) -> gh.Hand:
    h = gh.hand_from_dict(d)
    return gh.Hand(fingers=tuple(as_v3(f, fd.get("roll", 0)) for f, fd in zip(h.fingers, d["fingers"])))
MAX_TILT_DEG = 60
MM = 1e-3

# dynamics: the grammar hands' (urdf_export), unchanged
STIFFNESS, DAMPING, ARMATURE = 3.0, 0.0775033248319064, 0.00058
EFFORT, VELOCITY, DENSITY, FRICTION = 0.5, 5.0, 1750.0, 0.5


# ------------------------------------------------------------------ geometry (palm frame: x the normal, y / z the plate)
def finger_frame(f: gh.Finger) -> np.ndarray:
    """3x3, columns (along the finger, abduction axis, flexion axis) in the palm frame at q = 0. +flexion bends
    toward +x (the grasp side); the tilt turns the finger about its flexion axis toward +x."""
    fa, t = math.radians(f.facing), math.radians(f.tilt)
    d = np.array([0.0, math.sin(fa), math.cos(fa)])
    n = np.array([1.0, 0.0, 0.0])
    zf = np.cross(d, n)                                   # flexion axis: in the plate, square to the finger
    d_t = math.cos(t) * d + math.sin(t) * n
    n_t = np.cross(zf, d_t)                               # abduction axis, tilted with the finger
    return np.column_stack([d_t, n_t, zf])


def finger_solids_mm(f: gh.Finger, centre_yz=(0.0, 0.0)) -> List[Optional[Tuple[np.ndarray, float]]]:
    """Each link's solid at q = 0 (palm frame, mm, relative to the palm centre) as (points, radius) for the convex
    distance: a box's 8 corners (radius 0), the fingertip capsule's two segment ends (radius TIP_RADIUS_MM); None for a
    0 mm link."""
    B = finger_frame(f)
    d, a0, z0 = B[:, 0], B[:, 1], B[:, 2]
    rr = math.radians(roll_of(f))
    z = math.cos(rr) * z0 + math.sin(rr) * a0               # the box turns with the roll (as the joint axes do)
    a = math.cos(rr) * a0 - math.sin(rr) * z0
    p = np.array([0.0, f.y - centre_yz[0], f.z - centre_yz[1]])
    w = LINK_WIDTH_MM / 2.0
    out = []
    n = len(f.joints)
    for j, jt in enumerate(f.joints):
        q = p + jt.length * d
        if jt.length == 0:
            out.append(None)
        elif j == n - 1:
            r = TIP_RADIUS_MM
            out.append((np.array([p + min(r, jt.length / 2) * d, q - min(r, jt.length / 2) * d]), r))
        else:
            corners = np.array([e + sa * w * a + sz * w * z for e in (p, q) for sa in (-1, 1) for sz in (-1, 1)])
            out.append((corners, 0.0))
        p = q
    return out


def fingertip_mm(f: gh.Finger, centre_yz=(0.0, 0.0)) -> np.ndarray:
    B = finger_frame(f)
    return np.array([0.0, f.y - centre_yz[0], f.z - centre_yz[1]]) + sum(j.length for j in f.joints) * B[:, 0]


def palm_outline_mm(hand: gh.Hand, n_pts: int = 16) -> np.ndarray:
    """The palm's outline in the plate (y, z): the convex hull of the bases grown by half a link width."""
    from scipy.spatial import ConvexHull
    ang = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    ring = (LINK_WIDTH_MM / 2.0) * np.column_stack([np.cos(ang), np.sin(ang)])
    pts = np.concatenate([np.array([f.y, f.z], float) + ring for f in hand.fingers])
    return pts[ConvexHull(pts).vertices]


def palm_centre_mm(hand: gh.Hand) -> np.ndarray:
    """The palm outline's area centroid (y, z)."""
    P = palm_outline_mm(hand)
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2.0
    return np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (6.0 * A)


CONTACT_TOL_MM = 0.5            # solids exactly touching (e.g. 20 mm links on a 20 mm base grid) are not overlapping


def clear(hand: gh.Hand, gap_mm: float = -CONTACT_TOL_MM) -> bool:
    """No two solids that are not joint neighbours closer than `gap_mm` at q = 0 (negative: overlapping by more than
    that); the palm counts as a solid against every finger body past each finger's first two."""
    from .viability import gjk_distance
    c = palm_centre_mm(hand)
    P = palm_outline_mm(hand) - c
    h = PALM_THICKNESS_MM / 2.0
    palm = np.concatenate([np.column_stack([np.full(len(P), -h), P]), np.column_stack([np.full(len(P), h), P])])
    solids = []                                          # (finger, link index, points, radius)
    for i, f in enumerate(hand.fingers):
        for j, s in enumerate(finger_solids_mm(f, c)):
            if s is not None:
                solids.append((i, j, s[0], s[1]))
    for i, j, pts, r in solids:     # the palm vs every finger body but the first (and the second behind a knuckle)
        f = hand.fingers[i]
        if j == 0 or (j == 1 and f.joints[0].length == 0):
            continue
        if gjk_distance(palm, pts) - r < gap_mm:
            return False
    for a in range(len(solids)):
        for b in range(a + 1, len(solids)):
            ia, ja, pa, ra = solids[a]
            ib, jb, pb, rb = solids[b]
            if ia == ib:
                lo = min(ja, jb)
                if abs(ja - jb) <= 1:
                    continue
                if abs(ja - jb) == 2 and hand.fingers[ia].joints[lo + 1].length == 0:
                    continue                             # neighbours across a co-located knuckle
            if gjk_distance(pa, pb) - ra - rb < gap_mm:
                return False
    return True


# ------------------------------------------------------------------ the v3 rules
def base_type(f: gh.Finger) -> str:
    j = f.joints
    if len(j) >= 2 and j[0].length == KNUCKLE_LINK_MM:
        k0, k1 = kind_of(f, j[0]), kind_of(f, j[1])
        if (k0, k1) == ("flex", "abd"):
            return "knFA"
        if (k0, k1) == ("abd", "flex"):
            return "knAF"
        return "bad"
    return "kn1dof"


MUTANT_SLACK_DEG = 30          # mutants (as v2's mutation rules): a finger may point up to 30 deg past tangential
MUTANT_BASE_DISTANCE_MM = 110  # ... and its base sit up to 110 mm from the palm centre -- the centroid moves with edits


def violations(hand: gh.Hand, slack_deg: float = 0.0, max_base_mm: float = BASE_DISTANCE_MAX_MM) -> List[str]:
    out = []
    if not (2 <= len(hand.fingers) <= 6):
        return [f"{len(hand.fingers)} fingers (2-6)"]
    types, bases = set(), set()
    c = palm_centre_mm(hand)
    for i, f in enumerate(hand.fingers):
        if not (MIN_JOINTS <= len(f.joints) <= MAX_JOINTS):
            out.append(f"finger {i}: {len(f.joints)} joints ({MIN_JOINTS}-{MAX_JOINTS})")
        if f.facing % gh.GRID_DEG or f.tilt % gh.GRID_DEG:
            out.append(f"finger {i}: facing / tilt off the {gh.GRID_DEG} deg grid")
        t = base_type(f)
        types.add(t)
        if t == "bad":
            out.append(f"finger {i}: a 0 mm base link that is not a flexion / abduction knuckle")
        if roll_of(f) not in ROLLS:
            out.append(f"finger {i}: roll {roll_of(f)} (one of {ROLLS})")
        if f.tilt != 0 and not (t == "kn1dof" and f.joints and kind_of(f, f.joints[0]) == "abd"):
            out.append(f"finger {i}: tilt {f.tilt} on a {t} base (only a 1-DOF base with an abduction first)")
        if not (0 <= f.tilt <= MAX_TILT_DEG):
            out.append(f"finger {i}: tilt {f.tilt} outside 0-{MAX_TILT_DEG}")
        if math.hypot(f.y - c[0], f.z - c[1]) > max_base_mm:
            out.append(f"finger {i}: base {math.hypot(f.y - c[0], f.z - c[1]):.0f} mm from the palm centre")
        if (f.y, f.z) in bases:
            out.append(f"finger {i}: base shared with another finger")
        bases.add((f.y, f.z))
        for j, jt in enumerate(f.joints):
            if jt.type != "hinge" or kind_of(f, jt) == "bad":
                out.append(f"finger {i} joint {j}: {jt.type} {jt.axis} at roll {roll_of(f)} "
                           f"(flexion / abduction hinges only)")
            if jt.length == 0 and not (j == 0 and t in ("knFA", "knAF")):
                out.append(f"finger {i} joint {j}: 0 mm link off the base knuckle")
            elif jt.length != 0 and jt.length not in LINKS_MM:
                out.append(f"finger {i} joint {j}: link {jt.length} mm (one of {LINKS_MM})")
    if len(types) > 1:
        out.append(f"mixed base types {sorted(types)}")
    return out


def valid(hand: gh.Hand, mutant: bool = False) -> bool:
    """The v3 rules and the clearance; `mutant`: the relaxed priors (MUTANT_SLACK_DEG, MUTANT_BASE_DISTANCE_MM)."""
    v = violations(hand, MUTANT_SLACK_DEG, MUTANT_BASE_DISTANCE_MM) if mutant else violations(hand)
    return not v and clear(hand)


def centred(hand: gh.Hand) -> gh.Hand:
    """The hand moved by whole mm so the palm's centroid is at the origin."""
    c = np.round(palm_centre_mm(hand)).astype(int)
    if not c.any():
        return hand
    return gh.Hand(fingers=tuple(replace(f, y=int(f.y - c[0]), z=int(f.z - c[1])) for f in hand.fingers))  # keeps roll


# ------------------------------------------------------------------ design-level descriptors (fast; the URDF is exact)
def design_gap_deg(hand: gh.Hand) -> float:
    a = np.sort(np.mod([f.facing for f in hand.fingers], 360.0))
    return float(np.diff(np.concatenate([a, a[:1] + 360.0])).max()) if len(a) > 1 else 360.0


def design_reach_mm(hand: gh.Hand) -> float:
    c = palm_centre_mm(hand)
    return float(max(np.linalg.norm(fingertip_mm(f, c)) for f in hand.fingers))


def design_features(hand: gh.Hand) -> dict:
    return {"fingers": len(hand.fingers), "motors": sum(len(x.joints) for x in hand.fingers),
            "gap": design_gap_deg(hand), "reach_mm": design_reach_mm(hand),
            "knuckles_fa": sum(base_type(x) == "knFA" for x in hand.fingers),
            "knuckles_af": sum(base_type(x) == "knAF" for x in hand.fingers),
            "min_joints_per_finger": min(len(x.joints) for x in hand.fingers),
            "max_joints_per_finger": max(len(x.joints) for x in hand.fingers)}


# ------------------------------------------------------------------ the cell-conditioned generator
def parse_cell(cell: str) -> dict:
    """'f5_m18-20_gap240-360_knFA_mid' -> its pieces (descriptors_v3.cell's format)."""
    from coevolution.evolution import descriptors_v3 as d3
    f, m, gap, kn, rc = cell.split("_")
    lo, hi = (int(x) for x in m[1:].split("-"))
    g0, g1 = (float(x) for x in gap[3:].split("-"))
    ri = d3.REACH_NAMES.index(rc)
    return {"fingers": int(f[1:]), "motors": (lo, hi), "gap": (g0, g1), "base": kn,
            "reach": (d3.REACH_EDGES[ri], d3.REACH_EDGES[ri + 1])}


def _joint_counts(rng, n, lo, hi) -> Optional[List[int]]:
    for _ in range(200):
        js = [int(x) for x in rng.integers(MIN_JOINTS, MAX_JOINTS + 1, size=n)]
        if lo <= sum(js) <= hi:
            return js
    return None


def _facings(rng, n, g0, g1) -> Optional[List[int]]:
    """n facings on the 30 deg grid whose largest neighbouring gap G is in [g0, g1) (all other gaps <= G)."""
    step = gh.GRID_DEG
    Gs = [G for G in range(0, 361, step) if g0 <= G < g1 and G * n >= 360]
    if not Gs or n < 2:
        return None
    for _ in range(400):
        G = int(rng.choice(Gs))
        rest = 360 - G
        if rest < 0 or rest > (n - 1) * G:
            continue
        parts = np.zeros(n - 1, int)
        ok_all = True
        for _ in range(rest // step):
            ok = [i for i in range(n - 1) if (parts[i] + 1) * step <= G]
            if not ok:
                ok_all = False
                break
            parts[int(rng.choice(ok))] += 1
        if not ok_all:
            continue
        gaps = [G] + [int(p) * step for p in parts]
        gaps = [gaps[i] for i in rng.permutation(n)]
        a = int(rng.integers(0, 360 // step)) * step
        fac = []
        for g in gaps:
            fac.append(a % 360)
            a += g
        return fac
    return None


def _snap(v: float) -> int:
    return int(round(v / BASE_PITCH_MM) * BASE_PITCH_MM)


def _edge_bases(rng, facings) -> Optional[List[Tuple[int, int]]]:
    """Clustered facings: bases side by side on a line square to the fingers' mean facing, ordered by facing."""
    a = np.radians(np.asarray(facings, float))
    mean = math.atan2(np.sin(a).mean(), np.cos(a).mean())
    r = int(rng.choice(BASE_RADII_MM))
    s = int(rng.choice(EDGE_SPACINGS_MM))
    d = np.array([math.sin(mean), math.cos(mean)])
    t = np.array([d[1], -d[0]])
    rel = [((f - math.degrees(mean) + 180) % 360) - 180 for f in facings]
    pos = [None] * len(facings)
    for k, i in enumerate(np.argsort(rel)):
        q = r * d + (k - (len(facings) - 1) / 2.0) * s * t
        pos[i] = (_snap(q[0]), _snap(q[1]))
    return pos if len(set(pos)) == len(pos) else None


def _bases(rng, facings) -> Optional[List[Tuple[int, int]]]:
    a = np.sort(np.mod(facings, 360))
    span = 360 - float(np.diff(np.concatenate([a, a[:1] + 360])).max()) if len(a) > 1 else 0.0
    if span <= 180 and rng.random() < 0.5:
        return _edge_bases(rng, facings)
    r = int(rng.choice(BASE_RADII_MM))
    out, used = [], set()
    for fa in facings:
        ang = math.radians(fa)
        y, z = _snap(r * math.sin(ang)), _snap(r * math.cos(ang))
        step = 0
        while (y, z) in used and step < 8:
            step += 1
            sgn = 1 if step % 2 else -1
            y, z = _snap(y + sgn * BASE_PITCH_MM * math.cos(ang)), _snap(z - sgn * BASE_PITCH_MM * math.sin(ang))
        if (y, z) in used:
            return None
        used.add((y, z))
        out.append((y, z))
    return out


def _chain(rng, nj: int, base: str, roll: int = 0) -> List[gh.Joint]:
    if base == "knFA":
        kinds, lens = ["flex", "abd"], [KNUCKLE_LINK_MM]
    elif base == "knAF":
        kinds, lens = ["abd", "flex"], [KNUCKLE_LINK_MM]
    else:
        kinds, lens = ["flex" if rng.random() < 0.5 else "abd"], []
    while len(kinds) < nj:
        kinds.append("flex" if rng.random() < 0.5 else "abd")   # 50 / 50 after the base
    while len(lens) < nj:
        lens.append(int(rng.choice(LINKS_MM)))
    return [gh.Joint("hinge", axis_for(k, roll), L) for k, L in zip(kinds, lens)]


def _tilt(rng, base: str, first_kind: str) -> int:
    """0 unless the base is 1-DOF with an abduction first; then 0 / 30 / 60."""
    if base != "kn1dof" or first_kind != "abd":
        return 0
    return int(rng.choice(range(0, MAX_TILT_DEG + 1, gh.GRID_DEG)))


def _centred_bases(bases) -> np.ndarray:
    """The bases (y, z) moved so the palm outline's centroid (which depends on the bases only) is at the origin."""
    hand = gh.Hand(fingers=tuple(gh.Finger(y=int(y), z=int(z)) for y, z in bases))
    c = np.round(palm_centre_mm(hand)).astype(int)
    return np.asarray(bases, float) - c


def _bases_ok(B: np.ndarray, facings) -> bool:
    """Base-level rules, before any chain is drawn: distinct, within BASE_DISTANCE_MAX_MM of the palm centre, facing
    outward, and neighbouring bases at least a link width apart."""
    if len({(int(y), int(z)) for y, z in B}) < len(B):
        return False
    for (y, z), fa in zip(B, facings):
        if math.hypot(y, z) > BASE_DISTANCE_MAX_MM:
            return False
        a = math.radians(fa)
        if math.sin(a) * y + math.cos(a) * z < -0.5:
            return False
    d = np.linalg.norm(B[:, None, :] - B[None, :, :], axis=-1)
    np.fill_diagonal(d, 1e9)
    return bool(d.min() >= LINK_WIDTH_MM)


def _length_range(b: np.ndarray, fa: float, tilt: float, r_hi: float) -> float:
    """The longest straight finger from base b (palm frame, relative to the centre) at facing fa / tilt whose tip stays
    closer than r_hi to the centre: |b + L u|^2 < r_hi^2, u the unit finger direction (monotone in L when outward)."""
    a, t = math.radians(fa), math.radians(tilt)
    u = np.array([0.0, math.cos(t) * math.sin(a), math.cos(t) * math.cos(a)])
    bb = np.array([0.0, b[0], b[1]])
    p = float(bb @ u)
    disc = p * p - (bb @ bb - r_hi * r_hi)
    return -p + math.sqrt(disc) if disc > 0 else -1.0


def _tip_dist(b, fa, tilt, L) -> float:
    a, t = math.radians(fa), math.radians(tilt)
    u = np.array([0.0, math.cos(t) * math.sin(a), math.cos(t) * math.cos(a)])
    return float(np.linalg.norm(np.array([0.0, b[0], b[1]]) + L * u))


def _totals(n_links: int) -> List[int]:
    """Every total length n_links links of LINKS_MM can make (on the 10 mm grid)."""
    return list(range(min(LINKS_MM) * n_links, max(LINKS_MM) * n_links + 1, 10)) if n_links else [0]


def _split(rng, total: int, n_links: int) -> List[int]:
    """n_links links of LINKS_MM summing to `total`, uniformly among the ordered splits."""
    splits = [c for c in __import__("itertools").product(LINKS_MM, repeat=n_links) if sum(c) == total]
    return list(splits[int(rng.integers(len(splits)))])


def random_hand_in_cell(rng: np.random.Generator, cell: str, attempts: int = 200) -> Optional[gh.Hand]:
    """A v3-valid hand in `cell`, or None. Solved where it can be, retried only where it cannot:
      1. joint counts summing into the motor bin, skipping any whose shortest fingers already overshoot the reach bin;
      2. facings with exactly the cell's largest gap (constructed);
      3. bases, centred on the palm's centroid and checked (distinct, in range, outward, a link width apart) before
         any chain -- only the bases are redrawn on failure;
      4. chain kinds and tilts; then each finger's total length SOLVED from the reach bin (the tip distance is monotone
         in the length for an outward finger): every tip short of the bin's top, at least one at its bottom or past it,
         each total one the link grid can make; the totals split into links;
      5. the full rules and the clearance check (the only rejection left that the construction does not rule out)."""
    from coevolution.evolution import descriptors_v3 as d3
    c = parse_cell(cell)
    n, (mlo, mhi), (g0, g1), base, (r0, r1) = c["fingers"], c["motors"], c["gap"], c["base"], c["reach"]
    nlinks = (lambda nj: nj - 1) if base in ("knFA", "knAF") else (lambda nj: nj)
    comps = [js for js in __import__("itertools").product(range(MIN_JOINTS, MAX_JOINTS + 1), repeat=n)
             if mlo <= sum(js) <= mhi]
    if not comps:
        return None
    for _ in range(attempts):
        js = list(comps[int(rng.integers(len(comps)))])
        fac = _facings(rng, n, g0, g1)
        if fac is None:
            return None
        B = None
        for _ in range(30):
            raw = _bases(rng, fac)
            if raw is None:
                continue
            Bc = _centred_bases(raw)
            if _bases_ok(Bc, fac):
                B = Bc
                break
        if B is None:
            continue
        rolls = [int(rng.choice(SEED_ROLLS)) for _ in js]
        chains = [_chain(rng, nj, base, r_) for nj, r_ in zip(js, rolls)]
        tilts = [_tilt(rng, base, "flex" if (ch[0].axis[0] - r_) % 180 == 0 else "abd") for ch, r_ in zip(chains, rolls)]
        # 4. lengths: each finger's allowed totals (tip < r1), at least one finger with its tip >= r0
        allowed = []
        for (y, z), fa, t, nj in zip(B, fac, tilts, js):
            Lmax = _length_range(np.array([y, z]), fa, t, r1)
            allowed.append([T for T in _totals(nlinks(nj)) if T <= Lmax - 1e-6 and (T > 0 or nlinks(nj) == 0)])
        if any(not a for a in allowed):
            continue
        reachers = [i for i, ((y, z), fa, t, a) in enumerate(zip(B, fac, tilts, allowed))
                    if _tip_dist((y, z), fa, t, max(a)) >= r0]
        if not reachers:
            continue
        k = int(rng.choice(reachers))
        totals = []
        for i, ((y, z), fa, t, a) in enumerate(zip(B, fac, tilts, allowed)):
            opts = [T for T in a if _tip_dist((y, z), fa, t, T) >= r0] if i == k else a
            totals.append(int(rng.choice(opts)))
        fingers = []
        for (y, z), fa, t, ch, T, nj, r_ in zip(B, fac, tilts, chains, totals, js, rolls):
            links = _split(rng, T, nlinks(nj))
            lens = ([KNUCKLE_LINK_MM] if base in ("knFA", "knAF") else []) + links
            joints = tuple(replace(jt, length=L) for jt, L in zip(ch, lens))
            fingers.append(Finger(y=int(y), z=int(z), facing=int(fa), tilt=int(t), joints=joints, roll=int(r_)))
        hand = gh.Hand(fingers=tuple(fingers))
        if valid(hand) and d3.cell(design_features(hand)) == cell:
            return hand
    return None


# ------------------------------------------------------------------ URDF + spec (the template scene's format)
def _fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


def _rpy(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_euler("xyz")


def _inertial(link, m, com, I):
    el = ET.SubElement(link, "inertial")
    ET.SubElement(el, "origin", xyz=_fmt(com), rpy="0 0 0")
    ET.SubElement(el, "mass", value=f"{m:.9g}")
    ET.SubElement(el, "inertia", ixx=f"{I[0]:.9g}", ixy="0", ixz="0", iyy=f"{I[1]:.9g}", iyz="0", izz=f"{I[2]:.9g}")


def _shape(link, geom: Tuple[str, dict], xyz, rpy, rgba: str):
    for tag in ("visual", "collision"):
        el = ET.SubElement(link, tag)
        ET.SubElement(el, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
        g = ET.SubElement(el, "geometry")
        ET.SubElement(g, geom[0], **{k: (v if isinstance(v, str) else (_fmt(v) if np.ndim(v) else f"{float(v):.9g}"))
                                     for k, v in geom[1].items()})
        if tag == "visual":
            mat = ET.SubElement(el, "material", name=rgba.replace(" ", "_"))
            ET.SubElement(mat, "color", rgba=rgba)


FLEX_RGBA, ABD_RGBA, PALM_RGBA = "0.16 0.42 0.85 1", "0.86 0.20 0.18 1", "0.62 0.62 0.64 1"


def build(hand: gh.Hand, name: str, out_dir: Path, rules_name: str = "V3") -> dict:
    """Write the hand's URDF (palm plate, a box per link, a capsule fingertip), spec and design under `out_dir`; the
    root frame at the palm's centroid (where the wrist attaches)."""
    import trimesh
    from hand_sampler import REPO_ROOT as REPO
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c = palm_centre_mm(hand)
    root = ET.Element("robot", name=name)
    P = (palm_outline_mm(hand) - c) * MM
    h = PALM_THICKNESS_MM / 2.0 * MM
    pts = np.concatenate([np.column_stack([np.full(len(P), -h), P]), np.column_stack([np.full(len(P), h), P])])
    mesh = trimesh.convex.convex_hull(pts)
    mesh.export(out_dir / "palm.obj")
    palm = ET.SubElement(root, "link", name="palm")
    _shape(palm, ("mesh", {"filename": "palm.obj"}), [0, 0, 0], [0, 0, 0], PALM_RGBA)
    mesh.density = DENSITY
    _inertial(palm, float(mesh.mass), mesh.center_mass, np.diag(mesh.moment_inertia))
    joints, limits, adj = [], {}, {"palm": set()}
    w, r = LINK_WIDTH_MM * MM, TIP_RADIUS_MM * MM
    for i, f in enumerate(hand.fingers):
        B = finger_frame(f)
        base = np.array([0.0, f.y - c[0], f.z - c[1]]) * MM
        parent, n = "palm", len(f.joints)
        for j, jt in enumerate(f.joints):
            body = f"f{i}_link{j}"
            link = ET.SubElement(root, "link", name=body)
            L = jt.length * MM
            kd = kind_of(f, jt)
            rgba = FLEX_RGBA if kd == "flex" else ABD_RGBA
            if L <= 0:
                _inertial(link, 1e-4, [0, 0, 0], [1e-9, 1e-9, 1e-9])
            elif j == n - 1:                               # the fingertip: a capsule (cylinder + the importer's caps)
                cyl = max(L - 2 * r, 1e-4) if L > 2 * r else 1e-4
                _shape(link, ("cylinder", {"radius": r, "length": cyl}), [L / 2, 0, 0], [0, math.pi / 2, 0], rgba)
                m = DENSITY * (math.pi * r * r * cyl + 4.0 / 3.0 * math.pi * r ** 3)
                _inertial(link, m, [L / 2, 0, 0], [0.5 * m * r * r, m * (3 * r * r + L * L) / 12, m * (3 * r * r + L * L) / 12])
            else:                                          # a box from this joint to the next
                _shape(link, ("box", {"size": [L, w, w]}), [L / 2, 0, 0], [math.radians(roll_of(f)), 0, 0], rgba)
                m = DENSITY * L * w * w
                _inertial(link, m, [L / 2, 0, 0], [m * (2 * w * w) / 12, m * (L * L + w * w) / 12, m * (L * L + w * w) / 12])
            jn = f"f{i}_j{j}"
            el = ET.SubElement(root, "joint", name=jn, type="revolute")
            ET.SubElement(el, "parent", link=parent)
            ET.SubElement(el, "child", link=body)
            if j == 0:
                ET.SubElement(el, "origin", xyz=_fmt(base), rpy=_fmt(_rpy(B)))
            else:
                ET.SubElement(el, "origin", xyz=_fmt([f.joints[j - 1].length * MM, 0, 0]), rpy="0 0 0")
            ET.SubElement(el, "axis", xyz=_fmt(gh.axis_unit(tuple(jt.axis))))
            lo, hi = (math.radians(v) for v in KIND_RANGE_DEG[kd])
            ET.SubElement(el, "limit", effort=f"{EFFORT:g}", lower=f"{lo:.9g}", upper=f"{hi:.9g}",
                          velocity=f"{VELOCITY:g}")
            ET.SubElement(el, "dynamics", damping=f"{DAMPING:.6g}", friction="0")
            joints.append(jn)
            limits[jn] = [lo, hi]
            adj.setdefault(body, set()).add(parent)
            adj.setdefault(parent, set()).add(body)
            if j == 0 or (j == 1 and f.joints[0].length == 0):   # the palm and a finger's first body (+ a knuckle's)
                adj["palm"].add(body)
                adj[body].add("palm")
            if j >= 2 and f.joints[j - 1].length == 0:     # neighbours across a co-located knuckle
                adj[body].add(f"f{i}_link{j - 2}")
                adj[f"f{i}_link{j - 2}"].add(body)
            parent = body
    tree = ET.ElementTree(root)
    ET.indent(tree)
    urdf = out_dir / f"{name}.urdf"
    tree.write(urdf, encoding="unicode", xml_declaration=False)
    lo_yz, hi_yz = P.min(0), P.max(0)
    spec = {
        "hand": name, "side": "left", "urdf": str(urdf.relative_to(REPO)) if urdf.is_relative_to(REPO) else str(urdf),
        "palm_body": "palm", "hand_joint_names": joints, "joint_limits": limits,
        "hand_default_joint_pos": {j: 0.0 for j in joints},
        "fingertip_body_names": [f"f{i}_link{len(f.joints) - 1}" for i, f in enumerate(hand.fingers)],
        "fingertip_offsets": [[f.joints[-1].length * MM, 0.0, 0.0] for f in hand.fingers],
        "palm_frame": np.eye(3).tolist(), "palm_frame_det": 1.0,
        "palm_centre_in_palm_body": [0.0, 0.0, 0.0],
        "palm_extents_ours": [PALM_THICKNESS_MM * MM, *(hi_yz - lo_yz).tolist()],
        "palm_centre_ours": [0.0, 0.0, 0.0],
        "adjacent_links": {b: sorted(v) for b, v in adj.items()},
        "stiffness": STIFFNESS, "damping": DAMPING, "armature": ARMATURE,
        "effort_limit": EFFORT, "velocity_limit": VELOCITY, "link_density_kg_m3": DENSITY, "friction": FRICTION,
        "replace_cylinders_with_capsules": True, "rules": rules_name, "design": hand_to_dict(hand),
        "mount_in_root_m": [0.0, 0.0, 0.0],
        "geometry": {"link_width_mm": LINK_WIDTH_MM, "tip_radius_mm": TIP_RADIUS_MM,
                     "palm_thickness_mm": PALM_THICKNESS_MM},
    }
    (out_dir / f"{name}.spec.json").write_text(json.dumps(spec, indent=1))
    (out_dir / f"{name}.design.json").write_text(json.dumps(hand_to_dict(hand), indent=1))
    return {"name": name, "spec": spec["urdf"].replace(".urdf", ".spec.json"), "fingers": len(hand.fingers),
            "joints": len(joints), "joints_per_finger": [len(f.joints) for f in hand.fingers]}


__all__ = ["Finger", "LINK_WIDTH_MM", "PALM_THICKNESS_MM", "ROLLS", "TIP_RADIUS_MM", "axis_for", "build", "clear",
           "design_features", "hand_from_dict", "hand_to_dict", "kind_of", "parse_cell", "random_hand_in_cell",
           "roll_of", "valid", "violations"]
