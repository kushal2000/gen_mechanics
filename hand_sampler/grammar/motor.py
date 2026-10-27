"""The grammar's one motor: the Feetech HL-3915-C001 servo, in millimetres
(E15, `project-notes/grammar/experiments/E15_feetech_buildability/`).

The servo is a 34 x 20 x 29 mm box. Its output axis runs along the 29 mm
edges (the 29 mm include the horn and the idler), 9.5 mm from one end of the
34 mm side and centred in the 20 mm width. In a joint's frame the box has
axes (a, v, w): a the joint axis, v the motor's back (the finger direction x
reversed and made square to a; for a twist motor, whose axis is x, the
closing direction y reversed), w = a x v. The box centre lies 7.5 mm along v
from the joint point: the back 24.5 mm behind the axis, the front 9.5 mm
ahead. Parts that move relative to each other keep one printed wall, 2 mm,
between them.

A bend motor whose link before it is too short for it to lie along the
link is turned 90 degrees about its axis (Martin, 2026-10-16; which motors
turn is `hand.turned_servo`): its back v is the back of the finger (the
closing direction y reversed) instead of the finger direction reversed, so
only its 20 mm width lies along the link (10 mm behind the axis) and its
body stands 24.5 mm out behind the finger.

Every motor's body sits in the link before its joint (the palm for a
finger's first joint). The link after it holds the horn and the idler in a
yoke, a 33 x 24 x 24 mm box around the axis (the motor's length plus a wall
at each end, its front plus a wall).

Link floors (E15's plane rule): a plane square to the finger separates the
next motor's back from the corners of this motor's front, which swing on a
circle of sqrt(9.5^2 + 10^2) = 13.79 mm (13.23 mm for a spread motor, whose
+/-30 degree range never swings a corner fully forward), plus one wall.
A turned motor's back reaches 10 mm behind its axis instead of 24.5; its
body swings forward up to 20.9 mm (a back corner at the bend's -30 degree
hyperextension), worked out from its corners over the bend's range.
Rounded up to the 10 mm grid.

Crossing axes (a 0 mm link): the earlier motor slides along its own axis,
away from the joint point, until a plane square to that axis separates it
from the next motor by one wall. A rotation about that axis keeps the plane,
so the two motors stay clear over the whole range.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np

MOTOR_HALF_MM = np.array([14.5, 17.0, 10.0])   # along (a, v, w)
MOTOR_CENTRE_BACK_MM = 7.5                      # the box centre lies this far along v from the axis
MOTOR_FRONT_MM = 9.5                            # the front face, ahead of the axis
WALL_MM = 2.0                                   # one printed wall between parts that move
ENCLOSURE_HALF_MM = MOTOR_HALF_MM + WALL_MM     # the motor in its printed housing
YOKE_HALF_MM = np.array([16.5, 12.0, 12.0])     # the next link's bracket around the axis
PAD_MM = 10.0                                   # the fingertip pad beyond the last motor's front

FRONT_SWEEP_MM = math.hypot(MOTOR_FRONT_MM, MOTOR_HALF_MM[2])                      # 13.79
SPREAD_FRONT_SWEEP_MM = (MOTOR_FRONT_MM * math.cos(math.radians(30.0))
                         + MOTOR_HALF_MM[2] * math.sin(math.radians(30.0)))         # 13.23

X, Y, Z = np.eye(3)

BEND_RANGE_DEG = (-30.0, 90.0)
"""A bend's range (`hand.KIND_RANGE_DEG["flexion"]`): how far a turned motor's
body swings forward."""


def ceil10(x: float) -> int:
    """`x` rounded up to the 10 mm grid (exact multiples stay)."""
    return int(10 * math.ceil((float(x) - 1e-9) / 10.0))


def is_spread(a: Sequence[float]) -> bool:
    """Whether an axis (link frame) is nearest the spread (abduction) axis y;
    ties go to bend (z), then spread, as `derive.kind_of`."""
    a = np.abs(np.asarray(a, dtype=float))
    best = max((a[2], 2), (a[1], 1), (a[0], 0), key=lambda t: (round(t[0], 12), t[1]))
    return best[1] == 1


def is_bend(a: Sequence[float]) -> bool:
    """Whether an axis (link frame) is nearest the bend (flexion) axis z
    (ties go to bend)."""
    a = np.abs(np.asarray(a, dtype=float))
    best = max((a[2], 2), (a[1], 1), (a[0], 0), key=lambda t: (round(t[0], 12), t[1]))
    return best[1] == 2


def motor_axes(a: Sequence[float], turned: bool = False) -> np.ndarray:
    """3x3, columns (a, v, w): the motor's axis, its back and its width. A
    turned motor's back is the back of the finger (-y)."""
    a = np.asarray(a, dtype=float)
    a = a / np.linalg.norm(a)
    back = -Y if turned else (-X if abs(a[0]) < 1.0 - 1e-12 else -Y)
    v = back - a * float(a @ back)
    v = v / np.linalg.norm(v)
    return np.column_stack([a, v, np.cross(a, v)])


def motor_centre(a: Sequence[float], slide_mm: float = 0.0, turned: bool = False) -> np.ndarray:
    """The motor box's centre relative to its joint point (joint frame)."""
    M = motor_axes(a, turned)
    return slide_mm * M[:, 0] + MOTOR_CENTRE_BACK_MM * M[:, 1]


def support(centre: np.ndarray, axes: np.ndarray, half: np.ndarray, n: Sequence[float]) -> float:
    """How far a box (centre, axes as columns, half sizes) reaches along the
    unit direction `n`, from the origin."""
    n = np.asarray(n, dtype=float)
    return float(centre @ n + np.abs(n @ axes) @ half)


def front_sweep_mm(a: Sequence[float]) -> float:
    return SPREAD_FRONT_SWEEP_MM if is_spread(a) else FRONT_SWEEP_MM


def _turned_sweep_mm(a: np.ndarray) -> float:
    """How far a turned motor's body (axis `a`) reaches ahead of its joint
    along the next link (+x) as that link turns over the bend's range: the
    largest x of any corner, the body turned back by the joint angle."""
    a = a / np.linalg.norm(a)
    s = 1.0 if a[2] >= 0.0 else -1.0          # + closes (moves the next link toward +y)
    lo, hi = sorted(-s * math.radians(v) for v in BEND_RANGE_DEG)
    best = -math.inf
    for c in box_corners(motor_centre(a, 0.0, True), motor_axes(a, True), MOTOR_HALF_MM):
        par = float(c @ a) * a
        perp = c - par
        A, B, C = float(par[0]), float(perp[0]), float(np.cross(a, perp)[0])    # x(t) = A + B cos t + C sin t
        ts = [lo, hi]
        t0 = math.atan2(C, B)
        ts += [t for t in (t0 - 2.0 * math.pi, t0, t0 + 2.0 * math.pi) if lo <= t <= hi]
        best = max(best, max(A + B * math.cos(t) + C * math.sin(t) for t in ts))
    return best


def _key(v: Sequence[float]) -> Tuple[float, ...]:
    return tuple(round(float(x), 12) for x in v)


@lru_cache(maxsize=4096)
def _turned_sweep_cached(a: Tuple[float, ...]) -> float:
    return _turned_sweep_mm(np.array(a))


def swept_front_mm(a: Sequence[float], turned: bool = False) -> float:
    """How far motor `a`'s body reaches ahead of its joint along the next
    link, over the joint's range: E15's swinging front (13.79 mm for a bend,
    13.23 for a spread, plus half its length times the axis's lean along the
    finger); a turned motor's from its corners (20.9 mm for a bend)."""
    a = np.asarray(a, dtype=float)
    if turned:
        return _turned_sweep_cached(_key(a))
    t = abs(float(a[0]))
    return MOTOR_HALF_MM[0] * t + front_sweep_mm(a) * math.sqrt(max(0.0, 1.0 - t * t))


def back_mm(b: Sequence[float], turned: bool = False) -> float:
    """How far motor `b`'s body reaches behind its joint along the finger:
    24.5 mm for a bend, 10 for a turned bend, 14.5 for a twist."""
    return _back_cached(_key(b), bool(turned))


@lru_cache(maxsize=4096)
def _back_cached(b: Tuple[float, ...], turned: bool) -> float:
    v = np.array(b)
    return support(motor_centre(v, 0.0, turned), motor_axes(v, turned), MOTOR_HALF_MM, -X)


def plane_floor_mm(a: Sequence[float], b: Sequence[float], turned_a: bool = False, turned_b: bool = False) -> float:
    """E15's continuous floor between the earlier axis `a` and the next axis
    `b` (unit, link frame), mm: a wall, the earlier motor's swept front and
    the next motor's back."""
    return WALL_MM + swept_front_mm(a, turned_a) + back_mm(b, turned_b)


def link_floor_mm(a: Sequence[float], b: Sequence[float], turned_a: bool = False, turned_b: bool = False) -> int:
    """The shortest link between two joints with axes `a` then `b`, either
    turned, on the 10 mm grid: 30-50 mm (bend then bend 50, 30 when the next
    bend motor is turned and 40 when both are; spread then twist 30)."""
    return _link_floor_cached(_key(a), _key(b), bool(turned_a), bool(turned_b))


@lru_cache(maxsize=16384)
def _link_floor_cached(a: Tuple[float, ...], b: Tuple[float, ...], turned_a: bool, turned_b: bool) -> int:
    return ceil10(plane_floor_mm(np.array(a), np.array(b), turned_a, turned_b))


def tip_floor_mm(a: Sequence[float], turned: bool = False) -> int:
    """The shortest fingertip link after a joint with axis `a`: the last
    motor's swept front, a wall and a 10 mm pad, on the 10 mm grid (30 mm for
    every joint kind, 40 after a turned bend motor, at most 40)."""
    return ceil10(WALL_MM + swept_front_mm(a, turned) + PAD_MM)


def is_transverse(a: Sequence[float]) -> bool:
    """An axis square to the finger (a bend, a spread, or anything between)."""
    return abs(float(a[0])) < 1e-9


def crossing_ok(a: Sequence[float], b: Sequence[float]) -> bool:
    """Two joints may share a point (a 0 mm link) only if both axes are square
    to the finger and they are different lines."""
    return is_transverse(a) and is_transverse(b) and abs(abs(float(np.dot(a, b))) - 1.0) > 1e-9


def crossing_slide_mm(a: Sequence[float], b: Sequence[float], turned_b: bool = False) -> float:
    """How far the earlier motor (axis `a`) slides along -a so that it clears
    the next motor (axis `b`, at the same point, turned or not) by a wall:
    26.5 mm for square axes, 31 for parallel ones, 34.1 at 30 degrees; 41 for
    a spread crossing a turned bend."""
    a = np.asarray(a, dtype=float)
    B = motor_axes(b, turned_b)
    return max(0.0, MOTOR_HALF_MM[0] + WALL_MM + support(motor_centre(b, 0.0, turned_b), B, MOTOR_HALF_MM, -a))


def box_corners(centre: np.ndarray, axes: np.ndarray, half: np.ndarray) -> np.ndarray:
    """(8, 3): a box's corners."""
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return np.asarray(centre, dtype=float)[None] + (s * np.asarray(half, dtype=float)[None]) @ np.asarray(axes).T


def aabb(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(centre, half) of the axis-aligned box around `points`."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    return (lo + hi) / 2.0, (hi - lo) / 2.0
