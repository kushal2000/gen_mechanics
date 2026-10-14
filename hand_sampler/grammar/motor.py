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

Every motor's body sits in the link before its joint (the palm for a
finger's first joint). The link after it holds the horn and the idler in a
yoke, a 33 x 24 x 24 mm box around the axis (the motor's length plus a wall
at each end, its front plus a wall).

Link floors (E15's plane rule): a plane square to the finger separates the
next motor's back from the corners of this motor's front, which swing on a
circle of sqrt(9.5^2 + 10^2) = 13.79 mm (13.23 mm for a spread motor, whose
+/-30 degree range never swings a corner fully forward), plus one wall.
Rounded up to the 10 mm grid.

Crossing axes (a 0 mm link): the earlier motor slides along its own axis,
away from the joint point, until a plane square to that axis separates it
from the next motor by one wall. A rotation about that axis keeps the plane,
so the two motors stay clear over the whole range.
"""

from __future__ import annotations

import math
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


def ceil10(x: float) -> int:
    """`x` rounded up to the 10 mm grid (exact multiples stay)."""
    return int(10 * math.ceil((float(x) - 1e-9) / 10.0))


def is_spread(a: Sequence[float]) -> bool:
    """Whether an axis (link frame) is nearest the spread (abduction) axis y;
    ties go to bend (z), then spread, as `derive.kind_of`."""
    a = np.abs(np.asarray(a, dtype=float))
    best = max((a[2], 2), (a[1], 1), (a[0], 0), key=lambda t: (round(t[0], 12), t[1]))
    return best[1] == 1


def motor_axes(a: Sequence[float]) -> np.ndarray:
    """3x3, columns (a, v, w): the motor's axis, its back and its width."""
    a = np.asarray(a, dtype=float)
    a = a / np.linalg.norm(a)
    back = -X if abs(a[0]) < 1.0 - 1e-12 else -Y
    v = back - a * float(a @ back)
    v = v / np.linalg.norm(v)
    return np.column_stack([a, v, np.cross(a, v)])


def motor_centre(a: Sequence[float], slide_mm: float = 0.0) -> np.ndarray:
    """The motor box's centre relative to its joint point (joint frame)."""
    M = motor_axes(a)
    return slide_mm * M[:, 0] + MOTOR_CENTRE_BACK_MM * M[:, 1]


def support(centre: np.ndarray, axes: np.ndarray, half: np.ndarray, n: Sequence[float]) -> float:
    """How far a box (centre, axes as columns, half sizes) reaches along the
    unit direction `n`, from the origin."""
    n = np.asarray(n, dtype=float)
    return float(centre @ n + np.abs(n @ axes) @ half)


def front_sweep_mm(a: Sequence[float]) -> float:
    return SPREAD_FRONT_SWEEP_MM if is_spread(a) else FRONT_SWEEP_MM


def plane_floor_mm(a: Sequence[float], b: Sequence[float]) -> float:
    """E15's continuous floor between the earlier axis `a` and the next axis
    `b` (unit, link frame), mm."""
    ta, tb = abs(float(a[0])), abs(float(b[0]))
    return (WALL_MM + MOTOR_HALF_MM[0] * (ta + tb) + front_sweep_mm(a) * math.sqrt(max(0.0, 1.0 - ta * ta))
            + (MOTOR_HALF_MM[1] + MOTOR_CENTRE_BACK_MM) * math.sqrt(max(0.0, 1.0 - tb * tb)))


def link_floor_mm(a: Sequence[float], b: Sequence[float]) -> int:
    """The shortest link between two joints with axes `a` then `b`, on the
    10 mm grid: 30-50 mm (bend then bend 50, spread then bend 40, spread then
    twist 30)."""
    return ceil10(plane_floor_mm(a, b))


def tip_floor_mm(a: Sequence[float]) -> int:
    """The shortest fingertip link after a joint with axis `a`: the last
    motor's swinging front, a wall and a 10 mm pad, on the 10 mm grid (30 mm
    for every joint kind, at most 40)."""
    t = abs(float(a[0]))
    return ceil10(WALL_MM + MOTOR_HALF_MM[0] * t + front_sweep_mm(a) * math.sqrt(max(0.0, 1.0 - t * t)) + PAD_MM)


def is_transverse(a: Sequence[float]) -> bool:
    """An axis square to the finger (a bend, a spread, or anything between)."""
    return abs(float(a[0])) < 1e-9


def crossing_ok(a: Sequence[float], b: Sequence[float]) -> bool:
    """Two joints may share a point (a 0 mm link) only if both axes are square
    to the finger and they are different lines."""
    return is_transverse(a) and is_transverse(b) and abs(abs(float(np.dot(a, b))) - 1.0) > 1e-9


def crossing_slide_mm(a: Sequence[float], b: Sequence[float]) -> float:
    """How far the earlier motor (axis `a`) slides along -a so that it clears
    the next motor (axis `b`, at the same point) by a wall: 26.5 mm for square
    axes, 31 for parallel ones, 34.1 at 30 degrees."""
    a = np.asarray(a, dtype=float)
    B = motor_axes(b)
    return max(0.0, MOTOR_HALF_MM[0] + WALL_MM + support(motor_centre(b), B, MOTOR_HALF_MM, -a))


def box_corners(centre: np.ndarray, axes: np.ndarray, half: np.ndarray) -> np.ndarray:
    """(8, 3): a box's corners."""
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return np.asarray(centre, dtype=float)[None] + (s * np.asarray(half, dtype=float)[None]) @ np.asarray(axes).T


def aabb(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(centre, half) of the axis-aligned box around `points`."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    return (lo + hi) / 2.0, (hi - lo) / 2.0
