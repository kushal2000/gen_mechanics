"""Conform a commercial hand onto the grammar (its one grid: 10 mm, 30
degrees), and measure how well it fits. The aim is the hand's structure
(fingers, joints per finger, joint kinds, the palm joint), roughly in
place, not a close copy: the grid allows about half a step of error.

Fit metric (GRAMMAR-LOCK item 20): the largest joint-position error (mm),
axis error (degrees) and fingertip error (mm), target 10 mm / 20 degrees
(half the coarse grid's steps).
- A pure difference in the zero pose counts as a match: the grammar hand is
  compared at its own joint angles `q_off` (free), the commercial hand at
  its zero pose. Signs and ranges are global conventions, so an axis is a
  line (its error is at most 90 degrees) and ranges are not compared.
- A revolute joint's position is its axis line: sliding a joint along its own
  axis does not change any motion, so the position error is the distance
  from the grammar's joint point to the commercial axis line. A sliding
  joint's position is its child frame's origin.

The palm frame: origin at the hand root's origin (the wrist, where the hand
mounts: the grammar's mount disc sits there) moved onto the plate; x = the
normal of the plane fitted
through the finger bases, on the side the fingertips are when the fingers
leave the plate, else on the side the fingers close toward; z = from the
wrist toward the non-thumb bases. The plane's tilt and offset are then
refined to minimise the fit's worst error.

Per finger: each joint is slid along its axis onto the finger's centre line;
the base is where the first axis meets the plate; facing and tilt point the
finger at its next joint; then, joint by joint, the axis is expressed in the
current link frame and snapped to the 30 degree grid, the joint's zero-pose
difference turns the link toward the next joint, and the link length is
snapped to 10 mm (0 or 10-90 mm, fingertip 10-90 mm). A local search over
one-step changes of every value of the finger then lowers the worst error.
The grammar's finger frame is tilt-defined, so each finger is fitted on its
own; its base joint sits at the derived depth below the plate's front face.

Palm joints: the grammar has at most one, a straight cut of the plate. The
commercial palm joint carrying the most fingers is kept (joints carrying as
many are merged into one cut); the cut is the line on the grid that puts
exactly those fingers beyond it, nearest the commercial hinge axis.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import derive as dv
from . import motor as mt
from . import operators as ops
from . import viability as gvb
from .commercial import RealFinger, RealHand, RealJoint
from .kinematics import KinematicModel, Pose
from .hand import (
    COUPLING_RATIO,
    GRID_DEG,
    GRID_MM,
    LINK_MAX_MM,
    LINK_MIN_MM,
    MIN_CUT_OFFSET_MM,
    COMMERCIAL_RULES,
    NO_RULES,
    TILT_RANGE_DEG,
    TIP_MIN_MM,
    Finger,
    Hand,
    Joint,
    PalmJoint,
    Rules,
    SEAM_MM,
    allowed_lengths,
    bend_first,
    canonical_axis,
    check,
    cut_direction_yz,
    cut_normal_yz,
    shifted,
)

MM = 1e-3
DEG = math.pi / 180.0
TARGET_MM = 10.0
TARGET_DEG = 20.0
Q_WEIGHT = 0.2
"""Tie-breaker: prefer small zero-pose differences (per radian of mean
|q_off|), so a conformed finger lies where the real one does at rest."""
CLASH_WEIGHT = 5.0
"""Cost per millimetre a refitted finger's solids come inside the wall of
the rest of the hand, as C1 counts it (`Guard`, `refit_clear`): a 2 mm clash
outweighs a fit error of 20 mm."""
MAX_SLIDE_M = 0.04      # sliding a joint onto the finger's centre line
BASE_SHIFT_M = 0.002    # the first joint slides onto the plate if that moves the base at most this far
PALM_HINGE_SHIFT_M = 0.1  # the same for a palm joint's hinge


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0.0


def _rot(axis: np.ndarray, q: float) -> np.ndarray:
    return dv._rot_batch(np.asarray(axis, dtype=float), np.array([q]))[0]


def line_distance(p: np.ndarray, a: np.ndarray, u: np.ndarray) -> float:
    d = p - a
    return float(np.linalg.norm(d - u * float(d @ u)))


def axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = abs(float(np.clip(_unit(a) @ _unit(b), -1.0, 1.0)))
    return math.degrees(math.acos(min(1.0, c)))


def closest_on_line_to_line(p: np.ndarray, u: np.ndarray, q: np.ndarray, v: np.ndarray) -> Optional[float]:
    """s such that p + s u is the point of line (p, u) closest to line (q, v);
    None when the lines are (nearly) parallel."""
    w = p - q
    b = float(u @ v)
    den = 1.0 - b * b
    if den < 0.25 ** 2:          # within ~15 degrees: ill-conditioned, do not slide
        return None
    return (b * float(v @ w) - float(u @ w)) / den


def to_az_el(v: np.ndarray) -> Tuple[float, float]:
    """Continuous (az, el) degrees of the direction `v` (link-frame
    coordinates x, y, z)."""
    v = _unit(v)
    el = math.degrees(math.asin(max(-1.0, min(1.0, float(v[0])))))
    az = math.degrees(math.atan2(float(v[1]), float(v[2])))
    return az, el


def snap_axis(v: np.ndarray, step: int = GRID_DEG) -> Tuple[int, int]:
    """The grid line nearest the line through `v` (the four grid corners
    around it, in both orientations)."""
    best, best_ang = (0, 90), 1e9
    for s in (1.0, -1.0):
        az, el = to_az_el(s * v)
        for a in (math.floor(az / step) * step, math.ceil(az / step) * step):
            for e in (math.floor(el / step) * step, math.ceil(el / step) * step):
                if abs(e) > 90:
                    continue
                cand = canonical_axis(int(a), int(e))
                ang = axis_angle_deg(dv.direction(*cand), v)
                if ang < best_ang - 1e-12:
                    best, best_ang = cand, ang
    return best


def snap_length(L_mm: float, is_tip: bool, max_mm: int = LINK_MAX_MM) -> int:
    """The nearest length on the grid the grammar allows (0 or 10 mm up to
    `max_mm`; a fingertip from 10 mm)."""
    r = int(round(L_mm / GRID_MM)) * GRID_MM
    lo = TIP_MIN_MM if is_tip else 0
    return int(min(max_mm, max(lo, r)))


def snap_angle(v: float, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    r = int(round(v / GRID_DEG)) * GRID_DEG
    if lo is not None:
        r = max(lo, min(hi, r))
    return r


# --------------------------------------------------------------------------
# One finger
# --------------------------------------------------------------------------


@dataclass
class FingerTarget:
    """A commercial finger in the palm frame (metres)."""

    points: List[np.ndarray]     # slid joint points
    axes: List[np.ndarray]
    types: List[str]             # hinge / coupled / sliding
    tip: np.ndarray
    names: List[str]
    raw_points: List[np.ndarray]
    scale: float = 1.0           # the hand is this many times its real size; errors count at the real size
    q_samples: Optional[np.ndarray] = None    # (K, n) joint angles drawn over the commercial ranges (`tip_map_mm`)
    tip_samples: Optional[np.ndarray] = None  # (K, 3) the commercial fingertip at them, palm frame (m)


@dataclass
class FingerFit:
    finger: Finger
    q_off: np.ndarray
    joint_mm: List[float]
    axis_deg: List[float]
    tip_mm: float
    clash_mm: float = 0.0
    map_rms_mm: float = -1.0       # the fingertip map (`tip_map_mm`), mm at the real size; -1 = not measured
    map_worst_mm: float = -1.0

    @property
    def cost(self) -> float:
        worst = max([e / TARGET_MM for e in self.joint_mm] + [a / TARGET_DEG for a in self.axis_deg]
                    + [self.tip_mm / TARGET_MM])
        mean = (sum(self.joint_mm) / TARGET_MM + sum(self.axis_deg) / TARGET_DEG + self.tip_mm / TARGET_MM) / (
            2 * len(self.joint_mm) + 1)
        bend = float(np.mean(np.abs(self.q_off))) if len(self.q_off) else 0.0
        return worst + 0.1 * mean + Q_WEIGHT * bend + CLASH_WEIGHT * self.clash_mm


def _targets(fp: Sequence[np.ndarray], tip: np.ndarray, j: int) -> np.ndarray:
    """The first point after joint j at least 3 mm away from point j."""
    for k in range(j + 1, len(fp)):
        if np.linalg.norm(fp[k] - fp[j]) > 3e-3:
            return fp[k]
    return tip


def _aim(x: np.ndarray, u: np.ndarray, d: np.ndarray) -> float:
    """The angle about `u` that turns direction `x` closest to `d`."""
    xp = x - u * float(u @ x)
    if np.linalg.norm(xp) < 1e-9:
        return 0.0
    dp = d - u * float(u @ d)
    if np.linalg.norm(dp) < 1e-9:
        return 0.0
    return math.atan2(float(dp @ np.cross(u, xp)), float(dp @ xp))


def _frame(facing_deg: float, tilt_deg: float) -> np.ndarray:
    return dv.frame_of(facing_deg, tilt_deg)


def _tie_unsigned(B: np.ndarray, axes_link: Sequence[np.ndarray], types: Sequence[str], q: np.ndarray) -> np.ndarray:
    """`q` (angles about the stored, unsigned axis directions) with every
    coupled joint following the joint before it at 1.1 in the derived
    (signed) convention, offset 0."""
    q = np.array(q, dtype=float, copy=True)
    signs = [dv.axis_sign(_unit(a), B, t == "sliding") for a, t in zip(axes_link, types)]
    for j, t in enumerate(types):
        if t == "coupled" and j > 0:
            q[j] = signs[j] * COUPLING_RATIO * signs[j - 1] * q[j - 1]
    return q


def _chain(mount: np.ndarray, B: np.ndarray, axes_link: Sequence[np.ndarray], lengths_mm: Sequence[float],
           types: Sequence[str], q: np.ndarray):
    """Joint points, world axes and the fingertip of a grammar finger at its
    joint angles `q` about the stored axis directions (coupled joints follow
    their source; sliding joints are left at 0)."""
    q = _tie_unsigned(B, axes_link, types, q)
    G = mount.copy()
    F = B.copy()
    pts, axw = [], []
    for j, (a, L) in enumerate(zip(axes_link, lengths_mm)):
        pts.append(G.copy())
        axw.append(F @ a)
        if types[j] != "sliding":
            F = F @ _rot(a, float(q[j]))
        G = G + F[:, 0] * (L * MM)
    return pts, axw, G


def _errors(t: FingerTarget, pts, axw, tip) -> Tuple[List[float], List[float], float]:
    """Joint, axis and fingertip errors; millimetres at the real hand's size
    (a hand scaled up 2 x with a 20 mm error counts 10 mm), so the fit is
    the same whatever the scale."""
    u = MM * t.scale
    jmm, adeg = [], []
    for j, (G, a) in enumerate(zip(pts, axw)):
        if t.types[j] == "sliding":
            jmm.append(float(np.linalg.norm(G - t.points[j])) / u)
        else:
            jmm.append(line_distance(G, t.points[j], t.axes[j]) / u)
        adeg.append(axis_angle_deg(a, t.axes[j]))
    return jmm, adeg, float(np.linalg.norm(tip - t.tip)) / u


def _residuals(t: FingerTarget, pts, axw, tip) -> np.ndarray:
    out = []
    sin_target = math.sin(TARGET_DEG * DEG)
    for j, (G, a) in enumerate(zip(pts, axw)):
        d = G - t.points[j]
        if t.types[j] != "sliding":
            d = d - t.axes[j] * float(d @ t.axes[j])
        out.extend(d / (TARGET_MM * MM * t.scale))
        out.extend(np.cross(_unit(a), t.axes[j]) / sin_target)
    out.extend((tip - t.tip) / (TARGET_MM * MM * t.scale))
    return np.array(out)


def _lm(fun, x0: np.ndarray, iters: int = 40, eps: float = 1e-6) -> np.ndarray:
    x = x0.astype(float).copy()
    r = fun(x)
    c = float(r @ r)
    lam = 1e-2
    for _ in range(iters):
        J = np.zeros((len(r), len(x)))
        for k in range(len(x)):
            dx = x.copy()
            dx[k] += eps
            J[:, k] = (fun(dx) - r) / eps
        H = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(8):
            step = np.linalg.solve(H + lam * (np.diag(np.diag(H)) + 1e-9 * np.eye(len(x))), -g)
            xn = x + step
            rn = fun(xn)
            cn = float(rn @ rn)
            if cn < c:
                x, r, c = xn, rn, cn
                lam = max(lam / 3.0, 1e-7)
                improved = True
                break
            lam *= 5.0
        if not improved or float(np.abs(step).max()) < 1e-9:
            break
    return x


def evaluate_finger(f: Finger, t: FingerTarget, q: Optional[np.ndarray] = None,
                    guard: Optional["Guard"] = None) -> FingerFit:
    """The errors of `f` against `t` at joint angles `q` (default: each joint
    turns its link toward the next commercial joint); with `guard`, also how
    far its solids come inside the wall of the rest of the hand at rest."""
    B = dv.finger_frame(f)
    mount = dv.mount_point_m(f)
    axes = [dv.direction(*jt.axis) for jt in f.joints]
    lengths = [jt.length for jt in f.joints]
    if q is None:
        q = np.zeros(len(f.joints))
        G, F = mount.copy(), B.copy()
        for j, jt in enumerate(f.joints):
            if jt.type != "sliding":
                q[j] = _aim(F[:, 0], _unit(F @ axes[j]), _unit(_targets(t.points, t.tip, j) - G))
                F = F @ _rot(axes[j], q[j])
            G = G + F[:, 0] * (jt.length * MM)
    q = np.array([math.remainder(float(v), 2.0 * math.pi) for v in q])
    q = _tie_unsigned(B, axes, [jt.type for jt in f.joints], q)
    pts, axw, tip = _chain(mount, B, axes, lengths, t.types, q)
    jmm, adeg, tmm = _errors(t, pts, axw, tip)
    return FingerFit(f, q, jmm, adeg, tmm, guard.clash_mm(f) if guard is not None else 0.0)


def best_q(f: Finger, t: FingerTarget, q0: np.ndarray, iters: int = 15, guard: Optional["Guard"] = None) -> FingerFit:
    """The zero-pose differences that fit `f` best (least squares from `q0`)."""
    B = dv.finger_frame(f)
    mount = dv.mount_point_m(f)
    axes = [dv.direction(*jt.axis) for jt in f.joints]
    lengths = [jt.length for jt in f.joints]
    rev = [j for j, jt in enumerate(f.joints) if jt.type == "hinge"]
    if not rev:
        return evaluate_finger(f, t, np.zeros(len(f.joints)), guard)

    def fun(x):
        q = np.zeros(len(f.joints))
        q[rev] = x
        return np.concatenate([_residuals(t, *_chain(mount, B, axes, lengths, t.types, q)), Q_WEIGHT * x])

    x = _lm(fun, np.asarray(q0, dtype=float)[rev], iters=iters)
    q = np.zeros(len(f.joints))
    q[rev] = x
    return evaluate_finger(f, t, q, guard)


def continuous_fit(t: FingerTarget, mount: np.ndarray,
                   frame: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, dict]:
    """Least-squares fit of every finger value, unsnapped: facing, tilt, the
    axes (az, el), the link lengths and the zero-pose differences, the mount
    fixed on the plate (with `frame`, the facing and tilt fixed at it).
    Returns the parameter vector and its layout."""
    n = len(t.axes)
    d0 = _unit(_targets(t.points, t.tip, 0) - mount)
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, float(d0[0])))))
    facing = math.degrees(math.atan2(float(d0[1]), float(d0[2])))
    if frame is not None:
        facing, tilt = float(frame[0]), float(frame[1])
    B = _frame(facing, tilt)
    P, F = mount.copy(), B.copy()
    az, el, L, q = [], [], [], []
    for j in range(n):
        a_link = _unit(F.T @ t.axes[j])
        a1, e1 = to_az_el(a_link)
        az.append(a1)
        el.append(e1)
        qj = 0.0
        if t.types[j] != "sliding":
            qj = _aim(F[:, 0], t.axes[j], _unit(_targets(t.points, t.tip, j) - P))
            F = F @ _rot(a_link, qj)
        q.append(qj)
        nxt = t.points[j + 1] if j + 1 < n else t.tip
        Lj = max(0.0, float((nxt - P) @ F[:, 0])) / MM
        L.append(Lj)
        P = P + F[:, 0] * (Lj * MM)
    x0 = np.array([facing, tilt] + az + el + L + q, dtype=float)

    def unpack(x):
        fa, ti = x[0], x[1]
        A = [dv.direction(x[2 + j], x[2 + n + j]) for j in range(n)]
        Ls = list(np.maximum(x[2 + 2 * n: 2 + 3 * n], 0.0))
        qs = x[2 + 3 * n: 2 + 4 * n]
        return fa, ti, A, Ls, qs

    def fun(x):
        fa, ti, A, Ls, qs = unpack(x)
        return np.concatenate([_residuals(t, *_chain(mount, _frame(fa, ti), A, Ls, t.types, qs)),
                               Q_WEIGHT * qs])

    if frame is None:
        x = _lm(fun, x0, iters=60, eps=1e-5)
    else:
        x = np.concatenate([x0[:2], _lm(lambda y: fun(np.concatenate([x0[:2], y])), x0[2:], iters=60, eps=1e-5)])
    return x, {"n": n, "unpack": unpack}


def snap_continuous(x: np.ndarray, layout: dict, t: FingerTarget, mount_mm: Tuple[int, int],
                    rules: Rules = NO_RULES) -> Tuple[Finger, np.ndarray]:
    """Snap a continuous fit to the grid, joint by joint from the palm: each
    axis is snapped in the (already snapped) link frame; then every link to
    its nearest length `rules` allow (the servo floors for buildable rules)."""
    fa, ti, A, Ls, qs = layout["unpack"](x)
    facing = snap_angle(fa) % 360
    tilt = snap_angle(ti, *TILT_RANGE_DEG)
    f = Finger(y=mount_mm[0], z=mount_mm[1], facing=facing, tilt=tilt, joints=())
    B_cont = _frame(fa, ti)
    B = dv.finger_frame(f)
    # continuous world axes, re-expressed in the snapped frame chain
    F_c = B_cont.copy()
    F = B.copy()
    joints = []
    n = layout["n"]
    for j in range(n):
        a_world = F_c @ A[j]
        ax = snap_axis(F.T @ a_world)
        if t.types[j] != "sliding":
            F_c = F_c @ _rot(A[j], float(qs[j]))
            F = F @ _rot(dv.direction(*ax), float(qs[j]))
        joints.append(Joint(t.types[j], ax, snap_length(Ls[j], j == n - 1, rules.max_link_mm)))
    f = ops.fit_lengths(replace(f, joints=tuple(joints)), rules)
    qs = np.asarray(qs, dtype=float).copy()
    if f.tilt and bend_first(f):      # the tilt turns about the first bend's axis: fold it into that joint
        qs[0] += math.radians(f.tilt)
        f = replace(f, tilt=0)
    return f, qs


def _neighbours(f: Finger, mount_fixed: bool = False, rules: Rules = NO_RULES) -> List[Finger]:
    """Every finger one grid step away within `rules` (a turned axis moves
    the links next to it to their nearest allowed lengths)."""
    out = []
    if not mount_fixed:
        for c in ("y", "z"):
            for s in (-1, 1):
                out.append(replace(f, **{c: getattr(f, c) + s * GRID_MM}))
    for s in (-GRID_DEG, GRID_DEG):
        out.append(replace(f, facing=(f.facing + s) % 360))
        if TILT_RANGE_DEG[0] <= f.tilt + s <= TILT_RANGE_DEG[1]:
            out.append(replace(f, tilt=f.tilt + s))
    for j, jt in enumerate(f.joints):
        def put(new: Joint) -> Finger:
            js = list(f.joints)
            js[j] = new
            return replace(f, joints=tuple(js))

        for s in (-GRID_DEG, GRID_DEG):
            out.append(ops.fit_lengths(put(replace(jt, axis=canonical_axis(jt.axis[0] + s, jt.axis[1]))), rules))
            out.append(ops.fit_lengths(put(replace(jt, axis=canonical_axis(jt.axis[0], jt.axis[1] + s))), rules))
        allowed = allowed_lengths(f, j, rules.buildable_links, rules.max_link_mm)
        if jt.length in allowed:
            k = allowed.index(jt.length)
            for kk in (k - 1, k + 1):
                if 0 <= kk < len(allowed):
                    out.append(put(replace(jt, length=allowed[kk])))
    # a bend first has tilt 0 (the zero-pose fit takes the difference)
    out = [replace(g, tilt=0) if g.tilt and bend_first(g) else g for g in out]
    return [g for g in out if g != f]


def refine_finger(fit: FingerFit, t: FingerTarget, mount_fixed: bool = True,
                  max_passes: int = 30, rules: Rules = NO_RULES, guard: Optional["Guard"] = None) -> FingerFit:
    """Local search over one-step changes of every value of the finger (its
    base too unless `mount_fixed`), the zero-pose differences re-fitted
    (warm-started) for each candidate."""
    best = fit
    for _ in range(max_passes):
        improved = False
        for g in _neighbours(best.finger, mount_fixed=mount_fixed, rules=rules):
            r = best_q(g, t, best.q_off, iters=4, guard=guard)
            if r.cost < best.cost - 1e-9:
                best, improved = r, True
        if not improved:
            break
    return best_q(best.finger, t, best.q_off, iters=30, guard=guard)


class Guard:
    """The rest of a hand, for refitting finger i clear of it: `clash_mm(f)`
    is C1 on the hand with finger i replaced by f, counting only what
    involves f or a plate (the plates are the hull of the base servos, so
    they change with f): the summed overlaps in millimetres, at the zero and
    start poses, plus how far its base servo reaches into a cut's seam."""

    def __init__(self, hand: Hand, i: int):
        self.hand, self.i = hand, i

    def clash_mm(self, f: Finger) -> float:
        h = replace(self.hand, fingers=tuple(f if k == self.i else g for k, g in enumerate(self.hand.fingers)))
        total = sum(ov for (a, _b, ov) in gvb.seam_overlaps(h) if a[1] == self.i)
        for q in dv.check_poses(h):
            total += sum(ov for _a, _b, ov in gvb.overlaps(h, q, focus_finger=self.i))
        return float(total)


def kinds_kept(hand: Hand, palm_T: np.ndarray, q_off: np.ndarray, real: RealHand,
               fingers: Optional[Sequence[int]] = None) -> bool:
    """Whether every joint (of `fingers`, default all) keeps its commercial
    kind: the commercial axis, seen in the fitted link frame at the
    zero-pose difference, is nearest the same kind as the grammar joint, or
    lies within the fit target (20 degrees) of the grammar axis (a commercial
    axis about 45 degrees between two kinds, as on SHARPA's thumb, has no
    clear kind)."""
    R = palm_T[:3, :3]
    pose = dv.fk(hand, q_off)
    for i, (f, rf) in enumerate(zip(hand.fingers, real.fingers)):
        if fingers is not None and i not in fingers:
            continue
        for j, (jt, rj) in enumerate(zip(f.joints, rf.joints)):
            if jt.type == "sliding":
                continue
            a = pose.links[i][j][:3, :3].T @ (R.T @ rj.axis)
            if dv.kind_of(a) != dv.joint_kind(hand, i, j) and \
                    axis_angle_deg(a, dv.direction(*jt.axis)) > TARGET_DEG:
                return False
    return True


def _frame_sweep(t: FingerTarget, f: Finger, rules: Rules, guard: Guard) -> List[FingerFit]:
    """Finger f fitted in every grid frame (12 facings x 4 tilts) on its
    base, the clash with the rest of the hand counted, best first (not yet
    refined). A thumb fitted lying along the palm can come out standing off
    it, the difference taken into its zero-pose offsets."""
    mount = np.array([float(dv.mount_point_m(f)[0]), f.y * MM, f.z * MM])
    cands = []
    for fa in range(0, 360, GRID_DEG):
        for ti in range(TILT_RANGE_DEG[0], TILT_RANGE_DEG[1] + 1, GRID_DEG):
            x, layout = continuous_fit(t, mount, frame=(fa, ti))
            g, q = snap_continuous(x, layout, t, (f.y, f.z), rules)
            cands.append(best_q(g, t, q, iters=10, guard=guard))
    return sorted(cands, key=lambda r: r.cost)



# --------------------------------------------------------------------------
# Fitting the fingertip map (Martin, 2026-10-20)
# --------------------------------------------------------------------------

TIP_FIT_SAMPLES = 64
"""How many of the sampled configurations the fingertip-map fit uses."""

MAP_RMS_MM = 15.0
"""A scaled commercial fit counts when every finger's fingertip map stays
within this RMS (mm at the real size) ..."""
MAP_WORST_MM = 40.0
"""... and its worst sampled fingertip within this (Martin, 2026-10-20:
match the fingertips, not every joint)."""


def _map_fit(f: Finger, t: FingerTarget, q: np.ndarray, guard: Optional["Guard"] = None) -> FingerFit:
    r = evaluate_finger(f, t, np.asarray(q, dtype=float), guard)
    r.map_rms_mm, r.map_worst_mm = tip_map_mm(f, t, r.q_off)
    return r


def map_cost(r: FingerFit) -> float:
    """What the fingertip-map fit minimises: the map's RMS and worst miss
    against their limits, and any clash with the rest of the hand."""
    return r.map_rms_mm / MAP_RMS_MM + 0.5 * r.map_worst_mm / MAP_WORST_MM + CLASH_WEIGHT * r.clash_mm


def best_q_map(f: Finger, t: FingerTarget, q0: np.ndarray, iters: int = 15,
               guard: Optional["Guard"] = None) -> FingerFit:
    """The zero-pose differences that make `f`'s fingertip follow the
    commercial one best over the sampled configurations (least squares)."""
    rev = [j for j, jt in enumerate(f.joints) if jt.type == "hinge"]
    Qs, Ps = t.q_samples[:TIP_FIT_SAMPLES], t.tip_samples[:TIP_FIT_SAMPLES]
    q = np.asarray(q0, dtype=float).copy()
    if rev:
        def fun(x):
            qq = q.copy()
            qq[rev] = x
            return np.concatenate([((_tips_batch(f, t, qq, Qs) - Ps) / (TARGET_MM * MM * t.scale)).ravel(),
                                   Q_WEIGHT * x])

        q[rev] = _lm(fun, q[rev], iters=iters)
    return _map_fit(f, t, q, guard)


def refine_map(fit: FingerFit, t: FingerTarget, rules: Rules, guard: Optional["Guard"] = None,
               max_passes: int = 8) -> FingerFit:
    """Local search over one-step changes of every value of the finger, its
    base too, for the best fingertip map clear of the rest of the hand: the
    joints may move away from the commercial ones as long as the fingertip
    follows. Every joint keeps its kind."""
    kinds = [dv.kind_of(dv.direction(*jt.axis)) for jt in fit.finger.joints]
    best = fit if fit.map_rms_mm >= 0 else _map_fit(fit.finger, t, fit.q_off, guard)
    for _ in range(max_passes):
        improved = False
        for g in _neighbours(best.finger, mount_fixed=False, rules=rules):
            if [dv.kind_of(dv.direction(*jt.axis)) for jt in g.joints] != kinds:
                continue
            r = best_q_map(g, t, best.q_off, iters=6, guard=guard)
            if map_cost(r) < map_cost(best) - 1e-9:
                best, improved = r, True
        if not improved:
            break
    return best_q_map(best.finger, t, best.q_off, iters=30, guard=guard)


def map_refit(fit: "HandFit", real: RealHand, rules: Rules, rounds: int = 2) -> "HandFit":
    """Every finger refitted for its fingertip map (`refine_map`), clear of
    the rest of the hand, then the mount moved back to the palm's centre."""
    targets = _targets_for(real, fit.palm_T)
    for _ in range(rounds):
        changed = False
        for i in range(len(fit.hand.fingers)):
            guard = Guard(fit.hand, i)
            start = _map_fit(fit.hand.fingers[i], targets[i], fit.fingers[i].q_off, guard)
            new = refine_map(start, targets[i], rules, guard)
            fit.fingers[i] = replace(start, clash_mm=0.0)
            if map_cost(new) < map_cost(start) - 1e-9:
                old_hand, old = fit.hand, fit.fingers[i]
                fingers = list(fit.hand.fingers)
                fingers[i] = new.finger
                fit.hand = replace(fit.hand, fingers=tuple(fingers))
                fit.fingers[i] = replace(new, clash_mm=0.0)
                if kinds_kept(fit.hand, fit.palm_T, fit.q_off(), real, [i]):
                    changed = True
                else:          # moving the finger's frame can change how a commercial axis's kind reads
                    fit.hand, fit.fingers[i] = old_hand, old
        if not changed:
            break
    return recentre_fit(fit, real, rules)


def map_within(fit: "HandFit") -> bool:
    """Every finger's fingertip map within `MAP_RMS_MM` and `MAP_WORST_MM`."""
    return all(0.0 <= f.map_rms_mm <= MAP_RMS_MM and f.map_worst_mm <= MAP_WORST_MM for f in fit.fingers)


SWEEP_TRIES = 8
"""How many of the sweep's best frames `refit_clear` refines, in turn, before
giving up on a finger."""


def refit_clear(fit: "HandFit", real: RealHand, rules: Rules, rounds: int = 3) -> "HandFit":
    """Refit, with `Guard`, every finger whose solids fail C1 against another
    finger, a plate or the mount (a thumb resting across the palm, fingers
    crossing at rest), until the hand passes C1 or `rounds` run out: first by
    local search, then, if that refit still clashes or changes a joint's
    kind, in every grid frame (`_frame_sweep`, its best frames refined in
    turn). A refit may tilt or turn a finger clear and take the difference
    into its zero-pose offset. The first refit that is better (cost or
    clash) and keeps every joint's kind is kept."""
    targets = _targets_for(real, fit.palm_T)
    for _ in range(rounds):
        c1 = gvb.c1_self_overlap(fit.hand)
        if c1.ok:
            break
        bad = sorted({k[1] for a, b, _ in c1.pairs for k in (a, b) if k[0] in ("link", "motor")})
        if not bad:
            break
        for i in bad:
            guard = Guard(fit.hand, i)
            f = fit.hand.fingers[i]
            start = best_q(f, targets[i], fit.fingers[i].q_off, iters=30, guard=guard)

            def take(new: FingerFit) -> bool:
                if not (new.cost < start.cost - 1e-9 or new.clash_mm < start.clash_mm):
                    return False
                fingers = list(fit.hand.fingers)
                fingers[i] = new.finger
                old_hand, old_fit = fit.hand, fit.fingers[i]
                # the joint vector's signs come from the hand: swap both before reading it
                fit.hand, fit.fingers[i] = replace(fit.hand, fingers=tuple(fingers)), replace(new, clash_mm=0.0)
                if kinds_kept(fit.hand, fit.palm_T, fit.q_off(), real, [i]):
                    return True
                fit.hand, fit.fingers[i] = old_hand, old_fit
                return False

            local = refine_finger(start, targets[i], mount_fixed=True, rules=rules, guard=guard)
            if local.clash_mm == 0 and take(local):
                continue
            for c in _frame_sweep(targets[i], f, rules, guard)[:SWEEP_TRIES]:
                new = refine_finger(c, targets[i], mount_fixed=True, rules=rules, guard=guard)
                if new.clash_mm == 0 and take(new):
                    break
            else:
                take(local)
    return fit


def _grid_point(m: np.ndarray) -> Tuple[int, int]:
    return (int(round(m[1] / MM / GRID_MM)) * GRID_MM, int(round(m[2] / MM / GRID_MM)) * GRID_MM)


def fit_finger(t: FingerTarget, base_mm: Tuple[int, int], mount_fixed: bool = True,
               rules: Rules = NO_RULES) -> FingerFit:
    """Fit one finger starting from its base at `base_mm` (kept there when
    `mount_fixed`). The continuous fit starts on the plate's mid-plane, then
    once more from the depth the snapped finger's first motor puts its base
    at (an oblique or tilted first motor sits deeper), and the better start
    is refined."""
    y, z = base_mm
    best = None
    x0 = 0.0
    for _ in range(2):
        x, layout = continuous_fit(t, np.array([x0, y * MM, z * MM]))
        f, q = snap_continuous(x, layout, t, (y, z), rules)
        start = best_q(f, t, q, iters=30)
        aimed = evaluate_finger(f, t, None)
        if aimed.cost < start.cost:
            start = best_q(f, t, aimed.q_off, iters=30)
        if best is None or start.cost < best.cost:
            best = start
        x_next = float(dv.mount_point_m(f)[0])
        if abs(x_next - x0) < 1e-9:
            break
        x0 = x_next
    return refine_finger(best, t, mount_fixed, rules=rules)


def _fit_free(t: FingerTarget, base_mm: Tuple[int, int], rules: Rules = NO_RULES) -> FingerFit:
    """`fit_finger` with the base free, started from `base_mm` and, when the
    first two commercial joints are closer than the shortest link, also from
    the plate point under their midpoint (the base then sits between them)."""
    best = fit_finger(t, base_mm, mount_fixed=False, rules=rules)
    if len(t.points) > 1 and np.linalg.norm(t.points[1] - t.points[0]) < LINK_MIN_MM * MM:
        seed = _grid_point(0.5 * (t.points[0] + t.points[1]))
        if seed != tuple(base_mm):
            other = fit_finger(t, seed, mount_fixed=False, rules=rules)
            if other.cost < best.cost:
                best = other
    return best


# --------------------------------------------------------------------------
# The whole hand
# --------------------------------------------------------------------------


@dataclass
class HandFit:
    id: str
    hand: Hand
    palm_T: np.ndarray                 # 4x4: the palm frame in the hand-root frame
    fingers: List[FingerFit]
    palm_joint_mm: List[float]         # the cut's hinge from the commercial hinge axis (0 or 1 entry)
    palm_axis_deg: List[float]         # the cut's direction against the commercial hinge axis (reported only)
    name_map: Dict[str, str]           # grammar dof name -> commercial joint name
    notes: List[str] = field(default_factory=list)
    finger_order: List[int] = field(default_factory=list)   # grammar finger i <- real finger index
    section: List[int] = field(default_factory=list)        # fingers beyond the cut
    dropped_palm_joints: List[str] = field(default_factory=list)   # commercial palm joints with no cut of their own
    scale: float = 1.0                 # the commercial hand was scaled up by this before fitting

    @property
    def max_joint_mm(self) -> float:
        """The fingers' worst joint error (the palm cut's hinge is reported on
        its own: a straight cut that keeps the right fingers beyond it cannot
        always run along the commercial hinge axis)."""
        return max([e for f in self.fingers for e in f.joint_mm] + [0.0])

    @property
    def max_axis_deg(self) -> float:
        """The fingers' worst axis error (a straight cut in the plate cannot
        follow a hinge tilted out of it, so the palm joint's axis error is
        reported on its own)."""
        return max([e for f in self.fingers for e in f.axis_deg] + [0.0])

    @property
    def max_tip_mm(self) -> float:
        return max([f.tip_mm for f in self.fingers] + [0.0])

    @property
    def within_target(self) -> bool:
        return self.max_joint_mm <= TARGET_MM and self.max_tip_mm <= TARGET_MM and self.max_axis_deg <= TARGET_DEG

    def q_off(self) -> np.ndarray:
        """The zero-pose differences in `derive.dofs` order and the derived
        sign convention (the fit works about the stored axis directions)."""
        ds = dv.dofs(self.hand)
        q = np.zeros(len(ds))
        for k, d in enumerate(ds):
            if d.finger >= 0 and d.type != "sliding":
                q[k] = dv.joint_sign(self.hand, d.finger, d.index) * self.fingers[d.finger].q_off[d.index]
        return dv.tie(self.hand, q, ds)


def _slide_points(f: RealFinger, frame: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Joint points slid along their axes onto the finger's centre line
    (palm frame)."""
    R, o = frame[:3, :3], frame[:3, 3]
    P = [R.T @ (j.point - o) for j in f.joints]
    U = [_unit(R.T @ j.axis) for j in f.joints]
    tip = R.T @ (f.tip - o)
    S = [p.copy() for p in P]
    for _ in range(5):
        pts = np.array(S + [tip])
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c)
        v = vt[0] if float(vt[0] @ (tip - P[0])) >= 0 else -vt[0]
        newS = []
        for p, u, j in zip(P, U, f.joints):
            if j.type == "prismatic":
                newS.append(p)
                continue
            s = closest_on_line_to_line(p, u, c, v)
            if s is None or abs(s) > MAX_SLIDE_M:
                newS.append(p)
            else:
                newS.append(p + s * u)
        S = newS
    return S, U


def _mount_on_plate(S0: np.ndarray, u0: np.ndarray, jtype: str, max_shift: float = BASE_SHIFT_M) -> np.ndarray:
    """The finger's base on the plate (x = 0): where the first joint's axis
    line meets the plate if that moves the base at most `max_shift` across
    the plate (an axis near the plate's normal); otherwise the slid base
    dropped onto the plate, and its out-of-plane offset is an error the
    grammar keeps."""
    if jtype != "prismatic" and abs(u0[0]) > 1e-6:
        s = -S0[0] / u0[0]
        p = S0 + s * u0
        if float(np.hypot(p[1] - S0[1], p[2] - S0[2])) <= max_shift:
            return np.array([0.0, p[1], p[2]])
    return np.array([0.0, S0[1], S0[2]])


def palm_frame_guess(real: RealHand) -> np.ndarray:
    """4x4: the initial palm frame in the hand-root frame."""
    bases = []
    for f in real.fingers:
        S, _ = _slide_points(f, np.eye(4))
        bases.append(S[0])
    bases = np.array(bases)
    tips = np.array([f.tip for f in real.fingers])
    c = bases.mean(axis=0)
    if len(bases) >= 3:
        _, sv, vt = np.linalg.svd(bases - c)
        n = vt[2] if sv[1] > 1e-3 else None
    else:
        n = None
    # the closing direction: fingertips as each joint turns toward its larger limit
    from .fk import forward_kinematics

    model = real.model
    q = {}
    for j in model.joints:
        if j.type in ("revolute", "prismatic") and j.limits is not None:
            lo, hi = j.limits
            q[j.name] = 0.2 * (1.0 if abs(hi) >= abs(lo) else -1.0) * (1.0 if j.type == "revolute" else 0.05)
    W0 = forward_kinematics(model, {})
    W1 = forward_kinematics(model, q)
    move = np.zeros(3)
    for f in real.fingers:
        if f.thumb:
            continue
        b = f.joints[-1]
        body = next(jj.child for jj in model.joints if jj.name == b.name)
        T0, T1 = W0[body], W1[body]
        local = np.linalg.solve(T0, np.append(f.tip, 1.0))
        move += (T1 @ local)[:3] - f.tip
    out_dir = tips.mean(axis=0) - c
    if n is None:
        # bases on a line: if the fingers close across the plane holding the
        # line and the fingers, that plane is the plate; otherwise (jaws that
        # leave the plate) the normal points along the fingers
        d = _unit(bases[-1] - bases[0]) if len(bases) >= 2 else np.array([0.0, 0.0, 1.0])
        across = _unit(np.cross(d, out_dir))
        if np.linalg.norm(across) > 0.5 and abs(float(across @ move)) > 0.3 * np.linalg.norm(move) > 0:
            n = across
        else:
            n = _unit(out_dir - d * float(out_dir @ d))
        if np.linalg.norm(n) < 1e-9:
            n = np.array([1.0, 0.0, 0.0])
    if abs(float(n @ out_dir)) > 0.5 * np.linalg.norm(out_dir):
        # fingers that leave the plate (DClaw, Dex1, Barrett): the grasp side is where they point
        n = n if float(n @ out_dir) > 0 else -n
    elif abs(float(n @ move)) > 0.3 * np.linalg.norm(move) and np.linalg.norm(move) > 1e-6:
        n = n if float(n @ move) > 0 else -n
    else:
        n = n if float(n @ out_dir) >= 0 else -n
    return _frame_from(n, c, real)


def _frame_from(n: np.ndarray, plane_point: np.ndarray, real: RealHand) -> np.ndarray:
    n = _unit(n)
    origin = np.zeros(3) - n * float(n @ (np.zeros(3) - plane_point))
    nonthumb = [f for f in real.fingers if not f.thumb] or real.fingers
    cb = np.mean([f.joints[0].point for f in nonthumb], axis=0)
    zdir = cb - origin
    zdir = zdir - n * float(n @ zdir)
    if np.linalg.norm(zdir) < 5e-3:   # bases around the root: point z toward the mean fingertip
        zdir = np.mean([f.tip for f in real.fingers], axis=0) - origin
        zdir = zdir - n * float(n @ zdir)
    if np.linalg.norm(zdir) < 1e-6:
        zdir = np.array([0.0, 0.0, 1.0]) - n * n[2]
    z = _unit(zdir)
    y = np.cross(z, n)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([n, y, z])
    T[:3, 3] = origin
    return T


def _perturb_frame(T0: np.ndarray, p: np.ndarray, real: RealHand) -> np.ndarray:
    """The palm frame with its normal tilted by (p[0], p[1]) radians and its
    plane moved by p[2] m along the normal."""
    R = T0[:3, :3]
    n = R @ _unit(np.array([1.0, p[0], p[1]]))
    plane_point = T0[:3, 3] + R[:, 0] * p[2]
    return _frame_from(n, plane_point, real)


def _targets_for(real: RealHand, frame: np.ndarray) -> List[FingerTarget]:
    R, o = frame[:3, :3], frame[:3, 3]
    out = []
    for f in real.fingers:
        S, U = _slide_points(f, frame)
        names = [j.name for j in f.joints]
        types = []
        for k, j in enumerate(f.joints):
            if j.type == "prismatic":
                types.append("sliding")
            elif k > 0 and j.coupled_to in names[:k]:
                types.append("coupled")
            else:
                types.append("hinge")
        tip = R.T @ (f.tip - o)
        Q, P = _real_tip_samples(real, f, S, U, tip)
        out.append(FingerTarget(points=S, axes=U, types=types, tip=tip, names=names,
                                raw_points=[R.T @ (j.point - o) for j in f.joints], scale=real.scale,
                                q_samples=Q, tip_samples=P))
    return out


TIP_MAP_SAMPLES = 256
"""Joint configurations of each commercial finger, drawn uniformly over its
joint ranges (fixed seed), at which the fingertip map is compared."""


def _rot_points(a: np.ndarray, angles: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Each row of `V` turned about the unit axis `a` by its own angle."""
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    return V * c + np.cross(a[None], V) * s + a[None] * (V @ a)[:, None] * (1.0 - c)


def _real_tip_samples(real: RealHand, f: RealFinger, S: List[np.ndarray], U: List[np.ndarray],
                      tip: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(K, n) joint configurations of commercial finger `f`, uniform over its
    joint ranges (a joint coupled to another of the finger follows it), and
    (K, 3) its fingertip at each, palm frame: each joint turns (or slides)
    about its axis line at the zero pose, the last joint first."""
    lim = {j.name: j.limits for j in real.model.joints}
    names = [j.name for j in f.joints]
    rng = np.random.default_rng(TIP_MAP_SAMPLES)
    K, n = TIP_MAP_SAMPLES, len(f.joints)
    Q = np.zeros((K, n))
    for k, j in enumerate(f.joints):
        lo, hi = lim.get(j.name) or (-math.pi / 2, math.pi / 2)
        Q[:, k] = rng.uniform(lo, hi, K)
    for k, j in enumerate(f.joints):
        if j.coupled_to in names:
            Q[:, k] = j.ratio * Q[:, names.index(j.coupled_to)]
    P = np.repeat(np.asarray(tip, dtype=float)[None], K, axis=0)
    for k in range(n - 1, -1, -1):
        a, c = _unit(U[k]), np.asarray(S[k], dtype=float)
        if f.joints[k].type == "prismatic":
            P = P + Q[:, k:k + 1] * a[None]
        else:
            P = c[None] + _rot_points(a, Q[:, k], P - c[None])
    return Q, P


def _rot_mats(a: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """(K, 3, 3) rotations about the unit axis `a` by each angle."""
    a = _unit(a)
    c, s = np.cos(angles), np.sin(angles)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return (np.eye(3)[None] + s[:, None, None] * K[None] + (1.0 - c)[:, None, None] * (K @ K)[None])


def _tips_batch(f: Finger, t: FingerTarget, q_off: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """(K, 3) grammar fingertips (palm frame, m) at each row of the
    commercial joint angles `Q`: every grammar joint at its zero-pose
    difference plus the commercial angle, signed so the two turn the same
    way about their axes; coupled joints follow the joint before them."""
    B = dv.finger_frame(f)
    mount = dv.mount_point_m(f)
    axes = [dv.direction(*jt.axis) for jt in f.joints]
    lengths = [jt.length for jt in f.joints]
    q_off = np.asarray(q_off, dtype=float)
    _, axw, _ = _chain(mount, B, axes, lengths, t.types, q_off)
    sigma = np.array([1.0 if float(_unit(axw[j]) @ t.axes[j]) >= 0.0 else -1.0 for j in range(len(axes))])
    Qg = q_off[None] + sigma[None] * Q
    signs = [dv.axis_sign(_unit(a), B, ty == "sliding") for a, ty in zip(axes, t.types)]
    for j, ty in enumerate(t.types):
        if ty == "coupled" and j > 0:
            Qg[:, j] = signs[j] * COUPLING_RATIO * signs[j - 1] * Qg[:, j - 1]
    Kn = len(Q)
    F = np.repeat(B[None], Kn, axis=0)
    G = np.repeat(np.asarray(mount, dtype=float)[None], Kn, axis=0)
    for j, (a, L) in enumerate(zip(axes, lengths)):
        if t.types[j] != "sliding":
            F = F @ _rot_mats(a, Qg[:, j])
        G = G + F[:, :, 0] * (L * MM)
    return G


def tip_map_mm(f: Finger, t: FingerTarget, q_off: np.ndarray) -> Tuple[float, float]:
    """How closely grammar finger `f` (at its zero-pose differences `q_off`)
    follows the commercial fingertip over the commercial joint ranges: the
    RMS and the worst fingertip miss, mm at the real size, over the sampled
    configurations (`TIP_MAP_SAMPLES`)."""
    d = np.linalg.norm(_tips_batch(f, t, q_off, t.q_samples) - t.tip_samples, axis=1) / (MM * t.scale)
    return float(np.sqrt(np.mean(d * d))), float(d.max())


def _aimed_chain(t: FingerTarget, mount: np.ndarray):
    """The unsnapped finger built joint by joint (exact axes, each joint
    turning its link toward the next commercial joint, lengths by
    projection): a cheap estimate of the best continuous fit."""
    n = len(t.axes)
    d0 = _unit(_targets(t.points, t.tip, 0) - mount)
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, float(d0[0])))))
    facing = math.degrees(math.atan2(float(d0[1]), float(d0[2])))
    P, F = mount.copy(), _frame(facing, tilt)
    pts, axw = [], []
    for j in range(n):
        pts.append(P.copy())
        axw.append(t.axes[j])
        if t.types[j] != "sliding":
            a_link = _unit(F.T @ t.axes[j])
            F = F @ _rot(a_link, _aim(F[:, 0], t.axes[j], _unit(_targets(t.points, t.tip, j) - P)))
        nxt = t.points[j + 1] if j + 1 < n else t.tip
        P = P + F[:, 0] * max(0.0, float((nxt - P) @ F[:, 0]))
    return pts, axw, P


def _bases(targets: List[FingerTarget]) -> List[Tuple[int, int]]:
    """Every finger's base on the plate, on the grid."""
    return [_grid_point(_mount_on_plate(t.points[0], t.axes[0], "prismatic" if t.types[0] == "sliding" else "r"))
            for t in targets]


def _quick_cost(real: RealHand, frame: np.ndarray) -> float:
    """The worst finger cost of the cheap unsnapped fit (`_aimed_chain`)."""
    targets = _targets_for(real, frame)
    worst = 0.0
    for t, (y, z) in zip(targets, _bases(targets)):
        jmm, adeg, tmm = _errors(t, *_aimed_chain(t, np.array([0.0, y * MM, z * MM])))
        worst = max(worst, FingerFit(None, np.zeros(0), jmm, adeg, tmm).cost)
    return worst


def _nelder_mead(fn, x0: np.ndarray, scale: np.ndarray, iters: int = 60) -> np.ndarray:
    n = len(x0)
    pts = [x0] + [x0 + np.eye(n)[k] * scale[k] for k in range(n)]
    vals = [fn(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[k] for k in order]
        vals = [vals[k] for k in order]
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = fn(xr)
        if fr < vals[0]:
            xe = c + 2.0 * (c - pts[-1])
            fe = fn(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = fn(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [fn(p) for p in pts[1:]]
    return pts[int(np.argmin(vals))]


def _line_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two lines in the plate, 0-90 degrees."""
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    if n < 1e-12:
        return 90.0
    return math.degrees(math.acos(min(1.0, abs(float(a @ b)) / n)))


def _cut_fit(real: RealHand, frame: np.ndarray, fingers: Sequence[Finger]):
    """The cut standing for the commercial palm joints: `(cut or None,
    section, kept joint names, dropped joint names, notes)`. The palm joint
    carrying the most fingers is kept (joints carrying as many are merged:
    their fingers all go beyond the one cut); the cut is the grid line that
    puts exactly those fingers beyond it, keeps its seam clear of every base
    servo's housing if any such line does, and lies nearest the kept hinge's
    axis (direction in the plate, then distance)."""
    if not real.palm_joints:
        return None, [], [], [], []
    names = [j.name for j in real.palm_joints]
    carried = {n: [i for i, f in enumerate(real.fingers) if f.palm_joint == n] for n in names}
    most = max(len(v) for v in carried.values())
    kept = [n for n in names if len(carried[n]) == most]
    dropped = [n for n in names if n not in kept]
    section = sorted(i for n in kept for i in carried[n])
    notes = []
    R, o = frame[:3, :3], frame[:3, 3]
    rj = real.palm_joints[names.index(kept[0])]
    p = R.T @ (rj.point - o)
    u = _unit(R.T @ rj.axis)
    hinge = _mount_on_plate(p, u, "revolute", max_shift=PALM_HINGE_SHIFT_M)[1:] / MM
    direction = u[1:]
    if len(kept) > 1:
        q = R.T @ (real.palm_joints[names.index(kept[1])].point - o)
        direction = q[1:] - p[1:]
        notes.append(f"palm joints {kept} each carry {most} finger(s): one cut stands for them all")
    if dropped:
        notes.append(f"palm joints {dropped} dropped (the grammar has one palm joint; {kept} carries more fingers)")
    bases = np.array([[f.y, f.z] for f in fingers], dtype=float)
    # each finger's base housings seen from above, to keep the cut's seam clear of them
    feet = [np.concatenate([b.grown(mt.WALL_MM).corners()[:, 1:] for b in dv.base_motor_boxes_mm(f)]) for f in fingers]
    best = None
    for fa in range(0, 360, GRID_DEG):
        n = cut_normal_yz(PalmJoint(fa, 0))
        d = cut_direction_yz(PalmJoint(fa, 0))
        s = bases @ n
        for off in range(MIN_CUT_OFFSET_MM, int(math.ceil(float(s.max()))) + 1, GRID_MM):
            members = [i for i in range(len(fingers)) if s[i] > off]
            if not members or len(members) == len(fingers):
                continue
            mismatch = len(set(members) ^ set(section))
            seam = 0.0
            for i, ft in enumerate(feet):
                side = ft @ n - off
                seam = max(seam, (SEAM_MM / 2.0 - side.min()) if i in members else (side.max() + SEAM_MM / 2.0))
            cost = _line_angle_deg(d, direction) / TARGET_DEG + abs(float(hinge @ n) - off) / (TARGET_MM * real.scale)
            key = (mismatch, seam > 0.0, cost)
            if best is None or key < best[0]:
                best = (key, PalmJoint(facing=fa, offset=off), members)
    if best is None:
        notes.append(f"no straight cut separates fingers {section} from the rest: palm joint {kept} dropped")
        return None, [], [], names, notes
    (mismatch, _, _), cut, members = best
    if mismatch:
        notes.append(f"no straight cut puts exactly fingers {section} beyond it; fingers {members} are")
    return cut, members, kept, dropped, notes


def _palm_errors(real: RealHand, frame: np.ndarray, hand: Hand, kept: str) -> Tuple[float, float]:
    """The cut's hinge from the commercial hinge axis (mm) and the angle
    between the two axes (degrees)."""
    R, o = frame[:3, :3], frame[:3, 3]
    rj = next(j for j in real.palm_joints if j.name == kept)
    p = R.T @ (rj.point - o)
    u = _unit(R.T @ rj.axis)
    return line_distance(dv.palm_hinge_m(hand), p, u) / (MM * real.scale), axis_angle_deg(dv.palm_axis(hand), u)


def _variant_b(T0: np.ndarray, real: RealHand) -> np.ndarray:
    """The palm frame with its normal made perpendicular to the mean last
    axis of the non-thumb fingers (removes a tilt about the finger direction
    caused by staggered bases; palm-and-axis-study.md variant B)."""
    axes = []
    for f in real.fingers:
        if f.thumb or f.joints[-1].type == "prismatic":
            continue
        a = f.joints[-1].axis
        if axes and float(a @ axes[0]) < 0:
            a = -a
        axes.append(a)
    if not axes:
        return T0
    m = _unit(np.mean(axes, axis=0))
    n = T0[:3, 0] - m * float(T0[:3, 0] @ m)
    if np.linalg.norm(n) < 0.5:
        return T0
    return _frame_from(n, T0[:3, 3], real)


def conform(real: RealHand, palm_frame: Optional[np.ndarray] = None, refine_palm: bool = True,
            rules: Rules = NO_RULES) -> HandFit:
    """Conform `real` onto the grammar, its link lengths within `rules`
    (`COMMERCIAL_RULES`: the servo floors). `palm_frame`
    (4x4 in the hand-root frame) skips the palm-frame search; otherwise the
    plane fitted through the bases, its variant with the normal square to
    the fingers' last axes, and each of them moved to lower the worst error
    of a cheap unsnapped fit are all tried, and the best full fit is kept."""
    if palm_frame is not None:
        return _conform_in(real, palm_frame, rules)
    T0 = palm_frame_guess(real)
    frames = [T0, _variant_b(T0, real)]
    if refine_palm:
        for T in list(frames):
            p = _nelder_mead(lambda x: _quick_cost(real, _perturb_frame(T, x, real)), np.zeros(3),
                             np.array([0.05, 0.05, 0.004]))
            frames.append(_perturb_frame(T, p, real))
    best = None
    for T in frames:
        fit = _conform_in(real, T, rules)
        key = (not fit.within_target, max(fit.max_joint_mm / TARGET_MM, fit.max_tip_mm / TARGET_MM,
                                          fit.max_axis_deg / TARGET_DEG))
        if best is None or key < best[0]:
            best = (key, fit)
    return best[1]


def _recentred(fits: List[FingerFit], frame: np.ndarray, rules: Rules) -> Tuple[List[FingerFit], np.ndarray]:
    """The fitted fingers moved on the base grid so the mount sits at the
    palm's centre (`operators.recentre`), the palm frame moved back with them:
    the hand stays where it is, so every error is unchanged."""
    hand = Hand(fingers=tuple(r.finger for r in fits))
    moved = ops.recentre(hand, rules)
    if moved == hand:
        return fits, frame
    d = np.array([0.0, moved.fingers[0].y - hand.fingers[0].y, moved.fingers[0].z - hand.fingers[0].z]) * MM
    T = frame.copy()
    T[:3, 3] = frame[:3, 3] - frame[:3, :3] @ d
    return [replace(r, finger=g) for r, g in zip(fits, moved.fingers)], T


def recentre_fit(fit: "HandFit", real: RealHand, rules: Rules) -> "HandFit":
    """`fit` with the mount moved back to the palm's centre after a refit
    turned a base servo (the cut moves with the fingers, to the nearest 10
    mm; its errors are measured again)."""
    fits, T = _recentred(fit.fingers, fit.palm_T, rules)
    if T is fit.palm_T:
        return fit
    f0, g0 = fit.fingers[0].finger, fits[0].finger
    hand = shifted(fit.hand, g0.y - f0.y, g0.z - f0.z)
    fit.hand, fit.fingers, fit.palm_T = hand, fits, T
    if hand.palm_joints and "palm0_joint" in fit.name_map:
        e_mm, e_deg = _palm_errors(real, T, hand, fit.name_map["palm0_joint"])
        fit.palm_joint_mm, fit.palm_axis_deg = [e_mm], [e_deg]
    return fit


def _conform_in(real: RealHand, frame: np.ndarray, rules: Rules = NO_RULES) -> HandFit:
    notes = list(real.notes)
    targets = _targets_for(real, frame)
    fits = [_fit_free(t, b, rules) for t, b in zip(targets, _bases(targets))]
    fits, frame = _recentred(fits, frame, replace(rules, mount_at_centre=True))   # every fit: the mount at the centre
    for f, t in zip(real.fingers, targets):
        for j, jt in enumerate(f.joints):
            if jt.coupled_to and t.types[j] != "coupled":
                notes.append(f"{jt.name} follows {jt.coupled_to} (x{jt.ratio:.3g}) in the URDF; not a coupling to "
                             f"the joint before it in its finger, so it moves on its own here")
            elif t.types[j] == "coupled" and abs(jt.ratio - 1.1) > 1e-9:
                prev = f.joints[j - 1].name
                src = "" if jt.coupled_to == prev else f" (via {prev})"
                notes.append(f"{jt.name}: coupled x{jt.ratio:.3g} to {jt.coupled_to} in the URDF; x1.1 to the joint "
                             f"before it here{src}")
    fingers = tuple(r.finger for r in fits)
    cut, section, kept, dropped, cut_notes = _cut_fit(real, frame, fingers)
    notes += cut_notes
    hand = Hand(fingers=fingers, palm_joints=(cut,) if cut is not None else ())
    palm_mm, palm_deg = [], []
    name_map = {}
    if cut is not None:
        e_mm, e_deg = _palm_errors(real, frame, hand, kept[0])
        palm_mm, palm_deg = [e_mm], [e_deg]
        name_map["palm0_joint"] = kept[0]
    for i, f in enumerate(real.fingers):
        for j, jt in enumerate(f.joints):
            name_map[f"f{i}_j{j}"] = jt.name
    return HandFit(id=real.id, hand=hand, palm_T=frame, fingers=fits, palm_joint_mm=palm_mm, palm_axis_deg=palm_deg,
                   name_map=name_map, notes=notes, finger_order=list(range(len(real.fingers))), section=section,
                   dropped_palm_joints=dropped + (kept[1:] if len(kept) > 1 else []))


def conform_id(hand_id: str) -> HandFit:
    from .commercial import load_real_hand

    return conform_buildable(load_real_hand(hand_id))


def fit_violations(fit: HandFit) -> List[str]:
    return check(fit.hand, COMMERCIAL_RULES)


# --------------------------------------------------------------------------
# Scaling a commercial hand until Feetech servos fit
# --------------------------------------------------------------------------

SCALE_LADDER = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
SCALE_STEP = 0.05


def scale_model(model: KinematicModel, s: float) -> KinematicModel:
    """`model` with every length times `s` (joint origins, frames, sliding
    limits); angles unchanged."""
    def pose(p: Pose) -> Pose:
        return Pose(xyz=tuple(float(v) * s for v in p.xyz), rpy=p.rpy)

    joints = tuple(replace(j, origin=pose(j.origin),
                           limits=(tuple(float(v) * s for v in j.limits) if (j.type == "prismatic" and j.limits)
                                   else j.limits))
                   for j in model.joints)
    frames = tuple(replace(f, pose=pose(f.pose)) for f in model.frames)
    return replace(model, joints=joints, frames=frames)


def scaled(real: RealHand, s: float) -> RealHand:
    """The commercial hand scaled up by `s` about its root (the wrist)."""
    def sj(j: RealJoint) -> RealJoint:
        return replace(j, point=np.asarray(j.point, dtype=float) * s)

    fingers = [replace(f, joints=[sj(j) for j in f.joints], tip=np.asarray(f.tip, dtype=float) * s)
               for f in real.fingers]
    return replace(real, model=scale_model(real.model, s), fingers=fingers,
                   palm_joints=[sj(j) for j in real.palm_joints], scale=real.scale * s)


def buildable(fit: HandFit) -> bool:
    """The fitted hand keeps the servo floors and every solid clears every
    other by a wall (C1): it can be built from Feetech servos."""
    return not check(fit.hand, COMMERCIAL_RULES) and gvb.c1_self_overlap(fit.hand).ok


ACCEPT_FACTOR = 2.0
"""A scaled fit counts only within this many times the fit target (20 mm,
40 degrees): moving fingers clear can otherwise leave a fit that is no longer
the hand (Dex3 at its real size: 25 mm and 38 degrees, against 4 mm and 2
degrees at 2 x)."""


def acceptable(fit: HandFit, real: RealHand) -> bool:
    """Buildable, every joint keeps its commercial kind (the hand's
    structure), and every finger's fingertip map is within `MAP_RMS_MM` /
    `MAP_WORST_MM` (Martin, 2026-10-20: the fingertips must follow the
    commercial ones; a joint may sit elsewhere)."""
    return map_within(fit) and buildable(fit) and kinds_kept(fit.hand, fit.palm_T, fit.q_off(), real)


def fallback_fit(real: RealHand, top: float = SCALE_LADDER[-1]) -> HandFit:
    """A hand no scale up to `top` makes buildable: its fit at the real size
    under the grammar's own limits (no servo lengths: they would only spoil
    the fit), as before commercial hands were scaled."""
    fit = conform(real)
    fit.scale = 1.0
    for f, t, r in zip(fit.hand.fingers, _targets_for(real, fit.palm_T), fit.fingers):
        r.map_rms_mm, r.map_worst_mm = tip_map_mm(f, t, r.q_off)
    fit.notes.append(f"no scale up to {top:g} x gives a fit buildable from Feetech servos that keeps every joint's "
                     f"kind with every fingertip map within {MAP_RMS_MM:g} mm RMS and {MAP_WORST_MM:g} mm worst: kept "
                     f"at its real size, without servo lengths")
    return fit


def conform_buildable(real: RealHand, ladder: Sequence[float] = SCALE_LADDER, step: float = SCALE_STEP,
                      ) -> HandFit:
    """`real` scaled up by the smallest factor (to `step`) at which its fit,
    with servo link lengths, is buildable, keeps every joint's kind and stays
    within twice the fit target (`acceptable`): the scales in `ladder` in turn until one is, then every
    `step` between the last that was not and that one, lowest first. A fit
    that fails C1 is first refitted clear (`refit_clear`). The palm frame is
    searched once at scale 1 and scaled with the hand. If no scale on the
    ladder works, the hand's plain fit at its real size under the grammar's
    own limits is kept (`fallback_fit`) and a note says so; the simulator
    then filters its overlaps."""
    first = conform(real, rules=COMMERCIAL_RULES)
    R, o = first.palm_T[:3, :3], first.palm_T[:3, 3]

    def at(s: float) -> HandFit:
        real_s = scaled(real, s)
        if abs(s - 1.0) < 1e-9:
            fit = copy.deepcopy(first)            # the refit below changes a fit in place
        else:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, o * s
            fit = conform(real_s, palm_frame=T, rules=COMMERCIAL_RULES)
        if not check(fit.hand, COMMERCIAL_RULES) and not gvb.c1_self_overlap(fit.hand).ok:
            fit = refit_clear(fit, real_s, COMMERCIAL_RULES)
            fit = recentre_fit(fit, real_s, COMMERCIAL_RULES)     # a refit can turn a base servo
        fit = map_refit(fit, real_s, COMMERCIAL_RULES)             # the fingertips, clear of each other
        fit.scale = float(s)
        return fit

    prev, hit = None, None
    for s in ladder:
        fit = at(s)
        if acceptable(fit, scaled(real, s)):
            hit = fit
            break
        prev = s
    if hit is None:
        return fallback_fit(real, ladder[-1])
    if prev is not None:
        k = 1
        while prev + k * step < hit.scale - 1e-9:
            s = round(prev + k * step, 6)
            fit = at(s)
            if acceptable(fit, scaled(real, s)):
                hit = fit
                break
            k += 1
    hit.notes.append(f"scaled up {hit.scale:g} x so Feetech servos fit (servo link lengths, no collisions)")
    return hit


def urdf_joint_map(hand: Hand, palm_T: np.ndarray, q_off: np.ndarray, name_map: Dict[str, str],
                   real: RealHand) -> Dict[str, Tuple[int, float]]:
    """URDF joint name -> (index into `derive.dofs(hand)`, sign), so that a
    grammar joint vector `q` puts the commercial hand in the matching pose:
    q_urdf = sign * (q[index] - q_off[index]) (`urdf_joint_values`)."""
    R, o = palm_T[:3, :3], palm_T[:3, 3]
    ds = dv.dofs(hand)
    pose = dv.fk(hand, q_off)
    out = {}
    for k, d in enumerate(ds):
        urdf = name_map.get(d.name)
        if urdf is None:
            continue
        try:
            rj = real.joint(urdf)
        except KeyError:
            continue
        if d.finger < 0:
            a = dv.palm_axis(hand, d.index)
        else:
            a = pose.links[d.finger][d.index][:3, :3] @ dv.joint_axis(hand, d.finger, d.index)
        out[urdf] = (k, 1.0 if float(a @ (R.T @ rj.axis)) >= 0 else -1.0)
    return out


def urdf_joint_values(jmap: Dict[str, Tuple[int, float]], q: np.ndarray, q_off: np.ndarray) -> Dict[str, float]:
    return {n: s * float(q[k] - q_off[k]) for n, (k, s) in jmap.items()}
