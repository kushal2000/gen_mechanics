"""Conform a commercial hand onto the grammar at the fine resolution, and
measure how well it fits.

Fit metric (GRAMMAR-LOCK item 20): the largest joint-position error (mm),
axis error (degrees) and fingertip error (mm), target 5 mm / 10 degrees.
- A pure difference in the zero pose counts as a match: the grammar hand is
  compared at its own joint angles `q_off` (free), the commercial hand at
  its zero pose. Signs and ranges are global conventions, so an axis is a
  line (its error is at most 90 degrees) and ranges are not compared.
- A revolute joint's position is its axis line: sliding a joint along its own
  axis does not change any motion, so the position error is the distance
  from the grammar's joint point to the commercial axis line. A sliding
  joint's position is its child frame's origin.

The palm frame: origin at the hand root's origin (the wrist centre, where
the hand mounts) moved onto the plate; x = the normal of the plane fitted
through the finger bases, on the side the fingertips are when the fingers
leave the plate, else on the side the fingers close toward; z = from the
wrist toward the non-thumb bases. The plane's tilt and offset are then
refined to minimise the fit's worst error.

Per finger: each joint is slid along its axis onto the finger's centre line;
the base is where the first axis meets the plate; facing and tilt point the
finger at its next joint; then, joint by joint, the axis is expressed in the
current link frame and snapped to the 5 degree grid, the joint's zero-pose
difference turns the link toward the next joint, and the link length is
snapped to the millimetre (0 or 15-90 mm, fingertip 10-90 mm). A local
search over one-step changes of every value of the finger then lowers the
worst error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import derive as dv
from .commercial import RealFinger, RealHand, RealJoint
from .hand import (
    COUPLING_RATIO,
    LINK_MAX_MM,
    LINK_MIN_MM,
    NO_RULES,
    PALM_ANCHOR_MM,
    TIP_MIN_MM,
    Finger,
    Hand,
    Joint,
    PalmJoint,
    canonical_axis,
    check,
)

MM = 1e-3
DEG = math.pi / 180.0
TARGET_MM = 5.0
TARGET_DEG = 10.0
Q_WEIGHT = 0.2
"""Tie-breaker: prefer small zero-pose differences (per radian of mean
|q_off|), so a conformed finger lies where the real one does at rest."""
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


def snap_axis(v: np.ndarray, step: int = 5) -> Tuple[int, int]:
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


def snap_length(L_mm: float, is_tip: bool) -> int:
    if is_tip:
        return int(min(LINK_MAX_MM, max(TIP_MIN_MM, round(L_mm))))
    r = int(round(L_mm))
    if 0 < r < LINK_MIN_MM:
        r = 0 if L_mm < LINK_MIN_MM / 2.0 else LINK_MIN_MM
    return int(min(LINK_MAX_MM, max(0, r)))


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


@dataclass
class FingerFit:
    finger: Finger
    q_off: np.ndarray
    joint_mm: List[float]
    axis_deg: List[float]
    tip_mm: float

    @property
    def cost(self) -> float:
        worst = max([e / TARGET_MM for e in self.joint_mm] + [a / TARGET_DEG for a in self.axis_deg]
                    + [self.tip_mm / TARGET_MM])
        mean = (sum(self.joint_mm) / TARGET_MM + sum(self.axis_deg) / TARGET_DEG + self.tip_mm / TARGET_MM) / (
            2 * len(self.joint_mm) + 1)
        bend = float(np.mean(np.abs(self.q_off))) if len(self.q_off) else 0.0
        return worst + 0.1 * mean + Q_WEIGHT * bend


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
    return dv.finger_frame(Finger(y=0, z=0, facing=facing_deg, tilt=tilt_deg))


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
    jmm, adeg = [], []
    for j, (G, a) in enumerate(zip(pts, axw)):
        if t.types[j] == "sliding":
            jmm.append(float(np.linalg.norm(G - t.points[j])) / MM)
        else:
            jmm.append(line_distance(G, t.points[j], t.axes[j]) / MM)
        adeg.append(axis_angle_deg(a, t.axes[j]))
    return jmm, adeg, float(np.linalg.norm(tip - t.tip)) / MM


def _residuals(t: FingerTarget, pts, axw, tip) -> np.ndarray:
    out = []
    sin_target = math.sin(TARGET_DEG * DEG)
    for j, (G, a) in enumerate(zip(pts, axw)):
        d = G - t.points[j]
        if t.types[j] != "sliding":
            d = d - t.axes[j] * float(d @ t.axes[j])
        out.extend(d / (TARGET_MM * MM))
        out.extend(np.cross(_unit(a), t.axes[j]) / sin_target)
    out.extend((tip - t.tip) / (TARGET_MM * MM))
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


def evaluate_finger(f: Finger, t: FingerTarget, q: Optional[np.ndarray] = None) -> FingerFit:
    """The errors of `f` against `t` at joint angles `q` (default: each joint
    turns its link toward the next commercial joint)."""
    B = dv.finger_frame(f)
    mount = np.array([0.0, f.y * MM, f.z * MM])
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
    return FingerFit(f, q, jmm, adeg, tmm)


def best_q(f: Finger, t: FingerTarget, q0: np.ndarray, iters: int = 15) -> FingerFit:
    """The zero-pose differences that fit `f` best (least squares from `q0`)."""
    B = dv.finger_frame(f)
    mount = np.array([0.0, f.y * MM, f.z * MM])
    axes = [dv.direction(*jt.axis) for jt in f.joints]
    lengths = [jt.length for jt in f.joints]
    rev = [j for j, jt in enumerate(f.joints) if jt.type == "hinge"]
    if not rev:
        return evaluate_finger(f, t, np.zeros(len(f.joints)))

    def fun(x):
        q = np.zeros(len(f.joints))
        q[rev] = x
        return np.concatenate([_residuals(t, *_chain(mount, B, axes, lengths, t.types, q)), Q_WEIGHT * x])

    x = _lm(fun, np.asarray(q0, dtype=float)[rev], iters=iters)
    q = np.zeros(len(f.joints))
    q[rev] = x
    return evaluate_finger(f, t, q)


def continuous_fit(t: FingerTarget, mount: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Least-squares fit of every finger value, unsnapped: facing, tilt, the
    axes (az, el), the link lengths and the zero-pose differences, the mount
    fixed on the plate. Returns the parameter vector and its layout."""
    n = len(t.axes)
    d0 = _unit(_targets(t.points, t.tip, 0) - mount)
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, float(d0[0])))))
    facing = math.degrees(math.atan2(float(d0[1]), float(d0[2])))
    B = _frame(facing, tilt)
    G, F = mount.copy(), B.copy()
    az, el, L, q = [], [], [], []
    for j in range(n):
        a_link = _unit(F.T @ t.axes[j])
        a1, e1 = to_az_el(a_link)
        az.append(a1)
        el.append(e1)
        qj = 0.0
        if t.types[j] != "sliding":
            qj = _aim(F[:, 0], t.axes[j], _unit(_targets(t.points, t.tip, j) - G))
            F = F @ _rot(a_link, qj)
        q.append(qj)
        nxt = t.points[j + 1] if j + 1 < n else t.tip
        Lj = max(0.0, float((nxt - G) @ F[:, 0])) / MM
        L.append(Lj)
        G = G + F[:, 0] * (Lj * MM)
    x0 = np.array([facing, tilt] + az + el + L + q, dtype=float)

    def unpack(x):
        fa, ti = x[0], x[1]
        A = [dv.direction(x[2 + j], x[2 + n + j]) for j in range(n)]
        Ls = list(np.maximum(x[2 + 2 * n: 2 + 3 * n], 0.0))
        qs = x[2 + 3 * n: 2 + 4 * n]
        return fa, ti, A, Ls, qs

    def fun(x):
        fa, ti, A, Ls, qs = unpack(x)
        return np.concatenate([_residuals(t, *_chain(mount, _frame(fa, ti), A, Ls, t.types, qs)), Q_WEIGHT * qs])

    x = _lm(fun, x0, iters=60, eps=1e-5)
    return x, {"n": n, "unpack": unpack}


def snap_continuous(x: np.ndarray, layout: dict, t: FingerTarget, mount_mm: Tuple[int, int],
                    palm_joint: int) -> Tuple[Finger, np.ndarray]:
    """Snap a continuous fit to the fine grid, joint by joint from the palm:
    each axis is snapped in the (already snapped) link frame."""
    fa, ti, A, Ls, qs = layout["unpack"](x)
    facing = int(round(fa / 5.0) * 5) % 360
    tilt = int(round(ti / 5.0) * 5)
    tilt = max(-90, min(90, tilt))
    f = Finger(y=mount_mm[0], z=mount_mm[1], facing=facing, tilt=tilt, joints=(), palm_joint=palm_joint)
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
        joints.append(Joint(t.types[j], ax, snap_length(Ls[j], j == n - 1)))
    return replace(f, joints=tuple(joints)), np.asarray(qs, dtype=float)


def _neighbours(f: Finger, mount_fixed: bool = False) -> List[Finger]:
    out = []
    if not mount_fixed:
        for c in ("y", "z"):
            for s in (-1, 1):
                out.append(replace(f, **{c: getattr(f, c) + s}))
    for s in (-5, 5):
        out.append(replace(f, facing=(f.facing + s) % 360))
        if abs(f.tilt + s) <= 90:
            out.append(replace(f, tilt=f.tilt + s))
    n = len(f.joints)
    for j, jt in enumerate(f.joints):
        def put(new: Joint) -> Finger:
            js = list(f.joints)
            js[j] = new
            return replace(f, joints=tuple(js))

        for s in (-5, 5):
            out.append(put(replace(jt, axis=canonical_axis(jt.axis[0] + s, jt.axis[1]))))
            out.append(put(replace(jt, axis=canonical_axis(jt.axis[0], jt.axis[1] + s))))
        for s in (-1, 1):
            L = jt.length + s
            if j == n - 1:
                ok = TIP_MIN_MM <= L <= LINK_MAX_MM
            else:
                if 0 < L < LINK_MIN_MM:
                    L = 0 if s < 0 else LINK_MIN_MM
                ok = L == 0 or LINK_MIN_MM <= L <= LINK_MAX_MM
            if ok and L != jt.length:
                out.append(put(replace(jt, length=L)))
    return out


def refine_finger(fit: FingerFit, t: FingerTarget, mount_fixed: bool = False, max_passes: int = 30) -> FingerFit:
    """Local search over one-step changes of every value of the finger, the
    zero-pose differences re-fitted (warm-started) for each candidate."""
    best = fit
    for _ in range(max_passes):
        improved = False
        for g in _neighbours(best.finger, mount_fixed):
            r = best_q(g, t, best.q_off, iters=4)
            if r.cost < best.cost - 1e-9:
                best, improved = r, True
        if not improved:
            break
    return best_q(best.finger, t, best.q_off, iters=30)


def fit_finger(t: FingerTarget, mount: np.ndarray, palm_joint: int) -> FingerFit:
    y, z = int(round(mount[1] / MM)), int(round(mount[2] / MM))
    m = np.array([0.0, y * MM, z * MM])
    x, layout = continuous_fit(t, m)
    f, q = snap_continuous(x, layout, t, (y, z), palm_joint)
    start = best_q(f, t, q, iters=30)
    aimed = evaluate_finger(f, t)
    if aimed.cost < start.cost:
        start = best_q(f, t, aimed.q_off, iters=30)
    return refine_finger(start, t)


# --------------------------------------------------------------------------
# The whole hand
# --------------------------------------------------------------------------


@dataclass
class HandFit:
    id: str
    hand: Hand
    palm_T: np.ndarray                 # 4x4: the palm frame in the hand-root frame
    fingers: List[FingerFit]
    palm_joint_mm: List[float]
    palm_axis_deg: List[float]
    name_map: Dict[str, str]           # grammar dof name -> commercial joint name
    notes: List[str] = field(default_factory=list)
    finger_order: List[int] = field(default_factory=list)   # grammar finger i <- real finger index

    @property
    def max_joint_mm(self) -> float:
        return max([e for f in self.fingers for e in f.joint_mm] + list(self.palm_joint_mm) + [0.0])

    @property
    def max_axis_deg(self) -> float:
        return max([e for f in self.fingers for e in f.axis_deg] + list(self.palm_axis_deg) + [0.0])

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
        out.append(FingerTarget(points=S, axes=U, types=types, tip=R.T @ (f.tip - o), names=names,
                                raw_points=[R.T @ (j.point - o) for j in f.joints]))
    return out


def _aimed_chain(t: FingerTarget, mount: np.ndarray):
    """The unsnapped finger built joint by joint (exact axes, each joint
    turning its link toward the next commercial joint, lengths by
    projection): a cheap estimate of the best continuous fit."""
    n = len(t.axes)
    d0 = _unit(_targets(t.points, t.tip, 0) - mount)
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, float(d0[0])))))
    facing = math.degrees(math.atan2(float(d0[1]), float(d0[2])))
    G, F = mount.copy(), _frame(facing, tilt)
    pts, axw = [], []
    for j in range(n):
        pts.append(G.copy())
        axw.append(t.axes[j])
        if t.types[j] != "sliding":
            a_link = _unit(F.T @ t.axes[j])
            F = F @ _rot(a_link, _aim(F[:, 0], t.axes[j], _unit(_targets(t.points, t.tip, j) - G)))
        nxt = t.points[j + 1] if j + 1 < n else t.tip
        G = G + F[:, 0] * max(0.0, float((nxt - G) @ F[:, 0]))
    return pts, axw, G


def _quick_cost(real: RealHand, frame: np.ndarray) -> float:
    """The worst finger cost of the cheap unsnapped fit (`_aimed_chain`)."""
    worst = 0.0
    for t in _targets_for(real, frame):
        mount = _mount_on_plate(t.points[0], t.axes[0], "prismatic" if t.types[0] == "sliding" else "r")
        jmm, adeg, tmm = _errors(t, *_aimed_chain(t, mount))
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


def _palm_joint_fit(rj: RealJoint, frame: np.ndarray) -> Tuple[PalmJoint, float, float]:
    R, o = frame[:3, :3], frame[:3, 3]
    p = R.T @ (rj.point - o)
    u = _unit(R.T @ rj.axis)
    hinge = _mount_on_plate(p, u, "revolute", max_shift=PALM_HINGE_SHIFT_M)
    y, z = int(round(hinge[1] / MM)), int(round(hinge[2] / MM))
    hinge = np.array([0.0, y * MM, z * MM])
    ax = snap_axis(u)
    return PalmJoint(y=y, z=z, axis=ax), line_distance(hinge, p, u) / MM, axis_angle_deg(dv.direction(*ax), u)


def _enforce_spacing(hand: Hand, min_mm: int) -> Tuple[Hand, List[int]]:
    """Move bases closer than the spacing apart, symmetrically along their
    separation, to whole millimetres. Returns the hand and the fingers moved."""
    fs = list(hand.fingers)
    moved = set()
    for _ in range(200):
        bad = False
        for a in range(len(fs)):
            for b in range(a + 1, len(fs)):
                d = math.hypot(fs[a].y - fs[b].y, fs[a].z - fs[b].z)
                if d >= min_mm:
                    continue
                bad = True
                u = np.array([fs[b].y - fs[a].y, fs[b].z - fs[a].z], dtype=float)
                u = u / (np.linalg.norm(u) + 1e-12)
                push = (min_mm - d) / 2.0 + 0.5
                pa = np.array([fs[a].y, fs[a].z]) - u * push
                pb = np.array([fs[b].y, fs[b].z]) + u * push
                # round away from each other, so the distance only grows
                pa = np.where(u > 0, np.floor(pa), np.ceil(pa))
                pb = np.where(u > 0, np.ceil(pb), np.floor(pb))
                fs[a] = replace(fs[a], y=int(pa[0]), z=int(pa[1]))
                fs[b] = replace(fs[b], y=int(pb[0]), z=int(pb[1]))
                moved.update((a, b))
        if not bad:
            break
    return replace(hand, fingers=tuple(fs)), sorted(moved)


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
            enforce: bool = True) -> HandFit:
    """Conform `real` onto the grammar at the fine resolution. `palm_frame`
    (4x4 in the hand-root frame) skips the palm-frame search; otherwise the
    plane fitted through the bases, its variant with the normal square to
    the fingers' last axes, and each of them moved to lower the worst error
    of a cheap unsnapped fit are all tried, and the best full fit is kept."""
    if palm_frame is not None:
        return _conform_in(real, palm_frame, enforce)
    T0 = palm_frame_guess(real)
    frames = [T0, _variant_b(T0, real)]
    if refine_palm:
        for T in list(frames):
            p = _nelder_mead(lambda x: _quick_cost(real, _perturb_frame(T, x, real)), np.zeros(3),
                             np.array([0.05, 0.05, 0.004]))
            frames.append(_perturb_frame(T, p, real))
    best = None
    for T in frames:
        fit = _conform_in(real, T, enforce)
        key = (not fit.within_target, max(fit.max_joint_mm / TARGET_MM, fit.max_tip_mm / TARGET_MM,
                                          fit.max_axis_deg / TARGET_DEG))
        if best is None or key < best[0]:
            best = (key, fit)
    return best[1]


def _conform_in(real: RealHand, frame: np.ndarray, enforce: bool) -> HandFit:
    notes = list(real.notes)
    targets = _targets_for(real, frame)
    palm_names = [j.name for j in real.palm_joints]
    palm_fits = [_palm_joint_fit(j, frame) for j in real.palm_joints]
    fits: List[FingerFit] = []
    fingers: List[Finger] = []
    for f, t in zip(real.fingers, targets):
        k = palm_names.index(f.palm_joint) if f.palm_joint else -1
        mount = _mount_on_plate(t.points[0], t.axes[0], "prismatic" if t.types[0] == "sliding" else "r")
        fit = fit_finger(t, mount, k)
        fits.append(fit)
        fingers.append(fit.finger)
        for j, jt in enumerate(f.joints):
            if jt.coupled_to and t.types[j] != "coupled":
                notes.append(f"{jt.name} follows {jt.coupled_to} (x{jt.ratio:.3g}) in the URDF; not a coupling to "
                             f"the joint before it in its finger, so it moves on its own here")
            elif t.types[j] == "coupled" and abs(jt.ratio - 1.1) > 1e-9:
                prev = f.joints[j - 1].name
                src = "" if jt.coupled_to == prev else f" (via {prev})"
                notes.append(f"{jt.name}: coupled x{jt.ratio:.3g} to {jt.coupled_to} in the URDF; x1.1 to the joint "
                             f"before it here{src}")
    hand = Hand(fingers=tuple(fingers), palm_joints=tuple(p for p, _, _ in palm_fits))
    if enforce:
        spaced, moved = _enforce_spacing(hand, NO_RULES.min_spacing_mm)
        if moved:
            notes.append(f"bases of fingers {moved} moved apart to the {NO_RULES.min_spacing_mm} mm spacing")
            for i in moved:
                start = best_q(spaced.fingers[i], targets[i], fits[i].q_off, iters=30)
                fits[i] = refine_finger(start, targets[i], mount_fixed=True)
            hand = replace(spaced, fingers=tuple(f.finger for f in fits))
    name_map = {}
    for k, j in enumerate(real.palm_joints):
        name_map[f"palm{k}_joint"] = j.name
    for i, f in enumerate(real.fingers):
        for j, jt in enumerate(f.joints):
            name_map[f"f{i}_j{j}"] = jt.name
    return HandFit(id=real.id, hand=hand, palm_T=frame, fingers=fits,
                   palm_joint_mm=[e for _, e, _ in palm_fits], palm_axis_deg=[a for _, _, a in palm_fits],
                   name_map=name_map, notes=notes, finger_order=list(range(len(real.fingers))))


def conform_id(hand_id: str) -> HandFit:
    from .commercial import load_real_hand

    return conform(load_real_hand(hand_id))


def fit_violations(fit: HandFit) -> List[str]:
    return check(fit.hand, NO_RULES)


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
