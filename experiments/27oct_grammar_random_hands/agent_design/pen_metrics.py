"""Static, CPU-only metrics of a grammar hand for pen repose v2 (palm down, wrist translation only).

The pen is a 100 x 30 x 30 mm box. Success depends only on its long axis (4 keypoints along it), so the hand may hold
it at any roll. Any change of the long axis is one rotation about an axis u perpendicular to both the old and the new
axis (u is uniform on the sphere when goals are uniform): the core move is a *pivot* -- pinch the pen across its
30 mm width with the pinch line along u and swing it about that line.

Each finger is sampled over its own joint ranges (palm joint too, for a finger beyond a cut). For every sample we
keep the fingertip sphere centre (the far end of the tip capsule, radius 16.5 mm) and points along every link's
centre line with the link's half thickness as radius. Tip positions are binned into a voxel grid (`CELL_MM`), up
to `K_PER_CELL` samples per cell, so "can finger i put its tip at p without its links passing through X" is a
vectorised lookup over many targets.

Metrics (fractions in [0, 1]):

- `pickup`: the pen lies flat on the table at a random yaw (wrist yaw is locked). Fraction of yaws (12, 0-165 deg)
  at which two different fingers can pinch it across its width (or end to end) with the palm above the pen, no link
  below the table and no link inside the pen, for some wrist position.
- `hold2` / `hold3`: task goals (uniform orientation, centre height uniform from its floor to 12 cm above resting).
  Fraction of goals at which the pen can be held at that height in a 2-finger antipodal pinch (`hold3`: plus a third
  finger touching another face), for some wrist position inside its limits (root height <= 20 cm) and any pen roll.
- `pivot`: pivot axes u uniform on the sphere. Fraction of u for which two fingers can pinch the pen's centre across
  its width along u while every one of their links stays out of the disc the pen sweeps turning a full turn about u
  (radius 52.2 mm, 30 mm thick), the disc clear of the palm, and the pen reachable inside the wrist's height range.
  `pivot_h`: only axes within 30 deg of the palm plane (the end-over-end flips of a pen lying roughly flat).
- `pivot_reach`: the same pinch points reachable at all (no sweep check): an upper bound on `pivot`.

Approximations: fingers are checked one at a time (finger-finger collisions are left out; for fingers beyond a palm
cut the two pinching fingers may use different hinge angles); contact only through the fingertip's end sphere;
tolerance one cell.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from hand_sampler.grammar import derive as dv  # noqa: E402
from hand_sampler.grammar import hand as gh  # noqa: E402
from hand_sampler.grammar.hand import finger_section  # noqa: E402

PEN_HALF = np.array([50.0, 15.0, 15.0])          # mm; x = long axis
TIP_R = dv.TIP_RADIUS_MM                          # 16.5
FACE_OFF = PEN_HALF[1] + TIP_R                    # tip centre 31.5 mm from the pen axis
END_OFF = PEN_HALF[0] + TIP_R                     # 66.5 mm for an end-to-end pinch
SWEEP_R = float(np.hypot(PEN_HALF[0], PEN_HALF[1]))   # 52.2: the pen's corners turning about a width axis
PLATE_FACE = gh.PLATE_HALF_MM                     # palm front face at x = 12 mm
GAP = 2.0                                         # one wall
CELL_MM = 6.0
K_PER_CELL = 12
N_SAMPLES = 12000
WRIST_Z = (20.0, 200.0)                           # root height above the table, mm
REST_Z = 15.0
TOP_Z = REST_Z + 120.0                            # goals up to 12 cm above resting
FLOOR_CLEAR = 20.0
LATERAL = np.arange(-40.0, 40.1, 20.0)            # pen centre offsets in the plate (the wrist moves +-10 cm)
TOL_IN = 3.0                                      # how far a link may come into the pen (soft contact)


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def random_rotations(n, rng):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(phi), np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th)], 1)   # x first: palm normal


# --------------------------------------------------------------------------
# Per-finger samples
# --------------------------------------------------------------------------


@dataclass
class FingerSamples:
    tip: np.ndarray           # (N, 3) mm: tip sphere centre
    pts: np.ndarray           # (N, P, 3) mm: link centre-line points (tip capsule included)
    rad: np.ndarray           # (P,) mm: their radii
    lo: np.ndarray            # voxel grid origin
    shape: tuple
    table: np.ndarray         # (cells, K) sample indices, -1 empty
    park: np.ndarray          # sample indices tried for getting out of the way (0: the zero pose)
    q: Optional[np.ndarray] = None   # (N, dofs): each sample's joint vector (the hand's full dof order)


def _link_lines(f: gh.Finger):
    """Per link: (points along its centre line in its own frame (mm), radius)."""
    g = dv.finger_geometry(f)
    out = []
    n = len(f.joints)
    for j in range(n):
        if j == n - 1:
            c = g.tip
            s = np.linspace(0, 1, 5)[:, None]
            out.append((c.p0[None] * (1 - s) + c.p1[None] * s, c.radius))
        else:
            b = g.links[j]
            k = int(np.argmax(b.half))
            others = [b.half[m] for m in range(3) if m != k]
            r = float(min(others))
            reach = max(b.half[k] - r, 0.0)
            s = np.linspace(-1, 1, 4)[:, None]
            out.append((b.centre[None] + s * reach * b.axes[:, k][None], r))
    return out, g.tip.p1.copy()


def finger_samples(hand: gh.Hand, i: int, n: int = N_SAMPLES, seed: int = 0) -> FingerSamples:
    rng = np.random.default_rng(seed + 17 * i)
    ds = dv.dofs(hand)
    lo = np.zeros(len(ds))
    hi = np.zeros(len(ds))
    for k, d in enumerate(ds):
        if d.finger == i or (d.kind == "palm" and finger_section(hand, i) == 0):
            lo[k], hi[k] = d.limits
    q = rng.uniform(lo, hi, size=(n, len(ds)))
    q[0] = 0.0
    pose = dv.fk(hand, q)
    f = hand.fingers[i]
    lines, tip_local = _link_lines(f)
    pts, rad = [], []
    for j, (P, r) in enumerate(lines):
        T = pose.links[i][:, j]                               # (N, 4, 4) metres
        w = np.einsum("nab,pb->npa", T[:, :3, :3], P * dv.MM) + T[:, None, :3, 3]
        pts.append(w / dv.MM)
        rad += [r] * len(P)
    T = pose.links[i][:, -1]
    tip = (T[:, :3, :3] @ (tip_local * dv.MM) + T[:, :3, 3]) / dv.MM
    pts = np.concatenate(pts, 1)
    rad = np.array(rad)
    # voxel table
    glo = tip.min(0) - CELL_MM
    idx = np.floor((tip - glo) / CELL_MM).astype(int)
    shape = tuple(idx.max(0) + 2)
    flat = np.ravel_multi_index(idx.T, shape)
    order = rng.permutation(n)
    table = -np.ones((int(np.prod(shape)), K_PER_CELL), dtype=np.int32)
    fill = np.zeros(int(np.prod(shape)), dtype=np.int32)
    for s in order:
        c = flat[s]
        if fill[c] < K_PER_CELL:
            table[c, fill[c]] = s
            fill[c] += 1
    park = np.concatenate([[0], rng.choice(np.arange(1, n), N_PARK, replace=False)])
    return FingerSamples(tip, pts, rad, glo, shape, table, park, q)


def candidates(fs: FingerSamples, targets: np.ndarray):
    """(M, K) sample indices whose tips lie within one cell of each target (-1: none)."""
    idx = np.floor((targets - fs.lo) / CELL_MM).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(fs.shape)), axis=1)
    out = -np.ones((len(targets), K_PER_CELL), dtype=np.int32)
    if ok.any():
        flat = np.ravel_multi_index(idx[ok].T, fs.shape)
        out[ok] = fs.table[flat]
    cand = out
    valid = cand >= 0
    d = np.full(cand.shape, np.inf)
    d[valid] = np.linalg.norm(fs.tip[cand[valid]] - np.repeat(targets, K_PER_CELL, 0).reshape(-1, K_PER_CELL, 3)[valid],
                              axis=-1)
    cand = np.where(d <= CELL_MM, cand, -1)
    return cand


def box_sdf(p: np.ndarray, half: np.ndarray) -> np.ndarray:
    q = np.abs(p) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(-1), 0.0)


def _clear(P, rad, R, c, max_x, sweep_axis):
    """(h,) bool per sample row: points P (h, P, 3) with radii rad (P,) clear of the pen at (R, c) -- or of the disc
    it sweeps about `sweep_axis` -- and above the table (palm x + radius <= max_x)."""
    rel = P - c[:, None]
    if sweep_axis is None:
        loc = np.matmul(rel, R)                               # R^T (p - c), row vectors
        sd = box_sdf(loc, PEN_HALF)
    else:
        u = sweep_axis
        a = np.matmul(rel, u[:, :, None])[..., 0]
        radial = np.linalg.norm(rel - a[..., None] * u[:, None], axis=-1)
        # distance from a point to the solid disc (radius SWEEP_R, half thickness PEN_HALF[1])
        dz = np.abs(a) - PEN_HALF[1]
        dr = radial - SWEEP_R
        sd = np.where((dz > 0) & (dr > 0), np.hypot(np.maximum(dz, 0), np.maximum(dr, 0)), np.maximum(dz, dr))
    clear = np.all(sd >= rad[None] - TOL_IN, axis=1)
    below = np.all(P[..., 0] + rad[None] <= max_x[:, None], axis=1)
    return clear & below


def reach_ok(fs: FingerSamples, targets: np.ndarray, R: np.ndarray, c: np.ndarray, max_x: np.ndarray,
             sweep_axis: Optional[np.ndarray] = None) -> np.ndarray:
    """(M,) bool: some sample puts the tip at targets[m] with no link inside the pen (pose R[m], c[m], palm frame)
    -- or, with `sweep_axis`, inside the disc it sweeps about that axis -- and no point below the table
    (palm x + radius <= max_x[m])."""
    cand = candidates(fs, targets)                         # (M, K)
    M, K = cand.shape
    good = np.zeros((M, K), dtype=bool)
    has = cand >= 0
    if not has.any():
        return good.any(1)
    mi, ki = np.nonzero(has)
    s = cand[mi, ki]
    good[mi, ki] = _clear(fs.pts[s], fs.rad, R[mi], c[mi], max_x[mi],
                          None if sweep_axis is None else sweep_axis[mi])
    return good.any(1)


N_PARK = 150


def park_ok(fs: FingerSamples, R: np.ndarray, c: np.ndarray, max_x: np.ndarray,
            sweep_axis: Optional[np.ndarray] = None, chunk: int = 48) -> np.ndarray:
    """(M,) bool: the finger has some pose (of N_PARK fixed samples, plus its zero pose) entirely clear of the pen
    (or of its swept disc) and the table -- it can get out of the way."""
    M = len(c)
    out = np.zeros(M, dtype=bool)
    for k0 in range(0, len(fs.park), chunk):
        todo = np.nonzero(~out)[0]
        if not len(todo):
            break
        ids = fs.park[k0:k0 + chunk]
        nP = len(ids)
        P = np.broadcast_to(fs.pts[ids][None], (len(todo),) + fs.pts[ids].shape).reshape(-1, fs.pts.shape[1], 3)
        rep = lambda x: None if x is None else np.repeat(x[todo], nP, 0)    # noqa: E731
        ok = _clear(P, fs.rad, rep(R), rep(c), rep(max_x), rep(sweep_axis)).reshape(len(todo), nP)
        out[todo] = ok.any(1)
    return out


# --------------------------------------------------------------------------
# Pinch slots on the pen (its own frame)
# --------------------------------------------------------------------------

STATIONS = np.array([-30.0, -15.0, 0.0, 15.0, 30.0])


def pinch_slots(axes=(1, 2)):
    """(S, 2, 3) target pairs (tip centres, pen frame) for antipodal pinches: across the width along each axis in
    `axes` at each station along the pen, and end to end."""
    out = []
    for k in axes:
        e = np.zeros(3)
        e[k] = FACE_OFF
        for s in STATIONS:
            c = np.array([s, 0.0, 0.0])
            out.append([c + e, c - e])
    out.append([[END_OFF, 0, 0], [-END_OFF, 0, 0]])
    return np.array(out)


def touch_points():
    """(T, 3) single-contact targets for a third finger: the long faces at the stations, and both ends."""
    out = []
    for k in (1, 2):
        for sg in (1, -1):
            for s in STATIONS:
                p = np.array([s, 0.0, 0.0])
                p[k] = sg * FACE_OFF
                out.append(p)
    out += [[END_OFF, 0, 0], [-END_OFF, 0, 0]]
    return np.array(out)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------



def _feasible_matrix(samples: List[FingerSamples], targets_local, R, c, max_x):
    """(F, H, T) bool: finger f reaches target t of pose h. targets_local (T, 3) in the pen frame."""
    H, T = len(R), len(targets_local)
    W = np.einsum("hab,tb->hta", R, targets_local) + c[:, None]       # palm frame
    flatW = W.reshape(-1, 3)
    Rr = np.repeat(R, T, 0)
    cr = np.repeat(c, T, 0)
    mx = np.repeat(max_x, T, 0)
    return np.stack([reach_ok(fs, flatW, Rr, cr, mx).reshape(H, T) for fs in samples])


def _combine(A, B, park, touch=None):
    """A, B: (F, H, S) finger reaches slot s's first / second point; park (F, H); touch (F, H) or None.
    Returns (H,) bool: two different fingers pinch some slot [and a third touches], every other finger parks."""
    F, H, S = A.shape
    out = np.zeros(H, dtype=bool)
    for i in range(F):
        for j in range(F):
            if i == j:
                continue
            pinch = (A[i] & B[j]).any(1)
            if not pinch.any():
                continue
            rest = [k for k in range(F) if k not in (i, j)]
            if touch is None:
                ok = pinch.copy()
                for k in rest:
                    ok &= park[k]
                out |= ok
            else:
                for t in rest:
                    ok = pinch & touch[t]
                    for k in rest:
                        if k != t:
                            ok &= park[k]
                    out |= ok
    return out


def _lazy_park(samples, A, B, R, c, max_x, sweep_axis=None):
    """(F, H) park flags, computed only where some pinch exists (elsewhere False: no pinch to complete)."""
    F, H = A.shape[0], A.shape[1]
    park = np.zeros((F, H), dtype=bool)
    need = (A.any(2).sum(0) >= 1) & (B.any(2).sum(0) >= 1)
    if F <= 2 or not need.any():
        return park
    ids = np.nonzero(need)[0]
    for k, fs in enumerate(samples):
        park[k, ids] = park_ok(fs, R[ids], c[ids], max_x[ids], None if sweep_axis is None else sweep_axis[ids])
    return park


def _pinch_pose_set(samples, slots, touch_pts, R, c, max_x):
    """For poses (R, c, max_x): (pinch2, pinch3) (H,) bool."""
    S = len(slots)
    tl = slots.reshape(-1, 3)
    if touch_pts is not None:
        tl = np.concatenate([tl, touch_pts])
    feas = _feasible_matrix(samples, tl, R, c, max_x)            # (F, H, 2S [+T])
    A, B = feas[:, :, 0:2 * S:2], feas[:, :, 1:2 * S:2]
    park = _lazy_park(samples, A, B, R, c, max_x)
    p2 = _combine(A, B, park)
    p3 = None
    if touch_pts is not None and len(samples) >= 3:
        touch = feas.any(2)
        p3 = _combine(A, B, park, touch)
    elif touch_pts is not None:
        p3 = np.zeros(len(R), dtype=bool)
    return p2, p3


def evaluate(hand: gh.Hand, n_goals: int = 96, n_axes: int = 96, n_yaws: int = 12, seed: int = 0,
             samples: Optional[List[FingerSamples]] = None, n_samples: int = N_SAMPLES,
             details: bool = False) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    if samples is None:
        samples = [finger_samples(hand, i, n_samples, seed) for i in range(len(hand.fingers))]
    F = len(samples)
    res: Dict[str, float] = {}
    det: Dict[str, object] = {}
    lat = np.array([(y, z) for y in LATERAL for z in LATERAL])
    lat = lat[np.argsort(np.hypot(lat[:, 0], lat[:, 1]), kind="stable")]
    L = len(lat)

    # ---- pickup: pen flat on the table (pen z = palm -x), centre at REST_Z, random yaw
    yaws = np.arange(n_yaws) * (math.pi / n_yaws)
    pick = np.zeros(n_yaws, dtype=bool)
    pick_depth = np.full(n_yaws, np.nan)
    flat_slots = pinch_slots(axes=(1,))                   # across the horizontal width, or end to end
    Rp_y = []
    for a in yaws:
        xl = np.array([0.0, math.cos(a), math.sin(a)])
        zl = np.array([-1.0, 0.0, 0.0])
        Rp_y.append(np.column_stack([xl, np.cross(zl, xl), zl]))
    Rp_y = np.array(Rp_y)
    for depth in np.arange(PLATE_FACE + GAP + PEN_HALF[2], WRIST_Z[1] - REST_Z + 1, 10.0):
        todo = np.nonzero(~pick)[0]
        if not len(todo):
            break
        Rs = np.repeat(Rp_y[todo], L, 0)
        cs = np.concatenate([np.full((len(todo) * L, 1), depth), np.tile(lat, (len(todo), 1))], 1)
        p2, _ = _pinch_pose_set(samples, flat_slots, None, Rs, cs, np.full(len(Rs), depth + REST_Z))
        hit = p2.reshape(len(todo), L).any(1)
        pick_depth[todo[hit]] = depth
        pick[todo] |= hit
    res["pickup"] = float(pick.mean())
    det["pickup_depth_mm"] = pick_depth.tolist()

    # ---- hold: task goals
    Rg = random_rotations(n_goals, rng)
    W2P = np.array([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]])     # world -> palm (palm down): palm x = -world z
    Rp_goal = np.einsum("ab,hbc->hac", W2P, Rg)
    ext_world_z = np.abs(Rg[:, 2, :]) @ PEN_HALF          # vertical half extent
    floor = FLOOR_CLEAR + ext_world_z
    zg = rng.uniform(np.minimum(floor, TOP_Z), TOP_Z)
    held2 = np.zeros(n_goals, dtype=bool)
    held3 = np.zeros(n_goals, dtype=bool)
    slots, touch = pinch_slots(), touch_points()
    for roll in (0.0, math.pi / 6, math.pi / 3):
        Rroll = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        Rp = Rp_goal @ Rroll
        ext_x = np.abs(Rp[:, 0, :]) @ PEN_HALF               # half extent along the palm normal
        for depth_step in range(0, 20):
            todo = ~held3
            depth = PLATE_FACE + GAP + ext_x + 10.0 * depth_step
            ok_d = (depth + zg <= WRIST_Z[1]) & todo
            if not ok_d.any():
                break
            hs = np.nonzero(ok_d)[0]
            Rs = np.repeat(Rp[hs], L, 0)
            cs = np.concatenate([np.repeat(depth[hs], L)[:, None], np.tile(lat, (len(hs), 1))], 1)
            mx = np.repeat(depth[hs] + zg[hs], L)
            p2, p3 = _pinch_pose_set(samples, slots, touch, Rs, cs, mx)
            held2[hs] |= p2.reshape(len(hs), L).any(1)
            held3[hs] |= p3.reshape(len(hs), L).any(1)
    res["hold2"] = float(held2.mean())
    res["hold3"] = float(held3.mean())

    # ---- pivot about u: tips at c +- FACE_OFF u; their links, and every other finger, clear of the swept disc
    U = fibonacci_sphere(2 * n_axes)
    U = U[U[:, 0] >= 0][:n_axes]                          # a line: one hemisphere
    ext = SWEEP_R * np.sqrt(np.clip(1 - U[:, 0] ** 2, 0, 1)) + PEN_HALF[1] * np.abs(U[:, 0])
    piv = np.zeros(len(U), dtype=bool)
    reach = np.zeros(len(U), dtype=bool)
    piv_depth = np.full(len(U), np.nan)
    for depth_step in range(0, 12):
        depth = PLATE_FACE + GAP + ext + 10.0 * depth_step
        ok = depth + ext <= WRIST_Z[1]                      # pen centre high enough to clear the table, wrist <= 20
        hs = np.nonzero(ok & ~piv)[0]
        if not len(hs):
            break
        cs = np.concatenate([np.repeat(depth[hs], L)[:, None], np.tile(lat, (len(hs), 1))], 1)
        us = np.repeat(U[hs], L, 0)
        mx = np.full(len(cs), WRIST_Z[1])
        tp, tm = cs + FACE_OFF * us, cs - FACE_OFF * us
        Rd = np.repeat(np.eye(3)[None], len(cs), 0)
        A = np.stack([reach_ok(fs, tp, Rd, cs, mx, sweep_axis=us) for fs in samples])[:, :, None]
        B = np.stack([reach_ok(fs, tm, Rd, cs, mx, sweep_axis=us) for fs in samples])[:, :, None]
        park = _lazy_park(samples, A, B, Rd, cs, mx, us)
        ok2 = _combine(A, B, park).reshape(len(hs), L).any(1)
        piv_depth[hs[ok2]] = depth[hs[ok2]]
        piv[hs] |= ok2
        # reach only (static pen, its width along u, no sweep, no parking)
        Rpen = []
        for u in us:
            x = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
            x /= np.linalg.norm(x)
            Rpen.append(np.column_stack([x, u, np.cross(x, u)]))
        Rpen = np.array(Rpen)
        A2 = np.stack([reach_ok(fs, tp, Rpen, cs, mx) for fs in samples])[:, :, None]
        B2 = np.stack([reach_ok(fs, tm, Rpen, cs, mx) for fs in samples])[:, :, None]
        allp = np.ones((F, len(cs)), dtype=bool)
        reach[hs] |= _combine(A2, B2, allp).reshape(len(hs), L).any(1)
    horiz = np.abs(U[:, 0]) <= math.sin(math.radians(30))
    res["pivot"] = float(piv.mean())
    res["pivot_h"] = float(piv[horiz].mean())
    res["pivot_reach"] = float((reach | piv).mean())
    det.update(pivot_axes=U.tolist(), pivot_ok=piv.tolist(), pivot_depth_mm=piv_depth.tolist(),
               hold_ok=held3.tolist(), goal_axes=Rp_goal[:, :, 0].tolist(), goal_z=zg.tolist())

    # ---- structure
    res["fingers"] = F
    res["dof"] = len(dv.dofs(hand))
    res["finger_len_mm"] = float(np.mean([gh.finger_length_mm(f) for f in hand.fingers]))
    res["mass_g"] = float(sum(dv.link_mass_props(hand, i, j)[0] for i, f in enumerate(hand.fingers)
                              for j in range(len(f.joints))) * 1e3)
    if details:
        res["details"] = det
    return res


def score(m: Dict[str, float]) -> float:
    """One number for search: the task's three needs (pick up, hold any goal with a spare finger, pivot), less a
    small charge per joint beyond 8 (RL sample efficiency)."""
    return (0.2 * m["pickup"] + 0.3 * m["hold3"] + 0.4 * m["pivot"] + 0.1 * m["pivot_h"]
            - 0.01 * max(0, m["dof"] - 8))


RESET_ROOT_Z = 150.0          # WristCfg.default_pose z


def reset_clean(samples: List[FingerSamples], n: int = 3000, seed: int = 0) -> float:
    """Probability that the reset pose (wrist at 15 cm, every finger joint uniform over its range, the pen flat on
    the table under the palm centre at a random yaw) has no link inside the pen or the table (1 mm slack). Joints
    are drawn independently, so it is the product over fingers."""
    rng = np.random.default_rng(seed)
    p = 1.0
    for fs in samples:
        idx = np.arange(1, min(n, len(fs.tip)))
        P = fs.pts[idx]
        yaw = rng.uniform(0, math.pi, len(idx))
        xl = np.stack([np.zeros_like(yaw), np.cos(yaw), np.sin(yaw)], 1)
        zl = np.tile([-1.0, 0, 0], (len(idx), 1))
        R = np.stack([xl, np.cross(zl, xl), zl], 2)
        c = np.tile([RESET_ROOT_Z - REST_Z, 0.0, 0.0], (len(idx), 1))
        rel = P - c[:, None]
        sd = box_sdf(np.einsum("hba,hpb->hpa", R, rel), PEN_HALF)
        ok = np.all(sd >= fs.rad[None] - 1.0, 1) & np.all(P[..., 0] + fs.rad[None] <= RESET_ROOT_Z + 1.0, 1)
        p *= float(ok.mean())
    return p


def gait_hold(hand: gh.Hand, samples: List[FingerSamples], n_goals: int = 96, seed: int = 0) -> Dict[str, float]:
    """Finger gaiting under pushes. For the task's goals (the same draw as `evaluate`), at some held pose:
    `gait_any`: some finger can lift off (and park clear) while the others still hold a pinch plus a third contact;
    `gait_all`: every finger can, one at a time (a full gait cycle without ever dropping to a bare 2-finger pinch,
    which turns freely about its own axis under a push). Both need at least 4 fingers."""
    rng = np.random.default_rng(seed)
    F = len(samples)
    out = {"gait_any": 0.0, "gait_all": 0.0}
    if F < 4:
        return out
    lat = np.array([(y, z) for y in LATERAL for z in LATERAL])
    lat = lat[np.argsort(np.hypot(lat[:, 0], lat[:, 1]), kind="stable")]
    L = len(lat)
    Rg = random_rotations(n_goals, rng)
    W2P = np.array([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]])
    Rp_goal = np.einsum("ab,hbc->hac", W2P, Rg)
    floor = FLOOR_CLEAR + np.abs(Rg[:, 2, :]) @ PEN_HALF
    zg = rng.uniform(np.minimum(floor, TOP_Z), TOP_Z)
    g_any = np.zeros(n_goals, dtype=bool)
    g_all = np.zeros(n_goals, dtype=bool)
    slots, touch_pts = pinch_slots(), touch_points()
    S = len(slots)
    tl = np.concatenate([slots.reshape(-1, 3), touch_pts])
    for roll in (0.0, math.pi / 6, math.pi / 3):
        Rroll = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        Rp = Rp_goal @ Rroll
        ext_x = np.abs(Rp[:, 0, :]) @ PEN_HALF
        for depth_step in range(0, 20):
            depth = PLATE_FACE + GAP + ext_x + 10.0 * depth_step
            ok_d = (depth + zg <= WRIST_Z[1]) & ~g_all
            if not ok_d.any():
                break
            hs = np.nonzero(ok_d)[0]
            Rs = np.repeat(Rp[hs], L, 0)
            cs = np.concatenate([np.repeat(depth[hs], L)[:, None], np.tile(lat, (len(hs), 1))], 1)
            mx = np.repeat(depth[hs] + zg[hs], L)
            feas = _feasible_matrix(samples, tl, Rs, cs, mx)
            A, B = feas[:, :, 0:2 * S:2], feas[:, :, 1:2 * S:2]
            touch = feas.any(2)
            park = np.stack([park_ok(fs, Rs, cs, mx) for fs in samples])
            lift_ok = []
            for f in range(F):
                keep = [k for k in range(F) if k != f]
                ok = _combine(A[keep], B[keep], park[keep], touch[keep]) & park[f]
                lift_ok.append(ok)
            lift_ok = np.array(lift_ok)
            g_any[hs] |= lift_ok.any(0).reshape(len(hs), L).any(1)
            g_all[hs] |= lift_ok.all(0).reshape(len(hs), L).any(1)
    out["gait_any"] = float(g_any.mean())
    out["gait_all"] = float(g_all.mean())
    return out
