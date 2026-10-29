"""Stage 2: can the hand ROTATE the held pen, not just hold it? (CPU only.)

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_pick_random/rotate.py [--only 038,075] [--top 40]

For each hand (default: the --top hands of stage 1's metrics.json by C3 then zone volume), take the 4 pen centres
where stage 1's 3-finger hold covers the most axis directions. At each, for every axis direction a (hemisphere, 30)
and roll, and every rotation axis w perpendicular to a (6 over 180 deg: the rotations that move the long axis, which
is all the goals care about), ask whether some grasp can make that rotation:

- an antipodal pair (two distinct fingers on opposite faces, or on the two ends) whose contact velocities w x r
  (r: contact from the pen centre) both lie in the span of their finger's positional Jacobian at a sampled
  configuration that touches there (residual < 30 % of |v|; a contact within 8 mm of the rotation axis is a pivot and
  needs no motion);
- if both pair contacts are pivots (w nearly along the pinch line), a third finger that can touch another face and
  move along w x r there with a pushing component (|v.n| >= 30 % |v|).

Jacobian columns come from the grammar's own FK frames: w_j x (c - o_j), w_j the joint axis in the palm frame
(`derive.joint_axis` rotated by the joint frame before it turns), o_j the joint origin; directions with singular
value < 10 mm/rad do not count. The palm hinge is left out (it moves a whole palm section, not one finger).

ROT = the fraction of (axis direction, rotation axis) cells achievable at the best of the 4 centres;
ROT_mean = its mean over the 4 centres. Written to rotate.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screen as S  # noqa: E402
from screen import dv, gvb, finger_section, PLATE_HALF_MM  # noqa: E402

TOL = 7.5            # mm: a sample's sphere centre within this of the contact sphere centre
MAXC = 40            # candidate samples per (finger, contact)
RES = 0.30
PIVOT_MM = 8.0      # a contact closer than this to the rotation axis is a pivot (needs no motion)
PUSH = 0.30
SMIN = 10.0          # mm/rad
NW = 6


def finger_data(H, i, n, rng, hinge, footprint):
    ds = dv.dofs(H)
    cols = [k for k, d in enumerate(ds) if d.finger == i]
    q = np.zeros((n, len(ds)))
    for k in cols:
        q[:, k] = rng.uniform(ds[k].limits[0], ds[k].limits[1], n)
    if H.palm_joints:
        q[:, 0] = hinge
    q = dv.tie(H, q, ds)
    pose = dv.fk(H, q)
    f = H.fingers[i]
    T = pose.links[i][:, -1]
    R, o = T[:, :3, :3], T[:, :3, 3] / 1e-3
    cap = dv.finger_geometry(f).tip
    pts = [cap.p1, 0.5 * (cap.p0 + cap.p1), cap.p0]
    C = np.concatenate([o + R @ p for p in pts])
    D = np.tile(R[:, :, 0], (3, 1))
    K = np.repeat([0, 1, 1], n)
    # Jacobian columns per joint, for each sphere centre
    W, O = [], []
    for j in range(len(f.joints)):
        pre = pose.pre[i][:, j]
        W.append(pre[:, :3, :3] @ dv.joint_axis(H, i, j))
        O.append(pre[:, :3, 3] / 1e-3)
    W, O = np.stack(W, 1), np.stack(O, 1)                     # (n, k, 3)
    W3, O3 = np.tile(W, (3, 1, 1)), np.tile(O, (3, 1, 1))
    J = np.cross(W3, C[:, None, :] - O3)   # mm/rad, (3n, k, 3)
    # orthonormal basis of the significant range
    U, s, _ = np.linalg.svd(np.transpose(J, (0, 2, 1)), full_matrices=False)   # J^T: (3n, 3, k)
    U = U * (s[:, None, :] > SMIN)
    inside = gvb.dv.point_in_polygon(C[:, 1:], footprint, 0.0) & (C[:, 0] < PLATE_HALF_MM + S.R_TIP - 3.0)
    keep = ~inside
    return {"C": C[keep], "D": D[keep], "K": K[keep], "U": U[keep], "tree": cKDTree(C[keep])}


def candidates(fd, pt, nrm):
    idx = fd["tree"].query_ball_point(pt, TOL)
    if not idx:
        return None
    idx = np.array(idx)
    d = fd["D"][idx] @ nrm
    ok = np.where(fd["K"][idx] == 0, d <= S.DOT_END, np.abs(d) <= S.DOT_SIDE)
    idx = idx[ok]
    if len(idx) == 0:
        return None
    return fd["U"][idx[:MAXC]]                                 # (m, 3, r)


def can_move(U, v):
    """Some candidate's range holds v (residual < RES |v|)."""
    proj = np.einsum("mij,mkj,k->mi", U, U, v)
    return bool((np.linalg.norm(v - proj, axis=1) < RES * np.linalg.norm(v)).any())


def rot_at(H, fds, levels, nF, c, frames):
    O, Rn = frames.shape[:2]
    ach = np.zeros((O, NW), bool)
    for o in range(O):
        for r in range(Rn):
            F = frames[o, r]
            a = F[:, 0]
            ext = np.abs(F) @ S.PEN_HALF
            if c[0] - ext[0] < PLATE_HALF_MM + S.PALM_CLEAR:
                continue
            offs, nrm, pairs = S.contact_set(F)
            u, v = F[:, 1], F[:, 2]
            ws = [math.cos(t) * u + math.sin(t) * v for t in np.linspace(0, math.pi, NW, endpoint=False)]
            rvec = offs - S.R_TIP * nrm                           # contact points from the pen centre
            for li in range(len(levels)):
                cand = [[candidates(fds[(li, i)], c + offs[k], nrm[k]) for k in range(len(offs))] for i in range(nF)]
                for wi, w in enumerate(ws):
                    if ach[o, wi]:
                        continue
                    V = np.cross(w, rvec)
                    for ka, kb in pairs:
                        pa, pb = np.linalg.norm(V[ka]) < PIVOT_MM, np.linalg.norm(V[kb]) < PIVOT_MM
                        for i in range(nF):
                            if cand[i][ka] is None or not (pa or can_move(cand[i][ka], V[ka])):
                                continue
                            for j in range(nF):
                                if j == i or cand[j][kb] is None or not (pb or can_move(cand[j][kb], V[kb])):
                                    continue
                                if not (pa and pb):
                                    ach[o, wi] = True
                                else:   # a pivot pair: a third finger must push
                                    for m in range(nF):
                                        if m in (i, j):
                                            continue
                                        for k in range(len(offs)):
                                            if k in (ka, kb) or cand[m][k] is None:
                                                continue
                                            vk = V[k]
                                            nv = np.linalg.norm(vk)
                                            if nv < 1e-6 or abs(vk @ nrm[k]) < PUSH * nv:
                                                continue
                                            if can_move(cand[m][k], vk):
                                                ach[o, wi] = True
                                                break
                                        if ach[o, wi]:
                                            break
                                if ach[o, wi]:
                                    break
                            if ach[o, wi]:
                                break
                        if ach[o, wi]:
                            break
    return ach


def best_centres(H, grids, levels, nF, frames, k=4):
    xs = np.arange(25.0, 136.0, 10.0)
    ys = np.arange(-100.0, 101.0, 10.0)
    P = np.stack(np.meshgrid(xs, ys, ys, indexing="ij"), -1).reshape(-1, 3)
    O, Rn = frames.shape[:2]
    h3 = np.zeros((len(P), O), bool)
    for o in range(O):
        for r in range(Rn):
            F = frames[o, r]
            ext = np.abs(F) @ S.PEN_HALF
            fits = P[:, 0] - ext[0] >= PLATE_HALF_MM + S.PALM_CLEAR
            offs, nrm, pairs = S.contact_set(F)
            nb = S.nearest_bin(nrm)
            pts = P[:, None, :] + offs[None]
            for li in range(len(levels)):
                t = np.stack([S.lookup(grids[(li, i)], pts, nb[None, :]) for i in range(nF)])
                a2, nt = S.holds(t, pairs)
                h3[:, o] |= a2 & fits & (nt >= 3)
    cov = h3.mean(1)
    # k distinct-ish centres: greedy, at least 20 mm apart
    order = np.argsort(-cov, kind="stable")
    out = []
    for idx in order:
        if all(np.linalg.norm(P[idx] - P[j]) >= 20.0 for j in out):
            out.append(idx)
        if len(out) == k:
            break
    return [P[j] for j in out], [float(cov[j]) for j in out]


def evaluate(name, n=20000, seed=1):
    H = S.load(name)
    nF = len(H.fingers)
    rng = np.random.default_rng(seed)
    fp = gvb.footprint_mm(H)
    moving = [finger_section(H, i) == 0 for i in range(nF)]
    levels = S.HINGE_LEVELS if H.palm_joints else np.array([0.0])
    grids, fds = {}, {}
    for li, h in enumerate(levels):
        for i in range(nF):
            if li > 0 and not moving[i]:
                grids[(li, i)], fds[(li, i)] = grids[(0, i)], fds[(0, i)]
                continue
            grids[(li, i)], _ = S.finger_grid(H, i, n, np.random.default_rng(seed + 7 * li + i), h, fp)
            fds[(li, i)] = finger_data(H, i, n, rng, h, fp)
    frames = S.pen_frames()
    cs, covs = best_centres(H, grids, levels, nF, frames)
    rots = []
    for c in cs:
        ach = rot_at(H, fds, levels, nF, c, frames)
        rots.append(float(ach.mean()))
    return {"name": name, "centres": [c.tolist() for c in cs], "cov3": covs, "rot": rots,
            "ROT": max(rots), "ROT_mean": float(np.mean(rots))}


def _init(res):
    globals()["RES"] = res


def _run(nm):
    t0 = time.time()
    r = evaluate(nm)
    r["secs"] = time.time() - t0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--res", type=float, default=RES, help="range residual allowed, fraction of |v|")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "rotate.json"))
    a = ap.parse_args()
    globals()["RES"] = a.res
    if a.only:
        names = [x if x.startswith("coarse") else f"coarse_s0_{x}" for x in a.only.split(",")]
    else:
        m = json.load(open(HERE / "metrics.json"))
        m.sort(key=lambda r: (-r["C3"], -r["zone3_cm3"]))
        names = [r["name"] for r in m[:a.top]]
    rows = []
    from multiprocessing import Pool
    with Pool(a.jobs, initializer=_init, initargs=(a.res,)) as pool:
        for r in pool.imap(_run, names):
            rows.append(r)
            print(f"{r['name']} ROT={r['ROT']:.2f} ROT_mean={r['ROT_mean']:.2f} rot={[round(x, 2) for x in r['rot']]} "
                  f"cov3={[round(x, 2) for x in r['cov3']]} centres={r['centres']} ({r['secs']:.0f}s)", flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
