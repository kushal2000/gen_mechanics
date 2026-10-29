"""Stage 3: holds and rotations that are free of collisions (CPU only).

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_pick_random/clean.py --top 30 --jobs 10

Stages 1-2 only asked whether a fingertip can be at the contact. Here every link is a body: each link's box
(derive.finger_geometry: servo-sized, up to 57 mm thick) is approximated by a capsule along its longest axis
(radius = mean of the other two half sizes), the fingertip is its own capsule (r 16.5 mm). A grasp counts only if
- each finger's configuration keeps its links out of the palm plate (beyond 20 mm from the base joint) and out of the
  pen (fingertip capsule at most 4 mm into it, other links at most 3 mm);
- the fingers of the grasp keep clear of each other (capsules at most 3 mm into each other).

At 4 pen centres per hand (cells where stage 1's 3-finger hold covers >= 90 % of axis directions, ranked by how many
fingers can touch on average, >= 20 mm apart), for each of 30 axis directions x 3 rolls (x palm hinge levels):
- clean2: a collision-free antipodal pair; clean3: plus a third finger touching another contact, clear of both;
- rotc: per rotation axis w perpendicular to the pen axis (6), a collision-free pair whose contact velocities w x r
  are in its fingers' Jacobian spans (residual < 40 %), or a pivot pair plus a collision-free pushing third finger.

Per hand: CH2, CH3, ROTC = the best centre's fraction of axis directions (CH) or (axis, w) cells (ROTC); *_mean = mean
over the 4 centres. Writes clean.json.
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

TOL = 7.5
MAXC = 30
RES = 0.40
PIVOT_MM = 8.0
PUSH = 0.30
SMIN = 10.0
NW = 6
PEN_PEN_TIP = 4.0       # mm the fingertip capsule may sink into the pen
PEN_PEN_LINK = 3.0
FF_PEN = 3.0            # mm two fingers' capsules may overlap
SEG_PTS = 7


def finger_data(H, i, n, rng, hinge, footprint):
    ds = dv.dofs(H)
    q = np.zeros((n, len(ds)))
    for k, d in enumerate(ds):
        if d.finger == i:
            q[:, k] = rng.uniform(d.limits[0], d.limits[1], n)
    if H.palm_joints:
        q[:, 0] = hinge
    q = dv.tie(H, q, ds)
    pose = dv.fk(H, q)
    f = H.fingers[i]
    nj = len(f.joints)
    fg = dv.finger_geometry(f)
    # capsules: (n, nj, 2, 3) ends and (nj,) radii
    ends, rad = [], []
    for j in range(nj):
        T = pose.links[i][:, j]
        R, o = T[:, :3, :3], T[:, :3, 3] / 1e-3
        if j == nj - 1:
            p0, p1, r = fg.tip.p0, fg.tip.p1, fg.tip.radius
        else:
            b = fg.links[j]
            ax = int(np.argmax(b.half))
            others = [b.half[k] for k in range(3) if k != ax]
            r = float(np.mean(others))
            e = np.zeros(3); e[ax] = max(b.half[ax] - r, 0.0)
            M = np.asarray(b.axes) if getattr(b, "axes", None) is not None else np.eye(3)
            p0, p1 = b.centre - M @ e, b.centre + M @ e
        ends.append(np.stack([o + R @ p0, o + R @ p1], 1))
        rad.append(r)
    ends = np.stack(ends, 1)
    rad = np.array(rad)
    # palm check: points along every capsule (beyond 20 mm of the base joint) must stay off the plate
    base = pose.pre[i][:, 0, :3, 3] / 1e-3
    tt = np.linspace(0, 1, SEG_PTS)
    pts = ends[:, :, :1] * (1 - tt)[None, None, :, None] + ends[:, :, 1:] * tt[None, None, :, None]   # (n, nj, SEG, 3)
    far = np.linalg.norm(pts - base[:, None, None], axis=-1) > 20.0
    inplate = gvb.dv.point_in_polygon(pts.reshape(-1, 3)[:, 1:], footprint, 0.0).reshape(pts.shape[:3])
    deep = pts[..., 0] < PLATE_HALF_MM + rad[None, :, None] - 3.0
    palm_ok = ~(far & inplate & deep).any(axis=(1, 2))
    # contact spheres on the distal capsule
    T = pose.links[i][:, -1]
    R, o = T[:, :3, :3], T[:, :3, 3] / 1e-3
    cap = fg.tip
    sph = [cap.p1, 0.5 * (cap.p0 + cap.p1), cap.p0]
    C = np.concatenate([o + R @ p for p in sph])
    D = np.tile(R[:, :, 0], (3, 1))
    K = np.repeat([0, 1, 1], n)
    SID = np.tile(np.arange(n), 3)
    W, O = [], []
    for j in range(nj):
        pre = pose.pre[i][:, j]
        W.append(pre[:, :3, :3] @ dv.joint_axis(H, i, j))
        O.append(pre[:, :3, 3] / 1e-3)
    W, O = np.stack(W, 1), np.stack(O, 1)
    J = np.cross(np.tile(W, (3, 1, 1)), C[:, None, :] - np.tile(O, (3, 1, 1)))
    U, s, _ = np.linalg.svd(np.transpose(J, (0, 2, 1)), full_matrices=False)
    U = U * (s[:, None, :] > SMIN)
    keep = palm_ok[SID]
    return {"C": C[keep], "D": D[keep], "K": K[keep], "U": U[keep], "SID": SID[keep], "tree": cKDTree(C[keep]),
            "ends": ends, "rad": rad, "q": q}


def pen_clear(fd, sids, c, F):
    """Per candidate sample: links clear of the pen box (centre c, frame F)."""
    E = fd["ends"][sids]                                   # (m, nj, 2, 3)
    tt = np.linspace(0, 1, SEG_PTS)
    P = E[:, :, :1] * (1 - tt)[None, None, :, None] + E[:, :, 1:] * tt[None, None, :, None]
    L = (P - c) @ F                                        # pen frame
    d = np.linalg.norm(np.maximum(np.abs(L) - S.PEN_HALF, 0.0), axis=-1)       # outside distance (0 inside)
    inside = np.all(np.abs(L) < S.PEN_HALF, axis=-1)
    d = np.where(inside, -np.min(S.PEN_HALF - np.abs(L), axis=-1), d)
    gap = d.min(-1) - fd["rad"][None]                      # (m, nj)
    allow = np.full(len(fd["rad"]), -PEN_PEN_LINK); allow[-1] = -PEN_PEN_TIP
    return np.all(gap >= allow[None], axis=1)


def seg_dist(p1, q1, p2, q2):
    """Closest distance between segments, broadcast over leading dims."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = (d1 * d1).sum(-1); e = (d2 * d2).sum(-1); f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1); b = (d1 * d2).sum(-1)
    den = a * e - b * b
    s = np.where(den > 1e-9, np.clip((b * f - c * e) / np.maximum(den, 1e-9), 0, 1), 0.0)
    t = (b * s + f) / np.maximum(e, 1e-9)
    t1 = np.clip(t, 0, 1)
    s = np.where(t < 0, np.clip(-c / np.maximum(a, 1e-9), 0, 1), np.where(t > 1, np.clip((b - c) / np.maximum(a, 1e-9), 0, 1), s))
    t = t1
    x = (p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None])
    return np.linalg.norm(x, axis=-1)


def ff_clear(fa, sa, fb, sb):
    """(len(sa), len(sb)) bool: the two fingers' capsules clear of each other."""
    A, B = fa["ends"][sa], fb["ends"][sb]                # (ma, ja, 2, 3), (mb, jb, 2, 3)
    d = seg_dist(A[:, None, :, None, 0], A[:, None, :, None, 1], B[None, :, None, :, 0], B[None, :, None, :, 1])
    gap = d - fa["rad"][None, None, :, None] - fb["rad"][None, None, None, :]
    return np.all(gap >= -FF_PEN, axis=(2, 3))


def cands(fd, pt, nrm, c, F):
    idx = fd["tree"].query_ball_point(pt, TOL)
    if not idx:
        return None
    idx = np.array(idx)
    dd = fd["D"][idx] @ nrm
    ok = np.where(fd["K"][idx] == 0, dd <= S.DOT_END, np.abs(dd) <= S.DOT_SIDE)
    idx = idx[ok]
    if len(idx) == 0:
        return None
    # one row per distinct sample
    _, first = np.unique(fd["SID"][idx], return_index=True)
    idx = idx[np.sort(first)][:MAXC * 2]
    sids = fd["SID"][idx]
    okp = pen_clear(fd, sids, c, F)
    idx, sids = idx[okp][:MAXC], sids[okp][:MAXC]
    if len(idx) == 0:
        return None
    return {"sid": sids, "U": fd["U"][idx]}


def moves(U, v):
    nv = np.linalg.norm(v)
    if nv < PIVOT_MM:
        return np.ones(len(U), bool)
    proj = np.einsum("mij,mkj,k->mi", U, U, v)
    return np.linalg.norm(v - proj, axis=1) < RES * nv


def at_centre(fds, levels, nF, c, frames, arrays=False):
    O, Rn = frames.shape[:2]
    h2 = np.zeros(O, bool); h3 = np.zeros(O, bool); rot = np.zeros((O, NW), bool)
    for o in range(O):
        for r in range(Rn):
            if h3[o] and rot[o].all():
                break
            F = frames[o, r]
            ext = np.abs(F) @ S.PEN_HALF
            if c[0] - ext[0] < PLATE_HALF_MM + S.PALM_CLEAR:
                continue
            offs, nrm, pairs = S.contact_set(F)
            rvec = offs - S.R_TIP * nrm
            ws = [math.cos(t) * F[:, 1] + math.sin(t) * F[:, 2] for t in np.linspace(0, math.pi, NW, endpoint=False)]
            V = np.stack([np.cross(w, rvec) for w in ws])          # (NW, C, 3)
            for li in range(len(levels)):
                fl = [fds[(li, i)] for i in range(nF)]
                cd = [[cands(fl[i], c + offs[k], nrm[k], c, F) for k in range(len(offs))] for i in range(nF)]
                for ka, kb in pairs:
                    for i in range(nF):
                        A = cd[i][ka]
                        if A is None:
                            continue
                        for j in range(nF):
                            B = cd[j][kb]
                            if j == i or B is None:
                                continue
                            clr = ff_clear(fl[i], A["sid"], fl[j], B["sid"])
                            if not clr.any():
                                continue
                            h2[o] = True
                            need3 = not h3[o]
                            ai, bi = np.nonzero(clr)
                            # third finger for clean3 (and pivots)
                            third = []
                            for m in range(nF):
                                if m in (i, j):
                                    continue
                                for k in range(len(offs)):
                                    Cm = cd[m][k]
                                    if Cm is None or k in (ka, kb):
                                        continue
                                    third.append((m, k, Cm))
                            for wi in range(NW):
                                if rot[o, wi]:
                                    continue
                                va, vb = V[wi, ka], V[wi, kb]
                                ma, mb = moves(A["U"], va), moves(B["U"], vb)
                                good = ma[ai] & mb[bi]
                                if not good.any():
                                    continue
                                pivot = np.linalg.norm(va) < PIVOT_MM and np.linalg.norm(vb) < PIVOT_MM
                                if not pivot:
                                    rot[o, wi] = True
                                    continue
                                for m, k, Cm in third:
                                    vk = V[wi, k]
                                    nv = np.linalg.norm(vk)
                                    if nv < 1e-6 or abs(vk @ nrm[k]) < PUSH * nv:
                                        continue
                                    mm = moves(Cm["U"], vk)
                                    if not mm.any():
                                        continue
                                    sel = np.nonzero(good)[0][:8]
                                    ca = ff_clear(fl[m], Cm["sid"][mm], fl[i], A["sid"][ai[sel]])
                                    cb = ff_clear(fl[m], Cm["sid"][mm], fl[j], B["sid"][bi[sel]])
                                    if (ca & cb).any():
                                        rot[o, wi] = True
                                        break
                            if need3:
                                sel = np.arange(min(len(ai), 12))
                                for m, k, Cm in third:
                                    ca = ff_clear(fl[m], Cm["sid"], fl[i], A["sid"][ai[sel]])
                                    cb = ff_clear(fl[m], Cm["sid"], fl[j], B["sid"][bi[sel]])
                                    if (ca & cb).any():
                                        h3[o] = True
                                        break
                            if h3[o] and rot[o].all():
                                break
                        if h3[o] and rot[o].all():
                            break
                    if h3[o] and rot[o].all():
                        break
    if arrays:
        return h2, h3, rot
    return float(h2.mean()), float(h3.mean()), float(rot.mean())


def centres(H, grids, levels, nF, frames, k=4):
    xs = np.arange(25.0, 136.0, 10.0)
    ys = np.arange(-100.0, 101.0, 10.0)
    P = np.stack(np.meshgrid(xs, ys, ys, indexing="ij"), -1).reshape(-1, 3)
    O, Rn = frames.shape[:2]
    h3 = np.zeros((len(P), O), bool)
    ntm = np.zeros((len(P), O))
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
                ntm[:, o] = np.maximum(ntm[:, o], np.where(fits, nt, 0))
    cov = h3.mean(1)
    rich = ntm.mean(1)
    thr = min(0.9, cov.max())
    score = np.where(cov >= thr, rich, -1.0)
    order = np.argsort(-score, kind="stable")
    out = []
    for idx in order:
        if score[idx] < 0:
            break
        if all(np.linalg.norm(P[idx] - P[j]) >= 20.0 for j in out):
            out.append(idx)
        if len(out) == k:
            break
    return [P[j] for j in out], [float(cov[j]) for j in out], [float(rich[j]) for j in out]


def evaluate(name, n=30000, seed=5, k=4):
    t0 = time.time()
    H = S.load(name)
    nF = len(H.fingers)
    fp = gvb.footprint_mm(H)
    moving = [finger_section(H, i) == 0 for i in range(nF)]
    levels = S.HINGE_LEVELS if H.palm_joints else np.array([0.0])
    grids, fds = {}, {}
    for li, h in enumerate(levels):
        for i in range(nF):
            if li > 0 and not moving[i]:
                grids[(li, i)], fds[(li, i)] = grids[(0, i)], fds[(0, i)]
                continue
            grids[(li, i)], _ = S.finger_grid(H, i, n, np.random.default_rng(seed + 11 * li + i), h, fp)
            fds[(li, i)] = finger_data(H, i, n, np.random.default_rng(seed + 100 + 11 * li + i), h, fp)
    frames = S.pen_frames()
    cs, covs, rich = centres(H, grids, levels, nF, frames, k=k)
    res = [at_centre(fds, levels, nF, c, frames) for c in cs]
    if not res:
        res = [(0.0, 0.0, 0.0)]
    a = np.array(res)
    b = int(np.argmax(a[:, 1] + a[:, 2]))
    return {"name": name, "centres": [c.tolist() for c in cs], "stage1_cov3": covs, "mean_touching": rich,
            "per_centre": res, "CH2": float(a[b, 0]), "CH3": float(a[b, 1]), "ROTC": float(a[b, 2]),
            "CH3_mean": float(a[:, 1].mean()), "ROTC_mean": float(a[:, 2].mean()),
            "best_centre": cs[b].tolist() if cs else None, "secs": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--centres", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "clean.json"))
    a = ap.parse_args()
    if a.only:
        names = [x if x.startswith("coarse") else f"coarse_s0_{x}" for x in a.only.split(",")]
    else:
        rk = json.load(open(HERE / "ranking.json"))
        names = [r["name"] for r in rk[:a.top]]
    from multiprocessing import Pool
    rows = []
    with Pool(a.jobs) as pool:
        from functools import partial
        for r in pool.imap_unordered(partial(evaluate, seed=a.seed, k=a.centres), names):
            rows.append(r)
            print(f"{r['name']} CH2={r['CH2']:.2f} CH3={r['CH3']:.2f} ROTC={r['ROTC']:.2f} CH3m={r['CH3_mean']:.2f} "
                  f"ROTCm={r['ROTC_mean']:.2f} rich={[round(x, 1) for x in r['mean_touching']]} "
                  f"best={r['best_centre']} ({r['secs']:.0f}s)", flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
