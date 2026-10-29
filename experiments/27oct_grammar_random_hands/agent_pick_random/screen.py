"""Static screen of the 100 random grammar hands for pen repose v2 (CPU only, no sim).

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_pick_random/screen.py [--only 000,017] [--n 30000]

For every hand: sample each finger's joints uniformly over their ranges (the palm hinge on a shared grid of levels),
take the distal link's capsule (radius 16.5 mm) as the contact body, and voxelise where its sphere centres can be,
together with which pen-surface normals each sample may push on (the link must not come from inside the pen).
Then place the pen (10 x 3 x 3 cm box) at a grid of centres below the palm, in ~30 axis directions (a hemisphere:
holding is symmetric under a -> -a) x 3 rolls about its axis (keypoints lie on the axis, so the roll is free), and ask
which fingers can touch which face. Everything is in the palm frame, mm; x = palm normal toward the table.

Outputs (per hand): a row in metrics.json next to this script; see agent_pick_random.md for the metric definitions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from hand_sampler.grammar import derive as dv  # noqa: E402
from hand_sampler.grammar import viability as gvb  # noqa: E402
from hand_sampler.grammar.hand import PLATE_HALF_MM, finger_section, hand_from_dict  # noqa: E402

ASSETS = REPO / "assets/urdf/27oct_grammar_random_hands"
OUT = Path(__file__).resolve().parent

PEN_HALF = np.array([50.0, 15.0, 15.0])     # half length, half width, half width (mm)
R_TIP = dv.TIP_RADIUS_MM                    # 16.5 mm: the distal capsule
VOX = 5.0                                   # voxel size (mm)
LO = np.array([-60.0, -340.0, -340.0])      # voxel grid origin (mm)
SHAPE = (66, 136, 136)                      # covers x -60..270, y/z -340..340
DIL = 1.5                                   # dilation radius in voxels (~7.5 mm tolerance)
NB = 64                                     # normal bins (Fibonacci sphere), one bit each in a uint64
DOT_END = 0.35                              # distal sphere: link may not come from inside the pen (d.n <= this)
DOT_SIDE = 0.35                             # proximal/mid sphere: side-on contacts only (|d.n| <= this)
HINGE_LEVELS = np.radians([-30.0, -15.0, 0.0, 15.0, 30.0])
PALM_CLEAR = 2.0                            # pen kept this far below the palm's front face (mm)


def fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(phi), np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th)], 1)


NBINS = fib_sphere(NB)
BITS = (np.uint64(1) << np.arange(NB, dtype=np.uint64))
_r = int(math.ceil(DIL))
OFFS = np.array([(a, b, c) for a in range(-_r, _r + 1) for b in range(-_r, _r + 1) for c in range(-_r, _r + 1)
                 if a * a + b * b + c * c <= DIL * DIL + 1e-9])


def load(name: str):
    return hand_from_dict(json.load(open(ASSETS / name / f"{name}.design.json")))


def finger_spheres(H, i, n, rng, hinge):
    """(centres (n*3, 3) mm, link dirs (n*3, 3), kind (n*3,) 0=distal end / 1=side) of finger i's distal capsule."""
    ds = dv.dofs(H)
    q = np.zeros((n, len(ds)))
    for k, d in enumerate(ds):
        if d.finger == i:
            q[:, k] = rng.uniform(d.limits[0], d.limits[1], n)
    if H.palm_joints:
        q[:, 0] = hinge
    q = dv.tie(H, q, ds)
    pose = dv.fk(H, q)
    T = pose.links[i][:, -1]
    R, o = T[:, :3, :3], T[:, :3, 3] / 1e-3
    cap = dv.finger_geometry(H.fingers[i]).tip
    pts = [cap.p1, 0.5 * (cap.p0 + cap.p1), cap.p0]
    C = np.concatenate([o + R @ p for p in pts])
    D = np.tile(R[:, :, 0], (3, 1))
    K = np.repeat([0, 1, 1], n)
    return C, D, K, pose.tips[:, i] / 1e-3


def finger_grid(H, i, n, rng, hinge, footprint):
    C, D, K, tips = finger_spheres(H, i, n, rng, hinge)
    # drop samples whose capsule would sink into the palm plate
    inside = dv.point_in_polygon(C[:, 1:], footprint, 0.0) & (C[:, 0] < PLATE_HALF_MM + R_TIP - 3.0)
    keep = ~inside
    C, D, K = C[keep], D[keep], K[keep]
    dots = D @ NBINS.T                                                    # (m, NB)
    ok = np.where(K[:, None] == 0, dots <= DOT_END, np.abs(dots) <= DOT_SIDE)
    masks = (ok.astype(np.uint64) * BITS[None]).sum(1, dtype=np.uint64)  # bitwise OR == sum, distinct bits
    idx = np.floor((C - LO) / VOX).astype(np.int64)
    grid = np.zeros(SHAPE, dtype=np.uint64)
    for off in OFFS:
        j = idx + off
        m = np.all((j >= 0) & (j < np.array(SHAPE)), axis=1)
        np.bitwise_or.at(grid, (j[m, 0], j[m, 1], j[m, 2]), masks[m])
    return grid, tips


def lookup(grid, pts, nbin):
    """bool: some sample of this finger reaches the sphere centre pts with normal bin nbin allowed."""
    idx = np.floor((pts - LO) / VOX).astype(np.int64)
    m = np.all((idx >= 0) & (idx < np.array(SHAPE)), axis=-1)
    idx = np.where(m[..., None], idx, 0)
    v = grid[idx[..., 0], idx[..., 1], idx[..., 2]]
    return m & ((v & BITS[nbin]) != 0)


def nearest_bin(n):
    return np.argmax(n @ NBINS.T, axis=-1)


def pen_frames(n_dirs=60, rolls=(0.0, 30.0, 60.0)):
    """Axis directions on the hemisphere a_x >= 0 (area-uniform) x rolls -> (O, R, 3, 3) columns (a, u, v)."""
    A = fib_sphere(n_dirs)
    A = A[A[:, 0] >= 0]
    out = []
    for a in A:
        ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        u0 = np.cross(a, ref); u0 /= np.linalg.norm(u0)
        v0 = np.cross(a, u0)
        row = []
        for r in rolls:
            c, s = math.cos(math.radians(r)), math.sin(math.radians(r))
            u, v = c * u0 + s * v0, -s * u0 + c * v0
            row.append(np.stack([a, u, v], 1))
        out.append(row)
    return np.array(out)


T_AX = np.array([-35.0, -17.5, 0.0, 17.5, 35.0])   # contact positions along the pen's long sides


def contact_set(F):
    """For one pen frame F (3x3, columns a, u, v): contact sphere-centre offsets from the pen centre, normals, and
    antipodal pairs (index pairs). Side faces +-u, +-v at T_AX; the two end faces."""
    a, u, v = F[:, 0], F[:, 1], F[:, 2]
    offs, nrm = [], []
    for n in (u, -u, v, -v):
        for t in T_AX:
            offs.append(15.0 * n + t * a + R_TIP * n)
            nrm.append(n)
    for n in (a, -a):
        offs.append(50.0 * n + R_TIP * n)
        nrm.append(n)
    nt = len(T_AX)
    pairs = [(k, nt + k) for k in range(nt)] + [(2 * nt + k, 3 * nt + k) for k in range(nt)] + [(4 * nt, 4 * nt + 1)]
    return np.array(offs), np.array(nrm), pairs


def holds(touch, pairs):
    """touch: (F, P, C) bool -> hold2 (P,) exists antipodal pair by two distinct fingers, ntouch (P,)."""
    nF = touch.shape[0]
    h2 = np.zeros(touch.shape[1], bool)
    for ka, kb in pairs:
        A, B = touch[:, :, ka], touch[:, :, kb]
        na, nb = A.sum(0), B.sum(0)
        both = (A & B).sum(0)
        # distinct fingers exist unless the only finger on each side is the same one
        h2 |= (na >= 1) & (nb >= 1) & ~((na == 1) & (nb == 1) & (both == 1))
    nt = touch.any(2).sum(0)
    return h2, nt


def evaluate(name, n=30000, seed=0):
    H = load(name)
    nF = len(H.fingers)
    rng = np.random.default_rng(seed)
    fp = gvb.footprint_mm(H)
    moving = [finger_section(H, i) == 0 for i in range(nF)]
    levels = HINGE_LEVELS if H.palm_joints else np.array([0.0])
    grids = {}
    tips_all = {}
    for li, h in enumerate(levels):
        for i in range(nF):
            if li > 0 and not moving[i]:
                grids[(li, i)] = grids[(0, i)]
                continue
            grids[(li, i)], tips_all[(li, i)] = finger_grid(H, i, n, rng, h, fp)

    # ---- per-finger stats: reach below the palm face, at the zero hinge level
    reach_x = []
    for i in range(nF):
        t = tips_all[(0, i)] if (0, i) in tips_all else tips_all[(0, i)]
        reach_x.append(float(np.percentile(t[:, 0], 99)))

    # ---- in-air holding: pen centres on a grid below the palm, hemisphere of axes x 3 rolls
    xs = np.arange(25.0, 136.0, 10.0)
    ys = np.arange(-100.0, 101.0, 10.0)
    P = np.stack(np.meshgrid(xs, ys, ys, indexing="ij"), -1).reshape(-1, 3)
    frames = pen_frames()
    O, Rn = frames.shape[:2]
    h2 = np.zeros((len(levels), len(P), O), bool)
    h3 = np.zeros_like(h2)
    h4 = np.zeros_like(h2)
    for o in range(O):
        for r in range(Rn):
            F = frames[o, r]
            ext = np.abs(F) @ PEN_HALF                       # half extent along palm axes
            fits = P[:, 0] - ext[0] >= PLATE_HALF_MM + PALM_CLEAR
            offs, nrm, pairs = contact_set(F)
            nb = nearest_bin(nrm)
            pts = P[:, None, :] + offs[None]                 # (P, C, 3)
            for li in range(len(levels)):
                touch = np.stack([lookup(grids[(li, i)], pts, nb[None, :]) for i in range(nF)])   # (F, P, C)
                a2, nt = holds(touch, pairs)
                a2 &= fits
                h2[li, :, o] |= a2
                h3[li, :, o] |= a2 & (nt >= 3)
                h4[li, :, o] |= a2 & (nt >= 4)
    # a pose holds if it holds at some hinge level (the hinge is one more joint the policy sets)
    H2, H3, H4 = h2.any(0), h3.any(0), h4.any(0)
    cov2, cov3, cov4 = H2.mean(1), H3.mean(1), H4.mean(1)
    best3 = int(np.argmax(cov3 + 1e-3 * cov2))
    # robustness: the best 3x3x3 neighbourhood mean of cov3 (the pen drifts in the grasp)
    g3 = cov3.reshape(len(xs), len(ys), len(ys))
    from scipy.ndimage import uniform_filter
    nb3 = uniform_filter(g3, size=3, mode="constant")
    res = {
        "name": name, "fingers": nF, "joints": len(dv.dofs(H)), "palm_cut": bool(H.palm_joints),
        "reach_x_p99": reach_x,
        "C2": float(cov2.max()), "C3": float(cov3.max()), "C4": float(cov4.max()),
        "C3_nbhd": float(nb3.max()),
        "best_centre_mm": P[best3].tolist(),
        "R2": float(H2.any(0).mean()), "R3": float(H3.any(0).mean()),
        "zone3_cm3": int((cov3 >= 0.5).sum()),          # 1 cm^3 cells where >= half the axes hold with 3 fingers
        "zone3_75_cm3": int((cov3 >= 0.75).sum()),
    }
    # which fingers matter: touch participation at the best centre (any axis, any roll, any level)
    res["fingers_used_at_best"] = fingers_used(grids, levels, nF, P[best3], frames)
    res["pickup"] = pickup(grids, levels, nF)
    return res


def fingers_used(grids, levels, nF, c, frames):
    used = np.zeros(nF, bool)
    for o in range(frames.shape[0]):
        for r in range(frames.shape[1]):
            F = frames[o, r]
            offs, nrm, pairs = contact_set(F)
            nb = nearest_bin(nrm)
            pts = (c[None, :] + offs)[None]
            for li in range(len(levels)):
                t = np.stack([lookup(grids[(li, i)], pts, nb[None, :]) for i in range(nF)])
                a2, nt = holds(t, pairs)
                if a2[0]:
                    used |= t[:, 0].any(1)
    return [int(i) for i in np.nonzero(used)[0]]


def pickup(grids, levels, nF):
    """Pen flat on the table at yaw psi (axis in the palm plane), the palm face g mm above its top. A pick needs two
    distinct fingers on opposite long sides (same station) or on the two ends, their capsule clear of the table
    (sphere centre at least R_TIP above it), at some wrist offset (y, z) and gap g."""
    ys = np.arange(-100.0, 101.0, 10.0)
    Y = np.stack(np.meshgrid(ys, ys, indexing="ij"), -1).reshape(-1, 2)
    psis = np.arange(0.0, 180.0, 10.0)
    gaps = [0.0, 5.0, 15.0, 30.0]
    ok2 = np.zeros(len(psis), bool)
    ok3 = np.zeros(len(psis), bool)
    for k, psi in enumerate(psis):
        a = np.array([0.0, math.cos(math.radians(psi)), math.sin(math.radians(psi))])
        u = np.cross(np.array([1.0, 0, 0]), a)     # horizontal side normal
        for g in gaps:
            cx = PLATE_HALF_MM + g + 15.0
            table = PLATE_HALF_MM + g + 30.0
            offs, nrm = [], []
            for n in (u, -u):
                for t in T_AX:
                    for hgt in (-10.0, -4.0):      # upper half of the side face (the capsule must clear the table)
                        offs.append(15.0 * n + t * a + R_TIP * n + np.array([hgt, 0, 0]))
                        nrm.append(n)
            for n in (a, -a):
                for hgt in (-10.0, -4.0):
                    offs.append(50.0 * n + R_TIP * n + np.array([hgt, 0, 0]))
                    nrm.append(n)
            offs, nrm = np.array(offs), np.array(nrm)
            m = len(T_AX) * 2
            pairs = [(j, m + j) for j in range(m)] + [(2 * m, 2 * m + 2), (2 * m + 1, 2 * m + 3)]
            clear = offs[:, 0] + cx <= table - R_TIP + 1.0
            nb = nearest_bin(nrm)
            P = np.concatenate([np.full((len(Y), 1), cx), Y], 1)
            pts = P[:, None] + offs[None]
            for li in range(len(levels)):
                t = np.stack([lookup(grids[(li, i)], pts, nb[None, :]) for i in range(nF)]) & clear[None, None]
                a2, nt = holds(t, pairs)
                ok2[k] |= a2.any()
                ok3[k] |= (a2 & (nt >= 3)).any()
    return {"pick2": float(ok2.mean()), "pick3": float(ok3.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", default=str(OUT / "metrics.json"))
    args = ap.parse_args()
    man = json.load(open(ASSETS / "manifest.json"))
    names = [h["name"] for h in man["hands"]]
    if args.only:
        want = set(args.only.split(","))
        names = [nm for nm in names if nm[-3:] in want or nm in want]
    c2 = {h["name"]: h["c2_ok"] for h in man["hands"]}
    rows = []
    for nm in names:
        t0 = time.time()
        r = evaluate(nm, args.n)
        r["c2_ok"] = c2[nm]
        rows.append(r)
        print(f"{nm} F{r['fingers']} J{r['joints']:2d} C2={r['C2']:.2f} C3={r['C3']:.2f} C4={r['C4']:.2f} "
              f"C3n={r['C3_nbhd']:.2f} R3={r['R3']:.2f} zone={r['zone3_cm3']:4d} pick={r['pickup']['pick2']:.2f}/"
              f"{r['pickup']['pick3']:.2f} best={r['best_centre_mm']} used={r['fingers_used_at_best']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
