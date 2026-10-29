"""Pickup, redone with the palm free to sit higher (screen.py's pickup only let the palm come 0-30 mm above the pen).

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_pick_random/pickup.py [--jobs 4]

Protrusion: for each finger, the least depth below the palm's front face its lowest point can be pulled up to
(over sampled joint vectors; points: every joint origin + 12 mm, the distal capsule's ends + 16.5 mm). The palm can
only come as close to the table as the most protruding finger allows: palm face to table >= max_i(protrusion_i).

Pick: the pen flat at yaw psi (18 yaws over 180 deg), the palm face g above its top for g in 0..150 mm (kept only if
the table is below every finger's retracted lowest point), wrist offset (y, z) on a 10 mm grid within +-100 mm; two
distinct fingers on opposite long faces (same station, upper half of the face so the capsule clears the table) or on
the two ends. pick2 = fraction of yaws pickable; pick3 = with a third finger also touching the pen.
Also: g_min (the lowest the palm face can be above the pen top), and pick2 restricted to g <= 30 mm.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screen as S  # noqa: E402
from screen import dv, gvb, finger_section, PLATE_HALF_MM  # noqa: E402

GAPS = [0.0, 5.0, 15.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0]


def protrusion(H, i, n=20000, seed=3):
    ds = dv.dofs(H)
    rng = np.random.default_rng(seed)
    q = np.zeros((n, len(ds)))
    for k, d in enumerate(ds):
        if d.finger == i or (d.kind == "palm" and finger_section(H, i) == 0):
            q[:, k] = rng.uniform(d.limits[0], d.limits[1], n)
    q = dv.tie(H, q, ds)
    pose = dv.fk(H, q)
    low = np.full(n, -np.inf)
    for j in range(len(H.fingers[i].joints)):
        low = np.maximum(low, pose.pre[i][:, j, 0, 3] / 1e-3 + 12.0)
    T = pose.links[i][:, -1]
    cap = dv.finger_geometry(H.fingers[i]).tip
    for p in (cap.p0, cap.p1):
        x = (T[:, :3, :3] @ p)[:, 0] + T[:, 0, 3] / 1e-3
        low = np.maximum(low, x + cap.radius)
    return float(max(low.min(), PLATE_HALF_MM))


def pick(name, n=30000, seed=0):
    H = S.load(name)
    nF = len(H.fingers)
    rng = np.random.default_rng(seed)
    fp = gvb.footprint_mm(H)
    moving = [finger_section(H, i) == 0 for i in range(nF)]
    levels = S.HINGE_LEVELS if H.palm_joints else np.array([0.0])
    grids = {}
    for li, h in enumerate(levels):
        for i in range(nF):
            if li > 0 and not moving[i]:
                grids[(li, i)] = grids[(0, i)]
                continue
            grids[(li, i)], _ = S.finger_grid(H, i, n, rng, h, fp)
    prot = [protrusion(H, i) for i in range(nF)]
    floor = max(prot)
    gaps = [g for g in GAPS if PLATE_HALF_MM + g + 30.0 >= floor - 1.0]
    ys = np.arange(-100.0, 101.0, 10.0)
    Y = np.stack(np.meshgrid(ys, ys, indexing="ij"), -1).reshape(-1, 2)
    psis = np.arange(0.0, 180.0, 10.0)
    ok2 = np.zeros(len(psis), bool)
    ok3 = np.zeros(len(psis), bool)
    ok2_low = np.zeros(len(psis), bool)
    for k, psi in enumerate(psis):
        a = np.array([0.0, math.cos(math.radians(psi)), math.sin(math.radians(psi))])
        u = np.cross(np.array([1.0, 0, 0]), a)
        for g in gaps:
            cx = PLATE_HALF_MM + g + 15.0
            table = PLATE_HALF_MM + g + 30.0
            offs, nrm = [], []
            for nv in (u, -u):
                for t in S.T_AX:
                    for hgt in (-10.0, -4.0):
                        offs.append(15.0 * nv + t * a + S.R_TIP * nv + np.array([hgt, 0, 0]))
                        nrm.append(nv)
            for nv in (a, -a):
                for hgt in (-10.0, -4.0):
                    offs.append(50.0 * nv + S.R_TIP * nv + np.array([hgt, 0, 0]))
                    nrm.append(nv)
            offs, nrm = np.array(offs), np.array(nrm)
            m = len(S.T_AX) * 2
            pairs = [(j, m + j) for j in range(m)] + [(2 * m, 2 * m + 2), (2 * m + 1, 2 * m + 3)]
            clear = offs[:, 0] + cx <= table - S.R_TIP + 1.0
            nb = S.nearest_bin(nrm)
            P = np.concatenate([np.full((len(Y), 1), cx), Y], 1)
            pts = P[:, None] + offs[None]
            for li in range(len(levels)):
                t = np.stack([S.lookup(grids[(li, i)], pts, nb[None, :]) for i in range(nF)]) & clear[None, None]
                a2, nt = S.holds(t, pairs)
                ok2[k] |= a2.any()
                ok3[k] |= (a2 & (nt >= 3)).any()
                if g <= 30.0:
                    ok2_low[k] |= a2.any()
    return {"name": name, "protrusion_mm": prot, "g_min": (gaps[0] if gaps else None),
            "pick2": float(ok2.mean()), "pick3": float(ok3.mean()), "pick2_low": float(ok2_low.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(HERE / "pickup.json"))
    a = ap.parse_args()
    man = json.load(open(S.ASSETS / "manifest.json"))
    names = [h["name"] for h in man["hands"]]
    if a.only:
        names = [x if x.startswith("coarse") else f"coarse_s0_{x}" for x in a.only.split(",")]
    from multiprocessing import Pool
    rows = []
    with Pool(a.jobs) as pool:
        for r in pool.imap(pick, names):
            rows.append(r)
            print(f"{r['name']} g_min={r['g_min']} pick2={r['pick2']:.2f} pick3={r['pick3']:.2f} "
                  f"low={r['pick2_low']:.2f} prot={[round(x) for x in r['protrusion_mm']]}", flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
