"""Goal coverage: what fraction of pen repose goals can the hand hold (collision-free, 3 fingers) and rotate into,
given where the wrist can put the palm? (CPU only.)

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_pick_random/goal_cover.py 025,081 --jobs 10

The palm faces down; its front face is (wrist z - 12 mm) above the table, wrist z in [20, 200] mm. A goal is an axis
direction a (uniform) and a centre height h above the table, uniform between the orientation's floor (the pen's
lowest point 20 mm up) and 135 mm (12 cm above resting). Holding the pen at depth d below the palm face puts the face
at h + d, so d <= 188 - h (the wrist's ceiling) and h + d >= the hand's protrusion (pickup.py: the lowest the fingers
can be pulled up to; otherwise a finger is in the table).

For depths d = 35..135 mm (10 mm steps) take the 2 lateral pen centres where stage 1's 3-finger hold covers most axis
directions (ties: more fingers touching), and run stage 3 (clean.py) there over 30 axis directions x 3 rolls: per
direction, a collision-free 3-finger hold (H) and the collision-free rotations about the 6 axes perpendicular to the
pen (R). Then
    GC3  = P(goal holdable at some allowed depth),
    GROT = mean over goals of the fraction of the 6 rotation axes available at some allowed depth.
Writes goal_cover.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clean as CL  # noqa: E402
import screen as S  # noqa: E402
from screen import gvb, finger_section, PLATE_HALF_MM  # noqa: E402

DEPTHS = np.arange(35.0, 136.0, 10.0)
FACE_MAX = 200.0 - PLATE_HALF_MM
H_MAX = 15.0 + 120.0


def lateral_best(grids, levels, nF, frames, d, k=2):
    ys = np.arange(-100.0, 101.0, 10.0)
    P = np.stack(np.meshgrid([d], ys, ys, indexing="ij"), -1).reshape(-1, 3)
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
    score = h3.mean(1) + 0.01 * ntm.mean(1)
    order = np.argsort(-score, kind="stable")
    out = []
    for idx in order:
        if score[idx] <= 0:
            break
        if all(np.linalg.norm(P[idx] - P[j]) >= 20.0 for j in out):
            out.append(idx)
        if len(out) == k:
            break
    return [P[j] for j in out]


def run(name, prot, seed=41, n=30000):
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
            fds[(li, i)] = CL.finger_data(H, i, n, np.random.default_rng(seed + 300 + 11 * li + i), h, fp)
    frames = S.pen_frames()
    O = frames.shape[0]
    Hd = np.zeros((len(DEPTHS), O), bool)
    Rd = np.zeros((len(DEPTHS), O, CL.NW), bool)
    used = {}
    for di, d in enumerate(DEPTHS):
        cs = lateral_best(grids, levels, nF, frames, d)
        used[float(d)] = [c.tolist() for c in cs]
        for c in cs:
            h2, h3, rot = CL.at_centre(fds, levels, nF, c, frames, arrays=True)
            Hd[di] |= h3
            Rd[di] |= rot
    # goals: per axis direction, heights uniform on [floor, H_MAX]
    gc, gr, per_depth = [], [], Hd.mean(1).tolist()
    for o in range(O):
        a = frames[o, 0][:, 0]
        ext = np.abs(frames[o, 0]) @ S.PEN_HALF
        floor = 20.0 + ext[0]
        hs = np.linspace(floor, H_MAX, 41)
        cov_h, rot_h = [], []
        for h in hs:
            ok = (DEPTHS <= FACE_MAX - h) & (DEPTHS >= prot - h)
            cov_h.append(bool(Hd[ok, o].any()))
            rot_h.append(float(Rd[ok, o].any(0).mean()) if ok.any() else 0.0)
        gc.append(np.mean(cov_h))
        gr.append(np.mean(rot_h))
    return {"name": name, "GC3": float(np.mean(gc)), "GROT": float(np.mean(gr)), "hold_by_depth": per_depth,
            "rot_by_depth": Rd.mean((1, 2)).tolist(), "depths": DEPTHS.tolist(), "centres": used,
            "protrusion_face_mm": prot, "secs": time.time() - t0}


def main():
    names = [x if x.startswith("coarse") else f"coarse_s0_{x}" for x in sys.argv[1].split(",")]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 10
    pk = {r["name"]: r for r in json.load(open(HERE / "pickup.json"))}
    from multiprocessing import Pool
    rows = []
    with Pool(jobs) as pool:
        res = [pool.apply_async(run, (nm, max(pk[nm]["protrusion_mm"]))) for nm in names]
        for r in res:
            r = r.get()
            rows.append(r)
            print(f"{r['name']} GC3={r['GC3']:.2f} GROT={r['GROT']:.2f} hold_by_depth="
                  + " ".join(f"{int(d)}:{v:.2f}" for d, v in zip(r["depths"], r["hold_by_depth"]))
                  + f" prot={r['protrusion_face_mm']:.0f} ({r['secs']:.0f}s)", flush=True)
    json.dump(rows, open(HERE / "goal_cover.json", "w"), indent=1)


if __name__ == "__main__":
    main()
