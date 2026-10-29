"""Draw collision-free 3-finger grasps that stage 3 (clean.py) finds, for a few pen axis directions, palm down.

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_pick_random/grasp_viz.py 025 081

Per hand: the stage-3 best centre (clean_seed17_8c.json, else clean.json); the pen axis horizontal along palm y,
horizontal along palm z, tilted 45 deg, and vertical (along the palm normal). For each, the first collision-free
antipodal pair + clear third finger found (same tests as clean.py); the other fingers at their zero pose.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clean as CL  # noqa: E402
import render as RD  # noqa: E402
import screen as S  # noqa: E402
from screen import dv, gvb, finger_section  # noqa: E402


def frame(a, roll=0.0):
    a = np.asarray(a, float) / np.linalg.norm(a)
    ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, ref); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    return np.stack([a, c * u + s * v, -s * u + c * v], 1)


def find(H, fds, levels, c, F):
    nF = len(H.fingers)
    offs, nrm, pairs = S.contact_set(F)
    for li in range(len(levels)):
        fl = [fds[(li, i)] for i in range(nF)]
        cd = [[CL.cands(fl[i], c + offs[k], nrm[k], c, F) for k in range(len(offs))] for i in range(nF)]
        for ka, kb in pairs:
            for i in range(nF):
                for j in range(nF):
                    A, B = cd[i][ka], cd[j][kb]
                    if i == j or A is None or B is None:
                        continue
                    clr = CL.ff_clear(fl[i], A["sid"], fl[j], B["sid"])
                    for a_, b_ in zip(*np.nonzero(clr)):
                        for m in range(nF):
                            if m in (i, j):
                                continue
                            for k in range(len(offs)):
                                Cm = cd[m][k]
                                if Cm is None or k in (ka, kb):
                                    continue
                                ok = CL.ff_clear(fl[m], Cm["sid"], fl[i], A["sid"][[a_]])[:, 0] & \
                                    CL.ff_clear(fl[m], Cm["sid"], fl[j], B["sid"][[b_]])[:, 0]
                                if ok.any():
                                    q = np.zeros(len(dv.dofs(H)))
                                    ds = dv.dofs(H)
                                    for f_, sid in ((i, A["sid"][a_]), (j, B["sid"][b_]), (m, Cm["sid"][np.argmax(ok)])):
                                        for col, d in enumerate(ds):
                                            if d.finger == f_:
                                                q[col] = fl[f_]["q"][sid, col]
                                    if H.palm_joints:
                                        q[0] = levels[li]
                                    return q, (i, j, m)
    return None, None


def main():
    hands = [h if h.startswith("coarse") else f"coarse_s0_{h}" for h in sys.argv[1:]]
    src = HERE / "clean_seed17_8c.json"
    best = {r["name"]: r["best_centre"] for r in json.load(open(src))}
    best0 = {r["name"]: r["best_centre"] for r in json.load(open(HERE / "clean.json"))}
    tests = [("axis along palm y", (0, 1, 0)), ("axis along palm z", (0, 0, 1)),
             ("tilted 45 deg", (1, 1, 0)), ("vertical", (1, 0, 0))]
    for nm in hands:
        H = S.load(nm)
        nF = len(H.fingers)
        fp = gvb.footprint_mm(H)
        moving = [finger_section(H, i) == 0 for i in range(nF)]
        levels = S.HINGE_LEVELS if H.palm_joints else np.array([0.0])
        fds = {}
        for li, h in enumerate(levels):
            for i in range(nF):
                if li > 0 and not moving[i]:
                    fds[(li, i)] = fds[(0, i)]
                    continue
                fds[(li, i)] = CL.finger_data(H, i, 30000, np.random.default_rng(100 + 11 * li + i), h, fp)
        c = np.array(best.get(nm) or best0[nm])
        fig = plt.figure(figsize=(16, 4.6))
        for t, (lab, a) in enumerate(tests):
            q, who = None, None
            for roll in (0.0, 30.0, 60.0):
                F = frame(a, roll)
                q, who = find(H, fds, levels, c, F)
                if q is not None:
                    break
            ax = fig.add_subplot(1, 4, t + 1, projection="3d")
            RD.draw(ax, H, q, pen=(c, F), view=(12, -60))
            ax.set_title(f"{lab}: " + (f"fingers {who}" if q is not None else "no clean grasp found"), fontsize=9)
        fig.suptitle(f"{nm}: collision-free 3-finger grasps at centre {c.tolist()} mm below/along the palm "
                     f"(palm down; other fingers at zero pose)", fontsize=10)
        fig.tight_layout()
        out = RD.PLOTS / f"{nm}_grasps.png"
        fig.savefig(out, dpi=105)
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
